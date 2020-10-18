import os 
import json
import logging
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from text import extractor_factory
from models import model_factory

from utils.random import set_seed
from utils.split import random_splits, column_splits
from utils.evaluate import classification_report, discretize_output, find_thresholds
from utils.generators import SizedCallableWrapper, SizedBatchWrapper


from config import DataConfig, ExtractorConfig, ModelConfig, ExperimentConfig

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger('EMOTION')

PROJECT_ROOT_PATH = '.' # str(Path(__file__).parent.absolute().parent)
CONFIG_PATH = '{}/configs/GoEmotions/config.json'.format(PROJECT_ROOT_PATH)


def load_dataset(raw_path):
    LOGGER.debug('Path loading: {}'.format(raw_path))
    return pd.read_parquet(raw_path)


def preprocess_dataset(dataset, data_format, target_column, text_column):
    if data_format == 'vent':
        task_data = dataset[[text_column, target_column]].dropna()
    elif data_format == 'goemotions':
        task_data = dataset[[text_column, target_column, 'split']].dropna()
    else:
        task_data = dataset[[text_column, target_column]].dropna()
    return task_data


def save_dataset(dataset, path):
    dataset.to_parquet(path)


def load_splits(raw_path, 
                cache_path, 
                split_names, 
                split_portions,
                split_mode,
                dataset_format,
                target_column,
                text_column,
                save_cache=True,
                ignore_cache=False):
    LOGGER.debug('Path loading: {}'.format(raw_path))
    # Load the cache if it is available
    if split_mode == 'column':
        portions_string = 'cols'
    else:
        portions_string = '-'.join([str(pt * 100)[:5] for pt in split_portions])
    if os.path.exists(cache_path) and not ignore_cache:
        try:
            splits = []
            for split in split_names:
                split_path = '{}{}-{}.parquet'.format(cache_path, split, portions_string)
                splits.append(load_dataset(split_path))
            return splits
        except OSError:
            LOGGER.debug('Could not load cache, loading and preprocessing from raw...')

    # Otherwise load the full dataset and split it
    full_dataset = load_dataset(raw_path)
    clean_dataset = preprocess_dataset(full_dataset, dataset_format, target_column, text_column)

    if split_mode == 'random':
        all_splits = random_splits(clean_dataset, split_portions)
    elif split_mode == 'column':
        all_splits = column_splits(clean_dataset)
    else:
        raise RuntimeError('Unknown split mode "{}".'.format(split_mode))

    # Dump the cache
    if save_cache:
        if not os.path.exists(cache_path):
            os.mkdir(cache_path)
        for split, dataset in zip(split_names, all_splits):
            split_path = '{}{}-{}.parquet'.format(cache_path, split, portions_string)
            save_dataset(dataset, split_path)

    return all_splits


def log_report(report):
    for label, stats in report['labels'].items():
        p = stats['precision']
        r = stats['recall']
        f = stats['f1']
        s = stats['support']
        LOGGER.info('{}\tPrecision: {}\tRecall: {}\tF1-score: {}\tSupport: {}'.format(label, p, r, f, s))
    LOGGER.info('')
    LOGGER.info('Mean macro F1-score: {}'.format(report['macro_f1']))
    LOGGER.info('Mean micro F1-score: {}'.format(report['micro_f1']))


def load_config(config_path):
    with open(config_path, 'r') as f:
        as_string = f.read()
        as_dict = json.loads(as_string)
        LOGGER.info('Loaded configuration file "{}" with contents: \n{}'.format(config_path, as_string))
        parsed_config = ExperimentConfig.from_dict(as_dict)
        return parsed_config


def emotion_experiment(experiment_config):
    data_config = experiment_config.data_config
    extractor_config = experiment_config.extractor_config
    model_config = experiment_config.model_config
    if experiment_config.seed is not None:
        set_seed(experiment_config.seed)
        LOGGER.info('Seeding all libraries with seed "{}"...'.format(experiment_config.seed))

    splits = load_splits(**data_config._asdict())
    LOGGER.info('Splits are ready...')

    y_arrays = [np.asarray(split[data_config.target_column].to_list(), dtype=np.float32) 
                for split in splits]
    num_labels = np.unique(y_arrays[0]).size if len(y_arrays[0].shape) == 1 else y_arrays[0].shape[-1]
    LOGGER.info('Labels are ready, with a total number of {} possible targets...'.format(num_labels))

    extractor = extractor_factory(extractor_config.ex_type, 
                                  dataset=splits[0][data_config.text_column],
                                  **extractor_config.ex_args)
    X_batches = [SizedBatchWrapper(split[data_config.text_column], 
                                   model_config.batch_size) 
                 for split in splits]
    y_batches = [SizedBatchWrapper(y_split, model_config.batch_size) for y_split in y_arrays]
    X_splits  = [SizedCallableWrapper(X_split, extractor) for X_split in X_batches]
    LOGGER.info('Feature extractor is ready...')

    model = model_factory(model_config.model_name)
    clf = model(model_config.problem_type, input_size=extractor.vector_length(), output_size=num_labels, **model_config.model_conf)
    clf.fit(X_splits[0], y_batches[0])
    LOGGER.info('Training is done...')

    results = {}
    thresholds = []
    for i, data in enumerate(zip(X_splits, y_arrays)):
        X_split, y_split = data
        split_name = data_config.split_names[i]

        # In multilabel, report the best possible thresholds
        # In any split other than test, use its own, otherwise use the previous split's
        model_output = clf.predict(X_split)
        if model_config.problem_type == 'multilabel':
            split_thresholds, col_scores = find_thresholds(model_output, y_split)
            thresholds.append(split_thresholds)
            thresholds_to_use = split_thresholds if split_name != 'test' else thresholds[-1]
            y_p_split = discretize_output(model_output, model_config.problem_type, thresholds_to_use)
        else:
            y_p_split = discretize_output(model_output, model_config.problem_type)
        split_report = classification_report(y_split, y_p_split, experiment_config.label_names)
        results[split_name] = split_report

        LOGGER.info('Classification report -- {}'.format(split_name.capitalize()))
        LOGGER.info('#################################')
        LOGGER.info('')
        log_report(split_report)
        LOGGER.info('')
        LOGGER.info('')
    return results


def store_results(experiment_config, results):
    result_dict = {'config': experiment_config._as_flat_dict(),
                   'results': results}
    result_as_json = json.dumps(result_dict, indent=2)

    # Prepare the path for writing
    output_path = experiment_config.output_path
    Path(output_path).mkdir(parents=True, exist_ok=True)
    result_path = experiment_config.output_file()
    LOGGER.info('Storing the results in path: {}'.format(result_path))
    with open(result_path, 'w') as f:
        f.write(result_as_json)


def parse_arguments():
    main_args = argparse.ArgumentParser()
    main_args.add_argument('-c', '--config-path', help='Path to the configuration file to run.', type=str, default=CONFIG_PATH)
    return main_args.parse_args()


if __name__ == "__main__":
    arguments = parse_arguments()
    LOGGER.info('Running with args: {}'.format(arguments))
    config = load_config(arguments.config_path)
    if not os.path.exists(config.output_file()):
        results = emotion_experiment(config)
        store_results(config, results)
    else:
        LOGGER.info('Skipping experiment as "{}" already exists!'.format(config.output_file()))


