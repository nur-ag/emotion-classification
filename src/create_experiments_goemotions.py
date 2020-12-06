import os
import json
import hashlib
from itertools import product, chain

from config import DataConfig, ExtractorConfig, ModelConfig, ExperimentConfig

PROBLEM_TYPE = 'multilabel'
CLASSIC_BATCH_SIZE = 100000
CLASSIC_EXTRACTOR_TYPES = ['bow', 'tfidf']
BERT_BATCH_SIZE = 64
FASTTEXT_BATCH_SIZE = 1024

CONFIG_ROOT = 'configs/GoEmotions/'
CLASSIC_OUTPUT_PATH = 'output/GoEmotions/classic/'
NEURAL_OUTPUT_PATH = 'output/GoEmotions/neural/'

DATA_CONFIG = DataConfig(raw_path='preprocessed/GoEmotions.parquet', 
                         cache_path='preprocessed/GoEmotions-split-cache/', 
                         split_names=['train', 'valid', 'test'], 
                         split_portions=[0.8, 0.1, 0.1], 
                         split_mode='column', 
                         dataset_format='goemotions', 
                         target_column='emotions_label', 
                         text_column='text')

LABEL_NAMES = [l.strip() for l in open('data/GoEmotionsNames.txt')]

DNN_HIDDEN_SIZES = [100]
DNN_NUM_LAYERS = [1, 2, 3]
DNN_NUM_EPOCHS = [30, 40, 50, 60]
DNN_LEARNING_RATE = [0.01, 0.001, 0.0001]
DNN_EPSILON = [1e-5, 1e-6, 1e-7]
DNN_ACTIVATION = ['elu', 'tanh']
DNN_POOL_MODE = ['attention', 'max']
DNN_OPTIMIZER = ['adamw']

DNN_FIELDS = ['hidden_size', 'num_layers', 'num_epochs', 'learning_rate', 'epsilon', 'activation', 'pool_mode', 'optimizer']

LSTM_HIDDEN_SIZES = [100]
LSTM_NUM_LAYERS = [1, 2]
LSTM_NUM_EPOCHS = [30, 40, 50, 60]
LSTM_LEARNING_RATE = [0.01, 0.001, 0.0001]
LSTM_EPSILON = [1e-5, 1e-6, 1e-7]
LSTM_BIDIRECTIONAL = [True, False]
LSTM_OPTIMIZER = ['adamw']
LSTM_AGG_MODE = ['attention', 'mean', 'max']

LSTM_FIELDS = ['hidden_size', 'num_layers', 'num_epochs', 'learning_rate', 'epsilon', 'bidirectional', 'optimizer', 'aggregation_mode']

MAX_LENGTH = 25

BERT_FREEZE = [True, False]
BERT_MODEL = ['bert-base-uncased', 'bert-base-cased']

FASTTEXT_CONFIG = ExtractorConfig(ex_type='fasttext', ex_args={'ft_model_path': 'cc.en.300.bin', 'max_length': MAX_LENGTH})

TFIDF_NUM_WORDS = [5000, 10000, 20000, 40000]

NB_SMOOTHING = [1 / (10 ** i) for i in range(0, 10)]
SGD_NUM_EPOCHS = [1, 10, 50, 100]
SGD_ALPHA = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]

RF_TREES_PER_BATCH = [1000, 2000, 3000]
RF_MAX_DEPTH = [3, 4, 5]
RF_MAX_FEATURES = [0.05, 0.1, 0.2, 0.4]
RF_CRITERION = 'entropy'

def build_tfidf_pairs(problem_type, batch_size):
    ex_configs = []
    for words in TFIDF_NUM_WORDS:
        for ex_type in CLASSIC_EXTRACTOR_TYPES:
            cfg = ExtractorConfig(ex_type=ex_type, 
                                  ex_args={'num_words': words})
            ex_configs.append(cfg)

    model_configs = []
    for smoothing in NB_SMOOTHING:
        nb = ModelConfig(model_name='naivebayes', 
                         problem_type=problem_type, 
                         batch_size=batch_size, 
                         model_conf={'alpha': smoothing})
        model_configs.append(nb)

    for epochs in SGD_NUM_EPOCHS:
        for alpha in SGD_ALPHA:
            logreg = ModelConfig(model_name='sgd', 
                                 problem_type=problem_type, 
                                 batch_size=batch_size,
                                 model_conf={'loss': 'log',
                                             'tol': 0.001,
                                             'max_iter': epochs,
                                             'alpha': alpha})
            model_configs.append(logreg)

    for num_trees in RF_TREES_PER_BATCH:
        for depth in RF_MAX_DEPTH:
            for max_features in RF_MAX_FEATURES:
                rf = ModelConfig(model_name='rf', 
                                 problem_type=problem_type, 
                                 batch_size=batch_size,
                                 model_conf={'criterion': RF_CRITERION,
                                             'max_depth': depth,
                                             'n_estimators_per_chunk': num_trees,
                                             'max_features': max_features})
                model_configs.append(rf)
    return product(ex_configs, model_configs)


def build_bert_extractors():
    ex_configs = []
    for freeze, model in product(BERT_FREEZE, BERT_MODEL):
        cfg = ExtractorConfig(ex_type='bert', 
                              ex_args={'bert_model': model,
                                       'freeze_output': freeze,
                                       'max_length': MAX_LENGTH})
        ex_configs.append(cfg)
    return ex_configs


def build_neural_pairs(ex_configs, problem_type, batch_size):
    model_configs = []
    for input_tuple in product(DNN_HIDDEN_SIZES, DNN_NUM_LAYERS, 
                               DNN_NUM_EPOCHS, DNN_LEARNING_RATE, 
                               DNN_EPSILON, DNN_ACTIVATION, 
                               DNN_POOL_MODE, DNN_OPTIMIZER):
        dnn = ModelConfig(model_name='dnnpool', 
                          problem_type=problem_type, 
                          batch_size=batch_size, 
                          model_conf=dict(zip(DNN_FIELDS, input_tuple)))
        model_configs.append(dnn)
    for input_tuple in product(LSTM_HIDDEN_SIZES, LSTM_NUM_LAYERS,
                               LSTM_NUM_EPOCHS, LSTM_LEARNING_RATE,
                               LSTM_EPSILON, LSTM_BIDIRECTIONAL,
                               LSTM_OPTIMIZER, LSTM_AGG_MODE):
        lstm = ModelConfig(model_name='lstm', 
                           problem_type=problem_type, 
                           batch_size=batch_size, 
                           model_conf=dict(zip(LSTM_FIELDS, input_tuple)))
        model_configs.append(lstm)
    return product(ex_configs, model_configs)


def build_experiments(pairs, output_path, seed=123):
    experiment_configs = []
    for ex, model in pairs:
        experiment = ExperimentConfig(data_config=DATA_CONFIG,
                                      extractor_config=ex,
                                      model_config=model,
                                      label_names=LABEL_NAMES,
                                      seed=seed,
                                      output_path=output_path)
        experiment_configs.append(experiment)
    return experiment_configs


def store_experiments(pairs, base_path):
    for config in pairs:
        as_dict = config._as_flat_dict()
        as_hash = config.hash()
        as_json = json.dumps(as_dict, indent=2)
        config_path = '{}{}.json'.format(base_path, as_hash)
        with open(config_path, 'w') as f:
            f.write(as_json)


classic_pairs = build_tfidf_pairs(PROBLEM_TYPE, CLASSIC_BATCH_SIZE)
bert_exs = build_bert_extractors()
bert_pairs = build_neural_pairs(bert_exs, PROBLEM_TYPE, BERT_BATCH_SIZE)
fasttext_pairs = build_neural_pairs([FASTTEXT_BATCH_SIZE], PROBLEM_TYPE, FASTTEXT_BATCH_SIZE)
neural_pairs = chain(bert_pairs, fasttext_pairs)

classic_exp = build_experiments(classic_pairs, CLASSIC_OUTPUT_PATH)
neural_exp = build_experiments(neural_pairs, NEURAL_OUTPUT_PATH)

classic_path = '{}classic/'.format(CONFIG_ROOT)
neural_path = '{}neural/'.format(CONFIG_ROOT)

if not os.path.exists(classic_path):
    os.mkdir(classic_path)

if not os.path.exists(neural_path):
    os.mkdir(neural_path)

store_experiments(classic_exp, classic_path)
store_experiments(neural_exp, neural_path)

