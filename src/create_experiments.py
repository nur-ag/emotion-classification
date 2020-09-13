import os
import json
import hashlib
from itertools import product, chain

from config import DataConfig, ExtractorConfig, ModelConfig, ExperimentConfig

PROBLEM_TYPE = 'multilabel'
CLASSIC_BATCH_SIZE = 100000
NEURAL_BATCH_SIZE = 16

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

LSTM_FIELDS = ['hidden_size', 'num_layers', 'num_epochs', 'dropout', 'learning_rate', 'epsilon', 'bidirectional', 'optimizer', 'aggregation_mode']

MAX_LENGTH = 25

BERT_FREEZE = [True, False]
BERT_MODEL = ['bert-base-uncased', 'bert-base-cased']

FASTTEXT_CONFIG = ExtractorConfig(ex_type='fasttext', ex_args={'ft_model_path': 'cc.en.300.bin'})

TFIDF_NUM_WORDS = [5000, 25000, 125000]

NB_SMOOTHING = [1 / (10 ** i) for i in range(0, 10)]
SGD_NUM_EPOCHS = [1, 10, 50, 100]
SGD_ALPHA = [0.00001, 0.0001, 0.001, 0.01, 0.1, 1.0]

def build_tfidf_pairs(problem_type, batch_size):
    ex_configs = []
    for words in TFIDF_NUM_WORDS:
        cfg = ExtractorConfig(ex_type='tfidf', 
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
    return product(ex_configs, model_configs)


def build_neural_pairs(problem_type, batch_size):
    ex_configs = [FASTTEXT_CONFIG]
    for freeze, model in product(BERT_FREEZE, BERT_MODEL):
        cfg = ExtractorConfig(ex_type='bert', 
                              ex_args={'bert_model': model,
                                       'freeze_output': freeze,
                                       'max_length': MAX_LENGTH})
        ex_configs.append(cfg)

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
neural_pairs = build_neural_pairs(PROBLEM_TYPE, NEURAL_BATCH_SIZE)
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


