from .tfidf import TfidfExtractor
from .fasttext import FastTextTokenEmbeddingExtractor
from .bert import BertEmbeddingExtractor


def extractor_factory(extractor_type, dataset=None, **kwargs):
    if extractor_type == 'fasttext':
        return FastTextTokenEmbeddingExtractor(kwargs['ft_model_path'], 
                                               max_length=kwargs['max_length'])
    if extractor_type == 'tfidf':
        return TfidfExtractor(dataset, num_words=kwargs['num_words'])
    if extractor_type == 'bert':
        return BertEmbeddingExtractor(kwargs['bert_model'], 
                                      freeze_output=kwargs['freeze_output'], 
                                      max_length=kwargs['max_length'])
    return None

