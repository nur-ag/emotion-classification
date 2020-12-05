import torch
from sklearn.feature_extraction.text import CountVectorizer

from .base import BaseExtractor


class BOWExtractor(BaseExtractor):
    def __init__(self, input_dataset, num_words):
        self.num_words = num_words
        self.vectorizer = CountVectorizer(stop_words='english', max_features=num_words)
        self.vectorizer.fit(input_dataset)

    def __call__(self, input_text):
        output = self.vectorizer.transform(input_text)
        return output

    def vector_length(self):
        return self.num_words
