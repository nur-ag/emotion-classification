import torch
from sklearn.feature_extraction.text import TfidfVectorizer

from .base import BaseExtractor


class TfidfExtractor(BaseExtractor):
    def __init__(self, input_dataset, num_words):
        self.num_words = num_words
        self.vectorizer = TfidfVectorizer(stop_words='english', max_features=num_words)
        self.vectorizer.fit(input_dataset)

    def __call__(self, input_text):
        output = self.vectorizer.transform(input_text)
        return output

    def vector_length(self):
        return self.num_words
