import re
import torch
import numpy as np
from fasttext import load_model

from utils.cuda import safe_cuda_or_cpu
from .base import BaseExtractor


class FastTextDocEmbeddingExtractor(BaseExtractor):
    def __init__(self, ft_model_path):
        self.model = load_model(ft_model_path)

    def embed_doc(self, document):
        no_whitespace = re.sub(r'\s+', '', document)
        return self.model.get_sentence_vector(no_whitespace)

    def __call__(self, input_text):
        embeddings = torch.Tensor([self.embed_doc(line) for line in input_text])
        return safe_cuda_or_cpu(embeddings)

    def vector_length(self):
        return self.model.get_dimension()


class FastTextTokenEmbeddingExtractor(BaseExtractor):
    def __init__(self, ft_model_path, max_length=25):
        self.model = load_model(ft_model_path)
        self.max_length = max_length

    def tokenize_doc(self, document):
        no_whitespace = re.sub(r'\s+', ' ', document)
        tokens, _ = self.model.get_line(no_whitespace)
        end_token = tokens[-1]
        del tokens[self.max_length:]
        tokens[-1] = end_token
        return tokens

    def embed_tokens(self, document_tokens):
        return [self.model.get_word_vector(token) for token in document_tokens]

    def pad_document(self, document_embeddings, max_length):
        doc_length = len(document_embeddings)
        needed_padding = max_length - doc_length
        if needed_padding:
            vector_length = self.vector_length()
            zero_vector = np.zeros(vector_length)
            for _ in range(needed_padding):
                document_embeddings.append(zero_vector)
        return document_embeddings

    def embed_and_clamp_tokens(self, document_tokens, max_input_length):
        embedded = self.embed_tokens(document_tokens)
        padded = self.pad_document(embedded, max_input_length)
        return padded

    def embed(self, tokenized_documents):
        max_input_length = max(map(len, tokenized_documents))
        processed_docs = []
        for document_tokens in tokenized_documents:
            embedded = self.embed_and_clamp_tokens(document_tokens, max_input_length)
            processed_docs.append(embedded)
        return torch.Tensor(processed_docs)

    def __call__(self, documents):
        embeddings = self.embed([self.tokenize_doc(doc) for doc in documents])
        return safe_cuda_or_cpu(embeddings)

    def vector_length(self):
        return self.model.get_dimension()
