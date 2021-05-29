import re
from transformers import XLMRobertaTokenizer, XLMRobertaModel
import torch

from utils.cuda import safe_cuda_or_cpu
from .base import BaseExtractor


class XLMRobertaEmbeddingExtractor(BaseExtractor):
    def __init__(self, model_name, freeze_output=False, max_length=30):
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
        self.model = safe_cuda_or_cpu(XLMRobertaModel.from_pretrained(model_name))
        self.freeze_output = freeze_output
        self.max_length = max_length

    def __call__(self, input_text):
        input_ids = self.tokenizer(input_text, 
                                   return_tensors="pt", 
                                   truncation=True, 
                                   padding=True, 
                                   max_length=self.max_length)["input_ids"]
        input_ids = safe_cuda_or_cpu(input_ids)
        outputs = self.model(input_ids)[0]
        if self.freeze_output:
            outputs = outputs.detach()
        return safe_cuda_or_cpu(outputs)

    def vector_length(self):
        return self.model.config.hidden_size
