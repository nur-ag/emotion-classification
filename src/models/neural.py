import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils.cuda import safe_cuda_or_cpu


class LSTMClassifier(nn.Module):
    def __init__(self, 
                 criterion, 
                 input_size=300, 
                 output_size=9,
                 hidden_size=200, 
                 num_layers=1,
                 num_epochs=1, 
                 dropout=0.0, 
                 bidirectional=True,
                 aggregation_mode='attention'):
        super(LSTMClassifier, self).__init__()
        self.criterion = criterion
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.num_epochs = num_epochs
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.aggregation_mode = aggregation_mode
        self.model = safe_cuda_or_cpu(nn.LSTM(input_size, 
                                      hidden_size, 
                                      num_layers, 
                                      batch_first=True,
                                      dropout=dropout, 
                                      bidirectional=bidirectional))
        lstm_hidden_dim = hidden_size * (1 + self.bidirectional)
        if self.aggregation_mode == 'attention':
            self.attention_scorer = safe_cuda_or_cpu(nn.Linear(lstm_hidden_dim, 1))
        else:
            self.attention_scorer = None

        if self.aggregation_mode == 'slice':
            self.to_class = safe_cuda_or_cpu(nn.Linear(2 * lstm_hidden_dim, output_size))
        else:
            self.to_class = safe_cuda_or_cpu(nn.Linear(lstm_hidden_dim, output_size))
        self.optimizer = optim.Adam(self.parameters())
        self.softmax = nn.Softmax(dim=1)
        safe_cuda_or_cpu(self)

    def compute_attention(self, tensor):
        batch_len, seq_len, emb_len = tensor.shape
        scored = self.attention_scorer(tensor.reshape(batch_len * seq_len, emb_len))
        attention = self.softmax(scored.reshape(batch_len, seq_len, 1))
        scored = (attention * tensor).sum(axis=1).reshape(batch_len, emb_len)
        return scored

    def aggregate(self, tensor):
        batch_size = tensor.shape[0]
        if self.aggregation_mode == 'slice':
            return tensor[:, [0, -1], :].reshape(batch_size, -1)
        if self.aggregation_mode == 'sum':
            return tensor.sum(axis=1).reshape(batch_size, -1)
        if self.aggregation_mode == 'mean':
            return tensor.mean(axis=1).reshape(batch_size, -1)
        if self.aggregation_mode == 'attention':
            return self.compute_attention(tensor)
        raise ValueError('Unknown aggregation function "{}".'.format(self.aggregation_mode))

    def forward(self, X):
        encoded, _ = self.model(X)
        aggregate = self.aggregate(encoded)
        output = self.to_class(aggregate)
        return output

    def _fit_batch(self, X, y):
        self.train()
        self.optimizer.zero_grad()
        y_pred = self(X)
        loss = self.criterion(y_pred, y)
        loss.backward()
        self.optimizer.step()

    def fit(self, X, y):
        total_batches = min(len(X), len(y))
        for epoch in tqdm(range(self.num_epochs), desc='Epoch'):
            for X_batch, y_batch in tqdm(zip(X, y), total=total_batches, desc='Batch'):
                X_batch = safe_cuda_or_cpu(X_batch) # It should already be a tensor!
                y_batch = safe_cuda_or_cpu(torch.LongTensor(y_batch))
                self._fit_batch(X_batch, y_batch)
        return self

    def predict(self, X):
        results = []
        num_instances = 0
        with torch.no_grad():
            self.eval()
            for X_batch in tqdm(X, desc='Predict'):
                X_batch = safe_cuda_or_cpu(X_batch)
                num_instances += len(X_batch)
                results.extend(self(X_batch))
        output_logits = torch.cat(results).reshape(num_instances, -1).detach().cpu().numpy()
        return output_logits.argmax(axis=-1)
