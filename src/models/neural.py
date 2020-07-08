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
                 bidirectional=True):
        super(LSTMClassifier, self).__init__()
        self.criterion = criterion
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.num_epochs = num_epochs
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.model = safe_cuda_or_cpu(nn.LSTM(input_size, 
                                      hidden_size, 
                                      num_layers, 
                                      batch_first=True,
                                      dropout=dropout, 
                                      bidirectional=bidirectional))
        self.to_class = safe_cuda_or_cpu(nn.Linear(2 * hidden_size * (1 + self.bidirectional), output_size))
        self.optimizer = optim.Adam(self.parameters())
        safe_cuda_or_cpu(self)

    def forward(self, X):
        encoded, _ = self.model(X)
        batch_size = X.shape[0]
        aggregate = encoded[:, [0, -1], :].reshape(batch_size, -1)
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
