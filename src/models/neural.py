import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import numpy as np
from tqdm import tqdm

from utils.cuda import safe_cuda_or_cpu


ACTIVATIONS = {
    'elu': nn.ELU, 
    'relu': nn.ReLU, 
    'gelu': nn.GELU, 
    'relu6': nn.ReLU6, 
    'tanh': nn.Tanh, 
    'sigmoid': nn.Sigmoid, 
    'softmax': nn.Softmax,
    'linear': nn.Identity
}


OPTIMIZERS = {
    'adam': optim.Adam,
    'adamw': optim.AdamW
}


def loss_factory(problem_type):
    if problem_type == 'multiclass':
        return nn.CrossEntropyLoss()
    elif problem_type == 'multilabel':
        return nn.MultiLabelSoftMarginLoss()
    return None


class DNNPoolClassifier(nn.Module):
    def __init__(self, 
                 problem_type, 
                 input_size=300, 
                 output_size=9,
                 hidden_size=200, 
                 num_layers=1,
                 num_epochs=8, 
                 dropout=0.0, 
                 learning_rate=5e-5, 
                 epsilon=1e-8,
                 activation='linear',
                 pool_mode='attention',
                 optimizer='adamw'):
        super(DNNPoolClassifier, self).__init__()
        self.problem_type = problem_type
        self.criterion = loss_factory(problem_type)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.num_epochs = num_epochs
        self.dropout = dropout
        self.activation = activation
        self.pool_mode = pool_mode
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.optimizer_name = optimizer

        # Build the model as a simple sequential DNN
        layers = []
        activation_fn = ACTIVATIONS[self.activation]
        for depth in range(num_layers):
            layer_input_size = hidden_size
            layer_output_size = hidden_size

            # Compute layer bounds and sizes
            first_layer = depth == 0
            last_layer = depth == (num_layers - 1)
            if first_layer:
                layer_input_size = input_size
            if last_layer:
                layer_output_size = output_size

            # Add to the stack
            new_layer = nn.Linear(layer_input_size, layer_output_size)
            layers.append(new_layer)

            # Add activation and dropout if needed
            if not last_layer:
                layers.append(activation_fn())
                if self.dropout > 0.0:
                    layers.append(nn.Dropout(self.dropout))
        self.model = safe_cuda_or_cpu(nn.Sequential(*layers))

        # Include attention weights if specified
        self.attention = None
        if self.pool_mode == 'attention':
            self.attention = nn.Sequential(nn.Linear(input_size, 1), nn.Softmax(dim=1))
        safe_cuda_or_cpu(self)

        # Create the optimizer
        self.optimizer = OPTIMIZERS[self.optimizer_name](self.parameters(), 
                                                         lr=self.learning_rate, 
                                                         eps=self.epsilon)

    def aggregate(self, tensor, attention_mask=None):
        batch_size = tensor.shape[0]
        if self.pool_mode == 'last':
            return tensor[:, -1, :].reshape(batch_size, -1)
        if self.pool_mode == 'sum':
            return tensor.sum(axis=1).reshape(batch_size, -1)
        if self.pool_mode == 'mean':
            return tensor.mean(axis=1).reshape(batch_size, -1)
        if self.pool_mode == 'max':
            return tensor.max(axis=1).values.reshape(batch_size, -1)
        if self.pool_mode == 'attention':
            return (attention_mask * tensor).sum(axis=1).reshape(batch_size, -1)
        if self.pool_mode == 'pass':
            return tensor
        raise ValueError('Unknown pooling function "{}".'.format(self.pool_mode))

    def forward(self, X):
        encoded = self.model(X)
        attention_mask = None
        if self.pool_mode == 'attention' and self.attention is not None:
            attention_mask = self.attention(X)
        output = self.aggregate(encoded, attention_mask)
        return output

    def _fit_batch(self, X, y):
        try:
            self.train()
            self.optimizer.zero_grad()
            y_pred = self(X)
            loss = self.criterion(y_pred, y)
            loss.backward()
            self.optimizer.step()
        except RuntimeError:
            print(X.shape, y.shape)
            import sys
            sys.exit(1)

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
                output_logits = self(X_batch)
        
                # Apply transforms
                if self.problem_type == 'multiclass':
                    output = F.softmax(output_logits, dim=-1)
                elif self.problem_type == 'multilabel':
                    output = F.sigmoid(output_logits)
                else:
                    output = output_logits

                # Convert to numpy to reduce GPU memory cost
                output = output.detach().cpu().numpy()
                results.extend(output)
        return np.asarray(results).reshape(num_instances, -1)


class LSTMClassifier(nn.Module):
    def __init__(self, 
                 problem_type, 
                 input_size=300, 
                 output_size=9,
                 hidden_size=200, 
                 num_layers=1,
                 num_epochs=1, 
                 dropout=0.0, 
                 bidirectional=True,
                 learning_rate=5e-5, 
                 epsilon=1e-8,
                 optimizer='adamw',
                 aggregation_mode='attention'):
        super(LSTMClassifier, self).__init__()
        self.problem_type = problem_type
        self.criterion = loss_factory(problem_type)
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.num_epochs = num_epochs
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.aggregation_mode = aggregation_mode
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.optimizer_name = optimizer
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
        self.softmax = nn.Softmax(dim=1)
        safe_cuda_or_cpu(self)

        self.optimizer = OPTIMIZERS[self.optimizer_name](self.parameters(), 
                                                         lr=self.learning_rate, 
                                                         eps=self.epsilon)

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
                output_logits = self(X_batch)
        
                # Apply transforms
                if self.problem_type == 'multiclass':
                    output = F.softmax(output_logits, dim=-1)
                elif self.problem_type == 'multilabel':
                    output = F.sigmoid(output_logits)
                else:
                    output = output_logits

                # Convert to numpy to reduce GPU memory cost
                output = output.detach().cpu().numpy()
                results.extend(output)
        return np.asarray(results).reshape(num_instances, -1)

