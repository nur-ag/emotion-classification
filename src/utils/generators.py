class SizedCallableWrapper:
    def __init__(self, data, function):
        self.data = data
        self.function = function

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return (self.function(x) for x in self.data)


class SizedBatchWrapper:
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size

    def __len__(self):
        data_length = len(self.data)
        num_batches = (data_length + self.batch_size - 1) // self.batch_size
        return num_batches

    def __iter__(self):
        current_batch = []
        for datum in self.data:
            current_batch.append(datum)
            if len(current_batch) >= self.batch_size:
                yield current_batch
                current_batch.clear()
        if current_batch:
            yield current_batch

