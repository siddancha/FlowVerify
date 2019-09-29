import numpy as np


class SingleDataStream:
    """Takes a dataset and produces an infinite stream of batches of input batch size.
    Can save and load state of datastream to resume training.
    """
    def __init__(self, dataset, get_example_fn, collate_fn=None):
        self.dataset = dataset
        self.len_dataset = len(dataset)
        self.get_example_fn = get_example_fn
        self.collate_fn = collate_fn
        self.reset()

        # Number of successful and failed retrievals.
        self.num_retr_success = 0
        self.num_retr_failure = 0

    def reset(self):
        self.perm_seq = np.random.permutation(self.len_dataset)
        self.perm_ptr = 0

    def next(self, batchsize):
        """Returns a batch from the datastream with given batchsize
        NOTE: batchsize should be atmost the size of the dataset"""

        # Get next batch.
        batch = []
        while len(batch) < batchsize:

            # Reset if finished looping through the dataset.
            if self.perm_ptr == self.len_dataset:
                self.reset()

            example = self.get_example_fn(self.perm_seq[self.perm_ptr])
            self.perm_ptr += 1  # increment pointer

            # Append to batch if example has been successfully retrieved.
            if example is not None:
                batch.append(example)
                self.num_retr_success += 1
            else:
                self.num_retr_failure += 1

        if self.collate_fn:
            batch = self.collate_fn(batch)

        return batch

    def state(self):
        """Returns the state (dict) of the datastream"""
        ds_state = {'perm_seq': self.perm_seq, 'perm_ptr': self.perm_ptr}
        return ds_state

    def load_state(self, ds_state):
        """Loads the state (dict) of the datastream"""
        self.perm_seq = ds_state['perm_seq']
        self.perm_ptr = ds_state['perm_ptr']


class MultiDataStream:
    """Creates a multistream from multiple datasets.
    Maintains a SingleDataStream for each dataset. Draws from each of the
    streams simultaneously to create a multistream of batches of input batch size.
    The state of MultiDataStream is a union of states of all SingleDataStream, which
    can be loaded and saved.
    """
    def __init__(self, datasets, p, collate_fn):
        """
        Args:
            datasets: list(dataset) a list of datasets.
            p: list(float) p_i is the probability of sampling from the ith stream.
                sum(p_i) should equal to 1.
            collate_fn: (function) that specifies how to collate datapoints together.
        """
        assert(len(datasets) == len(p))
        self.datasets = datasets
        self.streams = [SingleDataStream(dataset, None) for dataset in datasets]
        self.p = p
        self.collate_fn = collate_fn

    @property
    def num_streams(self):
        return len(self.streams)

    def next(self, batchsize):
        """Returns a batch from the multi-datastream with given batchsize.
        Samples sub-batches from member streams according to self.p.
        NOTE: batchsize should be less than the size of each member dataset"""

        # Sub-batch (sbatch) sizes sampled from multinomial.
        sbatch_sizes = np.random.multinomial(batchsize, self.p)

        batch = []
        for i, sbatch_size in enumerate(sbatch_sizes):
            sbatch = self.streams[i].next(sbatch_size)
            batch.extend(sbatch)

        batch = self.collate_fn(batch)
        return batch

    def state(self):
        """Returns the state (list) of the multi-datastream"""
        ds_state = [stream.state() for stream in self.streams]
        return ds_state

    def load_state(self, ds_state):
        """Loads the state (list) of the multi-datastream"""
        assert(len(ds_state) == self.num_streams)
        for i in range(self.num_streams):
            self.streams[i].load_state(ds_state[i])
