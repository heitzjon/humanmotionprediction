from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import os


class Dataset(object):
    """
    Abstract Dataset.
    """
    def __init__(self, input, target):
        self._input_ = input
        self._target = target

    @property
    def input_(self):
        return self._input_

    @property
    def target(self):
        return self._target

    @input_.setter
    def input_(self, value):
        self._input_ = value

    @target.setter
    def target(self, value):
        self._target = value


class AbstractFeeder(object):
    """
    An abstract class that provides batch-wise streaming access to the data.
    """
    def _get_batch(self, batch_ptr, no_shuffle=False):
        """
        Get the specified batch.
        :param no_shuffle: If set, the underlying data is not shuffled and batches are returned in the same order
         as they originally were.
        :param batch_ptr: Which batch to access, i.e. index between 0 and n_batches().
        :return: The retrieved batch.
        """
        raise NotImplementedError('Method is abstract.')

    def next_batch(self, no_shuffle=False):
        """
        Returns the next available batch. Circular access if overflow happens.
        :param no_shuffle: If set, the underlying data is not shuffled and batches are returned in the same order
         as they originally were.
        :return: The next available batch
        """
        raise NotImplementedError('Method is abstract.')

    def all_batches(self, no_shuffle=False):
        """
        Generator function looping over all available batches.
        :param no_shuffle: If set, the underlying data is not shuffled and batches are returned in the same order
         as they originally were.
        """
        for i in range(self.n_batches):
            yield self.next_batch(no_shuffle=no_shuffle)

    @property
    def n_batches(self):
        """
        Returns the number of batches.
        """
        raise NotImplementedError('Method is abstract.')

    def random_batch(self, rng=np.random):
        """
        Returns a random batch.
        """
        batch_ptr = rng.randint(0, self.n_batches)
        batch = self._get_batch(batch_ptr)
        return batch

    def reshuffle(self, rng=np.random):
        """
        Reshuffles the data.
        """
        raise NotImplementedError('Method is abstract.')

    def batch_from_idxs(self, indices):
        """
        Return a batch consisting of the data points at the given indices.
        :param indices: Which data points to retrieve from the dataset.
        :return: A batch of size len(indices)
        """
        raise NotImplementedError('Method is abstract.')


class Feeder(AbstractFeeder):
    def __init__(self, dataset, batch_size, rng=np.random):
        self._dataset = dataset
        self._batch_size = batch_size
        self._rng = rng

        # how many batches we have
        self._n_batches = int(np.ceil(float(len(self._dataset.input_)) / float(batch_size)))

        # pointers to the next available batch
        self._batch_ptr = 0

        # indices into the data
        self._indices = np.arange(0, len(self._dataset.input_))

        # keep copy of indices for unshuffled access
        self._indices_unshuffled = np.arange(0, len(self._dataset.input_))

        # reshuffle the indices
        self._rng.shuffle(self._indices)

    def _update_batch_ptr(self):
        new_val = self._batch_ptr + 1
        new_val = new_val if new_val < self._n_batches else 0
        self._batch_ptr = new_val

    def _get_batch(self, batch_ptr, no_shuffle=False):
        assert 0 <= batch_ptr < self._n_batches, 'batch pointer out of range'

        start_idx = batch_ptr * self._batch_size
        end_idx = (batch_ptr + 1) * self._batch_size

        # because we want to use all available data, must be careful that `end_idx` is valid
        end_idx = end_idx if end_idx <= len(self._indices) else len(self._indices)
        indices_access = self._indices_unshuffled if no_shuffle else self._indices
        indices = indices_access[start_idx:end_idx]
        batch = self.batch_from_idxs(indices)
        return batch

    def next_batch(self, no_shuffle=False):
        next_batch = self._get_batch(self._batch_ptr, no_shuffle=no_shuffle)
        self._update_batch_ptr()
        return next_batch

    @property
    def n_batches(self):
        return self._n_batches

    def reshuffle(self, rng=np.random):
        rng.shuffle(self._indices)


class Batch(object):
    """
    Represents one minibatch that can have variable sequence lengths.
    """
    def __init__(self, input_, targets, ids, **kwargs):
        assert isinstance(input_, list) and (isinstance(targets, list) or targets is None), 'data expected in python lists'
        self.input_ = input_  # python list of numpy arrays
        self.target = targets  # python list of numpy arrays
        self.ids = ids  # numpy array of ids
        self.seq_lengths = np.array([b.shape[0] for b in self.input_])  # list of sequence lengths per batch entry
        self.batch_size = len(self.input_)
        self._mask = None
        for k in kwargs.keys():
            self.__setattr__(k, kwargs[k])

    @property
    def mask(self):
        """
        The mask is used to identify which entries in this minibatch are "real" values and which ones are "padded".
        The mask has shape (batch_size, max_seq_length) where an entry is 1 if it's a real value and 0 if it was padded.
        """
        if self._mask is None:
            max_seq_length = max(self.seq_lengths)
            ltri = np.tril(np.ones([max_seq_length, max_seq_length]))
            self._mask = ltri[self.seq_lengths - 1]
        return self._mask

    def get_padded_data(self, pad_target=True):
        """
        Pads the data with zeros, i.e. returns an np array of shape (batch_size, max_seq_length, dof). `max_seq_length`
        is the maximum occurring sequence length in the batch. Target is only padded if `pad_target` is True.
        """
        max_seq_length = max(self.seq_lengths)

        inputs = []
        targets = []
        for x, y in zip(self.input_, self.target):
            missing = max_seq_length - x.shape[0]
            x_padded, y_padded = x, y
            if missing > 0:
                # this batch entry has a smaller sequence length then the max sequence length, so pad it with zeros
                x_dof = x.shape[1]
                x_padded = np.concatenate([x, np.zeros(shape=[missing, x_dof])], axis=0)
                if pad_target:
                    y_dof = y.shape[1]
                    y_padded = np.concatenate([y, np.zeros(shape=[missing, y_dof])], axis=0)
            assert len(x_padded) == max_seq_length
            inputs.append(x_padded)

            if pad_target:
                assert len(y_padded) == max_seq_length
                targets.append(y_padded)

        return np.array(inputs), np.array(targets)


class MotionDataset(Dataset, Feeder):
    """
    Represents the motion data.
    """
    @classmethod
    def load(cls, data_path, split, seq_length, batch_size, rng=np.random):
        """
        Load the data from the hard disk.
        :param data_path: Where the *.npz files are stored.
        :param split: Which training split to load, must be one of ['train', 'valid', 'test'].
        :param seq_length: Desired sequence length. If -1, the input sequences will not be splitted.
        :param batch_size: Desired batch size.
        :param rng: Random number generator.
        :return: An instance of this class
        """
        assert split in ['train', 'valid', 'test']

        def _split(data):
            """
            Split data into chunks of size <= seq_length
            :param data: np array of shape (tot_length, dof)
            :return: A list of np arrays of shape (seq_length, dof)
            """
            if seq_length < 0:
                return [data]
            if seq_length == 0:
                raise ValueError('sequence length cannot be 0')

            tot_length = data.shape[0]
            return np.split(data, range(0, tot_length, seq_length)[1:], axis=0)

        data_file = os.path.join(data_path, '{}.npz'.format(split))
        print('load sequences of length {} from {}'.format(seq_length, data_file))

        data = np.load(data_file)['data']
        all_angles = []
        all_ids = []
        all_action_labels = []

        for d in data:
            angles = d['angles']
            angles_s = _split(angles)
            all_angles.extend(angles_s)
            all_ids.extend([d['id']]*len(angles_s))
            all_action_labels.extend([d['action_label']]*len(angles_s))

        assert len(all_angles) == len(all_ids)

        # create input and target
        input_ = all_angles

        # we want to predict the next pose given an input frame, so the target for frame t is just the frame t+1
        # for the last frame we have no clear target, so just repeat it
        if split == 'test':  # there's no targets in the test data
            target = None
        else:
            target = [np.concatenate([np.copy(x[1:]), np.copy(x[-1:])], axis=0) for x in all_angles]

        obj = cls(input_, target, all_ids, all_action_labels, batch_size, rng)
        return obj

    def __init__(self, input_, target, ids, action_labels, batch_size, rng=np.random):
        """
        Constructor.
        :param input_: list of np arrays of shape (seq_length, n_joints*3) representing the input to the model
        :param target: list of np arrays of shape (seq_length, n_joints*3) representing the ground truth
        :param ids: list of ids
        :aparam action_labels: list of action labels
        :param batch_size: desired batch size.
        :param rng: random number generator.
        """
        self.batch_size = batch_size
        self.rng = rng
        self.ids = ids
        self.action_labels = action_labels

        # initialize the dataset parent class
        Dataset.__init__(self, input_, target)

        # create batch_wise access by initializing the feeder parent class
        Feeder.__init__(self, self, batch_size, rng)

    def batch_from_idxs(self, indices):
        input_ = [np.copy(self.input_[i]) for i in indices]
        target = [np.copy(self.target[i]) for i in indices] if self.target is not None else None
        ids = [self.ids[i] for i in indices]
        action_labels = [self.action_labels[i] for i in indices]
        return Batch(input_, target, ids=ids, action_labels=action_labels)


