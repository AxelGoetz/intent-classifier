"""
Performs any necesarry data preprocessing such as:
- One hot encoding classes.
- Splitting data into mini-batches.
- padding traces within a mini-batch.
"""

import numpy as np

from sklearn import preprocessing
from os import path as ospath
from sys import path

# Hack to import from sibling directory
path.append(ospath.dirname(path[0]))
from utils import files

def get_batches(paths, batch_size=100, max_iterations=10000):
    """
    Divides the data up into mini-batches.

    Paramters:
      - paths: Array containing the absolute paths to the training files.
      - batch_size: Currently should be a multiple of 5 but this needs to be changed in the future *(Default 100)*.
      - max_iterations: *(Default 10,000)*.

    Returns:
      - An iterator with mini-batches where the array returned by the iterator is: `[(category, embedding)]`.
    """
    # TODO: Batch size needs to be multiple of 5.
    # TODO: Find a better way to temporarily store data in memory because
    # parsing an entire json file every time is probably too slow.
    # Stores the last accessed index for each file in file_indexes.

    file_indexes = {}
    for path in paths:
        file_indexes[path] = 0

    length = batch_size // 5

    for _ in range(max_iterations):
        np.random.shuffle(paths)

        mini_batch = []

        # Pick 5 files (just an arbitraty number)
        for i in range(5):
            data = files.open_file(paths[i])
            file_data = get_arr_index(data['data'], file_indexes[paths[i]], length)
            file_indexes[paths[i]] = (file_indexes[paths[i]] + length) % len(data['data'])

            mini_batch.extend([(data['intent'], val['embeddings']) for val in file_data])

        yield mini_batch

def get_arr_index(arr, init, length):
    """
    Basically performs index slicing with wrapping around.

    For instance when arr = [1, 2, 3], init = 1 and length = 10,
    then the function should return [2, 3, 1, 2, 3, 1, 2, 3, 1, 2].

    Parameters:
      - arr: A list object.
      - init: Initial index.
      - length: Length of the resulting array.
    """
    new_arr = arr[init:]

    while len(new_arr) <= length:
        new_arr.extend(arr[:])

    return new_arr[:length]

def pad_traces(data, reverse=False):
    """
    Pads all sequences such that they have the same length whilst leaving them in batch major form.

    Parameters:
      - data: A matrix of size `[sentence_length, embedding_size]`.
      - reverse: A boolean value that if true, reverses the traces *(Default False)*.

    Returns:
      - A tuple where the first element is the padded matrix and the second is an array of lengths of the original sequences.
    """
    sequence_lengths = [len(seq) for seq in data]
    embedding_size = len(data[0][0])
    batch_size = len(data)

    if reverse:
        for i, seq in enumerate(data):
            data[i] = list(reversed(seq))

    max_sequence_length = max(sequence_lengths)

    inputs_batch_major = np.zeros(shape=[batch_size, max_sequence_length, embedding_size], dtype=np.float32)

    for i, seq in enumerate(data):
        for j, element in enumerate(seq):
            inputs_batch_major[i][j] = element

    return inputs_batch_major, sequence_lengths

class OneHotEncoder():
    """
    Encodes a list of categorical features into a one-hot representation.
    The items in the array need to be hashable and comparable.
    """

    def __init__(self, categories):
        """
        Parameters:
          - categories: An array of hashable and comparable items, representing the categories.
        """
        self.encoder = preprocessing.LabelEncoder()
        self.encoder.fit(categories)
        self.num_categories = len(categories)

    def _get_one_hot_vector(self, value):
        """
        Given an integer value, representing the class, returns a one-hot vector.
        """
        vec = np.zeros(self.num_categories)
        vec[value] = 1
        return vec

    def _get_category(self, one_hot):
        """
        Gets an integer value of a category, given a one-hot vector.

        Returns:
          - Categorical value and -1 if vector is not a one-hot encoding.
        """
        for i in range(len(one_hot)):
            if one_hot[i] == 1:
                return i
        return -1

    def encode(self, categories):
        """
        Returns a 2D matrix, containing the one-hot vectors.

        Parameters:
          - An array of categorical values.
        """
        encoding = self.encoder.transform(categories)
        return [self._get_one_hot_vector(val) for val in encoding]

    def decode(self, one_hot_vectors):
        """
        Decodes an array of one-hot vectors into the original categorical values.

        Parameters:
          - one_hot_vectors: A 2D matrix, containing the one-hot vectors.
        """
        decoding = [self._get_category(val) for val in one_hot_vectors]
        return self.encoder.inverse_transform(decoding)

def get_classes(paths):
    """
    Gets all of the existing classes.

    Arguments:
      - paths: An array of absolute paths to the training data.
    """
    return [files.get_name(path) for path in paths]

def get_onehot_for_dir(dir_name):
    """
    Gets an instance of the `OneHotEncoder` class with all of the files in a directory as categories.

    Parameters:
      - dir_name: A string containing the absolute path to the directory.
    """
    paths = files.get_files(dir_name, '.json')
    classes = get_classes(paths)
    return OneHotEncoder(classes)

def get_batches_for_dir(dir_name, batch_size=100, max_iterations=10000):
    """
    Gets an iterator for the mini-batches in a given directory.

    Parameters:
      - dir_name: A string containing the absolute path to the directory.
      - batch_size: Currently should be a multiple of 5 but this needs to be changed in the future *(Default 100)*.
      - max_iterations: *(Default 10,000)*.

      Returns:
        - An iterator with mini-batches where the array returned by the iterator is: `[(category, embedding)]`.
    """
    paths = files.get_files(dir_name, '.json')
    return get_batches(paths, batch_size, max_iterations)
