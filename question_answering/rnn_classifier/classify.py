"""
Use a trained RNN for classification.
"""

import tensorflow as tf
import numpy as np

import rnn_classifier
import preprocessing

from tensorflow.contrib.rnn import LSTMCell, GRUCell

from os import path as ospath
from sys import path

DIR_NAME, _ = ospath.split(ospath.abspath(__file__))

# Hack to import from sibling directory
path.append(ospath.dirname(path[0]))

from utils import clean_text, word_embedding, save_object

def classify(sess, model, data_dir,
                       batch_size=1):
    """
    Classify using an existing RNN classifier.

    Parameters:
      - sess: A tensorflow session.
      - model: An instance of the `RNN` class.
      - data_dir: The directory with all of the training data.
      - batch_size: *(Default 1)*.

    """
    onehot_encoder = save_object.load_object(DIR_NAME + '/../onehot_encoder')

    while True:
        raw_text = input("Ask me a question: ")

        tokens = clean_text.clean_text(raw_text)
        embeddings = []

        for token in tokens:
            try:
                embedding = word_embedding.embed_word(token)
                embeddings.append(embedding)
            except KeyError:
                pass

        train_dict = {
            model.inputs: [embeddings],
            model.input_length: [len(embeddings)]
        }

        prediction_ = sess.run([model.prediction], train_dict)
        prediction_ = prediction_[0][0]

        maxi, maxval = 0, 0
        for i, val in enumerate(prediction_):
            if val > maxval:
                maxi, maxval = i, val

        print("Classified class is: " + onehot_encoder.decodes_ints(maxi) + " with probability " + str(maxval))

def run_model(data_dir):
    """
    Runs a rnn classifier model.

    @param data is
        if in_memory == True:
            ([[size, incoming]], [webpage_label])
        else:
            A list of paths
    """
    tf.reset_default_graph()
    tf.set_random_seed(123)

    # Only print small part of array
    np.set_printoptions(threshold=10)

    with tf.Session() as session:

        model = rnn_classifier.RNN(LSTMCell(300), 24,
                            batch_size=1,
                            embedding_size=300,
                            reverse=False,
                            softmax_on_hidden=False,
                            saved_graph=DIR_NAME + '/../rnn_classifier',
                            sess=session,
                            learning_rate=0.0001,
                            keep_prob=1,
                            is_training=False)

        classify(session, model, data_dir,
                           batch_size=1)

def main(_):
    run_model('/Users/axelg/Drive/Projects/nlp-classifier/data/classifier_data/')

if __name__ == '__main__':
    tf.app.run()
