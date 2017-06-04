"""
Training the RNN Classifier.
"""

import tensorflow as tf
import numpy as np

import rnn_classifier
import preprocessing

from tensorflow.contrib.rnn import LSTMCell, GRUCell
from sys import exit

from os import path as ospath
from sys import path

DIR_NAME, _ = ospath.split(ospath.abspath(__file__))

# Hack to import from sibling directory
path.append(ospath.dirname(path[0]))

from utils import clean_text, save_object


def train(sess, model, data_dir,
                       batch_size=100,
                       batches_in_epoch=10,
                       max_batches=10000,
                       verbose=False):
    """
    Train the `rnn_classifier` and display progress.

    Parameters:
      - sess: A tensorflow session.
      - model: An instance of the `RNN` class.
      - data_dir: The directory with all of the training data.
      - batch_size: *(Default 100)*.
      - batch_in_epoch: Extended amount of information is displayed after each epoch *(Default 10)*.
      - max_batches: *(Default 10,000)*.
      - verbose: Whether or not to display extended amount of information *(Default False)*.
    """
    batches = preprocessing.get_batches_for_dir(data_dir, batch_size=batch_size, max_iterations=max_batches)
    onehot_encoder = preprocessing.get_onehot_for_dir(data_dir)
    save_object.save_object(onehot_encoder, DIR_NAME + '/data/onehot_encoder')

    try:
        for batch in range(max_batches):
            print("Batch {}/{}".format(batch, max_batches))
            fd = model.next_batch(batches, onehot_encoder)

            _, _, summary = sess.run([model.train_op, model.loss, model.merged_summary], fd)

            if batch == 0 or batch % batches_in_epoch == 0:
                model.write_summary(summary, batch)

                model.save(sess, DIR_NAME + '/data/rnn_classifier')

                if verbose:
                    l, predict_ = sess.run([model.loss, model.prediction], fd)
                    print('  minibatch loss: {}\n'.format(l))

                    for i, (inp, pred) in enumerate(zip(fd[model.targets], predict_)):
                        print('  sample {}:\n'.format(i + 1))

                        maxi, maxval = 0, 0
                        for i, val in enumerate(inp):
                            if val > maxval:
                                maxi, maxval = i, val

                        print("    Target onehot at: " + str(maxi))

                        maxi, maxval = 0, 0
                        for i, val in enumerate(pred):
                            if val > maxval:
                                maxi, maxval = i, val

                        print("    Prediction onehot at: " + str(maxi) + " with probability " + str(maxval))

                        if i >= 2:
                            break
                    print('\n')

    except KeyboardInterrupt:
        print('training interrupted')
        model.save(sess, DIR_NAME + '/data/rnn_classifier')
        exit(0)

    model.save(sess, DIR_NAME + '/data/rnn_classifier')

def run_model(data_dir):
    """
    Runs a rnn classifier model.

    Parameters:
      - data_dir: A string, containing the absolute path to the directory containing the data.
    """
    tf.reset_default_graph()
    tf.set_random_seed(123)

    # Only print small part of array
    np.set_printoptions(threshold=10)

    with tf.Session() as session:

        model = rnn_classifier.RNN(LSTMCell(300), 24,
                            batch_size=100,
                            embedding_size=300,
                            reverse=False,
                            softmax_on_hidden=False,
                            saved_graph=None,
                            sess=None,
                            learning_rate=0.0001,
                            keep_prob=1,
                            is_training=True)

        model.set_summary_writer(session, DIR_NAME + '/data/classifier1_logs')

        session.run(tf.global_variables_initializer())

        train(session, model, data_dir,
                               batch_size=100,
                               batches_in_epoch=500,
                               max_batches=10000,
                               verbose=True)

def main(_):
    run_model(DIR_NAME + '/../../data/classifier_data/')

if __name__ == '__main__':
    tf.app.run()
