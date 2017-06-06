"""
Use a trained RNN for classification.
"""

import tensorflow as tf
import numpy as np

import question_answering.rnn_classifier.rnn_classifier as rnn_classifier
from question_answering.utils import clean_text, word_embedding, save_object

from tensorflow.contrib.rnn import LSTMCell, GRUCell

from os import path as ospath

DIR_NAME, _ = ospath.split(ospath.abspath(__file__))

class Classifier():
    """
    Uses a pretrained RNN classifier to classify queries.
    """

    def __init__(self):
        self.onehot_encoder = save_object.load_object(DIR_NAME + '/data/onehot_encoder')

        tf.reset_default_graph()
        tf.set_random_seed(123)

        # Only print small part of array
        np.set_printoptions(threshold=10)

        self.session = tf.Session()

        self.model = rnn_classifier.RNN(LSTMCell(300), 26,
                            batch_size=1,
                            embedding_size=300,
                            reverse=False,
                            softmax_on_hidden=False,
                            saved_graph=DIR_NAME + '/data/rnn_classifier',
                            sess=self.session,
                            learning_rate=0.0001,
                            keep_prob=1,
                            is_training=False)

    def predict(self, query):
        """
        Given a query, performs a classification task.

        Parameters:
          - query: A string, representing a query.

        Returns:
          - An array `[(class, probability)]`, which is sorted where the highest probability occurs first.
            Or if it cannot embed anything, the function just returns none.
        """

        tokens = clean_text.clean_text(query)
        embeddings = []

        for token in tokens:
            try:
                embedding = word_embedding.embed_word(token)
                embeddings.append(embedding)
            except KeyError:
                pass

        if len(embeddings) == 0:
            return None

        train_dict = {
            self.model.inputs: [embeddings],
            self.model.input_length: [len(embeddings)]
        }

        prediction_ = self.session.run([self.model.prediction], train_dict)
        prediction_ = prediction_[0][0]

        prediction = []

        for i, val in enumerate(prediction_):
            prediction.append({'class': self.onehot_encoder.decodes_ints(i), 'probability': val})

        prediction.sort(key=lambda x: x['probability'], reverse=True)
        return prediction


def main(_):
    classifier = Classifier()

    while True:
        raw_text = input("Ask me a question: ")
        print(classifier.predict(raw_text))

if __name__ == '__main__':
    tf.app.run()
