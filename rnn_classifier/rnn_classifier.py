"""
Simple RNN classifier (with a softmax stacked on the end).

Hyperparameters
--------------
- LSTM or GRU cells
- Size of cells
- Whether or not to use stacked RNN cells, if yes, how many?
- Whether or not to reverse sequences
- Learning rate
- Batch Size
- Use batch-normalisation
- Dropout (keep probability)
- Softmax on output or on memory state

Padding will be a 0 vector.
"""

import tensorflow as tf
import preprocessing

class RNN():
    """
    Implements a RNN with a softmax on top of it to perform intent classification in Tensorflow.

    """

    def __init__(self, rnn_cell, num_classes, batch_size=100, embedding_size=300,
        reverse=False, softmax_on_hidden=False, saved_graph=None, sess=None, learning_rate=0.0006, keep_prob=1, is_training=True):
        """
        Parameters:
          - rnn_cell: Either a LSTM or GRU cell (can also be stacked).
          - num_classes: Number of intents (+1 for default intent).
          - batch_size
          - embedding_size
          - reverse: If true, sentences will be reversed.
          - softmax_on_hidden: If true, the softmax will be applied to the hidden output and to the actual output otherwise.
          - saved_graph: If None, will be ignored and otherwise it will load the existing model.
          - sess: Must be defined if loading an existing graph.
          - learning_rate
          - keep_prob: If >= 1, no dropout layer will be added and otherwise it will be added with the appropriate `keep_prob`.
          - is_training
        """
        self.reverse = reverse
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate
        self.num_classes = num_classes

        self.keep_prob = keep_prob
        self.is_training = is_training
        self.softmax_on_hidden = softmax_on_hidden

        self.rnn_cell = rnn_cell

        self._make_graph()

        if saved_graph is not None and sess is not None:
            self.import_from_file(sess, saved_graph)

    def _make_graph(self):
        """
        Construct the computational graph.
        """

        self._init_placeholders()

        self._init_rnn()
        self._init_softmax()

        self._init_train()

    def _init_placeholders(self):
        """
        The main placeholders used for the input and output data.
        """
        # The usual format is: `[self.batch_size, max_sequence_length, self.seq_width]`
        # But we define `max_sequence_length` as None to make it dynamic so we only need to pad
        # each batch to the maximum sequence length
        self.inputs = tf.placeholder(tf.float32,
            [self.batch_size, None, self.embedding_size])

        self.input_length = tf.placeholder(tf.int32, [self.batch_size])

        self.targets = tf.placeholder(tf.int32,
            [self.batch_size, self.num_classes])

    def _init_rnn(self):
        """
        Creates the computational graph of the dynamic RNN cells.
        """
        if self.is_training and self.keep_prob < 1:
            self.rnn_cell = tf.contrib.rnn.DropoutWrapper(
                self.rnn_cell, output_keep_prob=self.keep_prob)

        self.rnn_outputs, self.rnn_final_state = tf.nn.dynamic_rnn(
            cell=self.rnn_cell,
            dtype=tf.float32,
            sequence_length=self.input_length,
            inputs=self.inputs,
            time_major=False
        )

    def _projection(self, inputs, projection_size, scope):
        """
        Projects the input with a known amount of features to a `projection_size` amount of features
        by using a simple feedforward neural network layer.

        Parameters:
          - inputs: Tensor to be projected.
          - projection_size: A int32 variable, representing the size of the projection.
          - scope
        """
        input_size = inputs.get_shape()[-1].value

        with tf.variable_scope(scope) as scope:
            W = tf.get_variable(name='W', shape=[input_size, projection_size],
                                dtype=tf.float32)

            b = tf.get_variable(name='b', shape=[projection_size],
                                dtype=tf.float32,
                                initializer=tf.constant_initializer(0, dtype=tf.float32))

        input_shape = tf.unstack(tf.shape(inputs))

        if len(input_shape) == 3:
            time, batch, _ = input_shape  # dynamic parts of shape
            inputs = tf.reshape(inputs, [-1, input_size])

        elif len(input_shape) == 2:
            batch, _depth = input_shape

        else:
            raise ValueError("Weird input shape: {}".format(inputs))

        linear = tf.add(tf.matmul(inputs, W), b)

        if len(input_shape) == 3:
            linear = tf.reshape(linear, [time, batch, projection_size])

        return linear

    def _extract_axis(self, data, ind):
        """
        Get specified elements along the first axis of tensor.

        Parameters:
          - data: Tensorflow tensor that will be subsetted.
          - ind: Indices to take (one for each element along axis 0 of data).
        """
        batch_range = tf.range(tf.shape(data)[0])
        indices = tf.stack([batch_range, ind], axis=1)
        res = tf.gather_nd(data, indices)

        return res

    def _init_softmax(self):
        """
        Computes the values of the logits.
        """
        with tf.variable_scope('softmax') as scope:
            self.logits = self._projection(self.rnn_final_state, self.num_classes, scope) if self.softmax_on_hidden else self._projection(self._extract_axis(self.rnn_outputs, self.input_length - 1), self.num_classes, scope)

            self.prediction = tf.nn.softmax(self.logits)


    def _init_train(self):
        """
        Creates the loss variables and initialises how to train the graph.
        """
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.targets,
            logits=self.logits
        )

        self.loss = tf.reduce_mean(cross_entropy)
        tf.summary.scalar('loss', self.loss)

        # TODO: Which optimizer to use? `GradientDescentOptimizer`, `AdamOptimizer` or `RMSProp`?
        self.train_op = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)


    def next_batch(self, batches, onehot_encoder):
        """
        Returns the next training mini-batch.

        Parameters:
          - batches: An iterator with all of the mini-batches
        - onehot_encoder: An instance of the class `preprocessing.OneHotEncoder`.

         Returns:
           - A dict for training.
        """
        batch = next(batches)

        inputs_ = [val[1] for val in batch]
        targets_ = [val[0] for val in batch]

        inputs_, input_lengths_ = preprocessing.pad_traces(inputs_, reverse=self.reverse)

        targets_ = onehot_encoder.encode(targets_)

        train_dict = {
            self.inputs: inputs_,
            self.input_length: input_lengths_,
            self.targets: targets_,
        }

        return train_dict

    def save(self, sess, file_name):
        """
        Save the model parameters.

        Parameters:
          - sess
          - file_name: The file name without the extension.
        """
        saver = tf.train.Saver()
        saver.save(sess, file_name)

    def import_from_file(self, sess, file_name):
        """
        Imports the graph from a file.

        Parameters:
          - sess
          - file_name: A string that represents the file name
            without the extension.
        """

        # Get the graph
        saver = tf.train.Saver()

        # Restore the variables
        saver.restore(sess, file_name)

    def set_summary_writer(self, sess):
        self.merged_summary = tf.summary.merge_all()
        self.summary_writer = tf.summary.FileWriter('./classifier1_logs', sess.graph)

    def write_summary(self, summary, i):
        self.summary_writer.add_summary(summary, i)
