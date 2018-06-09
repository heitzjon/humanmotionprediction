import tensorflow as tf
from tensorflow.python.ops.losses.losses_impl import Reduction
import numpy as np


class DAEModel(object):
    """
    Creates training and validation computational graphs.
    Note that tf.variable_scope enables parameter sharing so that both graphs are identical.
    """
    def __init__(self, config, mode):
        """
        Basic setup.
        :param config: configuration dictionary
        :param mode: training, validation or inference
        """
        assert mode in ['training', 'validation', 'inference']
        self.config = config
        self.mode = mode
        self.is_training = self.mode == 'training'
        self.reuse = self.mode == 'validation'

        # input_dim/output_dim is 75
        self.input_dim = config['input_dim']
        self.output_dim = config['output_dim']

        self.batch_size = self.config['batch_size']
        self.input_dim = self.config['input_dim']
        self.output_dim = self.config['output_dim']

        self.first_layer_dropout_rate = self.config['first_layer_dropout_ae']
        self.dense_layer_units = self.config['dense_layer_units_ae']
        self.l2_regularization = self.config['l2_regularization_ae']
        self.noise_std = self.config['gaussian_noise_standard_deviation_ae']

        self.summary_collection = 'training_summaries' if mode == 'training' else 'validation_summaries'

    def build_graph(self):
        self.build_model()
        self.build_loss()
        self.count_parameters()

    def build_model(self):
        """
        Builds the actual model.
        """

        with tf.variable_scope('dae_model', reuse=self.reuse):

            self.input = tf.placeholder(tf.float32, (self.batch_size, None, self.input_dim), name='input')
            self.target = tf.placeholder(tf.float32, (self.batch_size, None, self.output_dim), name='target')

            noisy_layer = gaussian_noise_layer(self.input, self.noise_std)

            dropout_layer = tf.layers.dropout(inputs=noisy_layer, rate=self.first_layer_dropout_rate, training=self.is_training)

            encoder_dense1 = tf.layers.dense(inputs=dropout_layer, units=self.dense_layer_units,
                                             activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_regularization))
            encoder_dense2 = tf.layers.dense(inputs=encoder_dense1, units=self.dense_layer_units,
                                             activation=tf.nn.relu, kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_regularization))

            decoder_dense1 = tf.layers.dense(inputs=encoder_dense2, units=self.dense_layer_units, activation=tf.nn.relu,
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_regularization))
            decoder_dense2 = tf.layers.dense(inputs=decoder_dense1, units=self.dense_layer_units, activation=tf.nn.relu,
                                             kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_regularization))

            self.prediction = tf.layers.dense(inputs=decoder_dense2, units=self.output_dim, activation=tf.nn.tanh,
                                              kernel_regularizer=tf.contrib.layers.l2_regularizer(self.l2_regularization))

    def build_loss(self):
        """
        Builds the loss function.
        """
        # only need loss if we are not in inference mode
        if self.mode is not 'inference':
            with tf.name_scope('loss'):

                self.loss = tf.losses.mean_squared_error(
                    labels=self.target,
                    predictions=self.prediction,
                    scope=None,
                    loss_collection=tf.GraphKeys.LOSSES,
                    reduction=Reduction.SUM_OVER_BATCH_SIZE
                ) + tf.losses.get_regularization_loss() # important! add the regularization-loss

                # self.loss = tf.sqrt(tf.reduce_mean(tf.square(self.target - self.prediction)))

                tf.summary.scalar('loss', self.loss, collections=[self.summary_collection])

    def count_parameters(self):
        """
        Counts the number of trainable parameters in this model
        """
        self.n_parameters = 0
        for v in tf.trainable_variables():
            params = 1
            for s in v.get_shape():
                params *= s.value
            self.n_parameters += params

    def get_feed_dict(self, batch):
        """
        Returns the feed dictionary required to run one training step with the model.
        :param batch: The mini batch of data to feed into the model
        :return: A feed dict that can be passed to a session.run call
        """

        # we are not interested in the target data - for the AE, the input is target at the same time!
        # the same about the sequence-lengths: we don't need them, we train a full batch at a time

        batch_input, _ = batch.get_padded_data(pad_target=False)

        feed_dict = {self.input: batch_input,
                     self.target: batch_input}

        return feed_dict


class RNNModel(object):
    """
    Creates training and validation computational graphs.
    Note that tf.variable_scope enables parameter sharing so that both graphs are identical.
    """
    def __init__(self, config, placeholders, mode):
        """
        Basic setup.
        :param config: configuration dictionary
        :param placeholders: dictionary of input placeholders
        :param mode: training, validation or inference
        """
        assert mode in ['training', 'validation', 'inference']
        self.config = config
        self.input_ = placeholders['input_pl']
        self.target = placeholders['target_pl']
        self.mask = placeholders['mask_pl']
        self.seq_lengths = placeholders['seq_lengths_pl']
        self.mode = mode
        self.is_training = self.mode == 'training'
        self.reuse = self.mode == 'validation'
        self.batch_size = tf.shape(self.input_)[0]  # dynamic size
        self.max_seq_length = tf.shape(self.input_)[1]  # dynamic size

        # input_dim/output_dim is 75
        self.input_dim = config['input_dim']
        self.output_dim = config['output_dim']
        self.hidden_units = config['hidden_units']
        self.num_layers = config['num_of_layers']
        self.init_scale_weights = config['init_scale_weights']
        self.summary_collection = 'training_summaries' if mode == 'training' else 'validation_summaries'

    def build_graph(self):
        self.build_model()
        self.build_loss()
        self.count_parameters()

    def build_model(self):
        """
        Builds the actual model.
        """
        # Some hints:
        #   1) You can access an input batch via `self.input_` and the corresponding targets via `self.target`. Note
        #      that the shape of each input and target is (batch_size, max_seq_length, input_dim)
        #
        #   2) The sequence length of each batch entry is variable, i.e. one entry in the batch might have length
        #      99 while another has length 67. No entry will be larger than what you supplied in
        #      `self.config['max_seq_length']`. This maximum sequence length is also available via `self.max_seq_length`
        #      Because TensorFlow cannot handle variable length sequences out-of-the-box, the data loader pads all
        #      batch entries with zeros so that they have size `self.max_seq_length`. The true sequence lengths are
        #      stored in `self.seq_lengths`. Furthermore, `self.mask` is a mask of shape
        #      `(batch_size, self.max_seq_length)` whose entries are 0 if this entry was padded and 1 otherwise.
        #
        #   3) You can access the config via `self.config`
        #
        #   4) The following member variables should be set after you complete this part:
        #      - `self.initial_state`: a reference to the initial state of the RNN
        #      - `self.final_state`: the final state of the RNN after the outputs have been obtained
        #      - `self.prediction`: the actual output of the model in shape `(batch_size, self.max_seq_length, output_dim)`

        with tf.variable_scope('rnn_model', reuse=self.reuse):

            batch_size = self.config['batch_size']

            cell = tf.contrib.rnn.MultiRNNCell([self.make_cell() for _ in range(self.num_layers)], state_is_tuple=True)

            # the idea of sharing the internal rnn-cell-state via tf-variables is based on this code:
            # https://stackoverflow.com/questions/38441589/is-rnn-initial-state-reset-for-subsequent-mini-batches
            states = create_empty_rnn_state_variables(batch_size, cell)

            output, new_state = tf.nn.dynamic_rnn(cell, self.input_, dtype=tf.float32, initial_state=states)

            # Add an operation to update the train states with the last state tensors.
            self.update_internal_rnn_state = update_rnn_state_variables(states, new_state)

            # we flatten down the output for the matrix multiplication to 700 * 650. (the 700 is 35 * 20, so just flatten the batches)
            output = tf.reshape(output, [-1, self.hidden_units])

            # we need to have a prediction with output size 20 * 35 * 75, so we multiply with a weight matrix of 650 * 75
            weight = tf.get_variable(name='output_weight', initializer=tf.random_uniform([self.hidden_units, self.output_dim], -1 * self.init_scale_weights, self.init_scale_weights))
            bias = tf.get_variable(name='output_bias', initializer=tf.random_uniform([self.output_dim], -1 * self.init_scale_weights, self.init_scale_weights))

            output_transformed = tf.nn.xw_plus_b(output, weight, bias)

            # Reshape logits to be a 3-D tensor for the loss function
            self.prediction = tf.reshape(output_transformed, [self.batch_size, self.max_seq_length, self.output_dim])

            # not sure if we really need this two variables, but the comments says so.
            self.final_state = new_state
            self.initial_state = states

    def make_cell(self):
        cell = tf.contrib.rnn.LSTMCell(self.hidden_units, forget_bias=1.0)
        # add a dropout wrapper if training
        if self.is_training:
            cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=self.config['dropout_on_lstm_cell'])

        return cell

    def build_loss(self):
        """
        Builds the loss function.
        """
        # only need loss if we are not in inference mode
        if self.mode is not 'inference':
            #with tf.name_scope('xloss'):
                # You can access the outputs of the model via `self.prediction` and the corresponding targets via
                # `self.target`. Hint 1: you will want to use the provided `self.mask` to make sure that padded values
                # do not influence the loss. Hint 2: L2 loss is probably a good starting point ...

                # Note Ursin: for the L2 loss function the mask should not be necessary to check


                # self.loss = tf.losses.mean_squared_error(
                #     labels=self.target,
                #     predictions=self.prediction,
                #     weights=1.0,
                #     scope=None,
                #     loss_collection=tf.GraphKeys.LOSSES,
                #     reduction=Reduction.SUM_BY_NONZERO_WEIGHTS
                # )
                #
                # #self.weighted_loss = self.loss * tf.reshape(self.mask, [-1])
                #
                # tf.summary.scalar('xloss', self.loss, collections=[self.summary_collection])

            with tf.name_scope('loss'):

                self.loss = tf.losses.mean_squared_error(
                    labels=self.target,
                    predictions=self.prediction,
                    weights=self.mask[..., None] * np.ones(self.input_dim),
                    scope=None,
                    loss_collection=tf.GraphKeys.LOSSES,
                    reduction=Reduction.SUM_BY_NONZERO_WEIGHTS
                )
                tf.summary.scalar('loss', self.loss, collections=[self.summary_collection])


    def count_parameters(self):
        """
        Counts the number of trainable parameters in this model
        """
        self.n_parameters = 0
        for v in tf.trainable_variables():
            params = 1
            for s in v.get_shape():
                params *= s.value
            self.n_parameters += params

    def get_feed_dict(self, batch):
        """
        Returns the feed dictionary required to run one training step with the model.
        :param batch: The mini batch of data to feed into the model
        :return: A feed dict that can be passed to a session.run call
        """
        input_padded, target_padded = batch.get_padded_data()

        feed_dict = {self.input_: input_padded,
                     self.target: target_padded,
                     self.seq_lengths: batch.seq_lengths,
                     self.mask: batch.mask}

        return feed_dict


def update_rnn_state_variables(state_variables, new_states):
    # Add an operation to update the train states with the last state tensors
    update_ops = []
    for state_variable, new_state in zip(state_variables, new_states):
        # Assign the new state to the state variables on this layer
        update_ops.extend([state_variable[0].assign(new_state[0]),
                           state_variable[1].assign(new_state[1])])
    # Return a tuple in order to combine all update_ops into a single operation.
    # The tuple's actual value should not be used.
    return tf.tuple(update_ops)


def create_empty_rnn_state_variables(batch_size, cell):
    # For each layer, get the initial state and make a variable out of it
    # to enable updating its value.
    # this variables are only used to transfer the internal state between the batches and they should not get trained!
    state_variables = []
    for idx, (state_c, state_h) in enumerate(cell.zero_state(batch_size, tf.float32)):
        state_variables.append(tf.contrib.rnn.LSTMStateTuple(
            tf.get_variable(name='transfer_state_c_{}'.format(idx), initializer=state_c, trainable=False),
            tf.get_variable(name='transfer_state_h_{}'.format(idx), initializer=state_h, trainable=False)))

    # Return as a tuple, so that it can be fed to dynamic_rnn as an initial state
    return tuple(state_variables)


def gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape=tf.shape(input_layer), mean=0.0, stddev=std, dtype=tf.float32)
    return input_layer + noise

