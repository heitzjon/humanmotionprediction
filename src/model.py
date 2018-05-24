import tensorflow as tf
from tensorflow.python.ops.losses.losses_impl import Reduction


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

            # it's a bit stupid, but we can't use the self.batch_size here as it's not yet initialized in the tensor.
            # this might be a solution: https://stackoverflow.com/questions/41630022/using-placeholder-as-shape-in-tensorflow
            self.initial_state = tf.placeholder(tf.float32, [self.num_layers, 2, self.config['batch_size'], self.hidden_units])

            state_per_layer_list = tf.unstack(self.initial_state, axis=0)

            rnn_tuple_state = tuple(
                [tf.contrib.rnn.LSTMStateTuple(state_per_layer_list[idx][0], state_per_layer_list[idx][1])
                 for idx in range(self.num_layers)]
            )

            cell = tf.contrib.rnn.MultiRNNCell([self.make_cell() for _ in range(self.num_layers)], state_is_tuple=True)

            output, self.final_state = tf.nn.dynamic_rnn(cell, self.input_, dtype=tf.float32, initial_state=rnn_tuple_state)

            # we flatten down the output for the matrix multiplication to 700 * 650. (the 700 is 35 * 20, so just flatten the batches)
            output = tf.reshape(output, [-1, self.hidden_units])

            # we need to have a prediction with output size 20 * 35 * 75, so we multiply with a weight matrix of 650 * 75
            softmax_w = tf.Variable(tf.random_uniform([self.hidden_units, self.output_dim], -1 * self.init_scale_weights, self.init_scale_weights))
            softmax_b = tf.Variable(tf.random_uniform([self.output_dim], -1 * self.init_scale_weights, self.init_scale_weights))

            logits = tf.nn.xw_plus_b(output, softmax_w, softmax_b)

            # Reshape logits to be a 3-D tensor for the loss function
            self.prediction = tf.reshape(logits, [self.batch_size, self.max_seq_length, self.output_dim])


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
            with tf.name_scope('loss'):
                # You can access the outputs of the model via `self.prediction` and the corresponding targets via
                # `self.target`. Hint 1: you will want to use the provided `self.mask` to make sure that padded values
                # do not influence the loss. Hint 2: L2 loss is probably a good starting point ...

                # Note Ursin: for the L2 loss function the mask should not be necessary to check

                self.loss = tf.losses.mean_squared_error(
                    labels=self.target,
                    predictions=self.prediction,
                    weights=1.0,
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
