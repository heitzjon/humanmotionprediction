import datetime
import os
import time

import tensorflow as tf

from model import get_model_and_placeholders
from utils import export_config


def train_rnn(config, data_train, data_valid):

    # get input placeholders and get the model that we want to train
    rnn_model_class, placeholders = get_model_and_placeholders(config)

    # Create a variable that stores how many training iterations we performed.
    # This is useful for saving/storing the network
    global_step = tf.Variable(1, name='global_step', trainable=False)

    # create a training graph, this is the graph we will use to optimize the parameters
    with tf.name_scope('training'):
        rnn_model = rnn_model_class(config, placeholders, mode='training')
        rnn_model.build_graph()
        print('created RNN model with {} parameters'.format(rnn_model.n_parameters))

        # configure learning rate
        if config['learning_rate_type_rnn'] == 'exponential':
            lr = tf.train.exponential_decay(config['learning_rate_rnn'],
                                            global_step=global_step,
                                            decay_steps=config['learning_rate_decay_steps_rnn'],
                                            decay_rate=config['learning_rate_decay_rate_rnn'],
                                            staircase=False)
            lr_decay_op = tf.identity(lr)
        elif config['learning_rate_type_rnn'] == 'linear':
            lr = tf.Variable(config['learning_rate_rnn'], trainable=False)
            lr_decay_op = lr.assign(tf.multiply(lr, config['learning_rate_decay_rate_rnn']))
        elif config['learning_rate_type_rnn'] == 'fixed':
            lr = config['learning_rate_rnn']
            lr_decay_op = tf.identity(lr)
        else:
            raise ValueError('learning rate type "{}" unknown.'.format(config['learning_rate_type_rnn']))

        # choose the optimizer you desire here and define `train_op. The loss should be accessible through rnn_model.loss
        optimizer = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-08)

        # params = tf.trainable_variables()
        # grads, _ = tf.clip_by_global_norm(tf.gradients(rnn_model.loss, params), config['max_grad_norm'])
        # train_op = optimizer.apply_gradients(
        #     zip(grads, params),
        #     global_step=tf.train.get_or_create_global_step())

        train_op = optimizer.minimize(loss=rnn_model.loss, global_step=tf.train.get_global_step())

    # create a graph for validation
    with tf.name_scope('validation'):
        rnn_model_valid = rnn_model_class(config, placeholders, mode='validation')
        rnn_model_valid.build_graph()

    # Create summary ops for monitoring the training
    # Each summary op annotates a node in the computational graph and collects data data from it
    tf.summary.scalar('learning_rate', lr, collections=['training_summaries'])

    # Merge summaries used during training and reported after every step
    summaries_training = tf.summary.merge(tf.get_collection('training_summaries'))

    # create summary ops for monitoring the validation
    # caveat: we want to store the performance on the entire validation set, not just one validation batch
    # Tensorflow does not directly support this, so we must process every batch independently and then aggregate
    # the results outside of the model
    # so, we create a placeholder where can feed the aggregated result back into the model
    loss_valid_pl = tf.placeholder(tf.float32, name='loss_valid_pl')
    loss_valid_s = tf.summary.scalar('loss_valid', loss_valid_pl, collections=['validation_summaries'])

    # merge validation summaries
    summaries_valid = tf.summary.merge([loss_valid_s])

    # dump the config to the model directory in case we later want to see it
    export_config(config, os.path.join(config['model_dir'], 'config.txt'))

    with tf.Session() as sess:
        # Add the ops to initialize variables.
        init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
        # Actually intialize the variables
        sess.run(init_op)

        # create file writers to dump summaries onto disk so that we can look at them with tensorboard
        train_summary_dir = os.path.join(config['model_dir'], "summary", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)
        valid_summary_dir = os.path.join(config['model_dir'], "summary", "validation")
        valid_summary_writer = tf.summary.FileWriter(valid_summary_dir, sess.graph)

        # create a saver for writing training checkpoints
        saver = tf.train.Saver(max_to_keep=config['n_keep_checkpoints'])

        # start training
        start_time = time.time()
        current_step = 0
        for e in range(config['n_epochs_rnn']):

            # reshuffle the batches
            data_train.reshuffle()

            # loop through all training batches
            for i, batch in enumerate(data_train.all_batches()):
                step = tf.train.global_step(sess, global_step)
                current_step += 1

                if config['learning_rate_type_rnn'] == 'linear' and current_step % config['learning_rate_decay_steps_rnn'] == 0:
                    sess.run(lr_decay_op)

                # we want to train, so must request at least the train_op
                fetches = {'summaries': summaries_training,
                           'loss': rnn_model.loss,
                           'train_op': train_op}

                # get the feed dict for the current batch
                feed_dict = rnn_model.get_feed_dict(batch)

                # feed data into the model and run optimization
                xyz = batch.mask
                training_out, _ = sess.run([fetches, rnn_model.update_internal_rnn_state], feed_dict)

                # write logs
                train_summary_writer.add_summary(training_out['summaries'], global_step=step)

                # print training performance of this batch onto console
                time_delta = str(datetime.timedelta(seconds=int(time.time() - start_time)))
                print('\rEpoch: {:3d} [{:4d}/{:4d}] time: {:>8} loss: {:.4f}'.format(
                    e + 1, i + 1, data_train.n_batches, time_delta, training_out['loss']), end='')

            # after every epoch evaluate the performance on the validation set
            total_valid_loss = 0.0
            n_valid_samples = 0
            for batch in data_valid.all_batches():
                fetches = {'loss': rnn_model_valid.loss}
                feed_dict = rnn_model_valid.get_feed_dict(batch)
                valid_out = sess.run(fetches, feed_dict)

                total_valid_loss += valid_out['loss'] * batch.batch_size
                n_valid_samples += batch.batch_size

            # write validation logs
            avg_valid_loss = total_valid_loss / n_valid_samples
            valid_summaries = sess.run(summaries_valid, {loss_valid_pl: avg_valid_loss})
            valid_summary_writer.add_summary(valid_summaries, global_step=tf.train.global_step(sess, global_step))

            # print validation performance onto console
            print(' | validation loss: {:.6f}'.format(avg_valid_loss))

            # save this checkpoint if necessary
            if (e + 1) % config['save_checkpoints_every_epoch'] == 0:
                saver.save(sess, os.path.join(config['model_dir'], 'model'), global_step)

        # Training finished, always save model before exiting
        print('Training finished')
        ckpt_path = saver.save(sess, os.path.join(config['model_dir'], 'model'), global_step)
        print('Model saved to file {}'.format(ckpt_path))
