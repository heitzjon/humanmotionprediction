import os
import tensorflow as tf
import numpy as np

import matplotlib

from model import DAEModel

matplotlib.use('TkAgg')

from config import test_config
from visualize import visualize_joint_angles
from utils import export_to_csv
from train import load_data, get_model_and_placeholders


def main(config):
    # load the data
    data_test = load_data(config, 'test')

    tf.reset_default_graph()

    config['input_dim'] = config['output_dim'] = data_test.input_[0].shape[-1]

    rnn_model_class, placeholders = get_model_and_placeholders(config)
    dae_model_class = DAEModel

    # restore the model by first creating the computational graph
    with tf.name_scope('inference'):
        rnn_model = rnn_model_class(config, placeholders, mode='inference')
        rnn_model.build_graph()

        dae_model = dae_model_class(config, mode='inference')
        dae_model.build_graph()

    with tf.Session() as session:

        # now restore the trained variables
        # this operation will fail if this `config` does not match the config you used during training
        ckpt_path_dae = restore_dae(config, session)
        ckpt_path_rnn = restore_rnn(config, session)

        print('Evaluating RNN ' + ckpt_path_rnn + ' together with DAE:' + ckpt_path_dae)

        # loop through all the test samples
        seeds = []
        predictions = []
        ids = []
        for batch in data_test.all_batches():

            # initialize the RNN with the known sequence (here 2 seconds)
            # no need to pad the batch because in the test set all batches have the same length
            input_, _ = batch.get_padded_data(pad_target=False)
            seeds.append(input_)

            # here we are propagating the seed (50 frames) through the rnn to initialize it. Remember: the hidden-state is
            # stored in a tf.variable, therefore we don't need to feed/fetch anything here.
            fetch = [rnn_model.update_internal_rnn_state]
            feed_dict = {placeholders['input_pl']: input_,
                         placeholders['seq_lengths_pl']: batch.seq_lengths}

            [_] = session.run(fetch, feed_dict)

            # now get the prediction by predicting one pose at a time and feeding this pose back into the model to
            # get the prediction for the subsequent time step
            next_pose = input_[:, -1:]
            predicted_poses = []
            for f in range(config['prediction_length']):
                fetch_rnn = [rnn_model.prediction, rnn_model.update_internal_rnn_state]
                feed_dict_rnn = {placeholders['input_pl']: next_pose,
                                 placeholders['seq_lengths_pl']: batch.batch_size * [1]}

                [predicted_pose, _] = session.run(fetch_rnn, feed_dict_rnn)

                # if 'use dae' is activated, feed the predicted_pose (from rnn) through the dae before appending it
                if config['use_dae']:
                    fetch_dae = dae_model.prediction
                    feed_dict_dae = {dae_model.input: predicted_pose}

                    predicted_pose = session.run(fetch_dae, feed_dict_dae)

                predicted_poses.append(np.copy(predicted_pose))
                next_pose = predicted_pose

            predicted_poses = np.concatenate(predicted_poses, axis=1)

            predictions.append(predicted_poses)
            ids.extend(batch.ids)

        seeds = np.concatenate(seeds, axis=0)
        predictions = np.concatenate(predictions, axis=0)

    seeds = seeds[0:len(data_test.input_)]
    predictions = predictions[0:len(data_test.input_)]
    ids = ids[0:len(data_test.input_)]

    # the predictions are now stored in test_predictions, you can do with them what you want
    # for example, visualize a random entry
    idx = np.random.randint(0, len(seeds))
    print('We display sample with idx {} '.format(idx))
    seed_and_prediction = np.concatenate([seeds[idx], predictions[idx]], axis=0)
    visualize_joint_angles([seed_and_prediction], change_color_after_frame=seeds[0].shape[0])

    # or, write out the test results to a csv file that you can upload to Kaggle
    model_name = config['model_dir_rnn'].split('/')[-1]
    model_name = config['model_dir_rnn'].split('/')[-2] if model_name == '' else model_name
    output_file = os.path.join(config['model_dir_rnn'], 'submit_to_kaggle_{}_{}.csv'.format(config['prediction_length'], model_name))
    export_to_csv(predictions, ids, output_file)


def restore_rnn(config, session):
    ckpt_path_rnn = tf.train.latest_checkpoint(config['model_dir_rnn'])

    # same as above. The [:-2] removes the index at the end of the name of the variable (e.g. :0),
    # as this is not stored in the checkpoint (use inspect_checkpoint to verify)
    rnn_varlist = {v.name[:-2]: v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="rnn_model/")}
    saver_rnn = tf.train.Saver(rnn_varlist)
    saver_rnn.restore(session, ckpt_path_rnn)

    return ckpt_path_rnn


def restore_dae(config, session):
    ckpt_path_dae = tf.train.latest_checkpoint(config['model_dir_dae'])

    # find all the dae-variables in the graph and initialize the Saver with them. If we don't do that, the saver
    # tries to restore all the variables in the graph, which is obviously not possible (as the graph consists of both, dae and rnn)
    dae_varlist = {v.name[:-2]: v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="dae_model/")}
    saver_dae = tf.train.Saver(dae_varlist)
    saver_dae.restore(session, ckpt_path_dae)

    return ckpt_path_dae


if __name__ == '__main__':
    main(test_config)