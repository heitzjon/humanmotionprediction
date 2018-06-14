import os

import matplotlib
import numpy as np
import tensorflow as tf

from model import CombinedModel

matplotlib.use('TkAgg')

from config import test_config
from visualize import visualize_multiple_poses
from utils import export_to_csv
from train import load_data, get_model_and_placeholders


def main(config):
    # load the data
    data_test = load_data(config, 'test')

    tf.reset_default_graph()

    config['input_dim'] = config['output_dim'] = data_test.input_[0].shape[-1]

    _, placeholders = get_model_and_placeholders(config)
    hybrid_model_class = CombinedModel

    # restore the model by first creating the computational graph
    with tf.name_scope('inference'):
        hybrid_model = hybrid_model_class(config, placeholders, mode='inference')
        hybrid_model.build_graph()

    with tf.Session() as session:
        # now restore the trained variables
        # this operation will fail if this `config` does not match the config you used during training
        saver = tf.train.Saver()
        ckpt_path = tf.train.latest_checkpoint(config['model_dir_hybrid'])

        print('Evaluating ' + ckpt_path)
        saver.restore(session, ckpt_path)

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
            fetch = [hybrid_model.update_internal_rnn_state]
            feed_dict = {placeholders['input_pl']: input_,
                         placeholders['seq_lengths_pl']: batch.seq_lengths}

            [_] = session.run(fetch, feed_dict)

            # now get the prediction by predicting one pose at a time and feeding this pose back into the model to
            # get the prediction for the subsequent time step
            next_pose = input_[:, -1:]
            predicted_poses = []
            for f in range(config['prediction_length']):
                fetch_rnn = [hybrid_model.prediction, hybrid_model.update_internal_rnn_state]
                feed_dict_rnn = {placeholders['input_pl']: next_pose,
                                 placeholders['seq_lengths_pl']: batch.batch_size * [1]}

                [predicted_pose, _] = session.run(fetch_rnn, feed_dict_rnn)

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
    if config['select_scenario']:
        labels = np.load(config['data_dir'] + '/test.npz')['data']
        if config['scenario_id'] is not None:
            idx=ids.index(config['scenario_id'])
            label_id=config['scenario_id']-180
        else:
            idx = np.random.randint(0, len(labels))
            while labels[idx]['action_label'] is not config['scenario']:
                idx = np.random.randint(0, len(labels))

            label_id = ids[idx] - 180
        label = labels[label_id]['action_label']
        print('We display sample with idx {} '.format(idx)+" and label {}".format(label))
    else:
        idx = np.random.randint(0, len(seeds))
        print('We display sample with idx {} '.format(idx));
        label=None
    seed_and_prediction = np.concatenate([seeds[idx], predictions[idx]], axis=0)
    visualize_multiple_poses([seed_and_prediction], change_color_after_frame=seeds[0].shape[0], action_label=label)

    # or, write out the test results to a csv file that you can upload to Kaggle
    model_name = config['model_dir_rnn'].split('/')[-1]
    model_name = config['model_dir_rnn'].split('/')[-2] if model_name == '' else model_name
    output_file = os.path.join(config['model_dir_rnn'], 'submit_to_kaggle_{}_{}.csv'.format(config['prediction_length'], model_name))
    export_to_csv(predictions, ids, output_file)


if __name__ == '__main__':
    main(test_config)