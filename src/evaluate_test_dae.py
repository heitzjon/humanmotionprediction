import os
import tensorflow as tf
import numpy as np

import matplotlib
matplotlib.use('TkAgg')

from config import test_config
from visualize import visualize_multiple_poses
from utils import export_to_csv
from train import load_data, get_model_and_placeholders
from model import DAEModel


def main(config):
    # load the data
    data_test = load_data(config, 'test')

    tf.reset_default_graph()

    config['input_dim'] = config['output_dim'] = data_test.input_[0].shape[-1]
    #dae_model, placeholders = get_model_and_placeholders(config)
    dae_model_class = DAEModel

    # restore the model by first creating the computational graph
    with tf.name_scope('inference'):
        dae_model = dae_model_class(config, mode='inference')
        dae_model.build_graph()

    with tf.Session() as sess:
        # now restore the trained variables
        # this operation will fail if this `config` does not match the config you used during training
        saver = tf.train.Saver()
        ckpt_id = config['checkpoint_id']
        if ckpt_id is None:
            ckpt_path = tf.train.latest_checkpoint(config['model_dir_dae'])
        else:
            ckpt_path = os.path.join(os.path.abspath(config['model_dir_dae']), 'model-{}'.format(ckpt_id))
        print('Evaluating ' + ckpt_path)
        saver.restore(sess, ckpt_path)

        # loop through all the test samples
        ground_truth = []
        predictions = []
        dropouts = []
        ids = []
        for batch in data_test.all_batches():

            input_, _ = batch.get_padded_data(pad_target=False)
            ground_truth.append(input_)

            predicted_poses = []
            dropout_poses = []

            fetch = [dae_model.prediction, dae_model.reshaped_dropout_input] #, dae_model.dropout_pose , dae_model.fig2
            feed_dict = dae_model.get_feed_dict(batch)

            [predicted_pose, dropout_pose] = sess.run(fetch, feed_dict) #, dropout_pose

            predicted_poses.append(np.copy(predicted_pose))
            dropout_poses.append(np.copy(dropout_pose))

            predicted_poses = np.concatenate(predicted_poses, axis=1)
            dropout_poses = np.concatenate(dropout_poses, axis=1)

            predictions.append(predicted_poses)
            dropouts.append(dropout_poses)
            ids.extend(batch.ids)

        ground_truth = np.concatenate(ground_truth, axis=0)
        predictions = np.concatenate(predictions, axis=0)
        dropouts = np.concatenate(dropouts, axis=0)

    if config['select_scenario']:
        labels = np.load(config['data_dir'] + '/test.npz')['data']
        idx = np.random.randint(0, len(labels))
        while labels[idx]['action_label'] is not config['scenario']:
            idx = np.random.randint(0, len(labels))
        label=labels[idx]['action_label']
        print('We display sample with idx {} '.format(idx)+" and label {}".format(label))
    else:
        idx = np.random.randint(0, len(ground_truth))
        print('We display sample with idx {} '.format(idx));
        label=None

    visualize_multiple_poses([ground_truth[idx]], [predictions[idx]], [dropouts[idx]], action_label=label)

if __name__ == '__main__':
    main(test_config)
