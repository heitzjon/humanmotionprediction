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
            ckpt_path = tf.train.latest_checkpoint(config['model_dir'])
        else:
            ckpt_path = os.path.join(os.path.abspath(config['model_dir']), 'model-{}'.format(ckpt_id))
        print('Evaluating ' + ckpt_path)
        saver.restore(sess, ckpt_path)

        # loop through all the test samples
        seeds = []
        predictions = []
        dropouts = []
        ids = []
        for batch in data_test.all_batches():

            # initialize the RNN with the known sequence (here 2 seconds)
            # no need to pad the batch because in the test set all batches have the same length
            input_, _ = batch.get_padded_data(pad_target=False)
            seeds.append(input_)


            predicted_poses = []
            dropout_poses = []
            fetch = [dae_model.prediction, dae_model.dropout_pose] #, dae_model.dropout_pose , dae_model.fig2
            feed_dict = dae_model.get_feed_dict(batch)
            [predicted_pose, dropout_pose] = sess.run(fetch, feed_dict) #, dropout_pose
            predicted_poses.append(np.copy(predicted_pose))
            dropout_poses.append(np.copy(dropout_pose))

            predicted_poses = np.concatenate(predicted_poses, axis=1)
            dropout_poses = np.concatenate(dropout_poses, axis=1)

            predictions.append(predicted_poses)
            dropouts.append(dropout_poses)
            ids.extend(batch.ids)

        seeds = np.concatenate(seeds, axis=0)
        predictions = np.concatenate(predictions, axis=0)
        dropouts = np.concatenate(dropouts, axis=0)

    seeds = seeds[0:len(data_test.input_)]
    predictions = predictions[0:len(data_test.input_)]
    dropouts = dropouts[0:len(data_test.input_)]
    ids = ids[0:len(data_test.input_)]

    # the predictions are now stored in test_predictions, you can do with them what you want
    # for example, visualize a random entry
    idx = np.random.randint(0, len(seeds))
    idy = np.random.randint(0, 50)
    seed_and_prediction = np.concatenate([seeds[idx], predictions[idx]], axis=0)

    #visualize_joint_angles2([seed_and_prediction])
    #visualize_multiple_poses([seeds[idx]],[predictions[idx]])
    visualize_multiple_poses([seeds[idx]],[predictions[idx]]) #[dropouts[idx]],

if __name__ == '__main__':
    main(test_config)
