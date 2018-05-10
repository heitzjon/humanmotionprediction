import os
import tensorflow as tf
import numpy as np

from config import test_config
from visualize import visualize_joint_angles
from utils import export_to_csv
from train import load_data, get_model_and_placeholders


def main(config):
    # load the data
    data_test = load_data(config, 'test')

    config['input_dim'] = config['output_dim'] = data_test.input_[0].shape[-1]
    rnn_model, placeholders = get_model_and_placeholders(config)

    # restore the model by first creating the computational graph
    with tf.name_scope('inference'):
        rnn_model = rnn_model(config, placeholders, mode='inference')
        rnn_model.build_graph()

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
        ids = []
        for batch in data_test.all_batches():

            # initialize the RNN with the known sequence (here 2 seconds)
            # no need to pad the batch because in the test set all batches have the same length
            input_ = np.array(batch.input_)
            seeds.append(input_)

            # here we are requesting the final state as we later want to supply this back into the RNN
            # this is why the model should have a member `self.final_state`
            fetch = [rnn_model.final_state]
            feed_dict = {placeholders['input_pl']: input_,
                         placeholders['seq_lengths_pl']: batch.seq_lengths}

            [state] = sess.run(fetch, feed_dict)

            # now get the prediction by predicting one pose at a time and feeding this pose back into the model to
            # get the prediction for the subsequent time step
            next_pose = input_[:, -1:]
            predicted_poses = []
            for f in range(config['prediction_length']):
                # TODO evaluate your model here frame-by-frame
                # To do so you should
                #   1) feed the previous final state of the model as the next initial state
                #   2) feed the previous output pose of the model as the new input (single frame only)
                #   3) fetch both the final state and prediction of the RNN model that are then re-used in the next
                #      iteration

                fetch = None
                feed_dict = None

                [state, predicted_pose] = sess.run(fetch, feed_dict)

                predicted_poses.append(np.copy(predicted_pose))
                next_pose = predicted_pose

            predicted_poses = np.concatenate(predicted_poses, axis=1)

            predictions.append(predicted_poses)
            ids.extend(batch.ids)

        seeds = np.concatenate(seeds, axis=0)
        predictions = np.concatenate(predictions, axis=0)


    # the predictions are now stored in test_predictions, you can do with them what you want
    # for example, visualize a random entry
    idx = np.random.randint(0, len(seeds))
    seed_and_prediction = np.concatenate([seeds[idx], predictions[idx]], axis=0)
    visualize_joint_angles([seed_and_prediction], change_color_after_frame=seeds[0].shape[0])

    # or, write out the test results to a csv file that you can upload to Kaggle
    model_name = config['model_dir'].split('/')[-1]
    model_name = config['model_dir'].split('/')[-2] if model_name == '' else model_name
    output_file = os.path.join(config['model_dir'], 'submit_to_kaggle_{}_{}.csv'.format(config['prediction_length'], model_name))
    export_to_csv(predictions, ids, output_file)


if __name__ == '__main__':
    main(test_config)
