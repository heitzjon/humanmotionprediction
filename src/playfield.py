import numpy as np

from model import RNNModel
from train import load_data, get_model_and_placeholders

from config import train_config
from sklearn.metrics import mean_squared_error




def dummy():

    config = train_config

    data_train = load_data(config, 'train')
    config['input_dim'] = data_train.input_[0].shape[-1]
    config['output_dim'] = data_train.target[0].shape[-1]

    data_train.reshuffle()

    rnn_model_class, placeholders = get_model_and_placeholders(config)

    rnn_model = RNNModel(config, placeholders, mode='training')

    # loop through all training batches
    for i, batch in enumerate(data_train.all_batches()):
        # get the feed dict for the current batch
        feed_dict = rnn_model.get_feed_dict(batch)

        for sequence_mask in batch.mask:
            if np.sum(sequence_mask) < 35:
                print('found it {0}'.format(np.sum(sequence_mask)))

                input_padded, target_padded = batch.get_padded_data()

                mse = mean_squared_error(input_padded[0], target_padded[0])
                mse2 = mean_squared_error(input_padded[1], target_padded[1])


if __name__ == '__main__':
    dummy()