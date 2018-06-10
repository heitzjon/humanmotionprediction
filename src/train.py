import sys

import datetime
import os
import tensorflow as tf
import time
from config import train_config

from model import DAEModel, get_model_and_placeholders
from load_data import MotionDataset
from train_dae import train_dae
from train_rnn import train_rnn
from utils import export_config


def load_data(config, split):
    print('Loading data from {} ...'.format(config['data_dir']))
    return MotionDataset.load(data_path=config['data_dir'],
                              split=split,
                              seq_length=config['max_seq_length'],
                              batch_size=config['batch_size'])


def init_data(config, model_type):
    # create unique output directory for this model
    timestamp = str(int(time.time()))
    config['model_dir'] = os.path.abspath(os.path.join(config['output_dir'], model_type + '_' + config['name'] + '_' + timestamp))
    os.makedirs(config['model_dir'])
    print('Writing checkpoints into {}'.format(config['model_dir']))

    # load the data, this requires that the *.npz files you downloaded from Kaggle be named `train.npz` and `valid.npz`
    data_train = load_data(config, 'train')
    data_valid = load_data(config, 'valid')
    config['input_dim'] = data_train.input_[0].shape[-1]
    config['output_dim'] = data_train.target[0].shape[-1]

    # TODO if you would like to do any preprocessing of the data, here would be a good opportunity
    return config, data_train, data_valid


def train_hybrid(config, data_train, data_valid):
    print('TODO')


if __name__ == '__main__':
    print('====== start program with parameters "' + sys.argv[1] + '" ======')

    if sys.argv[1] == 'rnn':
        train_rnn(*init_data(train_config, 'rnn'))

    if sys.argv[1] == 'dae':
        train_dae(*init_data(train_config, 'dae'))

    if sys.argv[1] == 'hybrid':
        print('not yet implemented!')