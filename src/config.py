# configuration used by the training and evaluation scripts
train_config = {}
train_config['data_dir'] = '../data'  # where the data downloaded from Kaggle is stored, i.e. the *.npz files
train_config['output_dir'] = '../trained_models/'  # where you want to store the checkpoints of different training runs
train_config['name'] = 'dummy'
train_config['batch_size'] = 20  # can not be zero!
train_config['max_seq_length'] = 50  # specify for how many time steps you want to unroll the RNN

# in paper https://arxiv.org/pdf/1508.00271.pdf they use 3 layer with 1000 units each.
# I assume the use the same in the ETH paper as they reference this work there
train_config['num_of_layers'] = 3
train_config['hidden_units'] = 1000


# see https://stackoverflow.com/questions/45507315/what-exactly-does-tf-contrib-rnn-dropoutwrapper-in-tensorflow-do-three-cit
train_config['dropout_on_lstm_cell'] = 0.5
train_config['lambda_l2_regularization'] = 0.0000001
train_config['init_scale_weights'] = 0.05
train_config['max_grad_norm'] = 5

train_config['n_epochs'] = 30

train_config['save_checkpoints_every_epoch'] = 1  # after how many epochs the trained model should be saved
train_config['n_keep_checkpoints'] = 3  # how many saved checkpoints to keep

# config for auto-encoder
train_config['first_layer_dropout_ae'] = 0.05
train_config['dense_layer_units_ae'] = 500
train_config['l2_regularization_ae'] = 0.001
train_config['gaussian_noise_standard_deviation_ae'] = 0.2

train_config['learning_rate'] = 0.0001
# some code to anneal the learning rate, this is implemented for you, you can just choose it here
train_config['learning_rate_type'] = 'linear'  # ['fixed', 'exponential', 'linear']
train_config['learning_rate_decay_steps'] = 1000
train_config['learning_rate_decay_rate'] = 0.99

# some additional configuration parameters required when the configured model is used at inference time
test_config = train_config.copy()
test_config['max_seq_length'] = 50  # want to use entire sequence during test, which is fixed to 50, don't change this
test_config['model_dir_rnn'] = '../trained_models/rnn_dummy_1528662933/'
test_config['checkpoint_id'] = None  # if None, the last checkpoint will be used
test_config['prediction_length'] = 25  # how many frames to predict into the future (assignment requires 25 frames, but you can experiment with more if you'd like)

test_config['model_dir_dae'] = '../trained_models/dae_dummy_1528667158/'
test_config['use_dae'] = False