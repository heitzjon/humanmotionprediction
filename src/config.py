# configuration used by the training and evaluation scripts
train_config = {}
train_config['data_dir'] = '../data'  # where the data downloaded from Kaggle is stored, i.e. the *.npz files
train_config['output_dir'] = '../trained_models/'  # where you want to store the checkpoints of different training runs
train_config['name'] = 'dummy'

train_config['save_checkpoints_every_epoch'] = 1  # after how many epochs the trained model should be saved
train_config['n_keep_checkpoints'] = 3  # how many saved checkpoints to keep

train_config['n_epochs_rnn'] = 25

train_config['learning_rate_rnn'] = 0.001
# some code to anneal the learning rate, this is implemented for you, you can just choose it here
train_config['learning_rate_type_rnn'] = 'fixed'  # ['fixed', 'exponential', 'linear']
train_config['learning_rate_decay_steps_rnn'] = 500
train_config['learning_rate_decay_rate_rnn'] = 0.9

train_config['batch_size'] = 20  # can not be zero!
train_config['max_seq_length'] = 50  # specify for how many time steps you want to unroll the RNN
train_config['num_of_layers'] = 3 #2 #3
train_config['hidden_units'] = 1500 #650 #1500

# see https://stackoverflow.com/questions/45507315/what-exactly-does-tf-contrib-rnn-dropoutwrapper-in-tensorflow-do-three-cit
train_config['dropout_on_lstm_cell'] = 0.5
train_config['init_scale_weights'] = 0.05
train_config['max_grad_norm'] = 5

# config for auto-encoder
train_config['first_layer_dropout_ae'] = 0.08
train_config['dense_layer_units_ae'] = 3000
train_config['n_epochs_ae'] = 30

train_config['learning_rate_ae'] = 0.005
train_config['learning_rate_type_ae'] = 'fixed'  # ['fixed', 'exponential', 'linear']
train_config['learning_rate_decay_steps_ae'] = 1000
train_config['learning_rate_decay_rate_ae'] = 0.90

# config for hybrid
train_config['n_epochs_hybrid'] = 20

train_config['learning_rate_hybrid'] = 0.001
train_config['learning_rate_type_hybrid'] = 'fixed'  # ['fixed', 'exponential', 'linear']
train_config['learning_rate_decay_steps_hybrid'] = 1000
train_config['learning_rate_decay_rate_hybrid'] = 0.95

# currently not used
# train_config['l2_regularization_ae'] = 0.001
# train_config['gaussian_noise_standard_deviation_ae'] = 0.2

# some additional configuration parameters required when the configured model is used at inference time
test_config = train_config.copy()


train_config['model_dir_rnn'] = test_config['model_dir_rnn'] = '../trained_models/rnn_25ep_adam'
train_config['model_dir_dae'] = test_config['model_dir_dae'] = '../trained_models/dae_trained_0_5_dropout_very_strict'


test_config['max_seq_length'] = 50  # want to use entire sequence during test, which is fixed to 50, don't change this
# train_config['model_dir_rnn'] = test_config['model_dir_rnn'] = '../trained_models/maskedloss_model_500_1528486275'
test_config['checkpoint_id'] = None  # if None, the last checkpoint will be used
test_config['prediction_length'] = 50  # how many frames to predict into the future (assignment requires 25 frames, but you can experiment with more if you'd like)

test_config['use_dae'] = False
test_config['scenario'] = 10
test_config['scenario_id'] = 675
test_config['select_scenario'] = True

test_config['model_dir_hybrid'] = '../trained_models/hybrid_dummy_1529059452'