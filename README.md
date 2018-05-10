# Human Motion Prediction Skeleton Code

Refer to our [website](https://ait.ethz.ch/teaching/courses/2018-SS-Machine-Perception/) for more information about the Machine Perception course. The Kaggle competition is available under [this link](https://www.kaggle.com/c/mp18-human-motion-prediction). Please address any questions regarding this project on [Piazza](https://piazza.com/class/jdbpmonr7fa26b) first. If you can't get any help on Piazza, feel free to contact me by [e-mail](mailto:manuel.kaufmann@inf.ethz.ch).

## Environment Setup
### Installing Dependencies

The following assumes that TensorFlow (GPU version) is already installed. If you need help installing it, visit their [official website](https://www.tensorflow.org/install/) or ask on Piazza. The skeleton code was tested with Python 3.5.4 and TensorFlow 1.4.0, but it should port without major issues to more recent versions.

I recommend using Anaconda to manage the dependencies, which should be pre-installed on MS Azure. To set up your environment you can follow these instructions (assuming you cd'ed into the directory where you extracted the skeleton code to):

```
# create a new environment
conda create -n tf python=3.5
source activate tf
# install all dependencies
python setup.py install
# install TF (this does not install CUDA and CuDNN, you need to do this manually)
# this installs TF 1.4 for linux and Python 3.5
python pip install --ignore-installed --upgrade https://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-1.4.0-cp35-cp35m-linux_x86_64.whl
```

### Getting the Data
Download the training, validation and test set directly from [Kaggle](https://www.kaggle.com/c/mp18-human-motion-prediction/data). More details on how the data is structured is also provided on Kaggle. The validation set is just a held-out subset from the training set, that you can use to simulate the performance of your model on unseen data. The provided code assumes that a validation set is available, however, you are free to change that and also train on the validation set if you want.

Move the downloaded `.npz` data files into a directory of your choice. Make sure that enter this path as `config['data_dir']` in `config.py`. Refer to the next section for more information about `config.py`.

Once a model was trained, you should evaluate it on the test set and then upload those results to Kaggle. Refer to the next section for more information about this.

## Using the Skeleton Code
The provided code takes care of a few common tasks associated with training a neural network, so that you can focus on the core parts of the problem to be solved. Those core parts have been stripped from the code and marked with `# TODO ...` comments. It is then explained what you are expected to do in these parts in order to be able to successfully train a model. In the following, the provided scripts are explained in a bit more detail. While a reasonable implementation of the missing `# TODO` sections should let you pass the easy baseline, the code is only meant as a guideline, i.e., you are free to modify it to whatever extent you deem useful.

### Visualizing the Data
`visualize.py` loads a random sample from the training set and displays it. This is useful for both inspecting the data at hand and looking at outputs produced by your model. You can directly run `python visualize.py`, just make sure to change the path 

### Training a Model
Run `train.py` to start a new training run. In this script, the data is loaded, a new model instantiated, and then optimized in the training loop. The whole training process is guided by the configuration parameters entered in `config.py`. The configuration parameters are all stored in a dictionary (called `config`) which is then passed to the training routine and the model. It is good practice to make certain (hyper-) parameters configurable through this `config` object. This can include parameters like learning rate, the choice of optimizer, regularization, but also others that drive the creation of the model, like number of hidden layers, type of RNN cells, etc. Some parameters are already present in `config.py`, but to create your own models, you will probably have to add new ones.

Whenever you start a new training, a new folder is created in `config['output_dir']` that stores all the checkpoints. This folder will also contain a file called `config.txt` which is just a human-readable dump of the `config` used to train this model. This is useful so that you can go back and check what configurations you used to train a certain model.

The implementation of a model is located in `models.py`. This is where you should implement the actual model. More details are commented directly in the code file. Note that whenever we train a model, we actually create two TensorFlow graphs: one is the training graph, whose weights are being optimized during training, and the other is a validation graph, which is sharing the weights with the training graph and only used at inference time. This is useful to evaluate your model's performance on a hold-out dataset. By default, the provided training routine evaluates the validation set after every epoch.

You might find it useful to monitor your training runs via tensorboard: `tensorboard --logdir config['output_path']`.

### Evaluating a Model
Finally, once you trained a model, you should evaluate it on the test set. This is what the script `evaluate_test.py` is for. It loads the training data and a model's checkpoint. Again, the functionality of this script is driven by the `config` dictionary. In `config['model_dir']` you can enter the path to the model's directory that you want to evaluate.

Note that in order to successfully load a pre-trained model, the remaining configuration parameters in `config` must match those that you used during training of this model. This only concerns parameters that influence the architecture of your model. E.g., if you created a model with a hidden layer of dimensionality 32, which was configured through `config['hidden_dim_size'] = 32`, you should make sure that this parameter is also present at test time. Otherwise you might get an error when TensorFlow is loading the model.

Once `evaluate_test.py` finishes, it currently displays the results of a random sample and then writes the Kaggle submission file into `config['model_dir']`, which you then can upload if you wish. 
