# imports

from matplotlib import rcParams
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import sklearn.metrics
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error


# data structuring functions
def filter_no_go_choice(data_set):
    # to do. make time window a parameter.
    """
    In order to reduce the complexity on the decode model(A3), we are removing the no-go trials
    so we kept binary choice (left or right) that fits with a LogisticRegression model
     Args:
        data_set: Subset of alldat

    Returns:
     new_data_set (dict): dat['spks']: neurons by trials.
                          dat['brain_area']: brain area for each neuron recorded.
                          dat['contrast_right']: contrast level for the right stimulus, which is always contralateral to the recorded brain areas.
                          dat['contrast_left']: contrast level for left stimulus.
                          dat['response']: which side the response was (-1,  1). Choices for the right stimulus are -1.
    """
    new_data_set = {}
    index_trials = data_set['response'].nonzero()
    # Remove baseline(first 50 bins) and get FR per neuron
    new_data_set.update({"spks": (data_set["spks"][:, index_trials[0], 50:].sum(axis=2) / trial_time).T})
    new_data_set.update({"brain_area": data_set["brain_area"]})
    new_data_set.update({"response": data_set["response"][index_trials]})
    new_data_set.update({"contrast_right": data_set["contrast_right"][index_trials]})
    new_data_set.update({"contrast_left": data_set["contrast_left"][index_trials]})
    return new_data_set



def get_spks_from_area(dat, brain_area):
    spks = dat["spks"].T
    n_neurons = spks.shape[0]
    index_neurons = np.zeros(n_neurons, dtype=bool)
    for neuron in range(n_neurons):
        index_neurons[neuron] = dat['brain_area'][neuron] in brain_area

    area_data = (spks[index_neurons, :].T)
    return area_data, index_neurons


### to do:
# create get methods that take a brain region argument:
# eg. get_region_data(data_set,'visual')
# data set should then have a field defining what sub areas are part of visual:
# data_set['areas']['visual'] = ["VISa", "VISam", "VISl", "VISp", "VISpm", "VISrl"]
def get_visual_ctx(data_set):
    visual_ctx = ["VISa", "VISam", "VISl", "VISp", "VISpm", "VISrl"]
    visual_data, _ = get_spks_from_area(data_set, visual_ctx)
    return visual_data


def get_motor_ctx(data_set):
    motor_ctx = ["CL", "LD", "LGd", "LH", "LP", "MD", "MG", "PO", "POL", "PT", "RT", "SPF", "TH", "VAL", "VPL", "VPM"]
    motor_data, _ = get_spks_from_area(data_set, motor_ctx)
    return motor_data


def get_stimulus(data_set):
    """
     Args:
        data_set: (map) Subset of alldat

     Returns:
        stims[nTrials x 3]  bias: constant with ones
                            stim_left: contrast level for the right stimulus.
                            stim_right: contrast level for left stimulus.
    """

    stims = data_set['contrast_left'], data_set['contrast_right']
    stims = np.array(stims).T
    n_trials = stims.shape[0]
    stims = np.column_stack((np.ones(n_trials), stims))
    return stims


def get_response(data_set):
    """
     Args:
        data_set: Subset of alldat

     Returns:
          np.array: which side the response was (-1, 0, 1). Choices for the right stimulus are -1.
    """
    response = np.array(data_set['response'])
    return response


def get_binary_response(data_set):
    """
     To simplify the model we ares going to use a binary response instead of the 3 possible values: left, no-action, right
     to true or false comparing the stimulus to the expected actions.

     Args:
        data_set: Subset of alldat

     Returns:
          binary (np.array boolean) : Returns True or False
    """
    vis_left, vis_right = [data_set["stims"][:, 1], data_set["stims"][:, 2]]
    response = get_response(data_set)
    binary_response = np.sign(response) == np.sign(vis_left - vis_right)
    return binary_response


## make test size an input parameter
def split_validation_set(data_set):
    """
    Get a subset of alldat for validation purposes. This should be ~5%-10% of all the data.


    Returns:
     train_set (dict): dat['spks']: neurons by trials.
                       dat['brain_area']: brain area for each neuron recorded.
                       dat['stims']: contrast level for the right stimulus, which is always contralateral to the recorded brain areas.
                       dat['response']: which side the response was (-1,  1). Choices for the right stimulus are -1.
    """

    stims = get_stimulus(data_set)
    response = get_response(data_set)
    spk = data_set['spks']
    # # create training and testing vars
    stim_train, stim_test, spk_train, spk_test, response_train, response_test = train_test_split(stims, spk, response,
                                                                                                 test_size=0.1)
    train_set = {
        "spks": spk_train,
        "stims": stim_train,
        "response": response_train,
        "brain_area": data_set['brain_area']
    }
    validation_set = {
        "spks": spk_test,
        "stims": stim_test,
        "response": response_test,
        "brain_area": data_set['brain_area']
    }
    return train_set, validation_set


# Draft of sigmoid calculation
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# models
def train_linear_encoder(dat_input, dat_output):
    """
    function that creates a linear mapping from input to output using least squares
    :param dat_input: n_trials x n_features
    :param dat_output: n_trials x n_neurons
    :return: encoder, coefs,
    -> encoder, the encoder takes an input matrix multiplies by coeff to create an output_hat
    -. goes from n_features to n_neurons.
    -> coefs: n_features x n_neurons

    example use:
    encoder, encoder_coef = train_linear_encoder(stim, visual_dat)
    visual_dat_hat = encoder(stim)
    # or, in this case.
    visual_dat_hat = stim @ encoder_coef
    """

    model = LinearRegression(fit_intercept=False).fit(dat_input, dat_output)
    coefs = model.coef_.T

    def encoder(x):
        return x@coefs

    return encoder, coefs

### to do:
def train_logistic_decoder(dat_input, dat_output):
    """
    Decoder that goes from neurons to a 1 dimensional binary outcome. (can be expanded to more than 1D)
    :param dat_input: n_trials x n_neurons
    :param dat_output: n_trials x 1 [outcome]
    :return: decoder, decoder_coef.
    -> decoder: goes from n_neurons to 1 dimension (probability)
    """

    raise NotImplementedError


### to do:
def train_linear_inner_transition(dat_input, dat_output):
    """
    Transition method that fits a matrix to map between input and output. least-squares (ridge?)
    :param dat_input: n_trials x n_neurons1
    :param dat_output: n_trials x n_neurons2
    :return: transition, decoder_coef.
    -> decoder: goes from n_neurons to 1 dimension (probability)
    """

    raise NotImplementedError


# scores
## to do:
def get_r2(y, y_hat):
    # get variance explained
    raise NotImplementedError

## to do:
def get_acc(y, y_hat):
    # for binary, obtain accuracy:
    return np.mean(y==y_hat)

## to do:
def get_rmse(y, y_hat):
    raise NotImplementedError

## to do:
def get_nrmse(y, y_hat):
    raise NotImplementedError


