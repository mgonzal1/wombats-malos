import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression


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


# to do:
def train_logistic_decoder(dat_input, dat_output):
    """
    Decoder that goes from neurons to a 1 dimensional binary outcome. (can be expanded to more than 1D)
    :param dat_input: n_trials x n_neurons
    :param dat_output: n_trials x 1 [outcome]
    :return: decoder, decoder_coef.
    -> decoder: goes from n_neurons to 1 dimension (probability)
    """
    model = LogisticRegression(fit_intercept=False),fit(dat_input,dat_output)
    coefs = model.coef_.T
    def encoder(x):
        return x@coefs

    return encoder, coefs

    raise NotImplementedError


# to do:
def train_linear_inner_transition(dat_input, dat_output):
    """
    Transition method that fits a matrix to map between input and output. least-squares (ridge?)
    :param dat_input: n_trials x n_neurons1
    :param dat_output: n_trials x n_neurons2
    :return: transition, decoder_coef.
    -> decoder: goes from n_neurons to 1 dimension (probability)
    """
   


    raise NotImplementedError
