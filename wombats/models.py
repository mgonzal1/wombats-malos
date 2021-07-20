import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from . import utils
from . import metrics


## high level interfaces functions
def get_a1(stim, region_dat):
    
    n_region_neurons = region_dat.shape[0]
    n_analysis_time_bins = region_dat.shape[2]
    
    if n_analysis_time_bins==1:
        region_dat = region_dat.squeeze().T  # reshapes from neurons x trial x 1 bin, to trials x neurons
        
        # Model Linear Regression
        encoder, encoder_coefs, encoder_model = train_linear_encoder(stim, region_dat)
        A1 = encoder_coefs
        
#         # get training data prediction
#         region_dat_hat = encoder(stim)  
    
#         # get performance
#         encoder_perf = metrics.get_xval_perf(model=encoder_model, input_data=stim, output_data=region_dat, scoring='explained_variance')
    else:
        # need to implement a method for time window iteration
        raise NotImplementedError
    
    return A1 #, encoder_perf, region_dat_hat


def get_a2(region1_dat, region2_dat):
    """
    :param region1_dat: generated output from a1
    """
    
    # Get data
    #region2_dat = utils.get_region_data(train_set, region=region2, data_type='fr') 
    n_region2_neurons = region2_dat.shape[0]
    n_analysis_time_bins = region2_dat.shape[2]
    
    if n_analysis_time_bins==1:
        region2_dat = region2_dat.squeeze().T  # reshapes from neurons x trial x 1 bin, to trials x neurons
        
        # Model Linear Regression
        transition, transition_coefs, transition_model = train_linear_transition(region1_dat, region2_dat)
        A2 = transition_coefs
        
#         # get training data prediction
#         region2_dat_hat = transition(region1_dat)

#         # get performance
#         transition_perf = metrics.get_xval_perf(model=transition_model, input_data=region1_dat, output_data=region2_dat, scoring='explained_variance')  

    else:
        # need to implement a method for time window iteration
        raise NotImplementedError
    
    return A2#, transition_perf, region2_dat_hat


def get_a3(region_dat, output_dat):
    n_region_neurons = region_dat.shape[0]
    n_analysis_time_bins = region_dat.shape[2]
    
    if n_analysis_time_bins==1:
        region_dat = region_dat.squeeze().T  # reshapes from neurons x trial x 1 bin, to trials x neurons
        
        # Model Linear Regression
        decoder, decoder_coefs, decoder_model = train_logistic_decoder(region_dat, output_dat)
        A3 = decoder_coefs
        
#         # get prediction
#         region_dat_hat = decoder(region_dat)
    
#         # get performance
#         decoder_perf = metrics.get_xval_perf(model=decoder_model, input_data=region_dat, output_data=output_dat, scoring='balanced_accuracy')
#     else:
#         # need to implement a method for time window iteration
#         raise NotImplementedError
    
    return A3#, decoder_perf, region_dat_hat

def get_ae(A):
    AE = A[0]
    for ii in range(1, len(A)):
        AE = AE@A[ii]
        
    return AE

def get_model_output(stim, AE=None, A1=None, A2=None, A3=None, output_type='bool'):
    
    if AE is None:
        linear_output = stim @ A1 @ A2 @ A3
    else:
        linear_output = stim @ AE
        
    if output_type=='prob':
        output = utils.sigmoid(linear_output).flatten()
    elif output_type == 'bool':
        output = (linear_output>0).flatten()
        
    return output


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

    model = LinearRegression(fit_intercept=False)
    fit = model.fit(dat_input, dat_output)
    coefs = fit.coef_.T

    def encoder(x):
        # for a linreg model equivalent to x@coef
        return fit.predict(x)

    return encoder, coefs, model


def train_logistic_decoder(dat_input, dat_output, output_type='bool'):
    """
    Decoder that goes from neurons to a 1 dimensional binary outcome. (can be expanded to more than 1D)
    :param dat_input: n_trials x n_neurons
    :param dat_output: n_trials x 1 [outcome]; bool type
    :param output_type: string ['prob', 'bool'], if prob, returns the probability of binary decision==1
    :return: decoder, decoder_coef.
    -> decoder: goes from n_neurons to 1 dimension (probability)
    """
    # unpenalized LR
    model = LogisticRegression(penalty="none",
                               fit_intercept=False,
                               class_weight="balanced",
                               max_iter=5000)
    fit = model.fit(dat_input, dat_output)

    coefs = fit.coef_.T

    if output_type == 'bool':
        def decoder(x):
            # this is equivalent to sigmoid(x@coefs)>0.5;
            return fit.predict(x)
    elif output_type == 'prob':
        def decoder(x):
            return utils.sigmoid(x@coefs)
    else:
        print(f"{output_type} not implemented.")
        raise NotImplementedError

    return decoder, coefs, model


def train_linear_transition(dat_input, dat_output):
    """
    Transition method that fits a matrix to map between input and output. least-squares (ridge?)
    :param dat_input: n_trials x n_neurons1
    :param dat_output: n_trials x n_neurons2
    :return: transition, decoder_coef.
    -> inner_trans: goes from n_neurons to 1 dimension (probability)
    """
    n_trials, n_neurons = dat_input.shape
    model = LinearRegression(fit_intercept=False)
    fit = model.fit(dat_input, dat_output)
    coefs = model.coef_.T

    def transition(x):
        # for a linreg model equivalent to x@coef
        return fit.predict(x)

    return transition, coefs, model


def model_stim_to_decision(stim, encoder, transition, decoder):
    """
    function that maps stim to decision based on
    Args:
        stim: n_trials x n_features
        encoder: output from train_{}_encoder
        transition: output from train_{}_transition
        decoder: output from train_{}_decoder

    Returns:
        decisions: n_trials x 1 of outputs as determined by the decoder

    """
    return decoder(transition(encoder(stim)))


class two_region_linear_model():
    def __init__(self, data_set, region1='visual', region2='motor', n_xval=10, data_type='fr', time_window=None, **kwargs):
        self.region1 = region1
        self.region2 = region2
        self.n_xval = n_xval
        self.data_type = data_type
        
        if time_window is None:
            self.time_window = np.array([0, 0.5])
        else:
            if len(time_window)==2:
                if time_window[1]>time_window[0]:
                    self.time_window = time_window
                else:
                    sys.exit("Invalid Time Window")
            else:
                sys.exit("Invalid Time Window")
        
        if 'encoder' not in kwargs:
            self.encoder_func = train_linear_encoder
        else:
            self.encoder_func = kwargs['encoder']
            
        if 'decoder' not in kwargs:
            self.decoder_func = train_logistic_decoder
        else:
            self.decoder_func = kwargs['decoder']

        if 'transition' not in kwargs:
            self.transition_func = train_linear_transition
        else:
            self.transition_func = kwargs['transition']
            
        
#     def train():
        
        
        
        
            
        
            