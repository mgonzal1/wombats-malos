import random
import numpy as np

# todo:
# generalized to a2, a3

# pertubation type:
# ablation => set neurons to zero
# white noise
# poisson noise (if using spike counts not firing rate) [needs rate x neuron]
# neural noise (get neuron specific firing probability) [needs stats x neuron]
# hyperactivate => set neuron to max fr [needs max fr x neurons]
# scramble conections (permute columns);
#
# next next todo: input pertubation idx
def pert_connections(connection_mat, pct_pert=0.2, pert_type="ablation", **kwargs):
    """
    inputs:
        connection_mat : matrix with shape n_trials x n_neurons (or weights)
        pct_pert : perturbation percentaje
        pert_type : one of the following perturbations
            - 'ablation'
            - 'white_noise'
            - 'poisson'
            - 'neural_noise'
            - 'hyperactive'
            - 'scramble'

        **kwargs : dictionary with neural population metrics (such as : mu, sigma, maxFR, etc)
        
    returns: 
        per_connection_mat : matrix with perturbed values 
        pertd_idx : index of neurons(or weights) that were perturbed

    """


    # ablation 
    n_trials, n_neurons = connection_mat.shape
    list_pert = list(np.arange(n_neurons-1))
    ablated_n = random.sample(list_pert, n_neurons*pct_pert)
    pert_connection_mat[:,ablated_n] = 0


    # white_noise
    


    # poisson



    # neural_noise



    # hyperactive



    # scramble
    



    
    return pert_connection_mat, pert_idx 




# dummy
def a1_pert(encoder_output, n_pert):
    # perturbar por columna/neurona
    # todo: n_pert <= n_neurons
    n_trials, n_neurons = encoder_output.shape
    list_pert = list(np.arange(n_neurons-1))
    ablated_neurons = random.sample(list_pert, n_pert)
    encoder_output[:,ablated_neurons] = 0
    new_encoder = encoder_output

    return new_encoder, ablated_neurons

