import random
import numpy as np

# todo:
# generalized to a2, a3
# pertubation type:
# ablation => set neurons to zero
# white noise
def a1_pert_whitenoise(encoder_output, n_pert):
    # perturbar por columna/neurona
    # todo: n_pert <= n_neurons
    n_trials, n_neurons = encoder_output.shape
    list_pert = list(np.arange(n_neurons-1))
    ablated_neurons = random.sample(list_pert, n_pert)
     # create white noise series
    noise_series = [gauss(0.0, 1.0) for i in range(n_neurons)]
    encoder_output_2[:,ablated_neurons] = noise_series
    # noise drawn from normal distribution + mean activity across time?
    noise = np.random.normal(0,1,original.shape)
    encoder_output_1[:,ablated_neurons] = noise + np.mean(encoder_output[:,ablated_neurons])
    new_encoder_1 = encoder_output_1
    new_encoder_2 = encoder_output_2


# poisson noise (if using spike counts not firing rate) [needs rate x neuron]
# neural noise (get neuron specific firing probability) [needs stats x neuron]
# hyperactivate => set neuron to max fr [needs max fr x neurons]
# scramble conections (permute columns);
#
# next next todo: input pertubation idx
def pert_connections(connection_mat, pct_pert=0.2, pert_type="ablation", **kwargs):
    """
    inputs:
    returns:
    """
   
    
    return pert_connection_mat, pert_idx 


def a1_pert(encoder_output, n_pert):
    # perturbar por columna/neurona
    # todo: n_pert <= n_neurons
    n_trials, n_neurons = encoder_output.shape
    list_pert = list(np.arange(n_neurons-1))
    ablated_neurons = random.sample(list_pert, n_pert)
    encoder_output[:,ablated_neurons] = 0
    new_encoder = encoder_output

    return new_encoder, ablated_neurons

