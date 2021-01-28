import random
import numpy as np

# todo:
# generalized to a2, a3
# pertubation type:
# ablation => set neurons to zero
# white noise
# poisson noise (if using spike counts not firing rate)
# neural noise (get neuron specific firing probability) [needs neuron stats]
# hyperactivate => set neuron to max fr [needs max fr as input]
# scramble conections (permute columns); [needs how many to scramble]

def pert_connections(connection_mat, pct_pert=0.2, pert_type="ablation"):
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

