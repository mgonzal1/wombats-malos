import random
import numpy as np


def a1_pert(encoder_output, n_pert):
    # perturbar por fila/row
    n_trials, n_neurons = encoder_output.shape
    list_pert = list(np.arange(n_neurons-1))
    ablated_neurons = random.sample(list_pert, n_pert)
    encoder_output[:,ablated_neurons] = 0
    new_encoder = encoder_output

    return new_encoder, ablated_neurons

