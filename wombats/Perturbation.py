import random
import numpy


def a1_pert(encoder_output, n_pert):
    # perturbar por fila/row
    trial, neuron = encoder_output.shape
    list_pert = list(np.arange(neuron-1))
    x = random.sample(list_pert, n_pert)
    encoder_output[:,x] = 0
    new_encoder = encoder_output

    return new_encoder, x

