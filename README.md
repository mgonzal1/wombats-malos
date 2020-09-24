# WOMBAT Project

Development of an end-to-end (stim-to-action) neural network trained only with an encoder (stim->sensory representation) and a decoder (motor choice representation -> action). The intermediate mapping can be trained with additional brain representations, or with a mapping between sensory and motor representations. 

The goal is to create a testbed model trained on real neural data, to test hypotheses ranging from number of neurons needed for computation, plasticity, timing, and the utility of other brain regions in the stim-to-action behavior. Because the model is never explicitly trained on stim-to-action, the internal representations and computations of the model can be compared to real neural data without any model bias.

__Model__

![Alt text](/img/model.png?raw=true "Title")
