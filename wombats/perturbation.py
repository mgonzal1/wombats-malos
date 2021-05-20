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

    params = {"normal_mu":0, "normal_sigma":1 ,"poisson_lam": 1, 'random_seed': 0}
    params.update(kwargs)
    
    np.random.seed(params['random_seed'])
    
    # ablation 
    n_trials, n_neurons = connection_mat.shape
   
    n_pert_neurons = np.ceil(n_neurons*pct_pert)
    pert_neurons = np.sort(np.random.permutation(n_neurons)[:n_pert_neurons])
    
    if pert_type=="ablation":
        pert_connection_mat[:,pert_neurons] = 0
    
    elif pert_type=="white_noise":
        # use connectivity mean & variance?
        pert_connection_mat[:,pert_neurons] = np.random.normal(params['normal_mu'], params['normal_sigma'],n_pert_neurons)
    
    # poisson
    elif pert_type == 'poisson':
        pert_connection_mat[:,pert_neurons] = np.random.poisson(params['poisson_lam'], n_pert_neurons)
    
    # scramble connectivity - [Milli] sample from weight distribution

    # scramble neurons - [Adriana] scramble n_pert_neurons columns

    # strengthen - [Fran] param is % increase of true connectivity



    
    



    
    return pert_connection_mat, pert_idx 


