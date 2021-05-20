# imports
from matplotlib import rcParams
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.signal as signal
import os
import requests
from pathlib import Path


# import data
def load_steinmentz(source_path=None):
    """
    Args:
        source_path: string indicating the location of the data
        if None, and data does not exists locally, the function will attempt to load data from the original URLs.
    Returns:
        alldat
    """
    file_names = []
    n_files = 3
    if source_path is None:
        source_path = Path(os.getcwd())
        source_path_flag = False
    else:
        source_path_flag = True

    file_found_flag = True
    for j in range(n_files):
        file_names.append(source_path / ('steinmetz_part%d.npz' % j))
        file_found_flag = file_found_flag and file_names[j].exists()

    # if no source given, and file does not exist, load from URL and save.
    if (not source_path_flag) and (not file_found_flag):
        print("No source folder given. Sourcing data from original URL & saving data.")
        print(f"Saving to {source_path}")
        url = ["https://osf.io/agvxh/download",
               "https://osf.io/uv3mw/download",
               "https://osf.io/ehmw2/download"]

        for jj, file in enumerate(file_names):
            try:
                r = requests.get(url[jj])
            except requests.ConnectionError:
                print("!!! Failed to download data !!!")
            else:
                if r.status_code != requests.codes.ok:
                    print("!!! Failed to download data !!!")
                else:
                    with open(file, "wb") as fid:
                        fid.write(r.content)

    alldat = np.array([])
    for file in file_names:
        if file.exists():
            alldat = np.hstack((alldat, np.load(file, allow_pickle=True)['dat']))
        else:
            print(f"File not found: {file}")
            raise FileNotFoundError

    return alldat


def spikes_to_fr(spk_data, samp_rate=0.01, smoothing_window=0.05, downsampling_factor=1, exclude_pre=None,
                 exclude_post=None, axis=2):
    """
    Args:
        spk_data: np.ndarray n_neurons x n_trials x n_trial_samps.
            operations performed along time (axis=2), otherwise specify axis.
        samp_rate: seconds per time bin [secs/bin]
        smoothing_window: float [secs], smoothing window in seconds
        downsampling_factor: int or 'all' factor to downsample signal, if 'all', a single estimate per neuron / trial \
        is returned and no filtering is performed. if int, it needs to be divisible by n_trial_samps
        exclude_pre: float [secs], seconds to ignore at the beginning of each trial
        exclude_post: float [secs], seconds to ignore at the end of each trial
        axis: int [au], dimension on which to operate.

        Note that exclusion of data for each trial is only for the output. Smoothing operations will still use these
        time windows to compute.

    Returns:
        spk_data: np.ndarray n_nuerons x n_trials x n_resulting_bins
            where the number of resulting bins depends on downsampling factor and the exclusion of data.

    """

    if not (spk_data.ndim == 3):
        print(f'Currently not supporting data not in 3 dimensions. Num of data dims = {spk_data.ndim}')
        raise NotImplementedError

    if axis == 2:
        n_neurons, n_trials, n_trial_samps = spk_data.shape
    else:
        print('Currently can only operate on axis=3')
        raise NotImplementedError

    if exclude_pre is None:
        trial_start_samp = 0
    else:
        trial_start_samp = np.round(exclude_pre / samp_rate).astype(int)

    if exclude_post is None:
        trial_end_samp = n_trial_samps
    else:
        trial_end_samp = n_trial_samps - np.round(exclude_pre / samp_rate).astype(int)

    trial_time = (trial_end_samp-trial_start_samp) * samp_rate

    if isinstance(downsampling_factor, str):
        if downsampling_factor == 'all':
            temp = spk_data[:, :, trial_start_samp:trial_end_samp].sum(axis=2)/trial_time
            return temp[:, :, np.newaxis]
        else:
            print(f'Downsampling {downsampling_factor} not understood.')
            raise NotImplementedError
    elif isinstance(downsampling_factor, int):
        if np.mod(n_trial_samps, downsampling_factor) > 0:
            print(f'Dowsampling of {downsampling_factor} not a multiple of n_trial_samps {n_trial_samps}')
            raise ValueError

        n_trial_samps = int(n_trial_samps/downsampling_factor)
        trial_start_samp = int(trial_start_samp/downsampling_factor)
        trial_end_samp = int(trial_end_samp/downsampling_factor)

    else:
        raise TypeError

    filter_len = np.round(smoothing_window / samp_rate).astype(int)

    if downsampling_factor >= 3*filter_len:
        print('The requested downsampling requires a larger filter. Increase the smoothing.')
        print(f'Min smoothing_window = {samp_rate*downsampling_factor/3:.3f} secs')
        raise ValueError

    # convert to float and spks/sec, and flatten
    fr_data = spk_data.astype(np.float32)/samp_rate
    fr_data = fr_data.reshape(n_neurons,-1)

    # filter by neuron
    fr_data = filter_data(fr_data, filter_len)

    # downsample
    fr_data_ds = fr_data[:, ::downsampling_factor]

    # re-arrange to trials
    fr_trial_data = fr_data_ds.reshape(n_neurons, n_trials, n_trial_samps)

    # select relevant samples and return
    return fr_trial_data[:, :, trial_start_samp:trial_end_samp]


def filter_data(data, filter_len):
    """
    Filter along 2nd dimension looping through first. this is hann FIR filter.
    Args:
        data: n_signals x n_samps
        filter_len: length of filter window for hann filter

    Returns: filtered data.

    """

    if data.ndim == 1:
        # add singleton dimension
        data = data.reshape(1, -1)

    filt_coef = signal.windows.hann(filter_len)
    filt_coef /= filt_coef.sum()

    out = np.zeros_like(data)
    for ii, sig in enumerate(data):
        out[ii] = signal.filtfilt(filt_coef, 1, sig)

    return out


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
    
    new_data_set.update({"spks": data_set["spks"][:, index_trials[0], :]})
    new_data_set.update({"brain_area": data_set["brain_area"]})
    new_data_set.update({"response": data_set["response"][index_trials]})
    new_data_set.update({"contrast_right": data_set["contrast_right"][index_trials]})
    new_data_set.update({"contrast_left": data_set["contrast_left"][index_trials]})
    return new_data_set


def get_dat_from_area(dat, brain_area, dat_type):

    n_neurons = dat[dat_type].shape[0]
    index_neurons = np.zeros(n_neurons, dtype=bool)
    for neuron in range(n_neurons):
        index_neurons[neuron] = dat['brain_area'][neuron] in brain_area

    area_fr = dat[dat_type][index_neurons]

    return area_fr, index_neurons


### to do:
# create get methods that take a brain region argument:
# eg. get_region_data(data_set,'visual')
# data set should then have a field defining what sub areas are part of visual:
# data_set['areas']['visual'] = ["VISa", "VISam", "VISl", "VISp", "VISpm", "VISrl"]
def get_region_data(data_set, region, data_type):
    regions = {"visual": ["VISa", "VISam", "VISl", "VISp", "VISpm", "VISrl"],
               "motor": ["MOp", "MOs"]}

    region_data, _ = get_dat_from_area(data_set, regions[region], data_type)
    return region_data


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


def get_data_splits(data_set, val_size=0.1, random_seed=None):
    """
    Get a subset of alldat for validation purposes. This should be ~5%-10% of all the data.
    Excludes no go or no response trials and returns these as separate sets

    Returns:
     train_set (dict): dat['spks']: neurons by trials.
                       dat['brain_area']: brain area for each neuron recorded.
                       dat['stims']: contrast level for the right stimulus, which is always contralateral to the recorded brain areas.
                       dat['response']: which side the response was (-1,  1). Choices for the right stimulus are -1.
                       
     val_set (dict): validation data split
     
     nogo_set (dict): no go trials & animal responded
     
     noresp_set (dict): no resp trials & go trials
     
     nono_set (dict): no go trials & no response (correct)
     
     ** note contrast values are expected to be positive.
    """

    if random_seed is None:
        np.random.seed(42)
    else:
        np.random.seed(random_seed)

    stims = get_stimulus(data_set)
    delta_contrast = data_set['contrast_left']-data_set['contrast_right']
    response = get_response(data_set)
    spks = data_set['spks']

    n_trials = len(response)
    all_trials = np.arange(n_trials)
    
    nono_trials = np.where((response==0) & (delta_contrast==0))[0]
    noresp_trials = np.setdiff1d( np.where(response==0)[0], nono_trials)
    nogo_trials = np.setdiff1d( np.where(delta_contrast==0)[0], nono_trials)
    no_trials = np.union1d(np.union1d(nogo_trials, noresp_trials), nono_trials)
    
    valid_trials = np.setdiff1d(all_trials, no_trials)
    n_valid_trials = len(valid_trials)
    
    n_val_trials = np.ceil(n_valid_trials*val_size).astype(int)
    n_train_trials = n_valid_trials-n_val_trials

    train_trials = np.sort( np.random.permutation(valid_trials)[:n_train_trials] )
    val_trials = np.setdiff1d(valid_trials, train_trials)
    
    train_set = create_trial_set(data_set, train_trials)
    val_set = create_trial_set(data_set, val_trials)
    nogo_set = create_trial_set(data_set, nogo_trials)
    noresp_set = create_trial_set(data_set, noresp_trials)
    nono_set = create_trial_set(data_set, nono_trials)
    
    return train_set, val_set, nogo_set, noresp_set, nono_set

def create_trial_set(data_set, trial_idx):
    stims = get_stimulus(data_set)
    delta_contrast = data_set['contrast_left']-data_set['contrast_right']
    response = get_response(data_set)
    spks = data_set['spks']

    new_set = {
                "spks": spks[:, trial_idx],
                "stims": stims[trial_idx],
                "delta_contrast": delta_contrast[trial_idx],
                "response": response[trial_idx],
                "brain_area": data_set['brain_area'],
                "trial_idx": trial_idx
                }
    
    if 'fr' in data_set.keys():
        new_set['fr'] = data_set['fr'][:, trial_idx]
    
    return new_set

def get_xval_trials(data_set, n_xval=10, random_seed=None):
    """
    
    """
    skf = StratifiedKFold(n_splits=n_xval, random_state=None, shuffle=False)
    
    y = data_set['response']
    n_trials = len(y)
    gskf = skf.split(X=np.zeros(n_trials), y=y)
    
    train = np.zeros(n_xval, dtype=object)
    test = np.zeros(n_xval, dtype=object)
    
    cnt = 0
    for train_idx, test_idx in gskf:
        train[cnt] = train_idx
        test[cnt] = test_idx
        cnt+=1
    return train, test


# Draft of sigmoid calculation
def sigmoid(z):
    return 1 / (1 + np.exp(-z))


# scores
## to do:
        
def session_unit_counter(data, regions):
    neurons_in_session = []
    session_unit_region_counts = pd.DataFrame(index=range(len(data)), columns=regions.keys())
    for i in range (0,len(data)):
        session = data[i]
        unit_brain_sub_area = session['brain_area']
        for region, sub_areas in regions.items():
            count = 0
            for sub_area in sub_areas:
                count += np.count_nonzero(unit_brain_sub_area == sub_area)    
            session_unit_region_counts.loc[i,region ] = count

    return session_unit_region_counts

class dataset():
    TRIAL_TIME_RANGE = np.array([-0.5, 2])
    SAMP_RATE = 0.01
    regions = {"visual": ["VISa", "VISam", "VISl", "VISp", "VISpm", "VISrl"],
               "motor": ["MOp", "MOs"]}
    
    def __init__(self, data, region1='visual', region2='motor', 
                 n_min_region_units=20, data_type='fr', downsampling_factor='all'):
        self.region1 = region1
        self.region2 = region2
        self.n_min_region_units=n_min_region_units
        self.data_type = data_type
        self.downsampling_factor = downsampling_factor
        
        self.valid_session_idx = self.select_sessions(data, region1, region2)
        
    def select_sessions(self, data, region1, region2):
        
        assert region1 in regions.keys(), f"Error {region1} not in available regions."
        assert region2 in regions.keys(), f"Error {region2} not in available regions."
        
        unit_table = session_unit_counter(data, regions)
        
        valid_sessions = (unit_table[region1]>=self.n_min_region_units) & (unit_table[region2]>=self.n_min_region_units)
        valid_sessions_idx = np.where(valid_sessions)[0]
        
        return valid_session_idx
        
    def select_data(self, data):
        valid_data = data[self.valid_sessions_idx]
        
        
        