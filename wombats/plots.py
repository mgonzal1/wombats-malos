import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats


def plot_perturbation_effect(accu, session):
    ## assuming

    mean_accu_SV = np.nanmean(accu[0,:], axis=1)
    mean_accu_VM = np.nanmean(accu[1,:], axis=1)
    mean_accu_MC = np.nanmean(accu[2,:], axis=1)

    std_accu_SV = np.nanstd(accu[0,:], axis=1)
    std_accu_VM = np.nanstd(accu[1,:], axis=1)
    std_accu_MC = np.nanstd(accu[2,:], axis=1)

    sem_accu_SV = stats.sem(accu[0,:], axis=1)
    sem_accu_VM = stats.sem(accu[1,:], axis=1)
    sew_accu_MC = stats.sem(accu[2,:], axis=1)

    x_axis = np.arange(len(mean_accu_SV))

    f, ax1 = plt.subplots(figsize=(5, 5))

    ax1.plot(x_axis, mean_accu_SV, label="Stim-Visual", color='royalblue')
    ax1.plot(x_axis, mean_accu_VM, label='Visual-Motor', color='orange')
    ax1.plot(x_axis, mean_accu_MC, label='Motor-Choice', color='lightgreen')

    ax1.fill_between(x_axis, mean_accu_SV-sem_accu_SV, mean_accu_SV+sem_accu_SV, color='royalblue', alpha=0.2)
    ax1.fill_between(x_axis, mean_accu_VM-sem_accu_VM, mean_accu_VM+sem_accu_VM, color='orange', alpha=0.2)
    ax1.fill_between(x_axis, mean_accu_MC-sew_accu_MC, mean_accu_MC+sew_accu_MC, color='lightgreen', alpha=0.2)

    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    ax1.set_xlabel('perturbation %')
    ax1.set_ylabel('Mean Prediction\nAccuracy')
