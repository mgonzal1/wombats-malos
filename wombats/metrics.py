import numpy as np


def r2(y, y_hat):
    """
        Coefficient of determination for vectors.
        :param y: array n_samps of dependent variable;
            can also work with matrices organized as n_samps x n_predictions
        :param y_hat: array with result of a prediction, same shape as y
        :return: R2. float. or array if y,y_hat are matrices. in that case r2 is an array of length n_predictions
        """
    if y.ndim == 1:
        y = y.reshape(1, -1)
        y_hat = y_hat.reshape(1, -1)
    y_bar = y.mean(axis=1)
    y_bar = y_bar.reshape(-1, 1)

    return 1 - ((y - y_hat) ** 2).sum(axis=1) / ((y - y_bar) ** 2).sum(axis=1)


def binarry_acc(y, y_hat):
    # for binary, obtain accuracy:
    return np.mean(y == y_hat)


def mse(y, y_hat):
    """
    Mean Square Error Calculation
    :param y: array n_samps of dependent variable;
        can also work with matrices organized as n_samps x n_predictions
    :param y_hat: array with result of a prediction, same shape as y
    :return: mse: float. mean square error. or array of length n_predictions
    """
    if y.ndim == 1:
        y = y.reshape(1, -1)
        y_hat = y_hat.reshape(1, -1)
    return np.mean((y - y_hat) ** 2, axis=1)


def rmse(y, y_hat):
    """
    Root Mean Square Error Calculation
    :param y: array n_samps of dependent variable;
        can also work with matrices organized as n_samps x n_predictions
    :param y_hat: array with result of a prediction, same shape as y
    :return: rmse: float. root mean square error, or array of length n_predictions
    """
    return np.sqrt(mse(y, y_hat))


def nrmse(y, y_hat):
    """
    Normalized Root Mean Square Error Calculation.
    Divides the RMSE by the mean of variable y
    :param y: array n_samps of dependent variable
    :param y_hat: array of n_samps as a result of a prediction
    :return: nrmse: float. normalize root mean square error, or array of length n_predictions
    """
    if y.ndim == 1:
        y = y.reshape(1, -1)
        y_hat = y_hat.reshape(1, -1)
        x = 0
        
    return rmse(y, y_hat) / np.mean(y, axis=1)

def get_xval_perf(model, input_data, output_data, n_xval=N_XVAL, scoring='explained_variance'):
    
    if output_data.ndim==1:
        output_data = output_data[:, np.newaxis]

    n_outputs = output_data.shape[1]
    
    perf = np.zeros((n_outputs, n_xval))
    for output_idx in range(n_outputs):
        perf[output_idx] = cross_val_score(model, input_data, output_data[:, output_idx], cv=n_xval, scoring=scoring)
        
    return perf

def regression_performance_by_neuron(data, data_hat, xval_perf):
    #MSE and R2 per unit
    scores = pd.DataFrame(columns = ['r2', 'xval_r2', 'mse', 'nrmse'])
    scores['r2'] = metrics.r2(data.T, data_hat.T)
    scores['mse'] = metrics.mse(data.T, data_hat.T)
    scores['nrmse'] = metrics.nrmse(data.T, data_hat.T)
    scores['xval_r2'] = np.median(xval_perf,axis=1)

    return scores.mean()
