import numpy as np
import scipy.linalg as la


def hankel_matrix(Xin, n_delay_coordinates, spacing=1):
    n_inputs, n_samples = Xin.shape

    X = np.zeros((n_inputs * (n_delay_coordinates), n_samples - spacing*(n_delay_coordinates-1)))
    for i in range(n_delay_coordinates):
        idxs = np.arange(spacing*i, spacing*(i+1) + n_samples - spacing*n_delay_coordinates)
        X[i*n_inputs:(i+1)*n_inputs] = Xin[:, idxs]
    return X


def differentiate(X, t, method='centered_difference', dt_max=None):
    if np.isscalar(t):
        # if t is a single number, assume uniform sampling with time step t
        return (X[:,2:]-X[:,:-2])/(2*t)
    else:
        # if t is an array, may have nonuniform sampling
        X_diff = X[:,2:] - X[:,:-2]
        t_diff = t[2:] - t[:-2]
        if dt_max is None:
            return X_diff*(1/t_diff)
        else:
            valid_idx = np.where(t_diff < 2*dt_max)[0]
            return X_diff[:,valid_idx]*(1/t_diff[valid_idx])


def integrate(X, t, dt_max=None):
    X_int = np.zeros(X.shape)
    if np.isscalar(t):
        # uniform time step
        X_diff = t/2*(X[:,1:] + X[:,:-1])
        X_int[:,1:] = np.cumsum(X_diff, axis=1)
    else:
        # nonuniform time step
        X_diff = (X[:,1:] + X[:,:-1])/2
        t_diff = t[1:] - t[:-1]
        if dt_max is None:
            X_int[:,1:] = np.cumsum(X_diff*t_diff, axis=1)
        else:
            invalid_idx = np.where(t_diff > dt_max)[0]
            if invalid_idx.size == 0:
                X_int[:,1:] = np.cumsum(X_diff*t_diff, axis=1)
            else:
                X_int[:,1:invalid_idx[0]+1] = np.cumsum(X_diff[:,:invalid_idx[0]]*t_diff[:invalid_idx[0]], axis=1)
                for i,idx in enumerate(invalid_idx):
                    if i == invalid_idx.size-1:
                        X_int[:,invalid_idx[i]+2:] = np.cumsum(X_diff[:,invalid_idx[i]+1:]*t_diff[invalid_idx[i]+1:], axis=1)
                    else:
                        X_int[:,invalid_idx[i]+2:invalid_idx[i+1]+1] = np.cumsum(X_diff[:,invalid_idx[i]+1:invalid_idx[i+1]] \
                                                                               *t_diff[invalid_idx[i]+1:invalid_idx[i+1]],
                                                                               axis=1)
    return X_int
