import numpy as np
import scipy.linalg as la
from .utils import hankel_matrix


def compute_dmd_rank(s, truncation_method, threshold=None):
    if truncation_method == 'soft':
        if threshold is None:
            raise ValueError('threshold must be defined for soft threshold truncation method')
        r = np.where(s > threshold)[0].size
    else:
        if threshold is None:
            r = s.size
        else:
            r = threshold
    return r


def compute_real_dmd_modes(Phi, omega, b):
    Phi_real = np.zeros(Phi.shape)
    omega_realpart = []
    omega_imagpart = []

    omega_tmp = []
    b_tmp = []

    omega_idx = np.arange(omega.size)
    omega_copy = omega.copy()
    i = 0
    while i < omega.size:
        if np.iscomplex(omega_copy[0]):
            Phi_real[:,i] = 2*np.real(Phi[:,omega_idx[0]])
            omega_realpart.append(np.real(omega_copy[0]))
            omega_realpart.append(np.real(omega_copy[0]))
            b_tmp.append(b[omega_idx[0]])

            Phi_real[:,i+1] = -2*np.imag(Phi[:,omega_idx[0]])
            omega_imagpart.append(np.imag(omega_copy[0]))
            omega_imagpart.append(-np.imag(omega_copy[0]))
            b_tmp.append(b[omega_idx[0]].conj())

            # find complex conjugate eval
            conj_idx = np.argsort(np.abs(np.conj(omega_copy[0]) - omega_copy))[0]

            # mask out this eigenvalue and its conjugate
            mask = np.ones(omega_idx.size, dtype=bool)
            mask[[0,conj_idx]] = False
            omega_idx = omega_idx[mask]
            omega_copy = omega_copy[mask]
            i += 2
        else:
            omega_realpart.append(np.real(omega_copy[0]))
            omega_imagpart.append(0.0)
            omega_tmp.append(omega_copy[0])
            b_tmp.append(np.real(b[omega_idx[0]]))
            Phi_real[:,i] = np.real(Phi[:,omega_idx[0]])
            omega_idx = omega_idx[1:]
            omega_copy = omega_copy[1:]
            i += 1

    omega = np.vstack((np.array(omega_realpart), np.array(omega_imagpart)))
    b = np.array(b_tmp)
    return Phi_real, omega, b


class DMD:
    def __init__(self, truncation='hard', threshold=None, time_delay=1, time_delay_spacing=1):
        self.truncation = truncation
        if (self.truncation == 'soft') and (threshold is None):
            self.threshold = 1e-10
        else:
            self.threshold = threshold
        self.time_delay = time_delay
        self.time_delay_spacing = time_delay_spacing

    def fit(self, X_fit, dt, real=None, t0=0.0, sample_spacing=1, dt_scale=1):
        self.dt = dt
        if real is None:
            self.real = (np.where(np.iscomplex(X_fit))[0].size < 1)
        else:
            self.real = real

        if self.time_delay > 1:
            H = hankel_matrix(X_fit, self.time_delay, spacing=self.time_delay_spacing)
            X = H[:,:-dt_scale:sample_spacing]
            Xp = H[:,dt_scale::sample_spacing]
        else:
            X = X_fit[:, :-dt_scale:sample_spacing]
            Xp = X_fit[:, dt_scale::sample_spacing]

        U,s,Vt = la.svd(X, full_matrices=False)
        r = compute_dmd_rank(s, self.truncation, threshold=self.threshold)
        self.rank = r
        U = U[:,:r]
        s = s[:r]
        V = Vt[:r].conj().T

        tmp = np.dot(Xp, V/s)
        A_tilde = np.dot(U.conj().T, tmp)
        evals, evecs = la.eig(A_tilde)

        # check for a negative real eigenvalue, which is sometimes an issue with time delay DMD
        if np.any(evals[~np.iscomplex(evals)] <= 0):
            raise ValueError('found negative eigenvalue')

        # get modes and normalize them
        Phi = np.dot(tmp, evecs)
        Phi = Phi / np.sqrt(np.sum(Phi**2, axis=0)) / np.sqrt(r)
        omega = np.log(evals)/(dt*dt_scale)

        b = la.lstsq(Phi, X[:,0])[0]

        sort_order = np.argsort(np.abs(b))[::-1]
        Phi = Phi[:,sort_order]
        omega = omega[sort_order]
        b = b[sort_order]

        # only take the beginning rows of Phi in the case of time delay
        Phi = Phi[:X_fit.shape[0]]

        if not self.real:
            self.Phi = Phi
            self.omega = omega
            self.b = b
        else:
            self.Phi, self.omega, self.b = compute_real_dmd_modes(Phi, omega, b)

        self.A = np.dot(tmp, U.conj().T)
        self.Atilde = A_tilde
        self.A_continuous = (self.A - np.eye(self.A.shape[0]))/(dt*dt_scale)
        self.Atilde_continuous = (self.Atilde - np.eye(self.Atilde.shape[0]))/(dt*dt_scale)
        self.P = U

    def reduced_dynamics(self, t, imaginary_evals=False):
        if self.omega.ndim == 2:
            x = np.zeros((self.rank, t.size))
            i = 0
            while i < self.omega.shape[1]:
                if imaginary_evals:
                    growth_rate = 0
                else:
                    growth_rate = self.omega[0,i]

                if self.omega[1,i] != 0:
                    x[i] = np.exp(growth_rate*t)*(np.real(self.b[i])*np.cos(self.omega[1,i]*t)
                                                      - np.imag(self.b[i])*np.sin(self.omega[1,i]*t))
                    x[i+1] = np.exp(growth_rate*t)*(np.imag(self.b[i])*np.cos(self.omega[1,i]*t)
                                                      + np.real(self.b[i])*np.sin(self.omega[1,i]*t))
                    i += 2
                else:
                    x[i] = np.exp(growth_rate*t)*np.real(self.b[i])
                    i += 1
            return x
        return (np.exp(np.outer(self.omega,t)).conj().T*self.b).conj().T

    def reconstruct(self, t, imaginary_evals=False):
        return np.dot(self.Phi, self.reduced_dynamics(t, imaginary_evals=imaginary_evals))
