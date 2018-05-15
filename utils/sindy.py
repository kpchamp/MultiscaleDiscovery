import numpy as np
from scipy.integrate import ode
from scipy.special import binom
from .utils import differentiate, integrate


def pool_data(Xin, poly_order=2, use_sine=False, include_constant=True, varname='x'):
    if poly_order > 5:
        poly_order = 5

    if Xin.ndim == 1:
        T = 1
        n = Xin.size
    else:
        n,T = Xin.shape
    n_vars = 0
    for k in range(poly_order+1):
        n_vars += int(binom(n+k-1,k))
    if use_sine:
        n_vars += 2*n
    Xout = np.zeros((n_vars,T))

    labels = []

    if include_constant:
        Xout[0] = np.ones(T)
        index = 1
        labels.append('1')
    else:
        index = 0

    for i in range(n):
        Xout[index] = Xin[i]
        index += 1
        labels.append('%s%d' % (varname,i+1))

    if (poly_order >= 2):
        for i in range(n):
            for j in range(i,n):
                Xout[index] = Xin[i]*Xin[j]
                index += 1
                labels.append('%s%d*%s%d' % (varname,i+1,varname,j+1))

    if (poly_order >= 3):
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    Xout[index] = Xin[i]*Xin[j]*Xin[k]
                    index += 1
                    labels.append('%s%d*%s%d*%s%d' % (varname,i+1,varname,j+1,varname,k+1))

    if (poly_order >= 4):
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    for l in range(k,n):
                        Xout[index] = Xin[i]*Xin[j]*Xin[k]*Xin[l]
                        index += 1
                        labels.append('%s%d*%s%d*%s%d*%s%d' % (varname,i+1,varname,j+1,varname,k+1,varname,l+1))

    if (poly_order >= 5):
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    for l in range(k,n):
                        for m in range(l,n):
                            Xout[index] = Xin[i]*Xin[j]*Xin[k]*Xin[l]*Xin[m]
                            index += 1
                            labels.append('%s%d*%s%d*%s%d*%s%d*%s%d' % (varname,i+1,varname,j+1,varname,k+1,varname,l+1,varname,m+1))

    if use_sine:
        for i in range(n):
            Xout[index] = np.sin(Xin[i])
            index += 1
            labels.append('sin(%s%d)' % (varname,i+1))
        for i in range(n):
            Xout[index] = np.cos(Xin[i])
            index += 1
            labels.append('cos(%s%d)' % (varname,i+1))

    return Xout,labels


def sindy_setup(Xin, poly_order, use_sine, t, method='derivative', dt_max=None):
    if method == 'integral':
        Theta, labels = pool_data(Xin, poly_order, use_sine)
        RHS = integrate(Theta, t, dt_max=dt_max)
        if np.isscalar(t) or (dt_max is None):
            LHS = Xin - np.broadcast_to(Xin[:,0], (Xin.shape[1],Xin.shape[0])).T
        else:
            time_gaps = np.where((t[1:] - t[:-1] > dt_max) | (t[1:] - t[:-1] < 0))[0]
            if time_gaps.size == 0:
                LHS = Xin - np.broadcast_to(Xin[:,0], (Xin.shape[1],Xin.shape[0])).T
            else:
                time_gaps += 1
                LHS = Xin.copy()
                LHS[:,:time_gaps[0]] -= np.broadcast_to(Xin[:,0], (time_gaps[0],Xin.shape[0])).T
                for i in range(time_gaps.size):
                    if i == time_gaps.size-1:
                        LHS[:,time_gaps[i]:] -= np.broadcast_to(Xin[:,time_gaps[i]],
                                                                (Xin.shape[1] - time_gaps[i],Xin.shape[0])).T
                    else:
                        LHS[:,time_gaps[i]:time_gaps[i+1]] -= np.broadcast_to(Xin[:,time_gaps[i]],
                                                                              (time_gaps[i+1] - time_gaps[i],
                                                                               Xin.shape[0])).T
        return RHS, LHS, labels
    else:
        LHS = differentiate(Xin, t, dt_max=dt_max)
        if np.isscalar(t) or (dt_max is None):
            RHS, labels = pool_data(Xin[:,1:-1], poly_order, use_sine)
        else:
            time_gaps = np.where((t[1:] - t[:-1] > dt_max) | (t[1:] - t[:-1] < 0))[0]
            if time_gaps.size == 0:
                RHS, labels = pool_data(Xin[:,1:-1], poly_order, use_sine)
            else:
                valid_idx = np.where((t[2:] - t[:-2] < 2*dt_max) & (t[2:] - t[:-2] > 0))[0]
                RHS, labels = pool_data(Xin[:,valid_idx+1], poly_order, use_sine)
        return RHS, LHS, labels


class SINDy:
    def __init__(self, use_sine=False, differentiation_method='derivative'):
        self.use_sine = use_sine
        self.differentiation_method = differentiation_method

    def fit(self, Xin, poly_order, t=None, Xprime=None, coefficient_threshold=.01, dt_max=None):
        RHS, LHS, labels = sindy_setup(Xin, poly_order, self.use_sine, t, method=self.differentiation_method,
                                       dt_max=dt_max)
        self.labels = labels

        n,T = LHS.shape
        Xi = np.linalg.lstsq(RHS.T,LHS.T)[0]

        for k in range(10):
            small_inds = (np.abs(Xi) < coefficient_threshold)
            Xi[small_inds] = 0
            for i in range(n):
                big_inds = ~small_inds[:,i]
                if np.where(big_inds)[0].size == 0:
                    continue
                Xi[big_inds,i] = np.linalg.lstsq(RHS[big_inds].T, LHS[i])[0]

        self.poly_order = poly_order
        self.Xi = Xi
        self.error = np.sum(np.mean((LHS - np.dot(Xi.T,RHS))**2,axis=1))

    def reconstruct(self, x0, t0, dt, n_timesteps):
        f = lambda t,x: np.dot(self.Xi.T, pool_data(np.real(x), poly_order=self.poly_order, use_sine=self.use_sine)[0])

        r = ode(f).set_integrator('zvode', method='bdf')
        r.set_initial_value(x0, t0)

        x = [x0]
        t = [t0]
        while r.successful() and len(x) < n_timesteps:
            r.integrate(r.t + dt)
            x.append(np.real(r.y))
            t.append(r.t)

        return np.array(x).T, np.array(t)

    def print(self, threshold=1e-10):
        for j in range(self.Xi.shape[1]):
            eqn = "x%d' =" % (j+1)
            for i,l in enumerate(self.labels):
                if np.abs(self.Xi[i,j]) > threshold:
                    eqn += " (%f)%s +" % (self.Xi[i,j],l)
            print(eqn.strip('+'))
