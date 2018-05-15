import numpy as np
from scipy.integrate import ode


def lorenz(sigma, rho, beta, tau=1.):
    f = lambda t,x : [sigma*(x[1] - x[0])/tau, (x[0]*(rho - x[2]) - x[1])/tau, (x[0]*x[1] - beta*x[2])/tau]
    jac = lambda t,x : [[-sigma/tau, sigma/tau, 0.],
                        [(rho - x[2])/tau, -1./tau, -x[0]/tau],
                        [x[1]/tau, x[0]/tau, -beta/tau]]
    return f,jac


def simulate_lorenz(dt, n_timesteps, x0=None, sigma=10., rho=28., beta=8/3, tau=1.):
    if x0 is None:
        x0 = [-8, 7, 27]

    f,jac = lorenz(sigma, rho, beta, tau=tau)
    r = ode(f,jac).set_integrator('zvode', method='bdf')
    r.set_initial_value(x0, 0.0)

    x = [x0]
    t = [0.0]
    xprime = [f(0.0,x0)]
    while r.successful() and len(x) < n_timesteps:
        r.integrate(r.t + dt)
        x.append(np.real(r.y))
        xprime.append(f(r.t,np.real(r.y)))
        t.append(r.t)

    return np.array(x).T, np.array(t), np.array(xprime).T


def rossler(a, b, c, tau=1.):
    f = lambda t,x : [(-x[1] - x[2])/tau, (x[0] + a*x[1])/tau, (b + x[2]*(x[0] - c))/tau]
    jac = lambda t,x : [[0., -1/tau, -1/tau],
                        [1/tau, a/tau, 0.],
                        [x[2]/tau, 0., x[0]/tau]]
    return f,jac


def simulate_rossler(dt, n_timesteps, x0=None, a=0.2, b=0.2, c=5.7, tau=1.):
    if x0 is None:
        x0 = [0, 10, 0]

    f,jac = rossler(a, b, c, tau=tau)
    r = ode(f,jac).set_integrator('zvode', method='bdf')
    r.set_initial_value(x0, 0.0)

    x = [x0]
    t = [0.0]
    xprime = [f(0.0,x0)]
    while r.successful() and len(x) < n_timesteps:
        r.integrate(r.t + dt)
        x.append(np.real(r.y))
        xprime.append(f(r.t,np.real(r.y)))
        t.append(r.t)

    return np.array(x).T, np.array(t), np.array(xprime).T


def vanderpol_oscillator(mu, tau=1):
    f = lambda t,x : [x[1]/tau, (mu*(1-x[0]**2)*x[1] - x[0])/tau]
    jac = lambda t,x : [[0., 1./tau], [(-2.*mu*x[0]*x[1] - 1.)/tau, -mu*x[0]**2/tau]]
    return f,jac


def simulate_vanderpol_oscillator(dt, n_timesteps, x0=None, mu=10., tau=1):
    if x0 is None:
        x0 = [2.,0.]

    f,jac = vanderpol_oscillator(mu,tau)
    r = ode(f,jac).set_integrator('zvode', method='bdf')
    r.set_initial_value(x0, 0.)

    x = [x0]
    t = [0.]
    xprime = [f(0.,x0)]
    while r.successful() and len(x) < n_timesteps:
        r.integrate(r.t + dt)
        x.append(np.real(r.y))
        xprime.append(f(r.t,np.real(r.y)))
        t.append(r.t)

    return np.array(x).T, np.array(t), np.array(xprime).T


def duffing_oscillator(alpha, beta, gamma, delta, omega, tau):
    f = lambda t,x : [x[1]/tau, (-delta*x[1] - alpha*x[0] - beta*x[0]**3 + gamma*np.cos(x[2]))/tau, omega]
    jac = lambda t,x : [[0., 1./tau, 0.], [(-alpha - 3*beta*x[0]**2)/tau, -delta/tau, -gamma*np.sin(x[2])/tau], [0., 0., 0.]]
    return f,jac


def simulate_duffing_oscillator(dt, n_timesteps, x0=None, alpha=1., beta=1, gamma=0., delta=0., omega=1., tau=1):
    if x0 is None:
        x0 = [1.,0.,0.]
    elif len(x0)==2:
        x0 += 0.

    f,jac = duffing_oscillator(alpha,beta,gamma,delta,omega,tau)
    r = ode(f,jac).set_integrator('zvode', method='bdf')
    r.set_initial_value(x0, x0[2])

    x = [x0]
    t = [x0[2]]
    xprime = [f(x0[2],x0)]
    while r.successful() and len(x) < n_timesteps:
        r.integrate(r.t + dt)
        x.append(np.real(r.y))
        xprime.append(np.real(f(r.t,r.y)))
        t.append(r.t)

    return np.array(x).T,np.array(t),np.array(xprime).T


def coupled_vdp(mu1, mu2, c1, c2, tau1=1, tau2=1):
    f = lambda t,x: [(x[1] + c1*x[2])/tau1, (mu1*(1-x[0]**2)*x[1] - x[0])/tau1,
                     (x[3] + c2*x[0])/tau2, (mu2*(1-x[2]**2)*x[3] - x[2])/tau2]
    jac = lambda t,x : [[0., 1./tau1, c1/tau1, 0.],
                        [(-2.*mu1*x[0]*x[1] - 1.)/tau1, (-mu1*x[0]**2)/tau1, 0., 0.],
                        [c2/tau2, 0., 0., 1./tau2],
                        [0., 0., (-2.*mu2*x[2]*x[3] - 1.)/tau2, (-mu2*x[2]**2)/tau2]]
    return f,jac


def simulate_coupled_vdp(dt, n_timesteps, x0=None, mu1=1., mu2=1., c1=1., c2=1., tau1=1, tau2=1):
    if x0 is None:
        x0 = [2.,0.,0.,2.]

    f,jac = coupled_vdp(mu1,mu2,c1,c2, tau1=tau1, tau2=tau2)
    r = ode(f,jac).set_integrator('zvode', method='bdf')
    r.set_initial_value(x0, 0.)

    x = [x0]
    t = [0.]
    xprime = [f(0.,x0)]
    while r.successful() and len(x) < n_timesteps:
        r.integrate(r.t + dt)
        x.append(np.real(r.y))
        xprime.append(f(r.t,np.real(r.y)))
        t.append(r.t)

    return np.array(x).T, np.array(t), np.array(xprime).T


def coupled_vdp_lorenz(mu, sigma, rho, beta, c1, c2, tau1=1, tau2=1):
    f = lambda t,x: [(x[1] + c1*x[2])/tau1, (mu*(1-x[0]**2)*x[1] - x[0])/tau1,
                     (sigma*(x[3] - x[2]) + c2*x[0])/tau2, (x[2]*(rho - x[4]) - x[3])/tau2, (x[2]*x[3] - beta*x[4])/tau2]
    jac = lambda t,x : [[0., 1./tau1, c1/tau1, 0., 0.],
                        [(-2.*mu*x[0]*x[1] - 1.)/tau1, (-mu*x[0]**2)/tau1, 0., 0., 0.],
                        [c2/tau2, 0., -sigma/tau2, sigma/tau2, 0.],
                        [0., 0., (rho - x[4])/tau2, -1./tau2, -x[2]/tau2],
                        [0., 0., x[3]/tau2, x[2]/tau2, -beta/tau2]]
    return f,jac


def simulate_coupled_vdp_lorenz(dt, n_timesteps, x0=None, mu=1., sigma=10., rho=28., beta=8/3, c1=1., c2=1., tau1=1, tau2=1):
    if x0 is None:
        x0 = [2., 0., -8., 7., 27.]

    f,jac = coupled_vdp_lorenz(mu,sigma,rho,beta,c1,c2,tau1=tau1,tau2=tau2)
    r = ode(f,jac).set_integrator('zvode', method='bdf')
    r.set_initial_value(x0, 0.)

    x = [x0]
    t = [0.]
    xprime = [f(0.,x0)]
    while r.successful() and len(x) < n_timesteps:
        r.integrate(r.t + dt)
        x.append(np.real(r.y))
        xprime.append(f(r.t,np.real(r.y)))
        t.append(r.t)

    return np.array(x).T, np.array(t), np.array(xprime).T
