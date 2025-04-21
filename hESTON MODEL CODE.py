import numpy as np
from scipy.integrate import quad

def chfun_heston(s0, v0, vbar, a, vvol, r, rho, t, w):
    alpha = -w**2/2 - 1j*w/2
    beta = a - rho*vvol*1j*w
    gamma = vvol**2/2
    h = np.sqrt(beta**2 - 4*alpha*gamma)
    rplus = (beta + h) / vvol**2
    rminus = (beta - h) / vvol**2
    g = rminus / rplus

    C = a * (rminus * t - (2 / vvol**2) * np.log((1 - g * np.exp(-h*t))/(1-g)))
    D = rminus * (1 - np.exp(-h * t)) / (1 - g * np.exp(-h*t))

    y = np.exp(C*vbar + D*v0 + 1j*w*np.log(s0*np.exp(r*t)))
    return y

def call_heston_cf(s0, v0, vbar, a, vvol, r, rho, t, k):
    int1 = lambda w: np.real(np.exp(-1j*w*np.log(k)) * chfun_heston(s0, v0, vbar, a, vvol, r, rho, t, w-1j) / (1j*w*chfun_heston(s0, v0, vbar, a, vvol, r, rho, t, -1j)))
    int1 = quad(int1, 0, 100)[0]
    pi1 = int1 / np.pi + 0.5

    int2 = lambda w: np.real(np.exp(-1j*w*np.log(k)) * chfun_heston(s0, v0, vbar, a, vvol, r, rho, t, w) / (1j*w))
    int2 = quad(int2, 0, 100)[0]
    pi2 = int2 / np.pi + 0.5

    y = s0 * pi1 - np.exp(-r*t) * k * pi2
    return y
