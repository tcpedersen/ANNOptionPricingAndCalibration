# -*- coding: utf-8 -*-
import sys
lbr_path = "C:\\Users\\tobia\\Anaconda3\\envs\\puk\\Lib\\site-packages\\lets_be_rational"
sys.path.append(lbr_path)
from lets_be_rational import implied_volatility_from_a_transformed_rational_guess

from scipy.stats import norm
from scipy.optimize import brentq, fminbound

import numpy as np

# ==============================================================================
# === Black and Scholes

def black_scholes(tau, forward, strike, rate, vol):
    d1 = ( np.log(forward / strike) + (vol**2 / 2) * tau ) / (vol * np.sqrt(tau))
    d2 = d1 - vol * np.sqrt(tau)
    
    return np.exp(-rate * tau) * (forward * norm.cdf(d1) - strike * norm.cdf(d2))


def implied_volatility(price, tau, forward, strike, rate, theta = 1):
    deflator = np.exp(-rate * tau)
    
    impvol = implied_volatility_from_a_transformed_rational_guess( \
                price/deflator, forward, strike, tau, theta)

    return impvol

# ==============================================================================
# === Heston

def explicit_complex_e(x):
    a, b = x.real, x.imag
    return np.exp(a) * ( np.cos(b) + 1j * np.sin(b) )


MACHINE_EPSILON = np.finfo(float).eps
NUMPY_COMPLEX128_MAX = np.finfo(np.complex128).max
NUMPY_LOG_COMPLEX128_MAX = np.log(NUMPY_COMPLEX128_MAX)

class HestonModel:
    def __init__(self, forward, vol, kappa, theta, sigma, rho):
        self.forward = forward
        self.vol = vol
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho
        
    def __str__(self):
        out_str = f"forward: {self.forward}\n\r" +\
                  f"vol: {self.vol}\n\r" +\
                  f"kappa: {self.kappa}\n\r" +\
                  f"theta: {self.theta}\n\r" +\
                  f"sigma: {self.sigma}\n\r" +\
                  f"rho: {self.rho}\n\r"
        return out_str
    
    def cf(self, z, tau):
        beta = self.kappa - 1j * self.sigma * self.rho * z
        D = np.sqrt(beta**2 + self.sigma**2 * z * (z + 1j))

        if beta.real * D.real + beta.imag * D.imag > 0:
            r = - self.sigma**2 * z * (z + 1j) / (beta + D)
        else:
            r = beta - D

        if D != 0:
            y = np.expm1(-D * tau) / (2 * D)
        else:
            y = -tau / 2

        A = self.kappa * self.theta / self.sigma**2 * \
            (r * tau - 2 * np.log1p(- r * y))
        
        B = z * (z + 1j) * y / (1 - r * y)
        
        exponent = A + B * self.vol
        
        if exponent > NUMPY_LOG_COMPLEX128_MAX:
            raise ValueError("Too large exponent in HestonModel.cf")
        
        return np.exp(exponent)
    
    def log_cf(self, alpha, tau):
        # Evaluation of ln phi(-1j * (1 + alpha))
        beta = self.kappa - self.rho * self.sigma * (1 + alpha)
        Dsq = beta**2 - self.sigma**2 * (1 + alpha) * alpha
        
        if Dsq > 0:
            D = np.sqrt(Dsq)
            coshdt = np.cosh(D * tau / 2)
            sinhdt = np.sinh(D * tau / 2) / D
            nume = coshdt + beta * sinhdt
            
        else:
            # D = 1j * x
            x = np.sqrt(-Dsq)
            coshdt = np.cos(x * tau / 2)
            sinhdt = np.sin(x * tau / 2) / x
            nume = coshdt + beta * sinhdt

        A = self.kappa * self.theta / self.sigma**2 *\
            (beta * tau - np.log(nume**2))
        B = alpha * (1 + alpha) * sinhdt / nume

        return A + B * self.vol

class EuropeanCallOption:
    def __init__(self, tau, strike, rate):
        self.tau = tau
        self.strike = strike
        self.rate = rate
    def __str__(self):
        out_str = f"tau: {self.tau}\n\r" +\
                  f"strike: {self.strike}\n\r" +\
                  f"rate: {self.rate}\n\r"
        return out_str


def call_option_price(model: HestonModel, option: EuropeanCallOption) -> float:
    omega = np.log(model.forward / option.strike)

    # Selection of phi
    r = model.rho - model.sigma * omega / \
        ( model.vol + model.kappa * model.theta * option.tau )
    if r * omega < 0:
        phi = np.pi / 12 * np.sign(omega)
    else:
        phi = 0

    # Selection of alpha
    eps = np.sqrt(MACHINE_EPSILON)
    alpha_min, alpha_max = alpha_min_max(model, option)
    if omega >= 0:
        alpha, val = locate_optimal_alpha(model, option, alpha_min, -1 - eps)
    elif omega < 0 and model.kappa - model.rho * model.sigma > 0:
        alpha, val = locate_optimal_alpha(model, option, eps, alpha_max)
    else:
        alpha, val = locate_optimal_alpha(model, option, eps, alpha_max)
        if val > 9:
            alpha, val = locate_optimal_alpha(model, option,
                                              alpha_min, -1 - eps)

    # Define integrand
    tphi = np.tan(phi)
    tphip = 1 + 1j * tphi
    
    def Q(z):
        return model.cf(z - 1j, option.tau) / (z * (z - 1j))

    def f(x):
        dexp = np.exp(-x * tphi * omega + 1j * x * omega)
        return (dexp * Q(-1j * alpha + x * tphip) * tphip).real    
    
    new_estimate, old_estimate = np.nan, np.nan
    err_tol = 1e-3
    h = 0.5
    for n in range(10000):
        old_estimate = new_estimate
        new_estimate = double_exp_quadrature(f, h)
        
        if abs( old_estimate - new_estimate ) < err_tol:
            break
        else:
            h /= 2
    else:
        raise ValueError(f"Integral did not converge. Final difference is {abs( old_estimate - new_estimate )}")
    
    I = np.exp(alpha * omega) * new_estimate
    R = model.forward * (alpha <= 0) - option.strike * (alpha <= -1) - \
        0.5 * (model.forward * (alpha == 0) - option.strike * (alpha == -1))
    return np.exp(- option.rate * option.tau) * (R - model.forward / np.pi * I)


def x(n, h):
    return np.exp(np.pi / 2 * np.sinh(n * h))


def w(n, h):
    return np.pi / 2 * np.cosh(n * h) * x(n, h)


def double_exp_quadrature(f, h):
    eps = np.sqrt(MACHINE_EPSILON)
    max_iter = 2**15

    # Positive half
    cum_sum_plus = 0
    old_delta, new_delta = -1, 1
    n = 0
    threshold = 0
    while abs(new_delta - old_delta) > threshold or n < 2:
        old_delta = new_delta
        new_delta = w(n, h) * f(x(n, h))
        
        if np.isnan(new_delta):
            raise ValueError(f"nan produced in double_exp_quadrature at step {n} with step size {h}.\n\r")
        elif n > max_iter:
            raise ValueError(f"no convergence of partial sum in double_exp_quadrature with step size {h}.\n" + 
                             f"Final change {abs(new_delta - old_delta)} vs threshold {threshold}.\n\r")
        else:
            cum_sum_plus += new_delta
            threshold = eps * abs(cum_sum_plus)
            n += 1

    # Negative half
    cum_sum_minus = 0
    old_delta, new_delta = -1, 1
    n = -1
    threshold = 0
    while abs(new_delta - old_delta) > threshold or n > -2:
        old_delta = new_delta
        new_delta = w(n, h) * f(x(n, h))
        
        if np.isnan(new_delta):
            raise ValueError(f"nan produced in double_exp_quadrature at step {n}.")
        elif n > max_iter:
            raise ValueError(f"no convergence of partial sum in double_exp_quadrature\n" + 
                             f"Final change {abs(new_delta - old_delta)} vs threshold {threshold}.")
        else:
            cum_sum_minus += new_delta
            threshold = eps * abs(cum_sum_minus)
            n -= 1

    return h * (cum_sum_plus + cum_sum_minus)



def locate_optimal_alpha(model, option, a, b):
    omega = np.log(model.forward / option.strike)
    obj_func = lambda alpha: model.log_cf(alpha, option.tau) -\
        np.log(alpha * (alpha + 1)) + alpha * omega

    alpha, val = fminbound(obj_func, a, b, full_output=True)[0:2]
    return alpha.real, val.real


def k_plus_minus(x: float, sign: int, model: HestonModel, 
                 option: EuropeanCallOption) -> float:
    A = model.sigma - 2 * model.rho * model.kappa
    B = (model.sigma - 2 * model.rho * model.kappa)**2 +\
        4 * (model.kappa**2 + x**2 / option.tau**2) * (1 - model.rho**2)
    C = 2 * model.sigma * (1 - model.rho**2)

    return (A + sign * np.sqrt(B)) / C


def critical_moments_func(k: float, model: HestonModel,
                          option: EuropeanCallOption) -> float:
    kminus, kplus = k_plus_minus(0, -1, model, option), k_plus_minus(0, 1, model, option)
    beta = model.kappa - model.rho * model.sigma * k
    D = np.sqrt(beta**2 + model.sigma**2 * (-1j * k) * ((-1j * k) + 1j))

    if k > kplus or k < kminus:
        D = abs(D)
        return np.cos(D * option.tau / 2) + \
            beta * np.sin(D * option.tau / 2) / D
    else:
        # Then D is real, but it will still return a complex type.
        D = D.real
        return np.cosh(D * option.tau / 2) + \
            beta * np.sinh(D * option.tau / 2) / D


def alpha_min_max(model: HestonModel,
                  option: EuropeanCallOption) -> (float, float):
    kminus, kplus = k_plus_minus(0, -1, model, option), k_plus_minus(0, 1, model, option)
    eps = np.sqrt(MACHINE_EPSILON)
    
    # Find kmin
    kmin2pi = k_plus_minus(2 * np.pi, -1, model, option)
    kmin = brentq(critical_moments_func, kmin2pi + eps, kminus - eps,
                  args=(model, option))

    # Find kmax
    kps = model.kappa - model.rho * model.sigma
    if kps > 0:
        a, b = kplus, k_plus_minus(2 * np.pi, 1, model, option)
    elif kps < 0:
        T = -2 / (model.kappa - model.rho * model.sigma * kplus)
        if option.tau < T:
            a, b = kplus, k_plus_minus(np.pi, 1, model, option)
        else:
            a, b = 1, kplus
    else:
        a, b = kplus, k_plus_minus(np.pi, 1, model, option)
    kmax = brentq(critical_moments_func, a + eps, b - eps,
                  args=(model, option))

    return kmin - 1, kmax - 1
