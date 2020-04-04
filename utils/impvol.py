# -*- coding: utf-8 -*-
import sys
from lets_be_rational import implied_volatility_from_a_transformed_rational_guess

from scipy.stats import norm
from scipy.optimize import brentq, fminbound

import numpy as np

def black_scholes(tau, forward, strike, rate, vol):
    d1 = ( np.log(forward / strike) + (vol**2 / 2) * tau ) / (vol * np.sqrt(tau))
    d2 = d1 - vol * np.sqrt(tau)
    
    return np.exp(-rate * tau) * (forward * norm.cdf(d1) - strike * norm.cdf(d2))


def implied_volatility(price, tau, forward, strike, rate, theta = 1):
    deflator = np.exp(-rate * tau)
    
    impvol = implied_volatility_from_a_transformed_rational_guess( \
                price/deflator, forward, strike, tau, theta)

    return impvol
