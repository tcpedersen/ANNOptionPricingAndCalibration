# -*- coding: utf-8 -*-sys

# If this import causes issues, try appending the folder where the 
# lets_be_rational package is installed to the sys.path as follows:
# import sys
# lbr_path = path\\to\\package
# sys.path.append(lbr_path)
from lets_be_rational import implied_volatility_from_a_transformed_rational_guess

from scipy.stats import norm
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

def load_data(data_set):
    output_grid = np.loadtxt("training-data/output-grid.csv")

    if data_set == "realistic":
        X_raw = np.loadtxt("training-data/realistic/input-grid-train-real.csv")
        Y_raw = np.loadtxt("training-data/realistic/impvol-train-real.csv")
        
        X_test_raw = np.loadtxt("training-data/realistic/input-grid-test-real.csv") 
        Y_test_raw = np.loadtxt("training-data/realistic/impvol-test-real.csv")
    elif data_set == "large":
        X_raw = np.loadtxt("training-data/large/input-grid-train-large.csv")
        Y_raw = np.loadtxt("training-data/large/impvol-train-large.csv")
        
        X_test_raw = np.loadtxt("training-data/large/input-grid-test-large.csv") 
        Y_test_raw = np.loadtxt("training-data/large/impvol-test-large.csv")
    
    idx_all_pos_train = (Y_raw > 0).all(axis=1)
    X_train = X_raw[idx_all_pos_train, :]
    Y_train = Y_raw[idx_all_pos_train, :]
    
    idx_all_pos_test = (Y_test_raw > 0).all(axis=1)
    X_test = X_test_raw[idx_all_pos_test, :]
    Y_test = Y_test_raw[idx_all_pos_test, :]
    
    return output_grid, X_train, Y_train, X_test, Y_test
