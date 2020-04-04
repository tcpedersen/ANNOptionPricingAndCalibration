# -*- coding: utf-8 -*-
import numpy as np
import torch
import itertools
from tqdm import tqdm

from utils.impvol import implied_volatility

from anderson_lake.integration import GaussianQuadrature
from anderson_lake.models import HestonModel
from anderson_lake.pricers import anderson_lake
from anderson_lake.options import EuropeanCallOption

# ==============================================================================
# === Functions
def scale_to_interval(x, a, b):
    xmin, xmax = np.min(x), np.max(x)
    return ( x - xmin ) / ( xmax - xmin ) * ( b - a ) + a

def generate_output_grid(x_min, x_max, y_min, y_max, num_dim):
    x_grid = np.linspace(x_min, x_max, num_dim[0])
    y_grid = np.linspace(y_min, y_max, num_dim[1])
    output_grid = np.array([x for x in itertools.product(x_grid, y_grid)])

    return output_grid

# ==============================================================================
# === Heston
def heston_closed_form(input_grid, output_grid):
    '''
    input_grid:     num_inputs x 7  (forward, vol, kappa, theta, sigma, rho, rate)
    output_grid:    num_outputs x 2 (tau, log-moneyness)
    '''
    
    m_impvol = np.ones((len(input_grid), len(output_grid))) * np.NaN
    m_prices = np.ones((len(input_grid), len(output_grid))) * np.NaN
    
    scheme = GaussianQuadrature(1e-8, 1e-8, 10000)
    for i, item_i in tqdm(enumerate(input_grid)):
        forward, vol, kappa, theta, sigma, rho, rate = item_i
        model = HestonModel(forward, vol, kappa, theta, sigma, rho, rate)

        for j, item_j in enumerate(output_grid):
            tau, log_moneyness = item_j
            strike = forward * np.exp(log_moneyness)
            option = EuropeanCallOption(tau, strike)
            
            # try calculating the implied volatility. If something fails
            # insert -1 instead.
            try:
                price = anderson_lake(model, option, scheme)
                impvol = implied_volatility(price, tau, forward, strike, rate)
            except:
                price = -1
                impvol = -1

            m_prices[i, j] = price
            m_impvol[i, j] = impvol

    return m_prices, m_impvol


def generate_heston_data(num_input, dim_output, skip=0):    
    # boundaries for parameter space
    forward_min, forward_max = 75, 125
    vol_min, vol_max = 0.05, 0.15

    kappa_min, kappa_max = 1, 5
    theta_min, theta_max = 0.05, 0.15
    sigma_min, sigma_max = 0.2, 0.5
    rho_min, rho_max = -1.0, -0.7
    rate_min, rate_max = 0, 0.1

    # volatility surface grid
    tau_min, tau_max = 0.1, 0.5
    log_moneyness_min, log_moneyness_max = -0.25, 0.25

    sobol = torch.quasirandom.SobolEngine(7)
    sobol.fast_forward(skip)

    # generate input grid
    input_grid = np.array(sobol.draw(int(num_input)))

    min_v = [forward_min, vol_min, kappa_min, theta_min, sigma_min, rho_min, rate_min]
    max_v = [forward_max, vol_max, kappa_max, theta_max, sigma_max, rho_max, rate_max]    

    for idx, (a, b) in enumerate(zip(min_v, max_v)):
        input_grid[:, idx] = scale_to_interval(input_grid[:, idx], a, b)

    # implied_volatility requires arguments to be of type float.   
    # Apparently np.float32 is not considered a float.
    input_grid = input_grid.astype(np.float64)

    # generate output grid
    output_grid = generate_output_grid(tau_min, tau_max, \
                        log_moneyness_min, log_moneyness_max, dim_output)

    m_prices, m_impvol = heston_closed_form(input_grid, output_grid)

    return input_grid, output_grid, m_prices, m_impvol

if __name__ == "__main__":
    # To change the parameter bounds, see generate_heston_data.
    num_training_samples = 100000
    num_test_samples = 10000

    # Training set
    input_grid, output_grid, m_prices, m_impvol = \
        generate_heston_data(num_training_samples, (8, 11), skip=0)
    np.savetxt("training-data\input-grid-train.csv", input_grid)
    np.savetxt("training-data\prices-train.csv", m_prices)
    np.savetxt("training-data\impvol-train.csv", m_impvol)

    # Test set
    input_grid, output_grid, m_prices, m_impvol = \
        generate_heston_data(num_test_samples, (8, 11), 
                             skip=num_training_samples)
    np.savetxt("training-data\input-grid-test.csv", input_grid)
    np.savetxt("training-data\prices-test.csv", m_prices)
    np.savetxt("training-data\impvol-test.csv", m_impvol)
