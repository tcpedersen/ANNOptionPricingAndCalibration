# -*- coding: utf-8 -*-
import numpy as np
from scipy.optimize import least_squares
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm


from utils.impvol import implied_volatility
from utils.load_data import load_data

from anderson_lake.integration import GaussianQuadrature
from anderson_lake.models import HestonModel
from anderson_lake.pricers import anderson_lake
from anderson_lake.options import EuropeanCallOption

# =============================================================================
# === Load data
# Either realistic or large
output_grid, X, Y, X_test, Y_test = load_data("realistic")

# =============================================================================
# === Normalise
norm_features = MinMaxScaler(feature_range = (-1, 1))
norm_labels = StandardScaler()

norm_x = norm_features.fit_transform(X)
norm_y = norm_labels.fit_transform(Y)

norm_x_test = norm_features.transform(X_test)
norm_y_test = norm_labels.transform(Y_test)

# =============================================================================
# === Define function to calculate a surface
def heston_surface(model, scheme, output_grid):
    m_impvol = np.ones((1, len(output_grid))) * np.NaN
    
    for j, item_j in enumerate(output_grid):
        tau, log_moneyness = item_j
        strike = model.forward * np.exp(log_moneyness)
        option = EuropeanCallOption(tau, strike)
        
        price = anderson_lake(model, option, scheme)
        impvol = implied_volatility(price, tau, model.forward, strike, \
                                    model.rate)

        m_impvol[0, j] = impvol
    return m_impvol

# ==============================================================================
# === Perform calibration
scheme = GaussianQuadrature(1e-8, 1e-8, 10000)

bounds = tuple((-1, 1) for _ in range(5))
calibrated_params = np.ones(norm_x_test[:, 1:6].shape) * np.nan
calibrated_status = np.ones(norm_x_test[:, 1:6].shape[0]) * np.nan

for idx, (features, labels) in tqdm(enumerate(zip(norm_x_test, norm_y_test))):
    def objective_ann(x, *args):
        input_v = np.hstack([features[0], x, features[-1]]).reshape(1, -1)
        forward, vol, kappa, theta, sigma, rho, r = \
            norm_features.inverse_transform(input_v)[0, :]
        model = HestonModel(forward, vol, kappa, theta, sigma, rho, r)
        
        prediction = norm_labels.transform(
                heston_surface(model, scheme, output_grid)
            )
    
        return prediction[0, :] - labels
    
    # Initial guess
    x0 = np.random.normal(loc=0, scale=0.001, size=5)
    bounds = ((-1, -1, -1, -1, -1), (1, 1, 1, 1, 1))
    
    try:
        result = least_squares(objective_ann, x0, bounds=bounds)
        calibrated_params[idx, :] = result.x
        calibrated_status[idx] = result.success
    except:
        calibrated_status[idx] = False

np.savetxt("calibration-data\calibrated-params-benchmark.csv", calibrated_params)
np.savetxt("calibration-data\calibrated-status-benchmark.csv", calibrated_status)
