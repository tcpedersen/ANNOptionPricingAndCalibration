# -*- coding: utf-8 -*-
import numpy as np
from keras.models import load_model
from scipy.optimize import differential_evolution
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm

from utils.load_data import load_data

import matplotlib.pyplot as plt

# ==============================================================================
# === Load model
ann_model = load_model("models/trained-model-realistic.h5")

# =============================================================================
# === Load data
# Either realistic or large
output_grid, X, Y, X_test, Y_test = load_data("realistic")

# ==============================================================================
# === Normalise
norm_features = MinMaxScaler(feature_range = (-1, 1))
norm_labels = StandardScaler()

norm_x = norm_features.fit_transform(X)
norm_y = norm_labels.fit_transform(Y)

norm_x_test = norm_features.transform(X_test)
norm_y_test = norm_labels.transform(Y_test)

# ==============================================================================
# === Perform calibration

bounds = tuple((-1, 1) for _ in range(5))
calibrated_params = np.ones(norm_x_test[:, 1:6].shape) * np.nan
calibrated_status = np.ones(norm_x_test[:, 1:6].shape[0]) * np.nan

for idx, (features, labels) in tqdm(enumerate(zip(norm_x_test, norm_y_test))):
    def objective_ann(x, *args):
        input_v = np.hstack([features[0], x, features[-1]]).reshape(1, -1)
        prediction = ann_model.predict(input_v)
    
        return np.linalg.norm(prediction[0, :] - labels)

    result = differential_evolution(objective_ann, bounds)
    calibrated_params[idx, :] = result.x
    calibrated_status[idx] = result.success

np.savetxt("calibration-data\calibrated-params-ann.csv", calibrated_params)
np.savetxt("calibration-data\calibrated-status-ann.csv", calibrated_status)

# Errors in normalised space
norm_error_abs = abs(calibrated_params - norm_x_test[:, 1:6])
norm_error_mean = norm_error_abs.mean(axis=0)
norm_error_std = norm_error_abs.std(axis=0)
norm_error_max = np.quantile(norm_error_abs, 0.99, axis=0)

# ==============================================================================
# === Visualise
split_idx = 8
num_moneyness = 11
idx = 3488 # pick the index of the surface you wish to see

true_params = norm_x_test[idx, 2:7]
forward, rate = norm_x_test[idx, 0], norm_x_test[idx, -1]

cal_params = calibrated_params[idx, ]

print([round(x, 4) for x in true_params])
print([round(x, 4) for x in cal_params])

for visualise_idx in range(split_idx):
    input_cal = np.hstack([forward, cal_params, rate]).reshape(1, -1)
    output_cal = ann_model.predict(input_cal)
    impvol_cal = norm_labels.inverse_transform(output_cal)
    surface_cal = np.vstack(np.hsplit(impvol_cal, split_idx))[visualise_idx, :]

    input_true = norm_x_test[idx, :].reshape(1, -1)
    output_true = ann_model.predict(input_true)
    impvol_true = norm_labels.inverse_transform(output_true)
    surface_true = np.vstack(np.hsplit(impvol_true, split_idx))[visualise_idx, :]

    surface_target = np.vstack(np.hsplit(Y_test[idx, :], split_idx))[visualise_idx, :]

    plt.figure()
    plt.plot(np.linspace(-0.25, 0.25, num_moneyness), surface_target, 
             label="Observed surface", color = "red")
    plt.plot(np.linspace(-0.25, 0.25, num_moneyness), surface_true, "--",
             label="ANN True Parameters", color = "green")
    plt.plot(np.linspace(-0.25, 0.25, num_moneyness), surface_cal, ".",
             label="ANN Calibrated Parameters", color = "blue")
    plt.legend()
    plt.show()
