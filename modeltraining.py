# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import LearningRateScheduler
from keras.optimizers import Adam
from keras.initializers import VarianceScaling

from sklearn.preprocessing import MinMaxScaler, StandardScaler

import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# =============================================================================
# === Learning rate scheduler
def lr_schedule(n, alpha):
    a, b = 1e-4, 1e-2
    n1, n2, n3 = 0, 24, 74

    if n <= n2:
        return (a - b)/(n1 - n2) * n - (a*n2 - b*n1) / (n1 - n2)
    elif n2 < n < n3:
        return -(a - b)/(n2 - n3) * n + (a*n2 - b*n3) / (n2 - n3)
    else:
        return a

# =============================================================================
# === Load data
# Change the paths to training-data/realistic/input-grid-train-real.csv etc.
# to train on the realistic parameter space instead.

output_grid = np.loadtxt("training-data/output-grid.csv")

# Training data
X_raw = np.loadtxt("training-data/large/input-grid-train-large.csv")
Y_raw = np.loadtxt("training-data/large/impvol-train-large.csv")

# Only use implied volatilities where there was no failure (see 
# datageneration.py) for details.
idx_all_pos_train = (Y_raw > 0).all(axis=1)
X = X_raw[idx_all_pos_train, :]
Y = Y_raw[idx_all_pos_train, :]

# Test data
X_test_raw = np.loadtxt("training-data/large/input-grid-test-large.csv") 
Y_test_raw = np.loadtxt("training-data/large/impvol-test-large.csv")

# Only use implied volatilities where there was no failure (see 
# datageneration.py) for details.
idx_all_pos_test = (Y_test_raw > 0).all(axis=1)
X_test = X_test_raw[idx_all_pos_test, :]
Y_test = Y_test_raw[idx_all_pos_test, :]

# =============================================================================
# === Extract and set variables for training
num_samples, num_input_units = X.shape
num_hidden_units = 30
num_hidden_layers = 4
num_samples, num_output_units = Y.shape
split_idx = 8 # a result of the 8 x 11 output grid.

# =============================================================================
# === Normalise
norm_features = MinMaxScaler(feature_range = (-1, 1))
norm_labels = StandardScaler()

norm_x = norm_features.fit_transform(X)
norm_y = norm_labels.fit_transform(Y)

norm_x_test = norm_features.transform(X_test)
norm_y_test = norm_labels.transform(Y_test)

# ==============================================================================
# === Fit model (note the model can be saved using 
# model.save( -- insert path --)
model = Sequential()
model.add(Dense(num_hidden_units, activation='softplus', use_bias=True, \
                input_dim=num_input_units, 
                kernel_initializer=VarianceScaling()))
for _ in range(num_hidden_layers - 1):
    model.add(Dense(num_hidden_units, activation = 'softplus', use_bias = True, 
                    kernel_initializer = VarianceScaling()))
model.add(Dense(num_output_units , activation = "linear", use_bias = True, 
                kernel_initializer = VarianceScaling()))

adam = Adam(lr=0.1)
model.compile(optimizer=adam, loss='mean_squared_error')

lr_scheduler = LearningRateScheduler(lr_schedule, verbose=0)

model.fit(norm_x, norm_y, epochs=100, batch_size=32, callbacks=[lr_scheduler], 
          verbose=2)

# ==============================================================================
# === Plot loss function
plt.figure()
plt.plot(model.history.history['loss'][2:], label = 'train error')
plt.show()

# ==============================================================================
# === Compute point-wise errors (test set)
prediction_test = norm_labels.inverse_transform(model.predict(norm_x_test))
error_rel_mat = abs(prediction_test / Y_test  - 1) 

# Values to plot
error_rel_mean_test = np.vstack(np.hsplit(error_rel_mat.mean(axis = 0), split_idx))
error_rel_std_test = np.vstack(np.hsplit(error_rel_mat.std(axis = 0), split_idx))
error_rel_max_test = np.vstack(np.hsplit(np.quantile(error_rel_mat, 0.99, axis = 0), split_idx))

errors = [error_rel_mean_test, error_rel_std_test, error_rel_max_test]
names = ["mean-relative-error", "std-relative-error", "quantile-relative-error"]
for error, name in zip(errors, names):
    # Figure mean relative error
    fig, ax = plt.subplots()
    
    im = plt.imshow(error)
    
    xlab = [f"{round(z, 2)}" for z in np.unique(output_grid[:, 1])]
    ylab = [z for z in [f"{round(z, 2)}" for z in np.unique(output_grid[:, 0])]]
    
    ax.set_xticks(np.arange(len(xlab)))
    ax.set_yticks(np.arange(len(ylab)))
    
    ax.set_xticklabels(xlab)
    ax.set_yticklabels(ylab)
    
    ax.set_xlabel("Inverse log-moneyness")
    ax.set_ylabel("Maturity")
    
    def fmt(x, pos):
        if name is names[0]:
            return "{:.2%}".format(x)
        elif name is names[1]:
            return "{:.1%}".format(x)
        elif name is names[2]:
            return "{:.1%}".format(x)
    
    cbar = ax.figure.colorbar(im, format=ticker.FuncFormatter(fmt))
    cbar.outline.set_visible(False)
    
    plt.show() # fmt does not work if this line is removed

# ==============================================================================
# === Plot random surface fits for training set
for idx in [int(x) for x in np.random.sample(5) * len(X)]:
    input_v = norm_features.transform(X[idx, :].reshape(1, -1))
    print(idx, [x for x in map(lambda x: round(x, 2), X[idx, :])])

    output_v = model.predict(input_v)
    impvol_all = norm_labels.inverse_transform(output_v)
    
    moneyness = np.unique(output_grid[:, 1])
    surface_prediction = np.hstack(np.split(impvol_all.T, split_idx)).T[0, :]
    surface_target = np.vstack(np.hsplit(Y[idx, :], split_idx))[0, :]

    plt.figure()
    
    plt.plot(moneyness, surface_target, 'r-', label='Closed-form target', linewidth=2)
    plt.plot(moneyness, surface_prediction, 'k--', label='ANN prediction', linewidth=2)
    
    plt.legend(loc='upper left', fontsize=15)
    plt.xlabel('Log-moneyness', fontsize=15)
    plt.ylabel('Implied volatility', fontsize=15)
    
    plt.show()

# ==============================================================================
# === Plot random surface fits for test set
for idx in [int(x) for x in np.random.sample(20) * len(X_test)]:
    input_v = norm_features.transform(X_test[idx, :].reshape(1, -1))
    print([x for x in map(lambda x: round(x, 2), X_test[idx, :])])
    
    output_v = model.predict(input_v)
    impvol_all = norm_labels.inverse_transform(output_v)
    
    moneyness = np.unique(output_grid[:, 1])
    surface_prediction = np.vstack(np.hsplit(impvol_all, split_idx))[0, :]
    surface_target = np.vstack(np.hsplit(Y_test[idx, :], split_idx))[0, :]
    
    plt.figure()
    
    plt.plot(moneyness, surface_target, 'r-', label = 'Closed-form target', 
             linewidth = 2)
    plt.plot(moneyness, surface_prediction, 'k--', label = 'ANN prediction', 
             linewidth = 2)
    
    plt.legend(loc='upper left', fontsize = 15)
    plt.xlabel('Log-moneyness', fontsize = 15)
    plt.ylabel('Implied volatility', fontsize = 15)
    
    plt.show()