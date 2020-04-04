# -*- coding: utf-8 -*-
import numpy as np

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
