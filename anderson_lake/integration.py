# -*- coding: utf-8 -*-
import numpy as np
import scipy

class GaussianQuadrature:
    def __init__(self, abs_tol, relative_tol, max_iter):
        self.abs_tol = abs_tol
        self.relative_tol = relative_tol
        self.max_iter = max_iter
    
    def __call__(self, func):
        return scipy.integrate.quad(func, 0, np.inf, epsabs=self.abs_tol, 
                                    epsrel=self.relative_tol, 
                                    limit=self.max_iter)[0]
