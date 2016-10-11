'''
emcee_chisq.py
@author: Benjamin Floyd

This script is to act as a sanity check against the MCMC code found in emcee_example.
It will compute a simple chi-squared fit to the parameters of the model and output the
reduced chi-squared value as well as the best fit parameter values.
'''

#%matplotlib inline
from __future__ import print_function
import numpy as np
import scipy.optimize as op

np.random.seed(100)

# Define the true parameters
a_true = 3.35
b_true = 0.37
c_true = 4.17

# Variable error factor
verr = 0

# Generate synthetic data from the model
N = 1000            # Number of data points
x = np.sort(2*np.random.rand(N))
y_true = a_true * x**2 + b_true * x + c_true
yerr = 0.1 * y_true/N + verr * np.random.rand(N)
y = y_true + yerr * np.random.randn(N)

# Define the Chi-Squared function
def ChiSq(param, x, y, yerr):
    a, b, c = param
    model = a * x**2 + b * x + c
    inv_sigma2 = 1.0/(yerr**2)
    return np.sum((y - model)**2 * inv_sigma2)

ChiSquare = lambda *args: ChiSq(*args)
result = op.minimize(ChiSquare, [a_true, b_true, c_true], args=(x, y, yerr),options={'disp': True},method = 'Nelder-Mead')
a_ChiSq, b_ChiSq, c_ChiSq = result.x

red_ChiSq = result.fun / ()

print("Maximum likelihood values of parameters are:\n a={0:.3f}, b={1:.3f}, and c={2:.3f} \n True values: a={3}, b={4},\
 c={5}".format(a_ChiSq, b_ChiSq, c_ChiSq, a_true, b_true, c_true))
