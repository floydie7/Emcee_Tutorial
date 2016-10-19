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

def modelfn(x, param):
    a, b, c = param
    return a * x**2 + b * x + c

def fit_bootstrap(p0, datax, datay, function, yerr_systematic=0.0):

    errfunc = lambda p, x, y: function(x,p) - y

    # Fit first time
    pfit, perr = op.leastsq(errfunc, p0, args=(datax, datay), full_output=0)


    # Get the stdev of the residuals
    residuals = errfunc(pfit, datax, datay)
    sigma_res = np.std(residuals)

    sigma_err_total = np.sqrt(sigma_res**2 + yerr_systematic**2)

    # 100 random data sets are generated and fitted
    ps = []
    for i in range(100):

        randomDelta = np.random.normal(0., sigma_err_total, len(datay))
        randomdataY = datay + randomDelta

        randomfit, randomcov = \
            op.leastsq(errfunc, p0, args=(datax, randomdataY),\
                             full_output=0)

        ps.append(randomfit) 

    ps = np.array(ps)
    mean_pfit = np.mean(ps,0)

    # You can choose the confidence interval that you want for your
    # parameter estimates: 
    Nsigma = 1. # 1sigma gets approximately the same as methods above
                # 1sigma corresponds to 68.3% confidence interval
                # 2sigma corresponds to 95.44% confidence interval
    err_pfit = Nsigma * np.std(ps,0) 

    pfit_bootstrap = mean_pfit
    perr_bootstrap = err_pfit
    return pfit_bootstrap, perr_bootstrap 

pfit, perr = fit_bootstrap([0,0,0], x, y, modelfn)

print("\nFit parameters and parameter errors from bootstrap method :")
print("pfit = ", pfit)
print("perr = ", perr)
print("True values = ",[a_true, b_true, c_true])