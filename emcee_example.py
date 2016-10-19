#!/usr/bin/python
'''
emcee_example.py
@author= Benjamin Floyd

This script is designed to provide a teaching example of using the emcee python package
'''
#%matplotlib inline
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import scipy.optimize as op
import emcee
import corner

np.random.seed(100)
#%%
'''
Let's pick something simple like a quadratic function to fit.
So our function will be y = a*x^2 + b*x + c.
'''

# First, let's define our "true" parameters.
a_true = 3.35
b_true = 0.37
c_true = 4.17

# Variable error factor
verr = 0
#%%
# Now, generate some synthetic data from our model.
N = 1000         # Number of data points
x = np.sort(2*np.random.rand(N))
y_true = a_true*x**2 + b_true*x + c_true
yerr = 0.1 * y_true + verr * np.random.rand(N)
y = a_true * x**2 + b_true * x + c_true + yerr*np.random.randn(N)

#%%
# First figure we'll show our true model in red and our synthetic data we just generated.
fig, ax = plt.subplots()
ax.plot(x, y_true, color='r')
ax.errorbar(x, y, yerr=yerr, fmt='k.')
plt.show();
fig.savefig('model_data.pdf', format='pdf')
#%%
'''
Here we want to define our maximum likelihood function of a least squares solution in order to optimize it.
Where the likelihood is defined as
ln P(y|x,sigma,m,b,f) = -1/2 * sum( ((y_n - a * x^2 + b * x + c)^2 / s_n^2) + ln(2 * pi * s_n^2))
 with s_n^2 = simga_n^2.
'''


def lnlike(param, x, y, yerr):
    a, b, c = param
    model = a * x**2 + b * x + c
    inv_sigma2 = 1.0/(yerr**2)
    return -0.5 * (np.sum((y - model)**2 * inv_sigma2))
#%%
'''
Now, we'll minimize the negative likelihood thus maximizing the likelihood function
'''
nll = lambda *args: -2.0 * lnlike(*args)  # Define negative likelihood
result = op.minimize(nll, [a_true, b_true, c_true], args=(x, y, yerr), method= 'Nelder-Mead')
a_ml, b_ml, c_ml = result.x
Chi_nu = result.fun / (N - 3)
print("Maximum likelihood values of parameters are:\n a={0}, b={1}, and c={2} \n True values: a={3}, b={4},\
 c={5}".format(a_ml, b_ml, c_ml, a_true, b_true, c_true))
print("Reduced chi-squared value = ",Chi_nu)
#%%
fig, ax = plt.subplots()
ax.plot(x, y_true, color='r')
ax.errorbar(x, y, yerr=yerr, fmt='k.')
ax.plot(x, a_ml * x**2 + b_ml * x + c_ml, 'k--', label='$\chi^2$ fit')
ax.legend(loc=2)
plt.show()
fig.savefig('max_like.pdf', format='pdf')
#%%
'''
In order to determine our posterior probablities we will need to use Bayes' Theorem
P(a,b,c|x,y,sigma) ~ P(a,b,c) * P(y|x,sigma,a,b,c)
'''


# For our example let's make no asumptions on the distributions on the paramers an use a uniform distribution.
def lnprior(param):
    a, b, c = param
    if 0.0 < a < 5.0 and 0.0 < b < 5.0 and 0.0 < c < 5.0:
        return 0.0
    return -np.inf


# Now the full probability function
def lnprob(param, x, y, yerr):
    lp = lnprior(param)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(param, x, y, yerr)

#%%
'''
Now we can set up our MCMC sampler to explore the possible values nearby our maximium likelihood result
'''

# Set the number of dimensions of parameter space and the nubmer of walkers to explore the space.
ndim, nwalkers = 3, 100
# Set the inital position of the walkers in the space.
#pos = [result['x'] + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]
pos = [[0,0,0] + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]

# Set up and run the sampler.
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))
sampler.run_mcmc(pos, 1000)    # Run sampler at initial position pos for 500 steps.
#%%
'''
To view the results let's look at the variation in the parameters
'''


#%%
fig2, (bx1, bx2, bx3) = plt.subplots(nrows=3, ncols=1, sharex=True)
bx1.plot(sampler.chain[:,:,0].T, color='k',alpha=0.4)
bx1.yaxis.set_major_locator(MaxNLocator(5))
bx1.axhline(a_true, color='blue', linewidth=2)
bx1.set_ylabel("$a$")

bx2.plot(sampler.chain[:,:,1].T, color='k',alpha=0.4)
bx2.yaxis.set_major_locator(MaxNLocator(5))
bx2.axhline(b_true, color='blue', linewidth=2)
bx2.set_ylabel("$b$")

bx3.plot(sampler.chain[:,:,2].T, color='k',alpha=0.4)
bx3.yaxis.set_major_locator(MaxNLocator(5))
bx3.axhline(c_true, color='blue', linewidth=2)
bx3.set_ylabel("$c$")

plt.show()
fig2.savefig('param_var.pdf', format='pdf')

burnin = 50
samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))

fig3 = corner.corner(samples, labels=["$a$", "$b$", "$c$"], truths=[a_true, b_true, c_true])
plt.show()
fig3.savefig('corner_plot.pdf', format='pdf')

plt.figure()
for a, b, c in samples[np.random.randint(len(samples), size=100)]:
    plt.plot(x, a*x**2+b*x+c, color="k", alpha=0.1)
plt.plot(x, a_true*x**2+b_true*x+c_true, color="r", lw=1, alpha=0.8)
plt.errorbar(x, y, yerr=yerr, fmt=".k")
plt.xlabel("$x$")
plt.ylabel("$y$")
plt.savefig('sampler_data.pdf',format='pdf')
plt.axis([1.8, 2.0, 15, 19])
plt.savefig('sampler_data_zoom.pdf', format='pdf')

a_mcmc, b_mcmc, c_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))
print("""MCMC result:
    a = {0[0]} +{0[1]} -{0[2]} (truth: {1})
    b = {2[0]} +{2[1]} -{2[2]} (truth: {3})
    c = {4[0]} +{4[1]} -{4[2]} (truth: {5})
""".format(a_mcmc, a_true, b_mcmc, b_true, c_mcmc, c_true))

print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))
