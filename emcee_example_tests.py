#!/usr/bin/python
'''
emcee_example_tests.py
@author: Benjamin Floyd

This script is designed to test the number of walkers and number of steps needed for the emcee sampler to converge
to a result.
'''

from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
import emcee

np.random.seed(100)

# First, let's define our "true" parameters.
a_true = 3.35
b_true = 0.37
c_true = 4.17

# Variable error factor
verr = 0

# Now, generate some synthetic data from our model.
N = 1000         # Number of data points
x = np.sort(2*np.random.rand(N))
y_true = a_true*x**2 + b_true*x + c_true
yerr = 0.1 * y_true + verr * np.random.rand(N)
y = a_true * x**2 + b_true * x + c_true + yerr*np.random.randn(N)

# Likelihood function.
def lnlike(param, x, y, yerr):
    a, b, c = param
    model = a * x**2 + b * x + c
    inv_sigma2 = 1.0/(yerr**2)
    return -0.5 * (np.sum((y - model)**2 * inv_sigma2))

# For our example let's make no assumptions on the distributions on the parameters and use a uniform distribution.
def lnprior(param, limits):
    a, b, c = param
    if limits[0] < a < limits[1] and limits[2] < b < limits[3] and limits[4] < c < limits[5]:
        return 0.0
    return -np.inf


# Now the full probability function
def lnprob(param, x, y, yerr, limits):
    lp = lnprior(param, limits)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(param, x, y, yerr)

'''
Now we can set up our MCMC sampler to explore the possible values nearby our maximum likelihood result
'''

# Initial Run to establish values

# Set the number of dimensions of parameter space and the nubmer of walkers to explore the space.
ndim, nwalkers = 3, 100
# Set the initial position of the walkers in the space. To start, set walkers uniformly distributed in space.
# pos = [result['x'] + 1e-4 * np.random.randn(ndim) for i in range(nwalkers)]
pos0 = [np.random.rand(ndim) for i in range(nwalkers)]

# Set the bounds on the prior distributions, defining our parameter space.
prior0_lim = np.array([0., 6., -1., 1., 0., 8.])
wprior = np.absolute(np.array([prior0_lim[i]-prior0_lim[i-1] for i in np.arange(1, len(prior0_lim), 2) ]))

# Set up and run the sampler.
nsteps, step_size = 300, 0.01*wprior.min()
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr, prior0_lim),a=step_size)
sampler.run_mcmc(pos0, nsteps)   # Run sampler at initial position pos for 300 steps.

# Remove the burn-in and calculate the mean sample values for the parameters.
burnin = 100    # Set burn-in to be 1/3 Number of steps.
samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
result0 = np.percentile(samples, 50, axis=0)
print("Initial result:",result0)
sampler.reset()


# Full run to with updated prior limits and using the initial run's results as starting values

# First vary the number of walkers from 6 to 100 with nsteps = 1000

run_results = []
for j in range(6,100,2):
    # Spread out original run's positions according to a standard normal distribution.
    nwalkers = j
    # pos1 = [result0 + 1e-2 * np.random.randn(ndim) for i in range(nwalkers)]
    pos1 = emcee.utils.sample_ball(result0, np.array([1e-2, 1e-2, 1e-2]), size=nwalkers)

    # Shrink prior bounds according to the initial run's results
    prior1_lim = np.array([result0[0] - 1.0, result0[0] + 1.0, result0[1] - 1.0, result0[1] + 1.0, result0[2] - 1.0, result0[2] + 1.0])
    wprior = np.absolute(np.array([prior1_lim[i]-prior1_lim[i-1] for i in np.arange(1, len(prior1_lim), 2) ]))

    # Set up and run the sampler again but with better a priori positions and the smaller prior ranges.
    nsteps, step_size = 1000, 1e-3*wprior.min()

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr, prior0_lim), threads=3)
    sampler.run_mcmc(pos1, nsteps)     # Run the sampler again starting at position pos1.

    burnin = nsteps/3.0    # Set burn-in to be 1/3 Number of steps.
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    a_mcmc, b_mcmc, c_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))
    run_results.append([a_mcmc, b_mcmc, c_mcmc])


# Plots
a_sampled = [run_results[i][0][0] for i in range(len(run_results))]
b_sampled = [run_results[i][1][0] for i in range(len(run_results))]
c_sampled = [run_results[i][2][0] for i in range(len(run_results))]
a_err = [[run_results[i][0][2] for i in range(len(run_results))], [run_results[i][0][1] for i in range(len(run_results))]]
b_err = [[run_results[i][1][2] for i in range(len(run_results))], [run_results[i][1][1] for i in range(len(run_results))]]
c_err = [[run_results[i][2][2] for i in range(len(run_results))], [run_results[i][2][1] for i in range(len(run_results))]]
Nwalkers = np.arange(6,100,2)

print("""Mean parameter fits:
    a = {0} (Truth: {1})
    b = {2} (Truth: {3})
    c = {4} (Truth: {5})
    """.format(np.mean(a_sampled), a_true, np.mean(b_sampled), b_true, np.mean(c_sampled), c_true))

fig, (ax1,ax2,ax3) = plt.subplots(nrows=3, ncols=1, sharex=True)
ax1.axhline(np.mean(a_sampled), color='red', linewidth=2)
ax1.errorbar(Nwalkers, a_sampled, yerr=a_err, fmt='.k')
ax1.axhline(a_true, color='blue', linewidth=2)
ax1.set_ylabel("$a$")
ax1.set_title("Fits with Nsteps = 1000")

ax2.axhline(np.mean(b_sampled), color='red', linewidth=2)
ax2.errorbar(Nwalkers, b_sampled, yerr=b_err, fmt='.k')
ax2.axhline(b_true, color='blue', linewidth=2)
ax2.set_ylabel("$b$")

ax3.axhline(np.mean(c_sampled), color='red', linewidth=2)
ax3.errorbar(Nwalkers, c_sampled, yerr=c_err, fmt='.k')
ax3.axhline(c_true, color='blue', linewidth=2)
ax3.set_ylabel("$c$")
ax3.set_xlabel("Walkers")
fig.savefig("Num_Walkers.pdf", format='pdf')


# Now vary the number of steps taken with nwalkers = 100

run_results = []
for j in range(1,1000):
    # Spread out original run's positions according to a standard normal distribution.
    nwalkers = 100
    # pos1 = [result0 + 1e-2 * np.random.randn(ndim) for i in range(nwalkers)]
    pos1 = emcee.utils.sample_ball(result0, np.array([1e-2, 1e-2, 1e-2]), size=nwalkers)

    # Shrink prior bounds according to the initial run's results
    prior1_lim = np.array([result0[0] - 1.0, result0[0] + 1.0, result0[1] - 1.0, result0[1] + 1.0, result0[2] - 1.0, result0[2] + 1.0])
    wprior = np.absolute(np.array([prior1_lim[i]-prior1_lim[i-1] for i in np.arange(1, len(prior1_lim), 2) ]))

    # Set up and run the sampler again but with better a priori positions and the smaller prior ranges.
    nsteps, step_size = j, 1e-3*wprior.min()

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr, prior0_lim), threads=3)
    sampler.run_mcmc(pos1, nsteps)     # Run the sampler again starting at position pos1.

    burnin = nsteps/3.0    # Set burn-in to be 1/3 Number of steps.
    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    a_mcmc, b_mcmc, c_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))
    run_results.append([a_mcmc, b_mcmc, c_mcmc])

# Plots
a_sampled = [run_results[i][0][0] for i in range(len(run_results))]
b_sampled = [run_results[i][1][0] for i in range(len(run_results))]
c_sampled = [run_results[i][2][0] for i in range(len(run_results))]
a_err = [[run_results[i][0][2] for i in range(len(run_results))], [run_results[i][0][1] for i in range(len(run_results))]]
b_err = [[run_results[i][1][2] for i in range(len(run_results))], [run_results[i][1][1] for i in range(len(run_results))]]
c_err = [[run_results[i][2][2] for i in range(len(run_results))], [run_results[i][2][1] for i in range(len(run_results))]]
Nsteps = np.arange(1,1000)

print("""Mean parameter fits:
    a = {0} (Truth: {1})
    b = {2} (Truth: {3})
    c = {4} (Truth: {5})
    """.format(np.mean(a_sampled), a_true, np.mean(b_sampled), b_true, np.mean(c_sampled), c_true))

fig, (ax1,ax2,ax3) = plt.subplots(nrows=3, ncols=1, sharex=True)
ax1.axhline(np.mean(a_sampled), color='red', linewidth=2)
ax1.errorbar(Nsteps, a_sampled, yerr=a_err, fmt='.k')
ax1.axhline(a_true, color='blue', linewidth=2)
ax1.set_ylabel("$a$")
ax1.set_title("Fits with Nwalkers = 100")

ax2.axhline(np.mean(b_sampled), color='red', linewidth=2)
ax2.errorbar(Nsteps, b_sampled, yerr=b_err, fmt='.k')
ax2.axhline(b_true, color='blue', linewidth=2)
ax2.set_ylabel("$b$")

ax3.axhline(np.mean(c_sampled), color='red', linewidth=2)
ax3.errorbar(Nsteps, c_sampled, yerr=c_err, fmt='.k')
ax3.axhline(c_true, color='blue', linewidth=2)
ax3.set_ylabel("$c$")
ax3.set_xlabel("Steps")
fig.savefig("Num_Steps.pdf", format='pdf')