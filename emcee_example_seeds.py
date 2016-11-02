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
import emcee
import corner


'''
Let's pick something simple like a quadratic function to fit.
So our function will be y = a*x^2 + b*x + c.
'''

np.random.seed(100) # Specify seed to generate the data
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


# First figure we'll show our true model in red and our synthetic data we just generated.
# fig, ax = plt.subplots()
# ax.plot(x, y_true, color='r')
# ax.errorbar(x, y, yerr=yerr, fmt='k.')
# plt.show();
# fig.savefig('model_data.pdf', format='pdf')

'''
Here we want to define our maximum likelihood function of a least squares solution in order to optimize it.
Where the likelihood is defined as
ln P(y|x,sigma,m,b,f) = -1/2 * sum( (y_n - a * x^2 + b * x + c)^2 / sigma_n^2 ).
'''


def lnlike(param, x, y, yerr):
    a, b, c = param
    model = a * x**2 + b * x + c
    inv_sigma2 = 1.0/(yerr**2)
    return -0.5 * (np.sum((y - model)**2 * inv_sigma2))

'''
In order to determine our posterior probabilities we will need to use Bayes' Theorem
P(a,b,c|x,y,sigma) ~ P(a,b,c) * P(y|x,sigma,a,b,c)
'''


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


data = []  # storage array for all runs
for runs in range(1000):
    print("Run # ",runs)

    np.random.seed() # Unspecify the seed to allow it to take on different values from this point on.
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
    # Spread out original run's positions according to a standard normal distribution.
    nwalkers = 1000
    # pos1 = [result0 + 1e-2 * np.random.randn(ndim) for i in range(nwalkers)]
    pos1 = emcee.utils.sample_ball(result0, np.array([1e-2, 1e-2, 1e-2]), size=nwalkers)

    # Shrink prior bounds according to the initial run's results
    prior1_lim = np.array([result0[0] - 0.8*result0[0], result0[0] + 0.8*result0[0], result0[1] - 0.8*result0[1], result0[1] + 0.8*result0[1], result0[2] - 0.8*result0[2], result0[2] + 0.8*result0[2]])
    wprior = np.absolute(np.array([prior1_lim[i]-prior1_lim[i-1] for i in np.arange(1, len(prior1_lim), 2) ]))

    # Set up and run the sampler again but with better a priori positions and the smaller prior ranges.
    nsteps, step_size = 10000, 0.01*wprior.min()

    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr, prior0_lim), a=step_size)
    sampler.run_mcmc(pos1, nsteps)     # Run the sampler again starting at position pos1.

    '''
    To view the results let's look at the variation in the parameters
    '''



    # fig2, (bx1, bx2, bx3) = plt.subplots(nrows=3, ncols=1, sharex=True)
    # bx1.plot(sampler.chain[:,:,0].T, color='k',alpha=0.4)
    # bx1.yaxis.set_major_locator(MaxNLocator(5))
    # bx1.axhline(a_true, color='blue', linewidth=2)
    # bx1.set_ylabel("$a$")
    #
    # bx2.plot(sampler.chain[:,:,1].T, color='k',alpha=0.4)
    # bx2.yaxis.set_major_locator(MaxNLocator(5))
    # bx2.axhline(b_true, color='blue', linewidth=2)
    # bx2.set_ylabel("$b$")
    #
    # bx3.plot(sampler.chain[:,:,2].T, color='k',alpha=0.4)
    # bx3.yaxis.set_major_locator(MaxNLocator(5))
    # bx3.axhline(c_true, color='blue', linewidth=2)
    # bx3.set_ylabel("$c$")
    #
    # # plt.show()
    # fig2.savefig('param_var.pdf', format='pdf')
    #
    # burnin = nsteps/3.0    # Set burn-in to be 1/3 Number of steps.
    # samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    #
    # fig3 = corner.corner(samples, labels=["$a$", "$b$", "$c$"], truths=[a_true, b_true, c_true])
    # # plt.show()
    # fig3.savefig('corner_plot.pdf', format='pdf')

    # plt.figure()
    # for a, b, c in samples[np.random.randint(len(samples), size=100)]:
    #     plt.plot(x, a*x**2+b*x+c, color="k", alpha=0.1)
    # plt.plot(x, a_true*x**2+b_true*x+c_true, color="r", lw=1, alpha=0.8)
    # plt.errorbar(x, y, yerr=yerr, fmt=".k")
    # plt.xlabel("$x$")
    # plt.ylabel("$y$")
    # plt.savefig('sampler_data.pdf',format='pdf')
    # plt.axis([1.8, 2.0, 15, 19])
    # plt.savefig('sampler_data_zoom.pdf', format='pdf')

    a_mcmc, b_mcmc, c_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)))
    print("""MCMC result:
        a = {0[0]} +{0[1]} -{0[2]} (truth: {1})
        b = {2[0]} +{2[1]} -{2[2]} (truth: {3})
        c = {4[0]} +{4[1]} -{4[2]} (truth: {5})
    """.format(a_mcmc, a_true, b_mcmc, b_true, c_mcmc, c_true))
    data.append([a_mcmc, b_mcmc, c_mcmc])

    print("Mean acceptance fraction:", np.mean(sampler.acceptance_fraction))
    # print("Autocorrelation time:", sampler.get_autocorr_time())

a_means = [data[i][0][0] for i in range(len(data))]
b_means = [data[i][1][0] for i in range(len(data))]
c_means = [data[i][2][0] for i in range(len(data))]

print("""Run-averaged parameter values:
    a = {0} (truth: {1})
    b = {2} (truth: {3})
    c = {4} (truth: {5})
    """.format(np.mean(a_means), a_true, np.mean(b_means), b_true, np.mean(c_means), c_true))


fig, ax = plt.subplots()
ax.hist(a_means, bins=500, align='mid', rwidth=0.9, linewidth=0, color='r')
ax.axvline(a_true, color='blue', linewidth=2)
ax.text(x=0.67, y=0.89, s="Mean = {0}\n Truth = {1}".format(np.mean(a_means), a_true), bbox=dict(facecolor='white', linewidth=0.3, pad=12), transform=ax.transAxes)
ax.set_title("1000 Runs with Random Seeds")
ax.set_xlabel("$a$")
fig.savefig('a_mean_distribution.pdf', format='pdf')

fig, ax = plt.subplots()
ax.hist(b_means, bins=500, align='mid', rwidth=0.9, linewidth=0, color='r')
ax.axvline(b_true, color='blue', linewidth=2)
ax.text(x=0.67, y=0.89, s="Mean = {0}\n Truth = {1}".format(np.mean(b_means), b_true), bbox=dict(facecolor='white', linewidth=0.3, pad=12), transform=ax.transAxes)
ax.set_title("1000 Runs with Random Seeds")
ax.set_xlabel("$b$")
fig.savefig('b_mean_distribution.pdf', format='pdf')

fig, ax = plt.subplots()
ax.hist(c_means, bins=500, align='mid', rwidth=0.9, linewidth=0, color='r')
ax.axvline(c_true, color='blue', linewidth=2)
ax.text(x=0.67, y=0.89, s="Mean = {0}\n Truth = {1}".format(np.mean(c_means), c_true), bbox=dict(facecolor='white', linewidth=0.3, pad=12), transform=ax.transAxes)
ax.set_title("1000 Runs with Random Seeds")
ax.set_xlabel("$c$")
fig.savefig('c_mean_distribution.pdf', format='pdf')