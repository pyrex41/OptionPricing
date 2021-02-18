import quadpy # pip install quadpy
from numpy import array, isreal
from numpy.polynomial import Polynomial
from math import sqrt, pi, exp, log
from pprint import pp, pprint

scheme = quadpy.e1r2.gauss_hermite(30)

# approach taken from Expectations.jl library.

# sets up lognormal distribution based on mean, var parameters
def expectation_weights(mu, sigma, gh=scheme):
    nodes = []
    for p in gh.points:
        nodes.append(exp((p * sqrt(2) * sigma) + mu))
    weights = gh.weights / sqrt(pi)
    return (array(nodes), weights)

# estimiates value of func(x) with x ~ LogNormal(a,b) via numerical method
def apply(func, nodes, weights):
    fn = map(lambda x: func(x),  nodes)
    arr = map(lambda x: x[0]*x[1], zip(fn, weights))
    s = sum(arr)
    return s


# sets up lognormal distribution and func based on parameters
def problem_func(verbose=True, **kwargs):
    # default params
    params  = {
        'cost': 0.0,
        'mu': 0.0,
        'T': 1,
        'S': 100,
        'X': 100,
        'sigma': .5,
        'r': 0.0
    }

    for (k,v) in kwargs.items():
        params[k] = v

    if verbose:
        pp("Parameters are:")
        pprint(params)

    # define mean, var of lognormal distribution
    a = params['T'] * (params['mu'] + params['r'] - params['sigma']**2/2) + log(params['S'])
    b = params['sigma'] * sqrt(params['T'])
    n,w = expectation_weights(a,b)

    # call payoff with s lognormally disstributed, were x is the 1st, 2nd, 3rd moment
    c = params['cost']
    def f(x):
        return exp(-params['r']*params['T']*(x)) * apply(lambda s: (max(s-params['X']-c,-c)/c)**(x), n, w)
    return f

# solve for kelly given vector of raw moments
def kelly(moms):
    coefs = []
    for i in range(0,len(moms)):
        coefs.append((-1)**(i)*moms[i])
    p = Polynomial(coefs)
    sol = p.roots()
    out = min(filter(isreal, sol))
    return out.real

# solve kelly for options
def kelly_opt(**kwargs):
    nm = kwargs.get('nmoms', 4)
    f = problem_func(**kwargs)
    # get a different function for each moment needed:
    moms = list(map(f, range(1,nm+1)))
    return kelly(moms)

def raw_moment(vec, x):
    n = len(vec)
    return 1/n * sum(map(lambda i: i**x, vec))
