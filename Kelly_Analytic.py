import quadpy # pip install quadpy
from numpy import array, isreal
from numpy.polynomial import Polynomial
from math import sqrt, pi, exp, log

scheme = quadpy.e1r2.gauss_hermite(30)

def expectation_weights(mu, sigma, gh=scheme):
    nodes = []
    for p in gh.points:
        nodes.append(exp((p * sqrt(2) * sigma) + mu))
    weights = gh.weights / sqrt(pi)
    return (array(nodes), weights)

def apply(func, nodes, weights):
    fn = map(lambda x: func(x),  nodes)
    arr = map(lambda x: x[0]*x[1], zip(fn, weights))
    s = sum(arr)
    return s

def problem_func(cost, mu, T, S, X, sigma, r):
    a = T * (mu + r - sigma**2/2) + log(S)
    b = sigma * sqrt(T)
    n,w = expectation_weights(a,b)
    def f(x):
        return exp(-r*T*(x+1)) * apply(lambda s: max(s-X,-cost)**(x+1), n, w)

    return f

def kelly(moms):
    coefs = []
    for i in range(0,len(moms)):
        coefs.append((-1)**(i)*moms[i])
    p = Polynomial(coefs)
    sol = p.roots()
    out = min(filter(isreal, sol))
    return out.real

def kelly_opt(*args, nmoms=4):
    f = problem_func(*args)
    moms = list(map(f, range(0,nmoms)))
    return kelly(moms)
