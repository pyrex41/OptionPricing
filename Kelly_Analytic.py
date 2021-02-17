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

def setup_problem(mu, sigma, r, T, S):
    def mom_func(k,x, cost):
        a = T * (mu + r - sigma**2/2) + log(S)
        b = sigma * sqrt(T)
        n,w = expectation_weights(a,b)
        return exp(-r*T*(x+1)) * apply(lambda s: max(s-k,-cost)**(x+1), n, w)
    return mom_func


def kelly(moms):
    coefs = []
    for i in range(0,len(moms)):
        coefs.append((-1)**(i)*moms[i])
    p = Polynomial(coefs)
    sol = p.roots()
    out = min(filter(isreal, sol))
    return out.real

def kelly_opt(cost, mu, T, S, X, sigma, r, nmoms=4):
    f = setup_problem(mu, sigma, r, T, S)
    moms = list(map(lambda x: f(X, x, cost), range(0,nmoms)))
    return kelly(moms)
