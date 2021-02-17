from random import gauss, seed
from math import sqrt, exp

def GBM(s0, mu, sigma):
    st = s0
    def generate_value():
        nonlocal st

        st *= exp((mu - 0.5 * sigma ** 2) * (1. / 365.) + sigma * sqrt(1./365.) * gauss(mu=0, sigma=1))
        return st

    return generate_value