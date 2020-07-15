# probably a more elegant way to do this by transforming columns in pandas
import math
from copy import copy

def get_returns(s, n=1):
    out = []
    v1 = s[:-1]
    v2 = s[n:]
    for i in range(0, len(v2), n):
        a = v1[i]
        b = v2[i]
        v = (b-a) / a     
        out.append(v)
    return out

# Hill estimator...very close to MLE
def Hill(vec, gamma = 0.01, tail = "right"):
    vvec = vec if tail == "right" else [-1 * v for v in vec]
        
    vs = sorted(filter(lambda x: x > gamma, vvec), reverse=True)
    k = len(vs) - 1
    log_v = map(math.log, vs[0:-1])
    ll_v = math.log(vs[-1])
    d = map(lambda x: x - ll_v, log_v)
    return k / sum(d)

# anderson darling --> method to set threshsold gamma for tails
# https://www.itl.nist.gov/div898/handbook/eda/section3/eda35e.htm
# method from here https://arxiv.org/pdf/cond-mat/0411161.pdf

def Asq(vec, gamma):
    s = sorted(filter(lambda x: x > gamma, vec))
    n_g = len(s)
    
    # step 1 from paper
    alpha = n_g / sum(map(lambda x: math.log(x / gamma), s))
    
    # cumulative distribution function
    z_arr = map(lambda x: (1-((gamma / x) ** alpha)), s)
        
    s_array = []
    for (j,si) in enumerate(s):
        i = j+1
        z_i = z_arr[j]
        z_ni = z_arr[n_g - i] # varies slightly from paper because of 0-index arrayss
        exp = (2*i - 1) * (math.log(z_i) + math.log(1 - z_ni))
        s_array.append(exp)
        
    asq = -n_g - sum(s_array) / n_g
    
    return (gamma, asq, alpha, n_g)

def fit_alpha(vvec, side = "right", nmin = 10, verbose=True):
    vec = vvec if side == "right" else [-1*x for x in vvec]
    vec = sorted(filter(lambda x: x > 0, vec))
    
    a = map(lambda gamma: Asq(vec, gamma), vec[:-1])
    
    # filter is my addition to make sure there is enough data for a fit
    aa = sorted(filter(lambda x: x[3] >= nmin, a), key=lambda x: x[1])
    gamma, asq, alpha, n = aa[0]
    
    if verbose:
        print("Fitted alpha stats ", side, " side is:")
        print("alpha ------> ", alpha)
        print("gamma ------> ", gamma)
        print("Asq --------> ",asq)
        print("n ----------> ", n)
        print("% of N: ----> ", round(n/len(vec)*100, ndigits=1),"%")
    
    return aa[0]
