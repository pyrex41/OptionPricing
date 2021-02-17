import math

def phi(x):
    return 1/2 * (1 + math.erf(x / math.sqrt(2)))

def GBlackScholes(side, S, X, T, r, sigma, q):
    s_side = side.lower()
    assert s_side in ["put", "call"]
    b = r - q
    d1 = (math.log(S/X) + (b + (sigma**2)/2)*T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    ii = 1 if s_side == "call" else -1
    out = ii*S*math.exp((b-r)*T) * phi(ii*d1) - ii*X*math.exp(-r*T) * phi(ii*d2)
    return out
    
def iter(v1,v2,p1,p2,p,vf,pf,epsilon=1e-10):
    vi = vf(v1,v2,p1,p2)
    pv = pf(vi)
    ee = abs(p-pv) < epsilon
    if pv < p:
        vv1 = vi
        pp1 = pv
        vv2 = v2
        pp2 = p2
        return vv1,vv2,pp1,pp2,ee,True
    else:
        vv1 = v1
        pp1 = p1
        vv2 = vi
        pp2 = pv
        return vv1, vv2, pp1, pp2, ee, False

def ImpliedVolatility(side, p, S, X, T, r, q, epsilon=1e-8):
    sside = side.lower()
    assert sside in ["put", "call"]
    if sside == "put":
        assert X <= S
    else:
        assert X >= S
    
    v1_init = 0.005
    v2_init = 4
    
    def vi_func(v1,v2,p1,p2):
        return v1 + (p-p1)*(v2-v1)/(p2-p1)
    def price_func(v):
        return GBlackScholes(sside, S, X, T, r, v, q)
    
    p1_init = price_func(v1_init)
    p2_init = price_func(v2_init)
    
    v1,v2,p1,p2, _, low = iter(v1_init, v2_init, p1_init, p2_init, p, vi_func, price_func, epsilon)
    
    i=0
    while i<1000:
        v1,v2,p1,p2,retBool,low = iter(v1,v2,p1,p2,p,vi_func, price_func)
        if retBool:
            if low:
                return v1
            else:
                return v2
        i += 1
    if low:
        return v1
    else:
        return v2