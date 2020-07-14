from copy import copy
import matplotlib.pyplot as plt

def F0(s, gamma, alpha):
    if s > gamma:
        return (1 - (gamma / s)**alpha)
    else:
        return 0
    
def fit_line(gamma, alpha, mx, s=.001):
    i = gamma
    xx = []
    yy = []
    while i < mx:
        xx.append(i)
        y = F0(i, gamma, alpha)
        yy.append(y)
        i += s
    return xx,yy

def density_scatter_points(vv, gamma, alpha):
    emp_x = []
    emp_y = []
    for i in vv:
        y = F0(i, gamma, alpha)
        if y != 0:
            emp_x.append(i)
            emp_y.append(y)
    return emp_x, emp_y

def density_plot_fit(p_arr, side="right", title = ""):
    vv_arr = [x[0] for x in p_arr]
    gmin = min([x[1] for x in p_arr])
    mx = max([max(x) for x in vv_arr])
    if len(p_arr) == 1:
        x,y = fit_line(gmin, 2.75, mx)
        plt.plot(x,y, label = "Alpha: 2.75")
    i = 1
    for (vv, gamma, alpha) in p_arr:
        v = copy(vv)
        if side == "left":
            v = [-1*x for x in v]
        xx,yy = fit_line(gamma, alpha, max(v))
        emp_x, emp_y = density_scatter_points(v, gamma, alpha)
        plt.plot(xx,yy, label="{}: Fit Alpha: {}".format(i, round(alpha, ndigits=2)))
        plt.plot(emp_x, emp_y, 'o')
        i += 1
    plt.title(title)
    plt.xlabel("Return (on the Tail)")
    plt.ylabel("CDF / Z-calc")
    plt.legend(loc='lower right')
    fsize = plt.gcf().get_size_inches()
    plt.gcf().set_size_inches(2*fsize)
    return plt.show()