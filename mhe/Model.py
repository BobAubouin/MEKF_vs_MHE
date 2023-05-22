import numpy as np

#This function models the dynamical system with the known theoretical parameters
def Model(t, x, u, param, T):
    a11, a12, a13, a21, a31, a41 = param.T
    uu = np.interp(t, T, u)
    y = np.array([[-a11*x[0]+a12*x[1]+a13*x[2]+uu], [a21*x[0]-a21*x[1]], [a31*x[0]-a31*x[2]], [a41*x[0]-a41*x[3]]])
    y = y.flatten()
    return y