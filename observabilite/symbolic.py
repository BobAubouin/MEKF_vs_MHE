import numpy as np
import pandas as pd
import casadi as cas
import python_anesthesia_simulator as pas
import scipy.linalg as la
import sympy as sym


def jacobian(f, x):
    n = len(x)
    J = sym.zeros(1, n)
    for i in range(n):
        J[0, i] = f.diff(x[i])
    return J


def Lie_derivative(f, h, x):
    return np.sum(jacobian(h, x) @ f)


def observability_matrix(f, h, x):
    n = len(x)
    Lie_dev = sym.zeros(n, 1)
    Lie_dev[0] = h
    obsv = sym.zeros(n, n)
    for i in range(1, n):
        Lie_dev[i] = Lie_derivative(f, Lie_dev[i-1], x)
    for i in range(n):
        for j in range(n):
            obsv[i, j] = Lie_dev[i].diff(x[j])
    return obsv

# %% test on simplified system


x = sym.symarray('x', 5)
k1, k2 = sym.symbols('k1 k2')
u1, u2 = sym.symbols('u1 u2')
f = sym.Matrix([-k1 * x[0] + u1, -k2 * x[1] + u2, 0, 0, 0])
C1, C2, gamma = sym.symbols('C1 C2 gamma')
y = x[0] * x[2] + x[1] * x[3] + x[4]
obsv1 = observability_matrix(f, y, x)

u12, u22 = sym.symbols('u12 u22')
f = sym.Matrix([-k1 * x[0] + u12, -k2 * x[1] + u22, 0, 0, 0])
C1, C2, gamma = sym.symbols('C1 C2 gamma')
y = x[0] * x[2] + x[1] * x[3] + x[4]
obsv2 = observability_matrix(f, y, x)

obsv = sym.Matrix([[obsv1, obsv2]]).T
print(obsv)
print(obsv1.rank())
# %%
A1 = obsv1.subs({x[0]: 4, x[1]: 3, x[2]: -8, x[3]: -2, x[4]: 100,
                 k1: 5, k2: 7, u1: 0.2, u2: 0.3})
print(A1.rank())
print(f"Rank of the observability matrix: {np.linalg.matrix_rank(np.array(A1).astype(np.float64))}")

A2 = obsv1.subs({x[0]: 4, x[1]: 3, x[2]: -8, x[3]: -2, x[4]: 100,
                 k1: 5, k2: 7, u1: 0.8, u2: 0.2})

A = np.concatenate((np.array(A1).astype(np.float64), np.array(A2).astype(np.float64)), axis=0)
print(f"Rank of the observability matrix: {np.linalg.matrix_rank(A.T)}")


# %% more complex system

x = sym.symarray('x', 7)
k1, k2, k3, k4 = sym.symbols('k1 k2 k3 k4')
u1, u2 = sym.symbols('u1 u2')
f = sym.Matrix([-k1 * x[0] + u1, -k2 * x[1] + u2, k3*(x[0]-x[2]), k4*(x[1] - x[3]), 0, 0, 0])
y = x[2] * x[4] + x[3] * x[5] + x[6]
obsv = observability_matrix(f, y, x)

print(obsv.rank())

A1 = obsv.subs({x[0]: 4, x[1]: 3, x[2]: 4, x[3]: 3, x[4]: -8, x[5]: -2, x[6]: 100,
                k1: 5, k2: 7, k3: 6, k4: 2, u1: 0.2, u2: 0.3})
print(A1.rank())


# %% Full system
k10p, k12p, k21p, k13p, k31p, ke1p, V1p = sym.symbols('k10p k12p k21p k13p k31p ke1p V1p')

Ap = sym.Matrix([[-(k10p + k12p + k13p), k12p, k13p, 0],
                 [k21p, -k21p, 0, 0],
                 [k31p, 0, - k31p, 0],
                 [ke1p, 0, 0, -ke1p]])

k10r, k12r, k21r, k13r, k31r, ke1r, V1r = sym.symbols('k10r k12r k21r k13r k31r ke1r V1r')

Ar = sym.Matrix([[-(k10r + k12r + k13r), k12r, k13r, 0],
                 [k21r, -k21r, 0, 0],
                 [k31r, 0, - k31r, 0],
                 [ke1r, 0, 0, -ke1r]])

A = sym.Matrix([[Ap, sym.zeros(4, 7)], [sym.zeros(4, 4), Ar, sym.zeros(4, 3)], [sym.zeros(3, 11)]])


x = sym.symarray('x', 11)
u = sym.symarray('u', 2)
B = sym.Matrix([[1/V1p, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1/V1r, 0, 0, 0, 0, 0, 0]]).T

f = A @ x + B @ u

y = x[3]*x[8] + x[7]*x[9] + x[10]

obsv = observability_matrix(f, y, x)

# get real values
age = 50
height = 170
weight = 70
gender = 0

simulator = pas.Patient([age, height, weight, gender])

A_p = simulator.propo_pk.continuous_sys.A[:4, :4]
A_r = simulator.remi_pk.continuous_sys.A[:4, :4]
B_p = simulator.propo_pk.continuous_sys.B[:4]
B_r = simulator.remi_pk.continuous_sys.B[:4]

k10p_value = A_p[0, 0] + A_p[0, 1] + A_p[0, 2]
k12p_value = A_p[0, 1]
k21p_value = A_p[1, 0]
k13p_value = A_p[0, 2]
k31p_value = A_p[2, 0]
ke1p_value = A_p[3, 0]
V1p_value = B_p[0, 0]

k10r_value = A_r[0, 0] + A_r[0, 1] + A_r[0, 2]
k12r_value = A_r[0, 1]
k21r_value = A_r[1, 0]
k13r_value = A_r[0, 2]
k31r_value = A_r[2, 0]
ke1r_value = A_r[3, 0]
V1r_value = B_r[0, 0]


A1 = obsv.subs({x[0]: 4, x[1]: 4, x[2]: 4, x[3]: 4,
                x[4]: 3, x[5]: 3, x[6]: 3, x[7]: 3,
                x[8]: -8, x[9]: -2, x[10]: 100,
                u[0]: 0.2, u[1]: 0.3,
                k10p: k10p_value, k12p: k12p_value, k21p: k21p_value,
                k13p: k13p_value, k31p: k31p_value, ke1p: ke1p_value, V1p: V1p_value,
                k10r: k10r_value, k12r: k12r_value, k21r: k21r_value,
                k13r: k13r_value, k31r: k31r_value, ke1r: ke1r_value, V1r: V1r_value})

print(A1.rank())
