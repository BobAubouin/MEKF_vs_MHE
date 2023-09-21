""" test emprirical grammian on a linear system """

import numpy as np
import matplotlib.pyplot as plt
from empiric_gramian import empiric_gramian
from scipy.linalg import expm
from scipy.integrate import quad

A = np.array([[0, 1],
              [-1, -1]])
B = np.array([[0],
              [1]])
C = np.array([[1, 0]])
D = np.array([[0]])


def h(x):
    return np.dot(C, x)


N_simu = int(1e6)


x_0 = np.array([[1],
                [1]])

u = np.ones((1, N_simu))*0

epsilon = 1e-3
ts = 1e-3

gram = empiric_gramian(A, B, h, u, x_0, epsilon, ts)[0]

print("empiric grammian:")
print(gram)


def f(t): return expm(A.T * t) @ C.T @ C @ expm(A * t)


th_gram = np.zeros((2, 2))
for i in range(2):
    for j in range(2):
        def f_temp(t): return f(t)[i, j]
        th_gram[i, j] = quad(f_temp, 0, np.Inf)[0]

print("theoretical grammian:")
print(th_gram)
