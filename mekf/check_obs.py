"""Theoretical work to verify the observability of the system."""

import numpy as np
import pandas as pd
import casadi as cas
import control
import python_anesthesia_simulator as pas
import scipy.linalg as la

# %% create patient
age = 50
height = 170
weight = 70
gender = 0

simulator = pas.Patient([age, height, weight, gender])

# %% get matrices of PK system
A_p = simulator.propo_pk.continuous_sys.A[:4, :4]
A_r = simulator.remi_pk.continuous_sys.A[:4, :4]
B_p = simulator.propo_pk.continuous_sys.B[:4]
B_r = simulator.remi_pk.continuous_sys.B[:4]

A = np.block([[A_p, np.zeros((4, 4))], [np.zeros((4, 4)), A_r]])
B = np.block([[B_p, np.zeros((4, 1))], [np.zeros((4, 1)), B_r]])

x = cas.MX.sym('x', 8)
BIS = simulator.bis_pd.compute_bis(x[3], x[7])

grad = cas.gradient(BIS, x)
grad = cas.Function('grad', [x], [grad])

# get the equilibrium concentration
ratio = 2
up, ur = simulator.find_bis_equilibrium_with_ratio(50, ratio)
xp = up * control.dcgain(simulator.propo_pk.continuous_sys)
xr = ur * control.dcgain(simulator.remi_pk.continuous_sys)

C = np.array(grad([xp]*4+[xr]*4)).reshape(1, 8)
# C = np.array([[0, 0, 0, 0, 0, 0, 0, 1],
#               [0, 0, 0, 1, 0, 0, 0, 0]])

C = np.array([[0, 0, 0, 1/simulator.bis_pd.c50p, 0, 0, 0, 1/simulator.bis_pd.c50r]])
# C = np.array([[0, 0, 0, 2, 0, 0, 0, 3]])
# %% check observability
obsv = control.obsv(A, C)
print(f"rank of the observability matrix: {np.linalg.matrix_rank(obsv)}")

# because the rank is 7, the system is not observable. However as the dynamic matrix A is stable the system is detectable.
# let's see wich node is not observable


# %% expended observability matrix
Aextented = np.block([[A, np.zeros((8, 3))], [np.zeros((3, 11))]])
Cextented = np.block([[C, np.zeros((1, 3))]])
obsv = control.obsv(Aextented, Cextented)
print(f"rank of the observability matrix: {np.linalg.matrix_rank(obsv)}")

# %% get the kernel of the observability matrix
kernel = la.null_space(obsv)
ortho_obs = la.orth(obsv)


# change base matrix
T = np.block([[ortho_obs, kernel]]).T
T.T @ T
# change base of the system
A_xi = T @ A @ T.T
B_xi = T @ B
C_xi = C @ T.T
C_xi[:, 6:] = 0

obsv_xi = control.obsv(A_xi, C_xi)
print(f"rank of the observability matrix: {np.linalg.matrix_rank(obsv_xi)}")

A_control = A_xi[:6, :6]
C_control = C_xi[:, :6]
obsv_control = control.obsv(A_control, C_control)
print(np.linalg.cond(obsv_control))
print(f"rank of the observability matrix: {np.linalg.matrix_rank(obsv_control)}")
kernel = la.null_space(obsv_control)
print(kernel)

L = control.place(A_control.T, C_control.T, [-1, -2, -3, -4, -5, -6]).T

A_control = A_xi[:6, :6] - L @ C_control

A_correct = np.block([[A_control, np.zeros((6, 2))], [np.zeros((2, 6)), A_xi[6:, 6:]]])
# eigenvalues of the system
print(f"eigen value of the observable system: {np.linalg.eigvals(A_control)}")
print(f"eigen value of the full observed system: {np.linalg.eigvals(A_correct)}")
