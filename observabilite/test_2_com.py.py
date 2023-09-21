""" Test of grammian on 4 compartment model """

import numpy as np
import matplotlib.pyplot as plt
from empiric_gramian import empiric_gramian

# define model

k1 = 1
k2 = 5
k3 = 1
k4 = 2

A = np.array([[-k1, 0, 0, 0, 0, 0, 0],
              [k2, -k2, 0, 0, 0, 0, 0],
              [0, 0, -k3, 0, 0, 0, 0],
              [0, 0, k4, -k4, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0],
              [0, 0, 0, 0, 0, 0, 0]])

B = np.array([[0, 0],
              [1, 0],
              [0, 0],
              [0, 1],
              [0, 0],
              [0, 0],
              [0, 0]])


def h(x):
    bis = x[1, :]*x[4, :] + x[3, :]*x[5, :] + x[6, :]
    # up = x[1, :]/x[4, :]
    # ur = x[3, :]/x[5, :]
    # I = (up + ur) ** x[6, :]
    # bis = 97.4 * (1 - I/(1+I))
    return bis


ts = 1e-4
N_simu = int(18.1/ts)

x = np.zeros((7, N_simu))
x[:, 0] = np.array([0, 0, 0, 0, 3, 1.2, 2])

u1_mean = 20
u2_mean = 4
u1 = np.sin(np.linspace(0, 4*np.pi * N_simu*ts, N_simu)) * 10 + u1_mean
u2 = np.sin(np.linspace(0, 2*np.pi * N_simu*ts, N_simu)) * 8 + u2_mean
u = np.array([u1, u2]) + np.random.randn(2, N_simu) * 1
u[:, N_simu//3:] = np.array([[u1_mean], [u2_mean]]) @ np.ones((1, N_simu - N_simu//3))

u = u/3

epsilon = 1e-3

gram_period = int(3/ts)
gram = np.zeros((7, 7, N_simu//gram_period))

for i in range(N_simu-1):
    x[:, i+1] = x[:, i] + ts*(A @ x[:, i] + B @ u[:, i])
    if (i % gram_period == 0 and i > 0) or i == N_simu-2:
        gram[:, :, i//gram_period-1] = empiric_gramian(A, B, h,
                                                       u[:, i-gram_period:i], x[:, [i-gram_period]], epsilon, ts)[0]

y = h(x)

min_eig = np.zeros((N_simu//gram_period))
for i in range(N_simu//gram_period):
    min_eig[i] = np.min(np.linalg.eig(gram[:, :, i])[0])
time = np.linspace(0, N_simu*ts, N_simu)
plt.figure()
plt.subplot(4, 1, 1)
plt.plot(time, u[0, :], label='u1')
plt.plot(time, u[1, :], label='u2')
plt.legend()
plt.xlim([0, N_simu*ts])
plt.grid()
plt.subplot(4, 1, 2)
plt.plot(time, y)
plt.xlim([0, N_simu*ts])
plt.grid()
plt.subplot(4, 1, 3)
plt.plot(time, x[1, :], label='x1')
plt.plot(time, x[3, :], label='x2')
plt.plot(time, x[4, :], '--', label='x1_50')
plt.plot(time, x[5, :], '--', label='x2_50')
plt.grid()
plt.xlim([0, N_simu*ts])
plt.legend()
plt.subplot(4, 1, 4)
plt.plot(time[gram_period::gram_period], min_eig, 'o--')
plt.xlim([0, N_simu*ts])
ax = plt.gca()
ax.set_yscale('symlog', linthresh=1e-17)
plt.grid()
plt.show()
