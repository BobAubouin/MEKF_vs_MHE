import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import expm
import matplotlib.ticker
from empiric_gramian import empiric_gramian

# %% Define the system
dt = 0.0001

k1 = 0.1 * 10
k2 = 0.6 * 10

A_simple = np.array([[-k1, 0],
                     [0, -k2]])
B_simple = np.array([[1, 0],
                     [0, 1]])

dM = expm(np.block([[A_simple, B_simple], [np.zeros((2, 2)), np.zeros((2, 2))]]) * dt)
Ad_simple = dM[:2, :2]
Bd_simple = dM[:2, 2:]
Ad_simple = np.array([[1 - k1*dt, 0],
                      [0, 1 - k2*dt]])
Bd_simple = np.array([[dt, 0],
                      [0, dt]])


# %% Define the observer

theta = 1e3
R = 1e-1
Q = np.diag([1e2]*2 + [1e-3]*3)
S0 = np.diag([1e1]*2 + [1e3]*2 + [1e3])*0.001
W = np.eye(1) * R
invW = np.eye(1) / R


def A_obs(u):
    A_obs = np.array([[0, 1, u[0], u[1], 0],
                      [0, 0, -k1*u[0], -k2*u[1], 1],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, -k1*k2, k1**2 * u[0], k2**2 * u[1], -(k1+k2)]])
    return A_obs

# def RG(x,u):


A_complex = np.array([[-k1, 0, 0, 0, 0],
                      [0, -k2, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0]])
B_complex = np.array([[1, 0],
                      [0, 1],
                      [0, 0],
                      [0, 0],
                      [0, 0]])

C_obs = np.array([[1, 0, 0, 0, 0]])


def x2z(x):
    z = np.array([x[0]*x[2] + x[1]*x[3] + x[4],
                  -k1*x[0]*x[2] - k2 * x[1]*x[3],
                  x[2],
                  x[3],
                  k1**2*x[0]*x[2] + k2**2*x[1]*x[3]])
    return z


def z2x(z):
    x = np.array([((k2*z[1] + z[4])/(k1*z[2]*(k1 - k2)),
                  -(k1*z[1] + z[4])/(k2*z[3]*(k1 - k2)),
                  z[2],
                  z[3],
                  (k1*k2*z[0] + (k1+k2)*z[1] + z[4])/(k1*k2))])
    return x

# %% perform the simulation


N_simu = int(1e5) * 2
time = np.linspace(0, N_simu*dt, N_simu)
x = np.zeros((2, N_simu))
y = np.zeros((N_simu))
a, b, c = 8, 6, 50
y[0] = a*x[0, 0] + b*x[1, 0] + c
u = np.random.randn(2, N_simu) + 5  # np.array([[4], [3]]) @ np.ones((1, N_simu))  #
u1 = np.sin(np.linspace(0, 1*np.pi * N_simu*dt, N_simu)) * 5 + 8
u2 = np.sin(np.linspace(0, 5*np.pi * N_simu*dt, N_simu)) * 3 + 6
u = np.array([u1, u2]) + np.random.randn(2, N_simu) * 1
u[:, N_simu//2:] = np.array([[8], [6]]) @ np.ones((1, N_simu - N_simu//2))

z = np.zeros((5, N_simu))
z[:, 0] = x2z(np.array([x[0, 0]+0.5, x[1, 0]-0.2, a+2, b+3, c+5]))
dz_liste = np.zeros((5, N_simu))
x_obs = np.zeros((5, N_simu))
x_obs[:, 0] = z2x(z[:, 0])
S = S0

u_period1 = 500
u_period2 = 700

grammian_period = int(2/dt)  # 1 secondes

grammian = np.zeros((5, 5, N_simu//grammian_period))
emp_gram = np.zeros((5, 5, N_simu//grammian_period))
def h(x): return x[0, :]*x[2, :] + x[1, :]*x[3, :] + x[4, :]


for i in range(N_simu-1):

    x[:, i+1] = Ad_simple @ x[:, i] + Bd_simple @ u[:, i]
    y[i+1] = a*x[0, i+1] + b*x[1, i+1] + c + np.random.randn()

    # observer
    K = S @ C_obs.T @ invW
    dz = A_obs(u[:, i]) @ z[:, i] - K @ (C_obs @ z[:, i] - y[i])
    z[:, i+1] = z[:, i] + dt * dz

    dS = S @ A_obs(u[:, i]).T + A_obs(u[:, i]) @ S - S @ C_obs.T @ invW @ C_obs @ S.T + Q  # + theta * S
    S = S + dt * dS
    # z[2:4, i+1] = np.clip(z[2:4, i+1], 1e-3, np.Inf)
    x_obs[:, i+1] = z2x(z[:, i+1])
    # x_obs[:, i+1] = np.clip(x_obs[:, i+1], 1e-3, np.Inf)
    z[:, i+1] = x2z(x_obs[:, i+1])

    id = i//grammian_period
    # grammian computation
    if i % grammian_period == 0:
        phi = np.eye(5)
        x0 = np.expand_dims(np.concatenate((x[:, i], np.array([a, b, c])), axis=0), axis=1)
        emp_gram[:, :, id] = empiric_gramian(
            A_complex, B_complex, h, u[:, i:i+grammian_period], x_obs[:, [i]], epsilon=1e-6, ts=dt)[0]

    phi = phi + dt * (A_obs(u[:, i]) @ phi)
    grammian[:, :, id] += phi.T @ C_obs.T @ C_obs @ phi*dt


vmin_grammian = np.zeros(N_simu//grammian_period-1)
vmax_grammian = np.zeros(N_simu//grammian_period-1)
v_min_emp = np.zeros(N_simu//grammian_period-1)
for i in range(N_simu//grammian_period-1):
    vp = np.linalg.eigvals(grammian[:, :, i])
    vmin_grammian[i] = np.min(vp)
    vmax_grammian[i] = np.max(vp)

    vp = np.linalg.eigvals(emp_gram[:, :, i])
    v_min_emp[i] = np.min(vp)

# %% plot the results

plt.figure()
plt.subplot(2, 1, 1)
plt.plot(time, u[0, :], label='u1')
plt.plot(time, u[1, :], label='u2')
plt.grid()
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(time, y, label='y')
plt.plot(time, x_obs[0, :]*x_obs[2, :] + x_obs[1, :]*x_obs[3, :] + x_obs[4, :], label='y_estimated')
plt.xlabel('time [s]')
plt.legend()
plt.grid()
plt.show()

# plot
param = [a, b, c]
plt.figure()
for i in range(5):
    plt.subplot(5, 1, i+1)
    if i < 2:
        plt.plot(time, x[i, :], label='x simulated')
    else:
        plt.plot(time, param[i-2]*np.ones(N_simu), label='x simulated')
    plt.plot(time, x_obs[i, :], '--', label='x estimated')
    plt.grid()
plt.xlabel('time [s]')
plt.legend()
plt.show()

# %% grammian plot

fig1, ax1 = plt.subplots()
ax1.set_title('grammian eigenvalues')
ax1.plot([time[i*grammian_period] for i in range(N_simu//grammian_period-1)],
         vmin_grammian, 'o--', label='theoretical min eigenv')
ax1.plot([time[i*grammian_period] for i in range(N_simu//grammian_period-1)],
         v_min_emp, 'o--', label='empiric min eigenv')
# plt.stairs(vmax_grammian, [time[i*grammian_period] for i in range(N_simu//grammian_period)], label='max eigenvalue')
ax1.grid()
ax1.set_yscale('symlog', linthresh=1e-12)
ax1.set_ylim([-1e-10, 1e-1])

# ax1.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
# ax1.set_yticks([1e-2, 1e-3, 1e-1])
ax1.set_xlabel('time [s]')
ax1.legend()
plt.show()

plt.figure()
plt.plot([time[i*grammian_period] for i in range(N_simu//grammian_period-1)], v_min_emp, label='empiric min eigenv')
plt.grid()
plt.show()
