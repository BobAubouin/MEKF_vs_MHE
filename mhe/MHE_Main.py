# %% Overview
# Author: Mohammad Ajami
# Date of creation: April 2023
# This moving horizon estimation code is based on the work of Mrs. Kaouther Moussa on 11/02/2023
# This code portrays a MIMO model that takes Propofol and Remifentanil infusion rates as inputs and the BIS as output
# This code estimates both the states and the PD parameters
# The states represent the concentration of the anesthetic agents in different compartments of the human body and in the effect site
# PK allows to characterize the evolution of the injected anesthetic agents in the human body
# PD allows to characterize the effect of the drug on the organism

# %% Library imports
import casadi as cas  # to be used for numerical optimization
import numpy as np  # to perform mathematical operations
import pandas as pd  # to be used as a data manipulation tool
# to solve ordinary differential equations
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt  # to plot figures
# to import the created dynamical system with the known theoretical parameters
from Model import Model
import time  # to measure the elapsed time
import python_anesthesia_simulator as pas

# Loading Patients' Data
Patient_Info = pd.read_csv("parameters.csv").values

# Picking Patient Characteristics
nb_Patient = int(input("Please enter the patient number: "))
age, height, weight, sex = Patient_Info[nb_Patient, 1], Patient_Info[nb_Patient, 2], Patient_Info[nb_Patient, 3], Patient_Info[nb_Patient, 4] 

Patient_Char = [age, height, weight, sex]

c50p, c50r, gamma, beta, E0, Emax = Patient_Info[nb_Patient, 5], Patient_Info[nb_Patient, 6], Patient_Info[nb_Patient, 7], Patient_Info[nb_Patient, 8], Patient_Info[nb_Patient, 9], Patient_Info[nb_Patient, 10] 
PD_param = [c50p, c50r, gamma, beta, Emax, E0]

Patient_Variables = pd.read_csv(f"simu_{nb_Patient}.csv").values
Time = Patient_Variables[:, 1]
BIS, Propo_rate, Remi_rate = Patient_Variables[:, 2], Patient_Variables[:, 6], Patient_Variables[:, 7]
Patient_Var = [BIS, Propo_rate, Remi_rate]

Patient = pas.Patient(patient_characteristic=Patient_Char, ts=1, model_propo='Eleveld', model_remi='Eleveld', co_update=True)
LBM = Patient.lbm

#Propo PK
Ap = Patient.propo_pk.A_init
a11, a12, a13, a21, a31, a41 = -Ap[0, 0], Ap[0, 1], Ap[0, 2], Ap[1, 0], Ap[2, 0], Ap[3, 0]
V1 = Patient.propo_pk.v1

#Remi PK
Ar = Patient.remi_pk.A_init
a11r, a12r, a13r, a21r, a31r, a41r = -Ar[0, 0], Ar[0, 1], Ar[0, 2], Ar[1, 0], Ar[2, 0], Ar[3, 0]
V1r = Patient.remi_pk.v1

# Data preparation for the MHE problem
#####################################
BIS = BIS.T
y_measurements = BIS
u_p = Propo_rate.T
u_r = Remi_rate.T

# MHE problem formulation
###########################
N_samp = 2
T = N_samp/60  # sampling period
N_MHE = 20  # estimation horizon

# Reduce the number of elements in the measured data
Time = Time[0::N_samp]
BIS = BIS[0::N_samp]
y_measurements = y_measurements[0::N_samp]
u_r = u_r[0::N_samp]
u_p = u_p[0::N_samp]

# BIS filtering
window_size = 5
y_measurements = pd.DataFrame(y_measurements)
y_measurements.reset_index(drop=True, inplace=True)
y_measurements = y_measurements.rolling(window=window_size, center=True, min_periods=1).mean().values

# Model simulation with the real input, to calibrate the PD params
Par = np.array([a11, a12, a13, a21, a31, a41])
xx0 = np.zeros([4, 1])
ODE_sol = solve_ivp(lambda t, x: Model(t, x, (u_p/V1).flatten(), Par, Time),
                    [Time[0], Time[-1]], xx0.flatten(), method='RK45', t_eval=Time, rtol=1e-6)

t = ODE_sol.t
yp = ODE_sol.y.T

Par = np.array([a11r, a12r, a13r, a21r, a31r, a41r])
Par = np.reshape(Par.T, (1, -1))
ODE_sol = solve_ivp(lambda t, x: Model(t, x, (u_r/V1r).flatten(), Par, Time),
                    [Time[0], Time[-1]], xx0.flatten(), method='RK45', t_eval=Time, rtol=1e-6)
tt = ODE_sol.t
yr = ODE_sol.y.T
xx1 = np.interp(Time, t, yp[:, 0])
xx2 = np.interp(Time, t, yp[:, 1])
xx3 = np.interp(Time, t, yp[:, 2])
xx4 = np.interp(Time, t, yp[:, 3])

xx5 = np.interp(Time, tt, yr[:, 0])
xx6 = np.interp(Time, tt, yr[:, 1])
xx7 = np.interp(Time, tt, yr[:, 2])
xx8 = np.interp(Time, tt, yr[:, 3])


# %% Data Extension

N_ex = N_MHE//1
tt1 = np.arange(0, N_samp*(N_ex), N_samp)/60
Time = np.concatenate((tt1, Time+tt1[-1]-Time[0]+T))
u_r = np.vstack((np.zeros([N_ex, 1]), u_r.reshape((-1, 1))))
u_p = np.vstack((np.zeros([N_ex, 1]), u_p.reshape((-1, 1))))
xx1 = np.vstack((np.zeros([N_ex, 1]), xx1.reshape(-1, 1))).T
xx2 = np.vstack((np.zeros([N_ex, 1]), xx2.reshape(-1, 1))).T
xx3 = np.vstack((np.zeros([N_ex, 1]), xx3.reshape(-1, 1))).T
xx4 = np.vstack((np.zeros([N_ex, 1]), xx4.reshape(-1, 1))).T
xx5 = np.vstack((np.zeros([N_ex, 1]), xx5.reshape(-1, 1))).T
xx6 = np.vstack((np.zeros([N_ex, 1]), xx6.reshape(-1, 1))).T
xx7 = np.vstack((np.zeros([N_ex, 1]), xx7.reshape(-1, 1))).T
xx8 = np.vstack((np.zeros([N_ex, 1]), xx8.reshape(-1, 1))).T
y_measurements = np.vstack((E0*np.ones([N_ex, 1]), np.reshape(y_measurements.T, (-1, 1))))
BIS = np.vstack((E0*np.ones([N_ex, 1]), BIS.reshape(-1, 1)))

# BIS computation
UUp = xx4/c50p
UUr = xx8/c50r

theta = np.divide(UUp, UUp + UUr + 1e-6)
Uu = np.divide(UUp + UUr, 1 - beta*theta + beta*theta**2)
BIS_mod = E0 - np.divide(Emax*(Uu**gamma), 1 + Uu**gamma)

# Plot the measured data
fig, axs = plt.subplots(2, 1)
axs[0].plot(Time, u_p, label='Propo')
axs[0].plot(Time, u_r, label='Remi')
axs[0].legend()
axs[0].set_title('Drug injection rates (input)')
axs[0].grid(True)

axs[1].plot(Time, BIS, label='BIS measured')
axs[1].plot(Time, y_measurements, linewidth=2, label='BIS filtered')
axs[1].plot(Time, BIS_mod.real.flatten(), linewidth=2, label='BIS estimated')
axs[1].legend()
axs[1].set_title('BIS (output)')
axs[1].grid(True)

# State bounds
x1_max = 1.3*round(np.max(xx1))
x2_max = 1.3*round(np.max(xx2))
x3_max = 1.3*round(np.max(xx3))
x4_max = 1.3*round(np.max(xx4))
x5_max = 1.3*round(np.max(xx5))
x6_max = 1.3*round(np.max(xx6))
x7_max = 1.3*round(np.max(xx7))
x8_max = 1.3*round(np.max(xx8))

x1_min = x2_min = x3_min = x4_min = x5_min = x6_min = x7_min = x8_min = 0

# Param bounds
x9_max = 90
x9_min = 30
x10_max = 9
x10_min = 2
x11_max = 25
x11_min = 6
x12_max = 5
x12_min = 1
x13_max = 3
x13_min = 0

# States definition
x1 = cas.SX.sym('x1')
x2 = cas.SX.sym('x2')
x3 = cas.SX.sym('x3')
x4 = cas.SX.sym('x4')
x5 = cas.SX.sym('x5')
x6 = cas.SX.sym('x6')
x7 = cas.SX.sym('x7')
x8 = cas.SX.sym('x8')
emax = cas.SX.sym('emax')
C50p = cas.SX.sym('C50p')
C50r = cas.SX.sym('C50r')
Gamma = cas.SX.sym('Gamma')
Beta = cas.SX.sym('Beta')

states = [x1, x2, x3, x4, x5, x6, x7, x8, emax, C50p, C50r, Gamma, Beta]
n_states = len(states)
states = cas.vertcat(*states)

# Inputs definition
vp = cas.SX.sym('vp')
vr = cas.SX.sym('vr')
controls = [vp, vr]
n_controls = len(controls)
controls = cas.vertcat(*controls)

# Model definition
rhs = [-a11*x1 + a12*x2 + a13*x3 + vp/V1, a21*x1 - a21*x2, a31*x1 - a31*x3, a41*x1 - a41*x4, -a11r*x5 +
       a12r*x6 + a13r*x7 + vr/V1r, a21r*x5 - a21r*x6, a31r*x5 - a31r*x7, a41r*x5 - a41r*x8, 0, 0, 0, 0, 0]
rhs = cas.vertcat(*rhs)
f = cas.Function('f', [states, controls], [rhs])

# Linear PK mapping function f(x, u)
Up = x4/C50p
Ur = x8/C50r
theta = Up/(Up+Ur+1e-6)
UU = (Up+Ur)/(1-Beta*theta+Beta*theta**2)

measurement_rhs = E0-emax*(UU**Gamma)/(1+UU**Gamma)
h = cas.Function('h', [states], [measurement_rhs])  # Measurement model

# Decision variables
X = cas.SX.sym('x', n_states, (N_MHE+1))  # states
P = cas.SX.sym('P', 1, n_controls*N_MHE +
               (N_MHE + 1) + (N_MHE + 1)*n_states + 1)

# Parameters (include BIS measurement, the controls measurements, the previous estimated states, and the estimation iteration)
V = 0.0001  # weighting matrices (output)  y_tilde - y

obj = 0  # objective function
g = []  # constraints vector

N_it = P[:, -1]

# Measurement Penalty
for k in range(0, N_MHE+1):
    st = X[:, k]
    h_x = h(st)
    y_tilde = P[:, k]
    obj += (y_tilde - h_x).T * V * (y_tilde - h_x)  # Calculate obj

# Model compatibility cost
R = np.eye(n_states - 5)
R[1, 1] = 550
R[2, 2] = 550
R[5, 5] = 50
R[6, 6] = 750

for k in range(0, N_MHE - 1):
    st1 = P[3*N_MHE + k+1: 1 + 3*N_MHE + k + n_states*(N_MHE) - 7: N_MHE + 1]
    con1 = np.hstack((P[:, N_MHE + k+1], P[:, (1 + N_MHE) + N_MHE + k]))
    f_value1 = f(st1, con1)
    st1_next = st1 + (T * f_value1.T)

    obj += (X[0:8, k].T - st1_next[0, 0:8]
            ) @ R @ (X[0:8, k].T - st1_next[0, 0:8]).T
    obj += (X[8, k].T - st1_next[0, 8]) * ((10 - 9.92) * np.exp(-300 * np.exp(-0.005*N_it))) * \
        np.eye(1) * (X[8, k - 1].T - st1_next[0, 8]).T
    obj += (X[9, k].T - st1_next[0, 9]) * (1e-3 + (200) * np.exp(-300 * np.exp(-0.005*N_it))) * \
        np.eye(1) * (X[9, k].T - st1_next[0, 9]).T
    obj += (X[10, k].T - st1_next[0, 10]) * (1e-5 + (0.16) * np.exp(-300 *
                                                                     np.exp(-0.005*N_it))) * np.eye(1) * (X[10, k].T - st1_next[0, 10]).T
    obj += (X[11, k].T - st1_next[0, 11]) * (1e-3 + (200) * np.exp(-300 * np.exp(-0.005*N_it))
                                             ) * np.eye(1) * (X[11, k].T - st1_next[0, 11]).T
    obj += (X[12, k].T - st1_next[0, 12]) * (5e-4 + (10.2) * np.exp(-300 *
                                                                 cas.exp(-0.005*N_it))) * cas.SX.eye(1) * (X[12, k].T - st1_next[0, 12]).T

# Multiple shooting constraints
for k in range(0, N_MHE):
    st = X[:, k]
    con = cas.horzcat(P[:, 1 + N_MHE + k], P[:, (1 + N_MHE) + N_MHE + k])
    st_next = X[:, k + 1]
    f_value = f(st, con)
    st_next_euler = st + (T * f_value)
    g = cas.vertcat(g, st_next - st_next_euler)  # compute constraints

# Make the decision variable one column vector
OPT_variables = cas.horzcat(cas.reshape(X, n_states*(N_MHE+1), 1))
nlp_mhe = {'f':  obj, 'x': OPT_variables, 'g': g, 'p': P}
opts = {'ipopt.max_iter': 2000,
        'ipopt.print_level': 0,
        'print_time': 0,
        'ipopt.acceptable_tol': 1e-8,
        'ipopt.acceptable_obj_change_tol': 1e-6}

solver = cas.nlpsol('solver', 'ipopt', nlp_mhe, opts)

lbg = np.zeros([1, n_states*N_MHE])
ubg = np.zeros([1, n_states*N_MHE])

lbx = np.zeros([n_states*(N_MHE+1), 1])
ubx = np.zeros([n_states*(N_MHE+1), 1])
lbx[0::n_states] = x1_min
ubx[0::n_states] = x1_max
lbx[1::n_states] = x2_min
ubx[1::n_states] = x2_max
lbx[2::n_states] = x3_min
ubx[2::n_states] = x3_max
lbx[3::n_states] = x4_min
ubx[3::n_states] = x4_max
lbx[4::n_states] = x5_min
ubx[4::n_states] = x5_max
lbx[5::n_states] = x6_min
ubx[5::n_states] = x6_max
lbx[6::n_states] = x7_min
ubx[6::n_states] = x7_max
lbx[7::n_states] = x8_min
ubx[7::n_states] = x8_max
lbx[8::n_states] = x9_min
ubx[8::n_states] = x9_max
lbx[9::n_states] = x10_min
ubx[9::n_states] = x10_max
lbx[10::n_states] = x11_min
ubx[10::n_states] = x11_max
lbx[11::n_states] = x12_min
ubx[11::n_states] = x12_max
lbx[12::n_states] = x13_min
ubx[12::n_states] = x13_max

args = {'lbx': lbx, 'ubx': ubx, 'lbg': lbg, 'ubg': ubg}

###########################################################
######## ALL OF THE ABOVE IS JUST A PROBLEM SET UP ########
###########################################################


###########################################################
############# MHE Simulation loop starts here #############
###########################################################

X_estimate = []  # contains the MHE estimate of the states

X0 = np.zeros([N_MHE + 1, n_states])
X0[:, : n_states - 5] = np.zeros([N_MHE + 1, n_states - 5])
X0[:, 8] = (Emax + 0.1*Emax)*np.ones([N_MHE + 1])
X0[:, 9] = (c50p + 0.1*c50p)*np.ones([N_MHE + 1])
X0[:, 10] = (c50r + 0.1*c50r)*np.ones([N_MHE + 1])
X0[:, 11] = (gamma + 0.1*gamma)*np.ones([N_MHE + 1])
X0[:, 12] = (beta + 0.1*beta)*np.ones([N_MHE + 1])

# Start MHE
MHEiter = 0
OBJ = []

# Initialize the previous estimated state
X_sol = np.zeros([N_MHE + 1, n_states])
X_sol[N_MHE, 8] = Emax + 0.1*Emax
X_sol[N_MHE, 9] = c50p + 0.1*c50p
X_sol[N_MHE, 10] = c50r + 0.1*c50r
X_sol[N_MHE, 11] = gamma + 0.1*gamma
X_sol[N_MHE, 12] = beta + 0.1*beta

start_time = time.time()
X_estimate = []
OBJ = []
for k in range(0, len(y_measurements) - N_MHE):
    p = np.hstack((y_measurements[k:k+N_MHE+1, :].T, u_p[k:k+N_MHE, :].T, u_r[k:k+N_MHE, :].T,
                  np.reshape(X_sol.T, (1, (N_MHE+1)*n_states)), np.array([k+1]).reshape(-1, 1)))
    x0 = X0.reshape(((N_MHE+1)*n_states, 1))
    sol = solver(x0=x0, p=p, lbx=lbx, ubx=ubx, lbg=lbg.T, ubg=ubg.T)
    X_sol = sol['x'][0:n_states*(N_MHE+1)].T.reshape((n_states, N_MHE+1)).T
    X_estimate.append(X_sol[N_MHE, :])
    X0 = np.vstack((X_sol[1:, :], X_sol[-1, :] + f(X_sol[-1, :], [u_p[k+N_MHE-1, :].T, u_r[k+N_MHE-1, :].T]).T))
    OBJ.append(sol['f'])
elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time} seconds.")

X_estimate = np.array(X_estimate).squeeze()
UUp = X_estimate[:, 3]/X_estimate[:, 9]
UUr = X_estimate[:, 7]/X_estimate[:, 10]

theta = UUp/(UUp+UUr)
U = (UUp + UUr)/(1 - X_estimate[:, 12]*theta + X_estimate[:, 12]*theta**2)
BIS_estimated = E0 - X_estimate[:, 8] * \
    (U**X_estimate[:, 11])/(1 + U**X_estimate[:, 11])

plt.figure()
plt.plot(Time[N_MHE:], BIS[N_MHE:],
         linewidth=1.5, label='BIS measured')
plt.plot(Time[N_MHE:], y_measurements[N_MHE:],
         linewidth=1.5, label='BIS filtered')
plt.plot(Time[N_MHE:], BIS_mod.T[N_MHE:],
         linewidth=1.5, label='BIS model')
plt.plot(Time[N_MHE:], BIS_estimated, '--',
         linewidth=1.5, label='BIS estimated')
plt.legend(['BIS measured', 'BIS filtered', 'BIS model', 'BIS estimated'])

# %% Params figures
fig, axs = plt.subplots(5, 1)
axs[0].plot(Time[N_MHE:], X_estimate[:, 8], linewidth=1.5)
axs[0].legend(['Emax'])
axs[0].grid(True)

axs[1].plot(Time[N_MHE:], X_estimate[:, 9], linewidth=1.5)
axs[1].legend(['C50p'])
axs[1].grid(True)

axs[2].plot(Time[N_MHE:], X_estimate[:, 10], linewidth=1.5)
axs[2].legend(['C50r'])
axs[2].grid(True)

axs[3].plot(Time[N_MHE:], X_estimate[:, 11], linewidth=1.5)
axs[3].legend(['Gamma'])
axs[3].grid(True)

axs[4].plot(Time[N_MHE:], X_estimate[:, 12], linewidth=1.5)
axs[4].legend(['Beta'])
axs[4].grid(True)

plt.show()