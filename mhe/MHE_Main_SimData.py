# Library imports
import casadi as cas  # to be used for numerical optimization
import numpy as np  # to perform mathematical operations
import pandas as pd  # to be used as a data manipulation tool
# to solve ordinary differential equations
from scipy.integrate import solve_ivp
from matplotlib import pyplot as plt  # to plot figures
from Model import Model 
import time  # to measure the elapsed time
import python_anesthesia_simulator as pas

#%% Loading the Data

# Loading Patients' Data
Patient_Info = pd.read_csv("./Data/parameters.csv").values

# Picking Patient Characteristics
nb_Patient = int(input("Please enter the patient number: "))
age, height, weight, sex = Patient_Info[nb_Patient, 1], Patient_Info[nb_Patient, 2], Patient_Info[nb_Patient, 3], Patient_Info[nb_Patient, 4] 

Patient_Char = [age, height, weight, sex]

c50pR, c50rR, gammaR, betaR, E0R, EmaxR = Patient_Info[nb_Patient, 5], Patient_Info[nb_Patient, 6], Patient_Info[nb_Patient, 7], Patient_Info[nb_Patient, 8], Patient_Info[nb_Patient, 9], Patient_Info[nb_Patient, 10] 
PD_paramR = [c50pR, c50rR, gammaR, betaR, EmaxR, E0R]

Patient_Variables = pd.read_csv(f"./Data/simu_{nb_Patient}.csv").values
Time = Patient_Variables[:, 1]

BIS, Propo_rate, Remi_rate = Patient_Variables[:, 2], Patient_Variables[:, 6], Patient_Variables[:, 7]
Patient_Var = [BIS, Propo_rate, Remi_rate]
Patient = pas.Patient(patient_characteristic=Patient_Char, ts=1, model_propo='Eleveld', model_remi='Eleveld', co_update=True)
c50p, c50r, gamma, beta, E0, Emax = Patient.hill_param
hill_param = [c50p, c50r, gamma, beta, Emax, E0]
LBM = Patient.lbm

#Propo PK
Ap = Patient.propo_pk.A_init
a11, a12, a13, a21, a31, a41 = -Ap[0, 0], Ap[0, 1], Ap[0, 2], Ap[1, 0], Ap[2, 0], Ap[3, 0]
V1 = Patient.propo_pk.v1

#Remi PK
Ar = Patient.remi_pk.A_init
a11r, a12r, a13r, a21r, a31r, a41r = -Ar[0, 0], Ar[0, 1], Ar[0, 2], Ar[1, 0], Ar[2, 0], Ar[3, 0]
V1r = Patient.remi_pk.v1

#%% Simulating the MISO System

# Input/Output Definition
BIS = BIS.T
u_p = Propo_rate.T
u_r = Remi_rate.T

# Sampling Measured Data
N_samp = 2

Time = Time[0::N_samp]
BIS = BIS[0::N_samp]
u_r = u_r[0::N_samp]
u_p = u_p[0::N_samp]

# Smoothing the Sampled Data
window_size = 5
BIS = pd.DataFrame(BIS)
BIS = BIS.rolling(window=window_size, center=True, min_periods=1).mean().values

# Model simulation with the PAS inputs
ParP = np.array([a11, a12, a13, a21, a31, a41])
xx0 = np.zeros([4, 1])
ODE_sol = solve_ivp(lambda t, x: Model(t, x, (u_p/V1).flatten(), ParP, Time),
                    [Time[0], Time[-1]], xx0.flatten(), method='RK45', t_eval=Time, rtol=1e-6)
t = ODE_sol.t
yp = ODE_sol.y.T

ParR = np.array([a11r, a12r, a13r, a21r, a31r, a41r])
ODE_sol = solve_ivp(lambda t, x: Model(t, x, (u_r/V1r).flatten(), ParR, Time),
                    [Time[0], Time[-1]], xx0.flatten(), method='RK45', t_eval=Time, rtol=1e-6)
tt = ODE_sol.t
yr = ODE_sol.y.T

xx1 = np.interp(Time, t, yp[:, 0]).T
xx2 = np.interp(Time, t, yp[:, 1]).T
xx3 = np.interp(Time, t, yp[:, 2]).T
xx4 = np.interp(Time, t, yp[:, 3]).T

xx5 = np.interp(Time, tt, yr[:, 0]).T
xx6 = np.interp(Time, tt, yr[:, 1]).T
xx7 = np.interp(Time, tt, yr[:, 2]).T
xx8 = np.interp(Time, tt, yr[:, 3]).T

# BIS computation
UUp = xx4/c50p
UUr = xx8/c50r
theta = np.divide(UUp, UUp + UUr + 1e-6)
Uu = np.divide(UUp + UUr, 1 - beta*theta + beta*theta**2)
BIS_mod = E0 - Emax*np.divide((Uu**gamma), 1 + Uu**gamma)

# Plot the measured data
fig, axs = plt.subplots(2, 1)
axs[0].plot(Time, u_p, label='Propo')
axs[0].plot(Time, u_r, label='Remi')
axs[0].legend()
axs[0].set_title('Drug injection rates (input)')
axs[0].grid(True)

axs[1].plot(Time, BIS, label='BIS Measured')
axs[1].plot(Time, BIS_mod, linewidth=2, label='BIS Model')
axs[1].legend()
axs[1].set_title('BIS (output)')
axs[1].grid(True)

#%% Running the MHE

# MHE problem formulation
###########################
T = N_samp/60  # sampling period
N_MHE = 20  # estimation horizon

# Data Extension
tt1 = np.arange(0, N_samp*(N_MHE), N_samp)/60
Time = np.concatenate((tt1, Time+tt1[-1]-Time[0]+T))
u_r = np.vstack((np.zeros([N_MHE, 1]), u_r.reshape((-1, 1))))
u_p = np.vstack((np.zeros([N_MHE, 1]), u_p.reshape((-1, 1))))
xx1 = np.vstack((np.zeros([N_MHE, 1]), xx1.reshape(-1, 1))).T
xx2 = np.vstack((np.zeros([N_MHE, 1]), xx2.reshape(-1, 1))).T
xx3 = np.vstack((np.zeros([N_MHE, 1]), xx3.reshape(-1, 1))).T
xx4 = np.vstack((np.zeros([N_MHE, 1]), xx4.reshape(-1, 1))).T
xx5 = np.vstack((np.zeros([N_MHE, 1]), xx5.reshape(-1, 1))).T
xx6 = np.vstack((np.zeros([N_MHE, 1]), xx6.reshape(-1, 1))).T
xx7 = np.vstack((np.zeros([N_MHE, 1]), xx7.reshape(-1, 1))).T
xx8 = np.vstack((np.zeros([N_MHE, 1]), xx8.reshape(-1, 1))).T
BIS = np.vstack((E0*np.ones([N_MHE, 1]), BIS.reshape(-1, 1)))
BIS_mod = np.vstack((E0*np.ones([N_MHE, 1]), BIS_mod.reshape(-1, 1)))

# State bounds
x1_max, x1_min = 1.4*round(np.max(xx1)), 0
x2_max, x2_min = 1.4*round(np.max(xx2)), 0
x3_max, x3_min = 1.4*round(np.max(xx3)), 0
x4_max, x4_min = 1.4*round(np.max(xx4)), 0
x5_max, x5_min = 1.4*round(np.max(xx5)), 0
x6_max, x6_min = 1.4*round(np.max(xx6)), 0
x7_max, x7_min = 1.4*round(np.max(xx7)), 0
x8_max, x8_min = 1.4*round(np.max(xx8)), 0

# Param bounds
x9_max, x9_min = c50p*1.4, c50p*0.6
x10_max, x10_min = c50r*1.4, c50r*0.6
x11_max, x11_min = gamma*1.4, gamma*0.6

# States definition
x1 = cas.SX.sym('x1')
x2 = cas.SX.sym('x2')
x3 = cas.SX.sym('x3')
x4 = cas.SX.sym('x4')
x5 = cas.SX.sym('x5')
x6 = cas.SX.sym('x6')
x7 = cas.SX.sym('x7')
x8 = cas.SX.sym('x8')
C50p = cas.SX.sym('C50p')
C50r = cas.SX.sym('C50r')
Gamma = cas.SX.sym('Gamma')

states = [x1, x2, x3, x4, x5, x6, x7, x8, C50p, C50r, Gamma]
n_states = len(states)
states = cas.vertcat(*states)

# Inputs definition
up = cas.SX.sym('up')
ur = cas.SX.sym('ur')
controls = [up, ur]
n_controls = len(controls)
controls = cas.vertcat(*controls)

# Model definition
rhs = [-a11*x1 + a12*x2 + a13*x3 + up/V1, a21*x1 - a21*x2, a31*x1 - a31*x3, a41*x1 - a41*x4, -a11r*x5 +
       a12r*x6 + a13r*x7 + ur/V1r, a21r*x5 - a21r*x6, a31r*x5 - a31r*x7, a41r*x5 - a41r*x8, 0, 0, 0]
rhs = cas.vertcat(*rhs)
f = cas.Function('f', [states, controls], [rhs]) # System model

# Linear PK mapping function f(x, u)
Up = x4/C50p
Ur = x8/C50r
theta = Up/(Up+Ur)
UU = (Up+Ur)/(1-beta*theta+beta*theta**2)

measurement_rhs = E0-Emax*(UU**Gamma)/(1+UU**Gamma)
h = cas.Function('h', [states], [measurement_rhs])  # Measurement model

# Decision variables
X = cas.SX.sym('x', n_states, (N_MHE+1))  # states
P = cas.SX.sym('P', 1, n_controls*N_MHE + (N_MHE + 1) + (N_MHE + 1)*n_states + 1)

# Parameters (include BIS measurement, the controls measurements, the previous estimated states, and the estimation iteration)
Q = 0.0001  # weighting matrices (output)  y_tilde - y

obj = 0  # objective function
g = []  # constraints vector

N_it = P[:, -1]

# Measurement Penalty
for k in range(0, N_MHE+1):
    st = X[:, k]
    h_x = h(st)
    y_tilde = P[:, k]
    obj += (y_tilde - h_x).T * Q * (y_tilde - h_x)  # Calculate obj

# Model compatibility cost
R = np.diag([1, 550, 550, 1, 1, 50, 750, 1])

for k in range(0, N_MHE):
    st1 = P[range(1 + 3*N_MHE + k, 1 + 3*N_MHE + k + n_states*(N_MHE), N_MHE + 1)]
    con1 = np.hstack((P[:, N_MHE + k+1], P[:, (1 + N_MHE) + N_MHE + k]))
    f_value1 = f(st1, con1)
    st1_next = st1 + (T * f_value1.T)

    obj += (X[0:8, k].T - st1_next[0, 0:8]) @ R @ (X[0:8, k].T - st1_next[0, 0:8]).T
    obj += (X[8, k].T - st1_next[0, 8]) * (1e-3 + 800*np.exp(-300*np.exp(-0.005*N_it))) * (X[8, k].T - st1_next[0, 8]).T
    obj += (X[9, k].T - st1_next[0, 9]) * (1e-5 + 2*np.exp(-300*np.exp(-0.005*N_it))) * (X[9, k].T - st1_next[0, 9]).T
    obj += (X[10, k].T - st1_next[0, 10]) * (1e-3 + 200*np.exp(-300*np.exp(-0.005*N_it))) * (X[10, k].T - st1_next[0, 10]).T

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

lbg = np.zeros([1, n_states*N_MHE]) #equality constraints
ubg = np.zeros([1, n_states*N_MHE]) #equality constraints

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
X0[:, 8] = (c50p + 0.1*c50p)*np.ones([N_MHE + 1])
X0[:, 9] = (c50r + 0.1*c50r)*np.ones([N_MHE + 1])
X0[:, 10] = (gamma + 0.1*gamma)*np.ones([N_MHE + 1])

# Start MHE
MHEiter = 0
OBJ = []

# Initialize the previous estimated state
X_sol = np.zeros([N_MHE + 1, n_states])
X_sol[N_MHE, 8] = c50p + 0.1*c50p
X_sol[N_MHE, 9] = c50r + 0.1*c50r
X_sol[N_MHE, 10] = gamma + 0.1*gamma


start_time = time.time()
X_estimate = []
OBJ = []
for k in range(0, len(BIS) - N_MHE):
    p = np.hstack((BIS[k:k+N_MHE+1, :].T, u_p[k:k+N_MHE, :].T, u_r[k:k+N_MHE, :].T,
                  np.reshape(X_sol.T, (1, (N_MHE+1)*n_states)), np.array([k+1]).reshape(-1, 1)))
    x0 = X0.reshape(((N_MHE+1)*n_states, 1))
    sol = solver(x0=x0, p=p, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
    X_sol = sol['x'][0:n_states*(N_MHE+1)].T.reshape((n_states, N_MHE+1)).T
    X_estimate.append(X_sol[N_MHE, :])
    X0 = np.vstack((X_sol[1:, :], X_sol[-1, :] + f(X_sol[-1, :], [u_p[k+N_MHE-1, :].T, u_r[k+N_MHE-1, :].T]).T))
    OBJ.append(sol['f'])
elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time} seconds.")

X_estimate = np.array(X_estimate).squeeze()
C50p_est = X_estimate[:, 8]
C50r_est = X_estimate[:, 9]
Gamma_est = X_estimate[:, 10]
UUp = X_estimate[:, 3]/C50p_est
UUr = X_estimate[:, 7]/C50r_est

theta = UUp/(UUp+UUr)
U = (UUp + UUr)/(1 - beta*theta + beta*theta**2)
BIS_estimated = E0 - Emax * \
    (U**Gamma_est)/(1 + U**Gamma_est)

plt.figure()
plt.plot(Time[N_MHE:], BIS[N_MHE:],
         linewidth=1.5, label='BIS Measured')
plt.plot(Time[N_MHE:], BIS_mod[N_MHE:],
         linewidth=1.5, label='BIS Model')
plt.plot(Time[N_MHE:], BIS_estimated, '--',
         linewidth=1.5, label='BIS Estimated')
plt.legend(['BIS Measured', 'BIS Model', 'BIS Estimated'])

# Params figures
fig, axs = plt.subplots(3, 1)
axs[0].plot(Time[N_MHE:], C50p_est, '--', linewidth=1.5)
axs[0].plot(Time[N_MHE:], np.full_like(Time[N_MHE:], c50pR), linewidth=1.5)
axs[0].legend(['C50p Estimated', 'C50p Real'])
axs[0].grid(True)

axs[1].plot(Time[N_MHE:], C50r_est, '--', linewidth=1.5)
axs[1].plot(Time[N_MHE:], np.full_like(Time[N_MHE:], c50rR), linewidth=1.5)
axs[1].legend(['C50r Estimated', 'C50r Real'])
axs[1].grid(True)

axs[2].plot(Time[N_MHE:], Gamma_est, '--', linewidth=1.5)
axs[2].plot(Time[N_MHE:], np.full_like(Time[N_MHE:], gammaR), linewidth=1.5)
axs[2].legend(['Gamma Estimated', 'Gamma Real'])
axs[2].grid(True)

plt.show()