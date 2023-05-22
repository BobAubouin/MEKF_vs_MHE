# Loading Libraries
import casadi as cas  # to be used for numerical optimization
import numpy as np  # to perform mathematical operations
import pandas as pd  # to be used as a data manipulation tool
from scipy.integrate import solve_ivp # to solve ordinary differential equations
from matplotlib import pyplot as plt  # to plot figures
from Model import Model # to import the created dynamical system with the known theoretical parameters
import python_anesthesia_simulator as pas

# %% Moving Horizon Estimator
class MHE():
    def __init__(self, nb_Patient: int, N_MHE = 20, ts = 1, N_samp = 2, model_propo = 'Eleveld', model_remi = 'Eleveld'):
        self.nb_Patient = nb_Patient  # patient number
        self.N_MHE = N_MHE  # estimation horizon
        self.ts = ts  # sampling time in ms
        self.N_samp = N_samp  # number of samples
        self.T = self.N_samp/60  # sampling period
        self.model_propo = model_propo
        self.model_remi = model_remi

        # Loading Patients' Data
        Patient_Info = pd.read_csv("parameters.csv").values
        age, height, weight, sex = Patient_Info[nb_Patient, 1], Patient_Info[nb_Patient, 2], Patient_Info[nb_Patient, 3], Patient_Info[nb_Patient, 4] 
        self.Patient_Char = [age, height, weight, sex]
        
        c50p, c50r, gamma, beta, E0, Emax = Patient_Info[nb_Patient, 5], Patient_Info[nb_Patient, 6], Patient_Info[nb_Patient, 7], Patient_Info[nb_Patient, 8], Patient_Info[nb_Patient, 9], Patient_Info[nb_Patient, 10] 
        self.Hill_Par = [c50p, c50r, gamma, beta, E0, Emax]

        Patient_Variables = pd.read_csv(f"simu_{nb_Patient}.csv").values
        self.Time = Patient_Variables[:, 1]
        BIS, Propo_rate, Remi_rate = Patient_Variables[:, 2], Patient_Variables[:, 6], Patient_Variables[:, 7]
        self.Patient_Var = [BIS, Propo_rate, Remi_rate]

        x1, x2, x3, x4, x5, x6, x7, x8, emax, C50p, C50r, Gamma, Beta = cas.SX.sym('x1'), cas.SX.sym('x2'), cas.SX.sym('x3'), cas.SX.sym('x4'), cas.SX.sym('x5'), cas.SX.sym('x6'), cas.SX.sym('x7'), cas.SX.sym('x8'), cas.SX.sym('emax'), cas.SX.sym('C50p'), cas.SX.sym('C50r'), cas.SX.sym('Gamma'), cas.SX.sym('Beta')
        states = [x1, x2, x3, x4, x5, x6, x7, x8, emax, C50p, C50r, Gamma, Beta]
        n_states = len(states)
        states = cas.vertcat(*states)
        self.States = [states, n_states]

        vp, vr = cas.SX.sym('vp'), cas.SX.sym('vr')
        controls = [vp, vr]
        n_controls = len(controls)
        controls = cas.vertcat(*controls)
        self.Inputs = [controls, n_controls]

        Patient = pas.Patient(patient_characteristic=self.Patient_Char, ts=self.ts, model_propo=self.model_propo, model_remi=self.model_remi, co_update=True)
        self.LBM = Patient.lbm

        #Propofol PK
        Ap = Patient.propo_pk.A_init
        a11, a12, a13, a21, a31, a41 = -Ap[0, 0], Ap[0, 1], Ap[0, 2], Ap[1, 0], Ap[2, 0], Ap[3, 0]
        self.PK_propo = np.array([a11, a12, a13, a21, a31, a41])
        self.V1 = Patient.propo_pk.v1

        #Remi PK
        Ar = Patient.remi_pk.A_init
        a11r, a12r, a13r, a21r, a31r, a41r = -Ar[0, 0], Ar[0, 1], Ar[0, 2], Ar[1, 0], Ar[2, 0], Ar[3, 0]
        self.PK_remi = np.array([a11r, a12r, a13r, a21r, a31r, a41r])
        self.V1r = Patient.remi_pk.v1

    def BIS_Real(self):
        Patient_Var = self.Patient_Var

        BIS = Patient_Var[0]
        BIS = BIS.T
        y_measurements = BIS

        Propo_rate = Patient_Var[1]
        u_p = Propo_rate.T

        Remi_rate = Patient_Var[2]
        u_r = Remi_rate.T

        # MHE problem formulation
        Time = self.Time
        N_samp = self.N_samp  # number of samples

        # Reduce the number of elements in the measured data
        Time = Time[0::N_samp]
        BIS = BIS[0::N_samp]  # measured BIS
        y_measurements = y_measurements[0::N_samp]
        u_p, u_r = u_p[0::N_samp], u_r[0::N_samp]

        # BIS filtering
        window_size = 5
        y_measurements = pd.DataFrame(y_measurements)
        y_measurements.reset_index(drop=True, inplace=True)
        y_measurements = y_measurements.rolling(window=window_size, center=True, min_periods=1).mean().values  # filtered BIS
        return BIS, y_measurements, Time, u_p, u_r

    def BIS_Model(self):
        # Model simulation with the real input, to calibrate the PD params
        BIS, y_measurements, Time, u_p, u_r = self.BIS_Real()

        # Propofol Model Simulation
        V1 = self.V1
        Par = self.PK_propo
        xx0 = np.zeros([4, 1])
        ODE_sol = solve_ivp(lambda t, x: Model(t, x, (u_p/V1).flatten(), Par, Time), [Time[0], Time[-1]], xx0.flatten(), method='RK45', t_eval=Time, rtol=1e-6)
        t = ODE_sol.t
        yp = ODE_sol.y.T

        # Remifentalin Model Simulation
        V1r = self.V1r
        Par = self.PK_remi
        ODE_sol = solve_ivp(lambda t, x: Model(t, x, (u_r/V1r).flatten(), Par, Time), [Time[0], Time[-1]], xx0.flatten(), method='RK45', t_eval=Time, rtol=1e-6)
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

        # Data Extension
        N_samp = self.N_samp
        T = self.T
        c50p, c50r, gamma, beta, E0, Emax = self.Hill_Par

        N_ex = self.N_MHE//1
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
        xx = [xx1, xx2, xx3, xx4, xx5, xx6, xx7, xx8]
        y_measurements = np.vstack((E0*np.ones([N_ex, 1]), np.reshape(y_measurements.T, (-1, 1))))  # filtered BIS
        BIS = np.vstack((E0*np.ones([N_ex, 1]), BIS.reshape(-1, 1)))  # measured BIS
        UUp, UUr = xx4/c50p, xx8/c50r
        theta = np.divide(UUp, UUp + UUr + 1e-6)
        Uu = np.divide(UUp + UUr, 1 - beta*theta + beta*theta**2)
        BIS_mod = E0 - np.divide(Emax*(Uu**gamma), 1 + Uu**gamma)  # Model BIS
        return BIS_mod, Time, u_p, u_r, BIS, y_measurements, xx

    def Plot_Comp(self):
        BIS_mod, Time, u_p, u_r, BIS, y_measurements, xx = self.BIS_Model()

        fig, axs = plt.subplots(2, 1)
        axs[0].plot(Time, u_p, label='Propo')
        axs[0].plot(Time, u_r, label='Remi')
        axs[0].legend()
        axs[0].set_title('Drug injection rates (input)')
        axs[0].grid(True)

        axs[1].plot(Time, BIS, label='BIS measured')
        axs[1].plot(Time, y_measurements, linewidth=2, label='BIS filtered')
        axs[1].plot(Time, BIS_mod.real.flatten(),
                    linewidth=2, label='BIS model')
        axs[1].legend()
        axs[1].set_title('BIS (output)')
        axs[1].grid(True)

        plt.show()

    def State_Bounds(self):
        xx1, xx2, xx3, xx4, xx5, xx6, xx7, xx8 = self.BIS_Model()[6]

        x1_max = 1.3*round(np.max(xx1))
        x2_max = 1.3*round(np.max(xx2))
        x3_max = 1.3*round(np.max(xx3))
        x4_max = 1.3*round(np.max(xx4))
        x5_max = 1.3*round(np.max(xx5))
        x6_max = 1.3*round(np.max(xx6))
        x7_max = 1.3*round(np.max(xx7))
        x8_max = 1.3*round(np.max(xx8))

        x1_min = x2_min = x3_min = x4_min = x5_min = x6_min = x7_min = x8_min = 0

        x_maxS = [x1_max, x2_max, x3_max, x4_max, x5_max, x6_max, x7_max, x8_max]
        x_minS = [x1_min, x2_min, x3_min, x4_min, x5_min, x6_min, x7_min, x8_min]
        return x_maxS, x_minS

    def Param_Bounds(self):
        x9_max, x9_min, x10_max, x10_min, x11_max, x11_min, x12_max, x12_min, x13_max, x13_min = 90, 30, 9, 2, 25, 6, 5, 1, 3, 0
        x_maxP = [x9_max, x10_max, x11_max, x12_max, x13_max]
        x_minP = [x9_min, x10_min, x11_min, x12_min, x13_min]
        return x_maxP, x_minP

    def Model_Def(self):
        PK_Propo = self.PK_propo
        PK_Remi = self.PK_remi
        states = self.States[0]
        controls = self.Inputs[0]
        E0 = self.Hill_Par[4]

        x1, x2, x3, x4, x5, x6, x7, x8, emax, C50p, C50r, Gamma, Beta = states[0], states[1], states[2], states[3], states[4], states[5], states[6], states[7], states[8], states[9], states[10], states[11], states[12]
        a11, a12, a13, a21, a31, a41 = self.PK_propo
        a11r, a12r, a13r, a21r, a31r, a41r = self.PK_remi
        V1 = self.V1
        V1r = self.V1r
        vp, vr = controls[0], controls[1]

        rhs = [-a11*x1 + a12*x2 + a13*x3 + vp/V1, a21*x1 - a21*x2, a31*x1 - a31*x3, a41*x1 - a41*x4, -a11r*x5 + a12r*x6 + a13r*x7 + vr/V1r, a21r*x5 - a21r*x6, a31r*x5 - a31r*x7, a41r*x5 - a41r*x8, 0, 0, 0, 0, 0]
        rhs = cas.vertcat(*rhs)
        # Linear PK mapping function f(x, u)
        f = cas.Function('f', [states, controls], [rhs])
        Up, Ur = x4/C50p, x8/C50r
        theta = Up/(Up+Ur+1e-6)
        UU = (Up+Ur)/(1-Beta*theta+Beta*theta**2)
        measurement_rhs = E0-emax*(UU**Gamma)/(1+UU**Gamma)
        # Measurement model
        h = cas.Function('h', [states], [measurement_rhs])
        return f, h

    def Decision_Var(self):
        n_states = self.States[1]
        n_controls = self.Inputs[1]
        N_MHE = self.N_MHE

        X = cas.SX.sym('x', n_states, (N_MHE+1))  # states
        P = cas.SX.sym('P', 1, n_controls*N_MHE + (N_MHE + 1) + (N_MHE + 1)*n_states + 1)

        V = 0.0001  # weighting matrices (output)  y_tilde - y
        obj = 0  # objective function
        N_it = P[:, -1]
        P = P
        return X, P, V, obj, N_it

    def Model_CompCost(self, X, P, V, obj, N_it):
        N_MHE = self.N_MHE
        #X, P, V, obj, N_it = self.Decision_Var()
        h = self.Model_Def()[1]
        n_states = self.States[1]
        f = self.Model_Def()[0]
        T = self.T

        for k in range(0, N_MHE+1):
            st = X[:, k]
            h_x = h(st)
            y_tilde = P[:, k]
            obj += (y_tilde - h_x).T * V * (y_tilde - h_x)  # Calculate obj

        R = np.eye(n_states - 5)
        R[1, 1], R[2, 2], R[5, 5], R[6, 6] = 550, 550, 50, 750

        for k in range(0, N_MHE - 1):
            st1 = P[3*N_MHE + k+1: 1 + 3*N_MHE + k + n_states*(N_MHE) - 7: N_MHE + 1]
            con1 = cas.horzcat(P[:, N_MHE + k+1], P[:, (1 + N_MHE) + N_MHE + k])
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
        return obj

    # Multiple shooting constraints
    def MS_Const(self):
        n_states = self.States[1]
        N_MHE = self.N_MHE
        X, P, V, obj, N_it = self.Decision_Var()
        T = self.T
        f = self.Model_Def()[0]
        obj = self.Model_CompCost(X, P, V, obj, N_it)
        x1_max, x2_max, x3_max, x4_max, x5_max, x6_max, x7_max, x8_max = self.State_Bounds()[0]
        x1_min, x2_min, x3_min, x4_min, x5_min, x6_min, x7_min, x8_min = self.State_Bounds()[1]
        x9_max, x10_max, x11_max, x12_max, x13_max = self.Param_Bounds()[0]
        x9_min, x10_min, x11_min, x12_min, x13_min = self.Param_Bounds()[1]

        g = [] #constraints vector
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

        return solver, lbg, ubg, lbx, ubx

###########################################################
######## ALL OF THE ABOVE IS JUST A PROBLEM SET UP ########
###########################################################

###########################################################
############# MHE Simulation loop starts here #############
###########################################################

    def BIS_MHE(self):
        N_MHE = self.N_MHE
        n_states = self.States[1]
        c50p, c50r, gamma, beta, E0, Emax = self.Hill_Par
        BIS_Model = self.BIS_Model()
        u_p, u_r, y_measurements = BIS_Model[2], BIS_Model[3], BIS_Model[5]
        solver, lbg, ubg, lbx, ubx = self.MS_Const()
        f = self.Model_Def()[0]

        X_estimate = []  # contains the MHE estimate of the states

        X0 = np.zeros([N_MHE + 1, n_states])
        X0[:, : n_states - 5] = np.zeros([N_MHE + 1, n_states - 5])
        X0[:, 8] = (Emax + 0.1*Emax)*np.ones([N_MHE + 1])
        X0[:, 9] = (c50p + 0.1*c50p)*np.ones([N_MHE + 1])
        X0[:, 10] = (c50r + 0.1*c50r)*np.ones([N_MHE + 1])
        X0[:, 11] = (gamma + 0.1*gamma)*np.ones([N_MHE + 1])
        X0[:, 12] = (beta + 0.1*beta)*np.ones([N_MHE + 1])

        OBJ = []

        # Initialize the previous estimated state
        X_sol = np.zeros([N_MHE + 1, n_states])
        X_sol[N_MHE, 8] = Emax + 0.1*Emax
        X_sol[N_MHE, 9] = c50p + 0.1*c50p
        X_sol[N_MHE, 10] = c50r + 0.1*c50r
        X_sol[N_MHE, 11] = gamma + 0.1*gamma
        X_sol[N_MHE, 12] = beta + 0.1*beta

        for k in range(0, len(y_measurements) - N_MHE):
            p = np.hstack((y_measurements[k:k+N_MHE+1, :].T, u_p[k:k+N_MHE, :].T, u_r[k:k+N_MHE, :].T, np.reshape(X_sol.T, (1, (N_MHE+1)*n_states)), np.array([k+1]).reshape(-1, 1)))
            x0 = X0.reshape(((N_MHE+1)*n_states, 1))
            sol = solver(x0=x0, p=p, lbx=lbx, ubx=ubx, lbg=lbg.T, ubg=ubg.T)
            X_sol = sol['x'][0:n_states * (N_MHE+1)].T.reshape((n_states, N_MHE+1)).T
            X_estimate.append(X_sol[N_MHE, :])
            X0 = np.vstack((X_sol[1:, :], X_sol[-1, :] + f(X_sol[-1, :], [u_p[k+N_MHE-1, :].T, u_r[k+N_MHE-1, :].T]).T))
            OBJ.append(sol['f'])

        X_estimate = np.array(X_estimate).squeeze()
        UUp, UUr = X_estimate[:, 3]/X_estimate[:, 9], X_estimate[:, 7]/X_estimate[:, 10]
        theta = UUp/(UUp+UUr)
        U = (UUp + UUr)/(1 - X_estimate[:, 12] * theta + X_estimate[:, 12]*theta**2)
        BIS_estimated = E0 - X_estimate[:, 8] * (U**X_estimate[:, 11])/(1 + U**X_estimate[:, 11])
        X_estimate = X_estimate
        return BIS_estimated, X_estimate

    def Plot_EstBIS(self):
        N_MHE = self.N_MHE
        BIS_mod, Time, u_p, u_r, BIS, y_measurements, xx = self.BIS_Model()
        BIS_estimated = self.BIS_MHE()[0]

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

        plt.show()

    def Plot_Params(self):
        Time = self.BIS_Model()[1]
        N_MHE = self.N_MHE
        X_estimate = self.BIS_MHE()[1]

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