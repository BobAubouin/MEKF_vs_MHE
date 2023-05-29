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
        self.ts = ts #sampling time
        self.N_samp = N_samp  # number of samples
        self.T = self.N_samp/60  # sampling period
        self.model_propo = model_propo
        self.model_remi = model_remi

        # Loading Patients' Data
        Patient_Info = pd.read_csv("./Data/parameters.csv").values
        age, height, weight, sex = Patient_Info[nb_Patient, 1], Patient_Info[nb_Patient, 2], Patient_Info[nb_Patient, 3], Patient_Info[nb_Patient, 4] 
        self.Patient_Char = [age, height, weight, sex]
        
        c50pR, c50rR, gammaR = Patient_Info[nb_Patient, 5], Patient_Info[nb_Patient, 6], Patient_Info[nb_Patient, 7]
        self.Par_Real = [c50pR, c50rR, gammaR] #Real Patient Params

        Patient_Variables = pd.read_csv(f"./Data/simu_{nb_Patient}.csv").values
        self.Time = Patient_Variables[:, 1]
        BIS, Propo_rate, Remi_rate = Patient_Variables[:, 2], Patient_Variables[:, 6], Patient_Variables[:, 7]
        self.Patient_Var = [BIS, Propo_rate, Remi_rate]

        Patient = pas.Patient(patient_characteristic=self.Patient_Char, ts=self.ts, model_propo=self.model_propo, model_remi=self.model_remi, co_update=True)
        c50p, c50r, gamma, beta, E0, Emax = Patient.hill_param
        self.Hill_Par = [c50p, c50r, gamma, beta, Emax, E0]
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

        #States Definition
        x1, x2, x3, x4, x5, x6, x7, x8, C50p, C50r, Gamma = cas.SX.sym('x1'), cas.SX.sym('x2'), cas.SX.sym('x3'), cas.SX.sym('x4'), cas.SX.sym('x5'), cas.SX.sym('x6'), cas.SX.sym('x7'), cas.SX.sym('x8'), cas.SX.sym('C50p'), cas.SX.sym('C50r'), cas.SX.sym('Gamma')
        states = [x1, x2, x3, x4, x5, x6, x7, x8, C50p, C50r, Gamma]
        n_states = len(states)
        states = cas.vertcat(*states)
        self.States = [states, n_states]

        up, ur = cas.SX.sym('up'), cas.SX.sym('ur')
        controls = [up, ur]
        n_controls = len(controls)
        controls = cas.vertcat(*controls)
        self.Inputs = [controls, n_controls]

        # Model definition
        rhs = [-a11*x1 + a12*x2 + a13*x3 + up/self.V1, a21*x1 - a21*x2, a31*x1 - a31*x3, a41*x1 - a41*x4, -a11r*x5 +
            a12r*x6 + a13r*x7 + ur/self.V1r, a21r*x5 - a21r*x6, a31r*x5 - a31r*x7, a41r*x5 - a41r*x8, 0, 0, 0]
        rhs = cas.vertcat(*rhs)
        self.f = cas.Function('f', [states, controls], [rhs]) # System model

        # Linear PK mapping function f(x, u)
        Up = x4/C50p
        Ur = x8/C50r
        theta = Up/(Up+Ur)
        UU = (Up+Ur)/(1-beta*theta+beta*theta**2)

        measurement_rhs = E0-Emax*(UU**Gamma)/(1+UU**Gamma)
        self.h = cas.Function('h', [states], [measurement_rhs])  # Measurement model

        # Decision variables
        self.X = cas.SX.sym('x', n_states, (N_MHE+1))  # states
        self.P = cas.SX.sym('P', 1, n_controls*N_MHE + (N_MHE + 1) + (N_MHE + 1)*n_states + 1)

        # Parameters
        self.Q = 0.0001  # weighting matrices (output)  y_tilde - y

        self.obj = 0  # objective function
        self.g = []  # constraints vector

        self.N_it = self.P[:, -1]


    def BIS(self):
        #Input/Output Definition
        Patient_Var = self.Patient_Var
        BIS = Patient_Var[0]
        BIS = BIS.T
        Propo_rate = Patient_Var[1]
        u_p = Propo_rate.T
        Remi_rate = Patient_Var[2]
        u_r = Remi_rate.T
        c50p, c50r, gamma, beta, Emax, E0 = self.Hill_Par

        # Sampling Measured Data
        Time = self.Time
        N_samp = self.N_samp  # number of samples

        Time = Time[0::N_samp]
        BIS = BIS[0::N_samp]
        u_p, u_r = u_p[0::N_samp], u_r[0::N_samp]

        # Smoothing the Sampled Data
        window_size = 5
        BIS = pd.DataFrame(BIS)
        BIS = BIS.rolling(window=window_size, center=True, min_periods=1).mean().values

        # Propofol Model Simulation
        V1 = self.V1
        ParP = self.PK_propo
        xx0 = np.zeros([4, 1])
        ODE_sol = solve_ivp(lambda t, x: Model(t, x, (u_p/V1).flatten(), ParP, Time), [Time[0], Time[-1]], xx0.flatten(), method='RK45', t_eval=Time, rtol=1e-6)
        t = ODE_sol.t
        yp = ODE_sol.y.T

        # Remifentalin Model Simulation
        V1r = self.V1r
        ParR = self.PK_remi
        ODE_sol = solve_ivp(lambda t, x: Model(t, x, (u_r/V1r).flatten(), ParR, Time), [Time[0], Time[-1]], xx0.flatten(), method='RK45', t_eval=Time, rtol=1e-6)
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
        
        xx = [xx1, xx2, xx3, xx4, xx5, xx6, xx7, xx8] #states

        # BIS computation
        UUp = xx4/c50p
        UUr = xx8/c50r
        theta = np.divide(UUp, UUp + UUr + 1e-6)
        Uu = np.divide(UUp + UUr, 1 - beta*theta + beta*theta**2)
        BIS_mod = E0 - Emax*np.divide((Uu**gamma), 1 + Uu**gamma)
        return BIS, BIS_mod, Time, u_p, u_r, xx

    def Plot_Comp(self):
        BIS, BIS_mod, Time, u_p, u_r, xx = self.BIS()
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

        plt.show()

    def Bounds(self):
        BIS, BIS_mod, Time, u_p, u_r, xx = self.BIS()
        xx1, xx2, xx3, xx4, xx5, xx6, xx7, xx8 = xx
        c50p, c50r, gamma, beta, Emax, E0 = self.Hill_Par
        N_MHE = self.N_MHE
        N_samp = self.N_samp
        T = self.T

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
        IOS = [u_r, u_p, BIS, BIS_mod, xx1, xx2, xx3, xx4, xx5, xx6, xx7, xx8, Time]

        # State bounds
        x1_max, x1_min = 1.4*round(np.max(xx1)), 0
        x2_max, x2_min = 1.4*round(np.max(xx2)), 0
        x3_max, x3_min = 1.4*round(np.max(xx3)), 0
        x4_max, x4_min = 1.4*round(np.max(xx4)), 0
        x5_max, x5_min = 1.4*round(np.max(xx5)), 0
        x6_max, x6_min = 1.4*round(np.max(xx6)), 0
        x7_max, x7_min = 1.4*round(np.max(xx7)), 0
        x8_max, x8_min = 1.4*round(np.max(xx8)), 0

        x_maxS = [x1_max, x2_max, x3_max, x4_max, x5_max, x6_max, x7_max, x8_max]
        x_minS = [x1_min, x2_min, x3_min, x4_min, x5_min, x6_min, x7_min, x8_min]

        # Param bounds
        x9_max, x9_min = c50p*1.4, c50p*0.6
        x10_max, x10_min = c50r*1.4, c50r*0.6
        x11_max, x11_min = gamma*1.4, gamma*0.6
        x_maxP = [x9_max, x10_max, x11_max]
        x_minP = [x9_min, x10_min, x11_min]

        return IOS, x_maxS, x_minS, x_maxP, x_minP

    def OBJ_Function(self, X, P, Q, obj, N_it):
        N_MHE = self.N_MHE
        h = self.h
        n_states = self.States[1]
        f = self.f
        T = self.T

        # Measurement Penalty
        for k in range(0, N_MHE+1):
            st = X[:, k]
            h_x = h(st)
            y_tilde = P[:, k]

            obj += (y_tilde - h_x).T * Q * (y_tilde - h_x)  # Calculate obj

        # Model Compatibility Cost
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
            
        return obj

    # Multiple shooting constraints
    def MS_Const(self):
        n_states = self.States[1]
        N_MHE = self.N_MHE
        X, P, Q, g, obj, N_it = self.X, self.P, self.Q, self.g, self.obj, self.N_it
        T = self.T
        f = self.f
        obj = self.OBJ_Function(X, P, Q, obj, N_it)
        x1_max, x2_max, x3_max, x4_max, x5_max, x6_max, x7_max, x8_max = self.Bounds()[1]
        x1_min, x2_min, x3_min, x4_min, x5_min, x6_min, x7_min, x8_min = self.Bounds()[2]
        x9_max, x10_max, x11_max = self.Bounds()[3]
        x9_min, x10_min, x11_min = self.Bounds()[4]

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

        return solver, lbg, ubg, lbx, ubx

    def BIS_MHE(self):
        N_MHE = self.N_MHE
        n_states = self.States[1]
        c50p, c50r, gamma, beta, E0, Emax = self.Hill_Par
        BIS = self.Bounds()[0]
        u_r, u_p, BIS = BIS[0], BIS[1], BIS[2]
        solver, lbg, ubg, lbx, ubx = self.MS_Const()
        f = self.f

        X0 = np.zeros([N_MHE + 1, n_states])
        X0[:, : n_states - 5] = np.zeros([N_MHE + 1, n_states - 5])
        X0[:, 8] = (c50p + 0.1*c50p)*np.ones([N_MHE + 1])
        X0[:, 9] = (c50r + 0.1*c50r)*np.ones([N_MHE + 1])
        X0[:, 10] = (gamma + 0.1*gamma)*np.ones([N_MHE + 1])

        # Initialize the previous estimated state
        X_sol = np.zeros([N_MHE + 1, n_states])
        X_sol[N_MHE, 8] = c50p + 0.1*c50p
        X_sol[N_MHE, 9] = c50r + 0.1*c50r
        X_sol[N_MHE, 10] = gamma + 0.1*gamma

        OBJ = []
        X_estimate = []  # contains the MHE estimate of the states
        for k in range(0, len(BIS) - N_MHE):
            p = np.hstack((BIS[k:k+N_MHE+1, :].T, u_p[k:k+N_MHE, :].T, u_r[k:k+N_MHE, :].T, np.reshape(X_sol.T, (1, (N_MHE+1)*n_states)), np.array([k+1]).reshape(-1, 1)))
            x0 = X0.reshape(((N_MHE+1)*n_states, 1))
            sol = solver(x0=x0, p=p, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)
            X_sol = sol['x'][0:n_states*(N_MHE+1)].T.reshape((n_states, N_MHE+1)).T
            X_estimate.append(X_sol[N_MHE, :])
            X0 = np.vstack((X_sol[1:, :], X_sol[-1, :] + f(X_sol[-1, :], [u_p[k+N_MHE-1, :].T, u_r[k+N_MHE-1, :].T]).T))
            OBJ.append(sol['f'])

        X_estimate = np.array(X_estimate).squeeze()
        C50p_est = X_estimate[:, 8]
        C50r_est = X_estimate[:, 9]
        Gamma_est = X_estimate[:, 10]
        UUp = X_estimate[:, 3]/C50p_est
        UUr = X_estimate[:, 7]/C50r_est
        theta = UUp/(UUp+UUr)
        U = (UUp + UUr)/(1 - beta*theta + beta*theta**2)
        BIS_estimated = E0 - Emax * (U**Gamma_est)/(1 + U**Gamma_est)
        Par_estimated = [C50p_est, C50r_est, Gamma_est]
        return BIS_estimated, Par_estimated, X_estimate

    def Plot_Est(self):
        N_MHE = self.N_MHE
        BIS = self.Bounds()[0]
        BIS, BIS_mod, Time = BIS[2], BIS[3], BIS[12]
        Estimates = self.BIS_MHE()
        BIS_estimated = Estimates[0]
        C50p_est, C50r_est, Gamma_est = Estimates[1]
        c50pR, c50rR, gammaR = self.Par_Real

        plt.figure()
        plt.plot(Time[N_MHE:], BIS[N_MHE:], linewidth=1.5, label='BIS Measured')
        plt.plot(Time[N_MHE:], BIS_mod[N_MHE:], linewidth=1.5, label='BIS Model')
        plt.plot(Time[N_MHE:], BIS_estimated, '--', linewidth=1.5, label='BIS Estimated')
        plt.legend(['BIS Measured', 'BIS Model', 'BIS Estimated'])

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