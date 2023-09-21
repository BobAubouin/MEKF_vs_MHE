import numpy as np
from typing import Callable
import control as ctrl
import matplotlib.pyplot as plt
from scipy.integrate import simpson
import python_anesthesia_simulator as pas
from TCI_control import TCI


def empiric_gramian(A: np.ndarray, B: np.ndarray, h: Callable, u: list, x_0: np.ndarray, epsilon: float = 1e-3, ts: float = 1):
    """_summary_

    Parameters
    ----------
    A : np.ndarray
        Dynamic matrix of the system such that dot(x) = Ax + Bu.
    B : np.ndarray
        Input matrix of the system such that dot(x) = Ax + Bu. 
    h : Callable
        Function that compute the output of the system. y = h(x).
    u : list
        Inputs values.over the horizon on which the gramian is computed.
    x_0 : np.ndarray
        Initial state of the system.
    epsilon : float, optional
        Perturbation value. The default is 1e-3.
    ts : float, optional
        Time step of the system. The default is 1s.

    Returns
    -------
    np.ndarray
        Gramian matrix.

    """
    n_states = len(x_0)
    n_inputs = len(u)
    n_horizon = len(u[0])

    syst = ctrl.ss(A, B, np.eye(n_states), np.zeros((n_states, n_inputs)))

    # get a base of the the Rn space
    e = np.eye(n_states)

    # init the gramian

    # compute all the trajectories
    Y = np.zeros((n_states, n_horizon))
    T = np.arange(n_horizon)*ts
    for states in range(n_states):
        x_plus = ctrl.forced_response(syst, U=u, X0=x_0 + epsilon * e[:, [states]], T=T, return_x=True)[2]
        x_moins = ctrl.forced_response(syst, U=u, X0=x_0 - epsilon * e[:, [states]], T=T, return_x=True)[2]
        Y[states, :] = (h(x_plus) - h(x_moins))/2/epsilon

    # compute the gramian
    Phi = np.zeros((n_states, n_states, n_horizon))
    for time in range(n_horizon):
        Phi[:, :, time] = np.dot(Y[:, [time]], Y[:, [time]].T)

    Grammian = simpson(Phi, T, axis=2)

    return Grammian, Y


""" Generate simulation files of anetshesia Induction with Propofol and Remifentanil. """


# %% Imports

if __name__ == '__main__':
    # %% Parameters

    sim_duration = 60*20  # 20 mintues
    sampling_time = 1  # 1 second
    BIS_target = 50
    first_propo_target = 4
    first_remi_target = 4
    model_PK = 'Eleveld'
    # %% Run simulation

    min_eigenvalue = []
    for l in range(10):

        # Generate parameters
        np.random.seed(l)
        age = np.random.randint(low=18, high=70)
        height = np.random.randint(low=150, high=190)
        weight = np.random.randint(low=50, high=100)
        gender = np.random.randint(low=0, high=2)

        patient_info = [age, height, weight, gender]
        # Define patient and controller
        patient = pas.Patient(patient_info, ts=sampling_time,
                              model_propo=model_PK, model_remi=model_PK,
                              random_PD=True, random_PK=True)

        controller_propo = TCI(patient_info=patient_info, drug_name='Propofol',
                               sampling_time=sampling_time, model_used=model_PK, control_time=10)
        controller_remi = TCI(patient_info=patient_info, drug_name='Remifentanil',
                              sampling_time=sampling_time, model_used=model_PK, control_time=10)

        target_propo = first_propo_target
        target_remi = first_remi_target
        control_time_propo = np.random.randint(low=8*60, high=9*60)
        control_time_remi = np.random.randint(low=8*60, high=9*60)

        # define parameters for the grammian computation
        grammian_period = 10*60//sampling_time  # 1 secondes
        emp_gram = np.zeros((11, 11, sim_duration//sampling_time//grammian_period))
        A_p = patient.propo_pk.continuous_sys.A[:4, :4]
        B_p = patient.propo_pk.continuous_sys.B[:4, :]
        A_r = patient.remi_pk.continuous_sys.A[:4, :4]
        B_r = patient.remi_pk.continuous_sys.B[:4, :]

        A = np.block([[A_p, np.zeros((4, 7))], [np.zeros((4, 4)), A_r, np.zeros((4, 3))], [np.zeros((3, 11))]])
        B = np.block([[B_p, np.zeros((4, 1))], [np.zeros((4, 1)), B_r], [np.zeros((3, 2))]])

        def h_g(x):
            up = x[3, :]/x[8, :]
            ur = x[7, :]/x[9, :]
            I = (up + ur) ** x[10, :]
            bis = 97.4 * (1 - I/(1+I))
            # bis = x[3, :]*x[8, :] + x[7, :]*x[9, :] + x[10, :]
            return bis

        N_simu = sim_duration//sampling_time
        u1 = np.sin(np.linspace(0, 1*np.pi*50 * N_simu*sampling_time, N_simu)) * 5 + 8
        u2 = np.sin(np.linspace(0, 3*np.pi*50 * N_simu*sampling_time, N_simu)+2) * 3 + 6
        u = np.array([u1, u2]) + np.random.randn(2, N_simu) * 2
        u[:, N_simu//2:] = np.array([[8], [6]]) @ np.ones((1, N_simu - N_simu//2))
        u = u/50
        x_save = np.zeros((11, N_simu))
        # run simulation
        for j in range(sim_duration//sampling_time):
            if j % 10 == 0:
                u_propo = controller_propo.one_step(target_propo) * 10/3600
                u_remi = controller_remi.one_step(target_remi) * 10/3600
            patient.one_step(u_propo, u_remi, noise=True)
            # at each control time if we are not in the intervall [40,60]
            if j % control_time_propo == 0:
                if patient.dataframe['BIS'].iloc[-1] < BIS_target-10:
                    target_propo -= 1
                elif patient.dataframe['BIS'].iloc[-1] < BIS_target-5:
                    target_propo -= .5
                elif patient.dataframe['BIS'].iloc[-1] > BIS_target+10:
                    target_propo += 1
                elif patient.dataframe['BIS'].iloc[-1] > BIS_target+5:
                    target_propo += .5
            if j % control_time_remi == 0:
                if patient.dataframe['BIS'].iloc[-1] < BIS_target-5:
                    target_remi -= 1
                elif patient.dataframe['BIS'].iloc[-1] > BIS_target+5:
                    target_remi += 1
            xp = patient.propo_pk.x[:4]
            xr = patient.remi_pk.x[:4]
            x_hill = np.array([patient.hill_param[0], patient.hill_param[1], patient.hill_param[2]])
            x = np.block([[xp, xr, x_hill]]).T
            x_save[:, j] = x[:, 0]
            patient.dataframe['BIS'].loc[-1] = h_g(x)
            if j % grammian_period == 0 and j > 0:

                up = patient.dataframe['u_propo'].iloc[-grammian_period:].to_numpy()
                ur = patient.dataframe['u_remi'].iloc[-grammian_period:].to_numpy()
                u_g = np.block([[up], [ur]])
                id = j//grammian_period - 1
                emp_gram[:, :, id] = empiric_gramian(
                    A, B, h_g, u_g, x_save[:, [j-grammian_period]], epsilon=1e-10, ts=sampling_time/4)[0]

        # get minimum eigen value of the grammian
        min_eig = np.zeros((sim_duration//sampling_time//grammian_period-1))
        for i in range(sim_duration//sampling_time//grammian_period-1):
            min_eig[i] = np.min(np.linalg.eig(emp_gram[:, :, i])[0])
        min_eigenvalue.append(min_eig)
    print(np.mean(min_eigenvalue, axis=0))

    # %% Plot results
    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(patient.dataframe['Time'], patient.dataframe['u_propo'], label='u_propo')
    plt.plot(patient.dataframe['Time'], patient.dataframe['u_remi'], label='u_remi')
    plt.xlim([0, sim_duration])
    plt.grid()
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.plot(patient.dataframe['Time'], patient.dataframe['BIS'], label='BIS')
    plt.xlim([0, sim_duration])
    plt.grid()
    plt.legend()
    plt.subplot(3, 1, 3)
    plt.plot([patient.dataframe['Time'].iloc[(i+1)*grammian_period]
              for i in range(sim_duration//sampling_time//grammian_period-1)], min_eig, 'o--', label='min grammian eigen value')
    plt.xlim([0, sim_duration])
    ax = plt.gca()
    ax.set_yscale('log')  # , linthresh=1e-9)
    plt.grid()
    plt.show()
