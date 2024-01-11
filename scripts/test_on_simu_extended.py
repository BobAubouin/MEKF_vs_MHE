"""Test MEKF on the simulated data."""

# %% Import
import time
import multiprocessing as mp
from functools import partial
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from filterpy.common import Q_continuous_white_noise
import python_anesthesia_simulator as pas
import scipy
import tqdm

from estimators_extended import MEKF, MHE_integrator, MEKF_MHE
# %% define simulation function


def simulation(patient_index: int, design_param: list, run_bool: list) -> tuple[list]:
    """_summary_

    Parameters
    ----------
    patient_index : int
        index of the patient to simulate
    design_param : list
        list of list of the design parameters [mekf, mhe, mekf_mhe]
         mekf = [R, Q, P0, eta0, grid_vector, lambda_1, lambda_2, nu, epsilon]
         mhe = [R, Q, theta, N_mhe]
        mekf_mhe = switch_time
    run_bool : list
        list of boolean to run the different estimators [mekf_p, mekf_n, mhe]

    Returns
    -------
    tuple[list]
        last parameter estimation.
    """
    # load the data
    Patient_info = pd.read_csv(
        './data/simulations/parameters.csv').iloc[patient_index][['age', 'height', 'weight', 'gender']].values
    Patient_simu = pd.read_csv(f'./data/simulations/simu_{patient_index}.csv')
    Time = Patient_simu['Time'].to_numpy()
    BIS = Patient_simu['BIS'].to_numpy()
    U_propo = Patient_simu['u_propo'].to_numpy()
    U_remi = Patient_simu['u_remi'].to_numpy()

    # define the model
    model = 'Eleveld'
    simulator = pas.Patient(Patient_info, model_propo=model, model_remi=model, ts=2)
    A_p = simulator.propo_pk.continuous_sys.A[:4, :4]
    A_r = simulator.remi_pk.continuous_sys.A[:4, :4]
    B_p = simulator.propo_pk.continuous_sys.B[:4]
    B_r = simulator.remi_pk.continuous_sys.B[:4]
    BIS_param_nominal = simulator.hill_param

    A = np.block([[A_p, np.zeros((4, 4))], [np.zeros((4, 4)), A_r]])
    B = np.block([[B_p, np.zeros((4, 1))], [np.zeros((4, 1)), B_r]])

    # init the estimators
    design_param_mekf = design_param[0]
    design_param_mhe = design_param[1]
    switch_time = design_param[2]

    if run_bool[0]:
        mekf_p = MEKF(A, B, design_param_mekf[4], ts=2, Q=design_param_mekf[1], R=design_param_mekf[0],
                      P0=design_param_mekf[2], eta0=design_param_mekf[3], design_param=design_param_mekf[5:])
        mekf_p.best_index = 93
    if run_bool[1]:
        mhe = MHE_integrator(A, B, BIS_param_nominal, ts=2, Q=design_param_mhe[1], R=design_param_mhe[0],
                             theta=design_param_mhe[2], N_MHE=design_param_mhe[3])
    if run_bool[2]:
        mekf_mhe = MEKF_MHE(A, B, BIS_param_nominal, mekf_param=design_param_mekf,
                            mhe_param=design_param_mhe, switch_time=switch_time, ts=2)

    # run the simulation
    x_mekf = np.zeros((8, len(BIS)))
    x_mhe = np.zeros((8, len(BIS)))
    x_mm = np.zeros((8, len(BIS)))
    bis_estimated_mekf = np.zeros((len(BIS), 1))
    bis_estimated_mhe = np.zeros((len(BIS), 1))
    bis_estimated_mm = np.zeros((len(BIS), 1))
    best_index_mekf = np.zeros((len(BIS), 1))
    best_index_mm = np.zeros((len(BIS), 1))
    estimated_parameters_mekf = np.zeros((len(BIS), 4))
    estimated_parameters_mhe = np.zeros((len(BIS), 4))
    estimated_parameters_mm = np.zeros((len(BIS), 4))
    time_mekf = np.zeros(len(BIS))
    time_mhe = np.zeros(len(BIS))
    time_mm = np.zeros(len(BIS))

    for i, bis in enumerate(BIS):
        u = np.array([[U_propo[i]], [U_remi[i]]])
        # MEKF
        if run_bool[0]:
            start = time.perf_counter()
            x, bis_estimated_mekf[i] = mekf_p.one_step(u, bis)
            time_mekf[i] = time.perf_counter() - start
            x_mekf[:, i] = x[:8]
            estimated_parameters_mekf[i] = x[-4:]
        # MHE
        if run_bool[1]:
            u = np.array([U_propo[i], U_remi[i]])
            start = time.perf_counter()
            x, bis_estimated_mhe[i] = mhe.one_step(u, bis)
            time_mhe[i] = time.perf_counter() - start
            x_mhe[:, i] = x[:8]
            estimated_parameters_mhe[i] = x[-4:]
        # MEKF Petri + MHE
        if run_bool[2]:
            u = np.array([U_propo[i], U_remi[i]])
            start = time.perf_counter()
            x, bis_estimated_mm[i] = mekf_mhe.one_step(u, bis)
            time_mm[i] = time.perf_counter() - start
            x_mm[:, i] = x[:8]
            estimated_parameters_mm[i] = x[-4:]

    # save bis_esttimated, x, and parameters in csv
    if run_bool[0]:
        pd.DataFrame(bis_estimated_mekf).to_csv(f'./data_extended/mekf/bis_estimated_{patient_index}.csv')
        states = pd.DataFrame(
            columns=['Time'] + [f'x_propo_{i}' for i in range(1, 5)] + [f'x_remi_{i}' for i in range(1, 5)])
        states['Time'] = Time
        states[[f'x_propo_{i}' for i in range(1, 5)]] = x_mekf[:4].T
        states[[f'x_remi_{i}' for i in range(1, 5)]] = x_mekf[4:].T
        states.to_csv(f'./data_extended/mekf/x_{patient_index}.csv')

        param = pd.DataFrame(columns=['Time', 'd', 'c50p', 'c50r', 'gamma'])
        param['Time'] = Time
        param['d'] = estimated_parameters_mekf[:, 0]
        param['c50p'] = estimated_parameters_mekf[:, 1]
        param['c50r'] = estimated_parameters_mekf[:, 2]
        param['gamma'] = estimated_parameters_mekf[:, 3]
        param.to_csv(f'./data_extended/mekf/parameters_{patient_index}.csv')
    if run_bool[1]:
        pd.DataFrame(bis_estimated_mhe).to_csv(f'./data_extended/mhe/bis_estimated_{patient_index}.csv')
        states = pd.DataFrame(
            columns=['Time'] + [f'x_propo_{i}' for i in range(1, 5)] + [f'x_remi_{i}' for i in range(1, 5)])
        states['Time'] = Time
        states[[f'x_propo_{i}' for i in range(1, 5)]] = x_mhe[:4].T
        states[[f'x_remi_{i}' for i in range(1, 5)]] = x_mhe[4:].T
        states.to_csv(f'./data_extended/mhe/x_{patient_index}.csv')

        param = pd.DataFrame(columns=['Time', 'd', 'c50p', 'c50r', 'gamma'])
        param['Time'] = Time
        param['d'] = estimated_parameters_mhe[:, 0]
        param['c50p'] = estimated_parameters_mhe[:, 1]
        param['c50r'] = estimated_parameters_mhe[:, 2]
        param['gamma'] = estimated_parameters_mhe[:, 3]
        param.to_csv(f'./data_extended/mhe/parameters_{patient_index}.csv')
    if run_bool[2]:
        pd.DataFrame(bis_estimated_mm).to_csv(f'./data_extended/mm/bis_estimated_{patient_index}.csv')
        states = pd.DataFrame(
            columns=['Time'] + [f'x_propo_{i}' for i in range(1, 5)] + [f'x_remi_{i}' for i in range(1, 5)])
        states['Time'] = Time
        states[[f'x_propo_{i}' for i in range(1, 5)]] = x_mm[:4].T
        states[[f'x_remi_{i}' for i in range(1, 5)]] = x_mm[4:].T
        states.to_csv(f'./data_extended/mm/x_{patient_index}.csv')

        param = pd.DataFrame(columns=['Time', 'd', 'c50p', 'c50r', 'gamma'])
        param['Time'] = Time
        param['d'] = estimated_parameters_mm[:, 0]
        param['c50p'] = estimated_parameters_mm[:, 1]
        param['c50r'] = estimated_parameters_mm[:, 2]
        param['gamma'] = estimated_parameters_mm[:, 3]
        param.to_csv(f'./data_extended/mm/parameters_{patient_index}.csv')

    return np.mean(time_mekf), np.mean(time_mhe), np.mean(time_mm)

# %% define the design parameters


if __name__ == '__main__':
    import optuna
    study_petri = optuna.load_study(study_name="petri_final_3", storage="sqlite:///data/petri_2.db")

    # Petri parameters
    P0 = 1e-3 * np.eye(9)
    P0[8, 8] = 1e-6
    # np.diag([1, 1/550, 1/550, 1, 1, 1/50, 1/750, 1])
    Q_p = study_petri.best_params['Q'] * np.diag([0.1, 0.1, 0.05, 0.05, 1, 1, 10, 1, 0.00001])
    R_p = study_petri.best_params['R']

    lambda_1 = 1
    lambda_2 = study_petri.best_params['lambda_2']
    nu = 1.e-5
    epsilon = study_petri.best_params['epsilon']

    # definition of the grid
    BIS_param_nominal = pas.BIS_model().hill_param

    cv_c50p = 0.182
    cv_c50r = 0.888
    cv_gamma = 0.304
    # estimation of log normal standard deviation
    w_c50p = np.sqrt(np.log(1+cv_c50p**2))
    w_c50r = np.sqrt(np.log(1+cv_c50r**2))
    w_gamma = np.sqrt(np.log(1+cv_gamma**2))

    c50p_list = BIS_param_nominal[0]*np.exp([-2*w_c50p, -w_c50p, -0.5*w_c50p, 0, w_c50p])  # , -w_c50p
    c50r_list = BIS_param_nominal[1]*np.exp([-2*w_c50r, -w_c50r, -0.5*w_c50r, 0, w_c50r])
    gamma_list = BIS_param_nominal[2]*np.exp([-2*w_gamma, -w_gamma, -0.5*w_gamma, 0, w_gamma])  #
    # surrender list by Inf value
    c50p_list = np.concatenate(([-np.Inf], c50p_list, [np.Inf]))
    c50r_list = np.concatenate(([-np.Inf], c50r_list, [np.Inf]))
    gamma_list = np.concatenate(([-np.Inf], gamma_list, [np.Inf]))

    def get_probability(c50p_set: list, c50r_set: list, gamma_set: list, method: str) -> float:
        """_summary_

        Parameters
        ----------
        c50p_set : float
            c50p set.
        c50r_set : float
            c50r set.
        gamma_set : float
            gamma set.
        method : str
            method to compute the probability. can be 'proportional' or 'uniform'.

        Returns
        -------
        float
            propability of the parameter set.
        """
        if method == 'proportional':
            mean_c50p = 4.47
            mean_c50r = 19.3
            mean_gamma = 1.13
            # cv_c50p = 0.182
            # cv_c50r = 0.888
            # cv_gamma = 0.304
            w_c50p = np.sqrt(np.log(1+cv_c50p**2))
            w_c50r = np.sqrt(np.log(1+cv_c50r**2))
            w_gamma = np.sqrt(np.log(1+cv_gamma**2))
            c50p_normal = scipy.stats.lognorm(scale=mean_c50p, s=w_c50p)
            proba_c50p = c50p_normal.cdf(c50p_set[1]) - c50p_normal.cdf(c50p_set[0])

            c50r_normal = scipy.stats.lognorm(scale=mean_c50r, s=w_c50r)
            proba_c50r = c50r_normal.cdf(c50r_set[1]) - c50r_normal.cdf(c50r_set[0])

            gamma_normal = scipy.stats.lognorm(scale=mean_gamma, s=w_gamma)
            proba_gamma = gamma_normal.cdf(gamma_set[1]) - gamma_normal.cdf(gamma_set[0])

            proba = proba_c50p * proba_c50r * proba_gamma
        elif method == 'uniform':
            proba = 1/(len(c50p_list))/(len(c50r_list))/(len(gamma_list))
        return proba

    grid_vector_p = []
    eta0_p = []
    proba = []
    alpha = 10
    for i, c50p in enumerate(c50p_list[1:-1]):
        for j, c50r in enumerate(c50r_list[1:-1]):
            for k, gamma in enumerate(gamma_list[1:-1]):
                grid_vector_p.append([c50p, c50r, gamma]+BIS_param_nominal[3:])
                c50p_set = [np.mean([c50p_list[i], c50p]),
                            np.mean([c50p_list[i+2], c50p])]

                c50r_set = [np.mean([c50r_list[j], c50r]),
                            np.mean([c50r_list[j+2], c50r])]

                gamma_set = [np.mean([gamma_list[k], gamma]),
                             np.mean([gamma_list[k+2], gamma])]

                eta0_p.append(alpha*(1-get_probability(c50p_set, c50r_set, gamma_set, 'proportional')))
                # proba.append(get_probability(c50p_set, c50r_set, gamma_set, 'proportional'))

    design_parameters_p = [R_p, Q_p, P0, eta0_p, grid_vector_p, lambda_1, lambda_2, nu, epsilon]

    # ----------------------------------------------------------------------------------------------
    # MHE parameters
    # theta = [0.001, 800, 1e2, 0.015]*3
    # theta[4] = 0.0001
    study_mhe = optuna.load_study(study_name="mhe_final_2", storage="sqlite:///data/mhe.db")
    gamma = 0.105  # study_mhe.best_params['eta']
    theta = [gamma, 800, 100, 0.005]*4
    theta[4] = gamma/100
    theta[12] = gamma*10
    theta[13] = 300
    theta[15] = 0.05
    Q = np.diag([1, 550, 550, 1, 1, 50, 750, 1])
    R = study_mhe.best_params['R']
    N_mhe = study_mhe.best_params['N_mhe']
    MHE_param = [R, Q, theta, N_mhe]

    # MEKF_MHE parameters
    switch_time = 180
    design_parameters = [design_parameters_p, MHE_param, switch_time]

    # %% run the simulation using multiprocessing
    patient_index_list = np.arange(0, 100)
    start = time.perf_counter()
    mekf_mhe_mm = [False, False, True]
    function = partial(simulation, design_param=design_parameters, run_bool=mekf_mhe_mm)
    with mp.Pool(mp.cpu_count()) as p:
        r = list(tqdm.tqdm(p.imap(function, patient_index_list), total=len(patient_index_list)))

    end = time.perf_counter()
    print(f'elapsed time: {end-start}')
    # print(f'average time per simulation: {(end-start)*mp.cpu_count/len(patient_index_list)}')
    time_mekf = []
    time_mhe = []
    time_mm = []
    for el in r:
        time_mekf.append(el[0])
        time_mhe.append(el[1])
        time_mm.append(el[2])
    print(f'mean time p: {np.mean(time_mekf)}')
    print(f'time mhe: {np.mean(time_mhe)}')
    print(f'time 2: {np.mean(time_mm)}')

    # %% plot the results
    path = './data_extended/mekf/'
    if False:
        from metrics_function import one_line
        patient_index_list = np.arange(5)
        for patient_index in patient_index_list:
            time_step = 2
            pred_time = 3*60
            stop_time_list = [i-1 for i in range(15, 15*60 - pred_time*time_step, 30)]
            r = one_line(patient_index, path, stop_time_list, pred_time, plot=True)

            bis_estimated = pd.read_csv(path + f'bis_estimated_{patient_index}.csv', index_col=0).values
            bis_measured = pd.read_csv(f'./data/simulations/simu_{patient_index}.csv', index_col=0)['BIS']
            parameters_estimated = pd.read_csv(path + f'parameters_{patient_index}.csv', index_col=0)
            true_parameters = pd.read_csv(f'./data/simulations/parameters.csv',
                                          index_col=0).iloc[patient_index].values[-6:]
            # get the effect site concentration
            x_propo = pd.read_csv(f'./data/simulations/simu_{patient_index}.csv', index_col=0)['x_propo_4']
            x_remi = pd.read_csv(f'./data/simulations/simu_{patient_index}.csv', index_col=0)['x_remi_4']
            x_estimated = pd.read_csv(path + f'x_{patient_index}.csv', index_col=0)

            plt.figure()
            plt.plot(bis_estimated, label='estimated')
            plt.plot(bis_measured, label='measured')
            plt.legend()
            plt.title(f'patient {patient_index}')
            plt.grid()
            plt.show()

            plt.figure()
            plt.subplot(3, 1, 1)
            for value in c50p_list[1:-1]:
                plt.plot(np.ones(len(parameters_estimated['c50p']))*value, 'k--', alpha=0.5)
            plt.plot(parameters_estimated['c50p'], label='estimated')
            plt.plot(np.ones(len(parameters_estimated['c50p']))*true_parameters[0], label='true')
            plt.legend()
            plt.title(f'patient {patient_index}')
            plt.grid()
            plt.ylabel('C50p')

            plt.subplot(3, 1, 2)
            for value in c50r_list[1:-1]:
                plt.plot(np.ones(len(parameters_estimated['c50r']))*value, 'k--', alpha=0.5)
            plt.plot(parameters_estimated['c50r'], label='estimated')
            plt.plot(np.ones(len(parameters_estimated['c50r']))*true_parameters[1], label='true')
            plt.legend()
            plt.grid()
            plt.ylabel('C50r')

            plt.subplot(3, 1, 3)
            for value in gamma_list[1:-1]:
                plt.plot(np.ones(len(parameters_estimated['gamma']))*value, 'k--', alpha=0.5)
            plt.plot(parameters_estimated['gamma'], label='estimated')
            plt.plot(np.ones(len(parameters_estimated['gamma']))*true_parameters[2], label='true')
            plt.legend()
            plt.grid()
            plt.ylabel('gamma')
            plt.show()

            # plot the effect site concentration
            plt.figure()
            plt.plot(x_propo, 'r', label='propo')
            plt.plot(x_remi, 'b', label='remi')
            plt.plot(x_estimated['x_propo_4'], 'r--', label='propo estimated')
            plt.plot(x_estimated['x_remi_4'], 'b--', label='remi estimated')
            plt.legend()
            plt.title(f'patient {patient_index}')
            plt.grid()
            plt.show()

    # %%
