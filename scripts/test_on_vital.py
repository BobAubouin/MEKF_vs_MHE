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
import optuna

from estimators import MEKF_Petri, MEKF_Narendra, MHE
# %% define simulation function

Patient_data = pd.read_csv(r"./scripts/info_clinic_vitalDB.csv")
caseid_list = list(np.loadtxt('./scripts/caseid_list.txt', dtype=int))
caseid_list.remove(104)
caseid_list.remove(859)
caseid_list.remove(29)

mean_c50p = 4.40
mean_c50r = 33.45
mean_gamma = 1.73


def simulation(patient_index: int, design_param: list, run_bool: list) -> tuple[list]:
    """_summary_

    Parameters
    ----------
    patient_index : int
        index of the patient to simulate
    design_param : list
        list of the design parameters [R, Q, P0, eta0, grid_vector, lambda_1, lambda_2, nu, epsilon]
    run_bool : list
        list of boolean to run the different estimators [mekf_p, mekf_n, mhe]

    Returns
    -------
    tuple[list]
        last parameter estimation.
    """
    # load the data
    patient_index = caseid_list[patient_index]
    age = int(Patient_data[Patient_data['caseid'] == str(patient_index)]['age'])
    height = float(Patient_data[Patient_data['caseid'] == str(patient_index)]['height'])
    weight = float(Patient_data[Patient_data['caseid'] == str(patient_index)]['weight'])
    sex = str(Patient_data[Patient_data['caseid'] == str(patient_index)]['sex'])
    if sex == "M":
        sex = 1  # Male (M)
    else:
        sex = 0  # Female (F)
    ts = 2
    Patient_info = [age, height, weight, sex]
    Patient_simu = pd.read_csv(f'./data/vital/case_{patient_index}.csv')
    BIS = Patient_simu['BIS/BIS'].to_numpy()
    Time = np.arange(0, len(BIS)*ts, ts)
    U_propo = Patient_simu['Orchestra/PPF20_RATE'].to_numpy() * 20/3600
    U_remi = Patient_simu['Orchestra/RFTN20_RATE'].to_numpy() * 20/3600

    # define the model
    model = 'Eleveld'
    simulator = pas.Patient(Patient_info, model_propo=model, model_remi=model, ts=2)
    A_p = simulator.propo_pk.continuous_sys.A[:4, :4]
    A_r = simulator.remi_pk.continuous_sys.A[:4, :4]
    B_p = simulator.propo_pk.continuous_sys.B[:4]
    B_r = simulator.remi_pk.continuous_sys.B[:4]
    BIS_param_nominal = simulator.hill_param
    BIS_param_nominal = [mean_c50p, mean_c50r, mean_gamma, 0, 97.4, 97.4]
    A = np.block([[A_p, np.zeros((4, 4))], [np.zeros((4, 4)), A_r]])
    B = np.block([[B_p, np.zeros((4, 1))], [np.zeros((4, 1)), B_r]])

    # init the estimators
    design_param_p = design_param[0]
    design_param_n = design_param[1]
    design_param_mhe = design_param[2]

    mekf_p = MEKF_Petri(A, B, design_param_p[4], ts=ts, Q=design_param_p[1], R=design_param_p[0],
                        P0=design_param_p[2], eta0=design_param_p[3], design_param=design_param_p[5:])
    mekf_p.best_index = 93
    mekf_n = MEKF_Narendra(A, B, design_param_n[4], ts=ts, Q=design_param_n[1], R=design_param_n[0],
                           P0=design_param_n[2], eta0=design_param_n[3], design_param=design_param_n[5:])
    mhe = MHE(A, B, BIS_param_nominal, ts=ts, Q=design_param_mhe[1], R=design_param_mhe[0],
              theta=design_param_mhe[2], N_MHE=design_param_mhe[3])

    # run the simulation
    x_p = np.zeros((8, len(BIS)))
    x_n = np.zeros((8, len(BIS)))
    x_mhe = np.zeros((8, len(BIS)))
    bis_estimated_p = np.zeros((len(BIS), 1))
    bis_estimated_n = np.zeros((len(BIS), 1))
    bis_estimated_mhe = np.zeros((len(BIS), 1))
    best_index_p = np.zeros((len(BIS), 1))
    best_index_n = np.zeros((len(BIS), 1))
    estimated_parameters_p = np.zeros((len(BIS), 3))
    estimated_parameters_n = np.zeros((len(BIS), 3))
    estimated_parameters_mhe = np.zeros((len(BIS), 3))
    time_max_n = 0
    time_max_p = 0
    time_max_mhe = 0
    for i, bis in enumerate(BIS):
        u = np.array([[U_propo[i]], [U_remi[i]]])
        # MEKF Petri
        if run_bool[0]:
            start = time.perf_counter()
            x_p[:, i], bis_estimated_p[i], best_index_p[i] = mekf_p.one_step(u, bis)
            time_max_p = max(time_max_p, time.perf_counter() - start)
            estimated_parameters_p[i] = mekf_p.EKF_list[int(best_index_p[i][0])].BIS_param[:3]
        # MEKF Narendra
        if run_bool[1]:
            start = time.perf_counter()
            x_n[:, i], bis_estimated_n[i], best_index_n[i] = mekf_n.one_step(u, bis)
            time_max_n = max(time_max_n, time.perf_counter() - start)
            estimated_parameters_n[i] = mekf_n.EKF_list[int(best_index_n[i][0])].BIS_param[:3]
        # MHE
        if run_bool[2]:
            u = np.array([U_propo[i], U_remi[i]])
            start = time.perf_counter()
            x, bis_estimated_mhe[i] = mhe.one_step(u, bis)
            time_max_mhe = max(time_max_mhe, time.perf_counter() - start)
            x_mhe[:, [i]] = x[:8]
            estimated_parameters_mhe[[i]] = x[8:11].T

    # save bis_esttimated, x, and parameters in csv
    if run_bool[0]:
        pd.DataFrame(bis_estimated_p).to_csv(f'./data/vital/mekf_p/bis_estimated_{patient_index}.csv')
        states = pd.DataFrame(
            columns=['Time'] + [f'x_propo_{i}' for i in range(1, 5)] + [f'x_remi_{i}' for i in range(1, 5)])
        states['Time'] = Time
        states[[f'x_propo_{i}' for i in range(1, 5)]] = x_p[:4].T
        states[[f'x_remi_{i}' for i in range(1, 5)]] = x_p[4:].T
        states.to_csv(f'./data/vital/mekf_p/x_{patient_index}.csv')

        param = pd.DataFrame(columns=['Time', 'c50p', 'c50r', 'gamma'])
        param['Time'] = Time
        param['c50p'] = estimated_parameters_p[:, 0]
        param['c50r'] = estimated_parameters_p[:, 1]
        param['gamma'] = estimated_parameters_p[:, 2]
        param.to_csv(f'./data/vital/mekf_p/parameters_{patient_index}.csv')
    if run_bool[1]:
        pd.DataFrame(bis_estimated_n).to_csv(f'./data/vital/mekf_n/bis_estimated_{patient_index}.csv')
        states = pd.DataFrame(
            columns=['Time'] + [f'x_propo_{i}' for i in range(1, 5)] + [f'x_remi_{i}' for i in range(1, 5)])
        states['Time'] = Time
        states[[f'x_propo_{i}' for i in range(1, 5)]] = x_n[:4].T
        states[[f'x_remi_{i}' for i in range(1, 5)]] = x_n[4:].T
        states.to_csv(f'./data/vital/mekf_n/x_{patient_index}.csv')

        param = pd.DataFrame(columns=['Time', 'c50p', 'c50r', 'gamma'])
        param['Time'] = Time
        param['c50p'] = estimated_parameters_n[:, 0]
        param['c50r'] = estimated_parameters_n[:, 1]
        param['gamma'] = estimated_parameters_n[:, 2]
        param.to_csv(f'./data/vital/mekf_n/parameters_{patient_index}.csv')
    if run_bool[2]:
        pd.DataFrame(bis_estimated_mhe).to_csv(f'./data/vital/mhe/bis_estimated_{patient_index}.csv')
        states = pd.DataFrame(
            columns=['Time'] + [f'x_propo_{i}' for i in range(1, 5)] + [f'x_remi_{i}' for i in range(1, 5)])
        states['Time'] = Time
        states[[f'x_propo_{i}' for i in range(1, 5)]] = x_mhe[:4].T
        states[[f'x_remi_{i}' for i in range(1, 5)]] = x_mhe[4:].T
        states.to_csv(f'./data/vital/mhe/x_{patient_index}.csv')

        param = pd.DataFrame(columns=['Time', 'c50p', 'c50r', 'gamma'])
        param['Time'] = Time
        param['c50p'] = estimated_parameters_mhe[:, 0]
        param['c50r'] = estimated_parameters_mhe[:, 1]
        param['gamma'] = estimated_parameters_mhe[:, 2]
        param.to_csv(f'./data/vital/mhe/parameters_{patient_index}.csv')

    return time_max_p, time_max_n, time_max_mhe

# %% define the design parameters


if __name__ == '__main__':
    study_petri = optuna.load_study(study_name="petri_final_3", storage="sqlite:///data/petri_2.db")

    # Petri parameters
    P0 = 1e-3 * np.eye(8)
    # np.diag([1, 1/550, 1/550, 1, 1, 1/50, 1/750, 1])
    Q_p = study_petri.best_params['Q'] * np.diag([0.1, 0.1, 0.05, 0.05, 1, 1, 10, 1])
    R_p = study_petri.best_params['R']

    lambda_1 = 1
    lambda_2 = study_petri.best_params['lambda_2']
    nu = 1.e-5
    epsilon = study_petri.best_params['epsilon']

    # definition of the grid
    BIS_param_nominal = [mean_c50p, mean_c50r, mean_gamma, 0, 97.4, 97.4]

    cv_c50p = 0.36
    cv_c50r = 0.11
    cv_gamma = 0.60
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
    # MEKF_Narendra parameters
    c50p_list = BIS_param_nominal[0]*np.exp([-2*w_c50p, -w_c50p, 0, w_c50p])  # , -w_c50p
    c50r_list = BIS_param_nominal[1]*np.exp([-2*w_c50r, -1*w_c50r, 0, w_c50r, ])
    gamma_list = BIS_param_nominal[2]*np.exp([-2*w_gamma, -w_gamma, 0, w_gamma])  # , -w_gamma
    # surrender list by Inf value
    c50p_list = np.concatenate(([-np.Inf], c50p_list, [np.Inf]))
    c50r_list = np.concatenate(([-np.Inf], c50r_list, [np.Inf]))
    gamma_list = np.concatenate(([-np.Inf], gamma_list, [np.Inf]))

    grid_vector_n = []
    eta0_n = []
    proba = []
    alpha = 10
    for i, c50p in enumerate(c50p_list[1:-1]):
        for j, c50r in enumerate(c50r_list[1:-1]):
            for k, gamma in enumerate(gamma_list[1:-1]):
                grid_vector_n.append([c50p, c50r, gamma]+BIS_param_nominal[3:])
                c50p_set = [np.mean([c50p_list[i], c50p]),
                            np.mean([c50p_list[i+2], c50p])]

                c50r_set = [np.mean([c50r_list[j], c50r]),
                            np.mean([c50r_list[j+2], c50r])]

                gamma_set = [np.mean([gamma_list[k], gamma]),
                             np.mean([gamma_list[k+2], gamma])]

                eta0_n.append(alpha*(1-get_probability(c50p_set, c50r_set, gamma_set, 'proportional')))
                # proba.append(get_probability(c50p_set, c50r_set, gamma_set, 'proportional'))

    # grid_narendra = pd.read_csv('./data/grid_search_narendra.csv', index_col=0)
    # grid_narendra.sort_values('objective_function', inplace=True)
    # best_index = grid_narendra.index[0]

    Q_n = 1e0 * np.diag([0.01]*4+[1]*4)  # np.diag([1, 1/550, 1/550, 1, 1, 1/50, 1/750, 1])
    R_n = 1  # float(grid_narendra.loc[best_index, 'R']) * np.eye(1)

    alpha = 0
    beta = 1
    lambda_p = 1  # float(grid_narendra.loc[best_index, 'lambda'])
    hysteresis = 1  # float(grid_narendra.loc[best_index, 'epsilon'])
    window_length = 1  # int(grid_narendra.loc[best_index, 'N'])

    design_parameters_n = [R_n, Q_n, P0, eta0_n, grid_vector_n, alpha, beta, lambda_p, hysteresis, window_length]

    # MHE parameters
    # theta = [0.001, 800, 1e2, 0.015]*3
    # theta[4] = 0.0001
    study_mhe = optuna.load_study(study_name="mhe_final_2", storage="sqlite:///data/mhe.db")
    gamma = study_mhe.best_params['eta']
    theta = [gamma, 1, 300, 0.005]*3
    theta[4] = gamma
    theta[8] = gamma*10
    Q = np.diag([1, 550, 550, 1, 1, 50, 750, 1])
    R = study_mhe.best_params['R']
    N_mhe = study_mhe.best_params['N_mhe']
    MHE_param = [R, Q, theta, N_mhe]

    design_parameters = [design_parameters_p, design_parameters_n, MHE_param]

    # %% run the simulation using multiprocessing
    patient_index_list = np.arange(0, len(caseid_list))  # len(caseid_list))
    start = time.perf_counter()
    ekf_P_ekf_N_MHE = [True, False, False]
    function = partial(simulation, design_param=design_parameters, run_bool=ekf_P_ekf_N_MHE)
    with mp.Pool(mp.cpu_count()) as p:
        r = list(tqdm.tqdm(p.imap(function, patient_index_list), total=len(patient_index_list)))

    end = time.perf_counter()
    print(f'elapsed time: {end-start}')
    # print(f'average time per simulation: {(end-start)*mp.cpu_count/len(patient_index_list)}')
    time_p = []
    time_n = []
    time_mhe = []
    for el in r:
        time_p.append(el[0])
        time_n.append(el[1])
        time_mhe.append(el[2])
    print(f'time max p: {np.mean(time_p)}')
    print(f'time max n: {np.mean(time_n)}')
    print(f'time max mhe: {np.mean(time_mhe)}')


# %% plot the results
    path = './data/vital/mhe/'
    if False:
        from metrics_function import one_line
        patient_index_list = caseid_list[:5]
        for patient_index in patient_index_list:
            # time_step = 2
            # pred_time = 3*60
            # stop_time_list = [i-1 for i in range(15, 15*60 - pred_time*time_step, 30)]
            # r = one_line(patient_index, path, stop_time_list, pred_time, plot=True)

            bis_estimated = pd.read_csv(path + f'bis_estimated_{patient_index}.csv', index_col=0).values
            Patient_simu = pd.read_csv(f'./data/vital/case_{patient_index}.csv')
            bis_measured = Patient_simu['BIS/BIS'].to_numpy()
            parameters_estimated = pd.read_csv(path + f'parameters_{patient_index}.csv', index_col=0)
            # get the effect site concentration
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
            plt.legend()
            plt.title(f'patient {patient_index}')
            plt.grid()
            plt.ylabel('C50p')

            plt.subplot(3, 1, 2)
            for value in c50r_list[1:-1]:
                plt.plot(np.ones(len(parameters_estimated['c50r']))*value, 'k--', alpha=0.5)
            plt.plot(parameters_estimated['c50r'], label='estimated')
            plt.grid()
            plt.ylabel('C50r')

            plt.subplot(3, 1, 3)
            for value in gamma_list[1:-1]:
                plt.plot(np.ones(len(parameters_estimated['gamma']))*value, 'k--', alpha=0.5)
            plt.plot(parameters_estimated['gamma'], label='estimated')
            plt.grid()
            plt.ylabel('gamma')
            plt.show()

# %%
