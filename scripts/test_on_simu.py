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

from estimators import MEKF_Petri, MEKF_Narendra, MHE, MHE_standard
# %% define simulation function


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
    design_param_p = design_param[0]
    design_param_n = design_param[1]
    design_param_mhe = design_param[2]
    design_param_mhe_std = design_param[3]

    if run_bool[0]:
        mekf_p = MEKF_Petri(A, B, design_param_p[4], ts=2, Q=design_param_p[1], R=design_param_p[0],
                            P0=design_param_p[2], eta0=design_param_p[3], design_param=design_param_p[5:])
        mekf_p.best_index = 93
    if run_bool[1]:
        mekf_n = MEKF_Narendra(A, B, design_param_n[4], ts=2, Q=design_param_n[1], R=design_param_n[0],
                               P0=design_param_n[2], eta0=design_param_n[3], design_param=design_param_n[5:])
    if run_bool[2]:
        mhe = MHE(A, B, BIS_param_nominal, ts=2, Q=design_param_mhe[1], R=design_param_mhe[0],
                  theta=design_param_mhe[2], N_MHE=design_param_mhe[3])

    if run_bool[3]:
        mekf_2 = MEKF_Petri(A, B, design_param_p[4], ts=2, Q=design_param_p[1], R=design_param_p[0],
                            P0=design_param_p[2], eta0=design_param_p[3], design_param=design_param_p[5:])
        mekf_2.best_index = 93
        mhe_2 = MHE(A, B, BIS_param_nominal, ts=2, Q=design_param_mhe[1], R=design_param_mhe[0],
                    theta=design_param_mhe[2], N_MHE=design_param_mhe[3])
    if run_bool[4]:
        mhe_std = MHE_standard(A, B, BIS_param_nominal, ts=2, Q=design_param_mhe_std[1], R=design_param_mhe_std[0],
                               theta=design_param_mhe_std[2], N_MHE=design_param_mhe_std[3], P=design_param_mhe_std[4])

    # run the simulation
    x_p = np.zeros((8, len(BIS)))
    x_n = np.zeros((8, len(BIS)))
    x_mhe = np.zeros((8, len(BIS)))
    x_2 = np.zeros((8, len(BIS)))
    x_mhe_std = np.zeros((8, len(BIS)))
    bis_estimated_p = np.zeros((len(BIS), 1))
    bis_estimated_n = np.zeros((len(BIS), 1))
    bis_estimated_mhe = np.zeros((len(BIS), 1))
    bis_estimated_2 = np.zeros((len(BIS), 1))
    bis_estimated_mhe_std = np.zeros((len(BIS), 1))
    best_index_p = np.zeros((len(BIS), 1))
    best_index_n = np.zeros((len(BIS), 1))
    best_index_2 = np.zeros((len(BIS), 1))
    best_index_mhe_std = np.zeros((len(BIS), 1))
    estimated_parameters_p = np.zeros((len(BIS), 3))
    estimated_parameters_n = np.zeros((len(BIS), 3))
    estimated_parameters_mhe = np.zeros((len(BIS), 3))
    estimated_parameters_2 = np.zeros((len(BIS), 3))
    estimated_parameters_mhe_std = np.zeros((len(BIS), 3))
    time_n = np.zeros(len(BIS))
    time_p = np.zeros(len(BIS))
    time_mhe = np.zeros(len(BIS))
    time_2 = np.zeros(len(BIS))
    time_mhe_std = np.zeros(len(BIS))

    for i, bis in enumerate(BIS):
        u = np.array([[U_propo[i]], [U_remi[i]]])
        # MEKF Petri
        if run_bool[0]:
            start = time.perf_counter()
            x_p[:, i], bis_estimated_p[i], best_index_p[i] = mekf_p.one_step(u, bis)
            time_p[i] = time.perf_counter() - start
            estimated_parameters_p[i] = mekf_p.EKF_list[int(best_index_p[i][0])].BIS_param[:3]
        # MEKF Narendra
        if run_bool[1]:
            start = time.perf_counter()
            x_n[:, i], bis_estimated_n[i], best_index_n[i] = mekf_n.one_step(u, bis)
            time_n[i] = time.perf_counter() - start
            estimated_parameters_n[i] = mekf_n.EKF_list[int(best_index_n[i][0])].BIS_param[:3]
        # MHE
        if run_bool[2]:
            u = np.array([U_propo[i], U_remi[i]])
            start = time.perf_counter()
            x, bis_estimated_mhe[i] = mhe.one_step(u, bis)
            time_mhe[i] = time.perf_counter() - start
            x_mhe[:, [i]] = x[:8]
            estimated_parameters_mhe[[i]] = x[8:11].T
        # MEKF Petri + MHE
        if run_bool[3]:
            start = time.perf_counter()
            if i < 120//2:
                x_2[:, i], bis_estimated_2[i], best_index_2[i] = mekf_2.one_step(u, bis)
                estimated_parameters_2[i] = mekf_2.EKF_list[int(best_index_2[i][0])].BIS_param[:3]
            elif i == 120//2:
                temp = np.vstack(
                    (x_2[:, i-mhe_2.N_mhe:i], np.repeat(estimated_parameters_2[[i-1], :].T, mhe_2.N_mhe, axis=1)))
                mhe_2.x_pred = temp.reshape(mhe_2.nb_states*mhe_2.N_mhe, order='F')
                mhe_2.y = list(BIS[i-mhe_2.N_mhe:i])
                u_temp = np.array([U_propo[i-1], U_remi[i-1]])
                for j in range(1, mhe_2.N_mhe):
                    u_temp = np.hstack((np.array([U_propo[i-1-j], U_remi[i-1-j]]), u_temp))
                mhe_2.u = u_temp

                u = np.array([U_propo[i], U_remi[i]])
                x, bis_estimated_2[i] = mhe_2.one_step(u, bis)
                x_2[:, [i]] = x[:8]
                estimated_parameters_2[[i]] = x[8:11].T

            else:
                u = np.array([U_propo[i], U_remi[i]])
                x, bis_estimated_2[i] = mhe_2.one_step(u, bis)
                x_2[:, [i]] = x[:8]
                estimated_parameters_2[[i]] = x[8:11].T
            time_2[i] = time.perf_counter() - start
        # MHE standard
        if run_bool[4]:
            u = np.array([U_propo[i], U_remi[i]])
            start = time.perf_counter()
            x, bis_estimated_mhe_std[i] = mhe_std.one_step(u, bis)
            time_mhe_std[i] = time.perf_counter() - start
            x_mhe_std[:, [i]] = x[:8]
            estimated_parameters_mhe_std[[i]] = x[8:11].T

    # save bis_esttimated, x, and parameters in csv
    if run_bool[0]:
        pd.DataFrame(bis_estimated_p).to_csv(f'./data/mekf_p/bis_estimated_{patient_index}.csv')
        states = pd.DataFrame(
            columns=['Time'] + [f'x_propo_{i}' for i in range(1, 5)] + [f'x_remi_{i}' for i in range(1, 5)])
        states['Time'] = Time
        states[[f'x_propo_{i}' for i in range(1, 5)]] = x_p[:4].T
        states[[f'x_remi_{i}' for i in range(1, 5)]] = x_p[4:].T
        states.to_csv(f'./data/mekf_p/x_{patient_index}.csv')

        param = pd.DataFrame(columns=['Time', 'c50p', 'c50r', 'gamma'])
        param['Time'] = Time
        param['c50p'] = estimated_parameters_p[:, 0]
        param['c50r'] = estimated_parameters_p[:, 1]
        param['gamma'] = estimated_parameters_p[:, 2]
        param.to_csv(f'./data/mekf_p/parameters_{patient_index}.csv')
    if run_bool[1]:
        pd.DataFrame(bis_estimated_n).to_csv(f'./data/mekf_n/bis_estimated_{patient_index}.csv')
        states = pd.DataFrame(
            columns=['Time'] + [f'x_propo_{i}' for i in range(1, 5)] + [f'x_remi_{i}' for i in range(1, 5)])
        states['Time'] = Time
        states[[f'x_propo_{i}' for i in range(1, 5)]] = x_n[:4].T
        states[[f'x_remi_{i}' for i in range(1, 5)]] = x_n[4:].T
        states.to_csv(f'./data/mekf_n/x_{patient_index}.csv')

        param = pd.DataFrame(columns=['Time', 'c50p', 'c50r', 'gamma'])
        param['Time'] = Time
        param['c50p'] = estimated_parameters_n[:, 0]
        param['c50r'] = estimated_parameters_n[:, 1]
        param['gamma'] = estimated_parameters_n[:, 2]
        param.to_csv(f'./data/mekf_n/parameters_{patient_index}.csv')
    if run_bool[2]:
        pd.DataFrame(bis_estimated_mhe).to_csv(f'./data/mhe/bis_estimated_{patient_index}.csv')
        states = pd.DataFrame(
            columns=['Time'] + [f'x_propo_{i}' for i in range(1, 5)] + [f'x_remi_{i}' for i in range(1, 5)])
        states['Time'] = Time
        states[[f'x_propo_{i}' for i in range(1, 5)]] = x_mhe[:4].T
        states[[f'x_remi_{i}' for i in range(1, 5)]] = x_mhe[4:].T
        states.to_csv(f'./data/mhe/x_{patient_index}.csv')

        param = pd.DataFrame(columns=['Time', 'c50p', 'c50r', 'gamma'])
        param['Time'] = Time
        param['c50p'] = estimated_parameters_mhe[:, 0]
        param['c50r'] = estimated_parameters_mhe[:, 1]
        param['gamma'] = estimated_parameters_mhe[:, 2]
        param.to_csv(f'./data/mhe/parameters_{patient_index}.csv')
    if run_bool[3]:
        pd.DataFrame(bis_estimated_2).to_csv(f'./data/plus/bis_estimated_{patient_index}.csv')
        states = pd.DataFrame(
            columns=['Time'] + [f'x_propo_{i}' for i in range(1, 5)] + [f'x_remi_{i}' for i in range(1, 5)])
        states['Time'] = Time
        states[[f'x_propo_{i}' for i in range(1, 5)]] = x_2[:4].T
        states[[f'x_remi_{i}' for i in range(1, 5)]] = x_2[4:].T
        states.to_csv(f'./data/plus/x_{patient_index}.csv')

        param = pd.DataFrame(columns=['Time', 'c50p', 'c50r', 'gamma'])
        param['Time'] = Time
        param['c50p'] = estimated_parameters_2[:, 0]
        param['c50r'] = estimated_parameters_2[:, 1]
        param['gamma'] = estimated_parameters_2[:, 2]
        param.to_csv(f'./data/plus/parameters_{patient_index}.csv')
    if run_bool[4]:
        pd.DataFrame(bis_estimated_mhe_std).to_csv(f'./data/mhe_std/bis_estimated_{patient_index}.csv')
        states = pd.DataFrame(
            columns=['Time'] + [f'x_propo_{i}' for i in range(1, 5)] + [f'x_remi_{i}' for i in range(1, 5)])
        states['Time'] = Time
        states[[f'x_propo_{i}' for i in range(1, 5)]] = x_mhe_std[:4].T
        states[[f'x_remi_{i}' for i in range(1, 5)]] = x_mhe_std[4:].T
        states.to_csv(f'./data/mhe_std/x_{patient_index}.csv')

        param = pd.DataFrame(columns=['Time', 'c50p', 'c50r', 'gamma'])
        param['Time'] = Time
        param['c50p'] = estimated_parameters_mhe_std[:, 0]
        param['c50r'] = estimated_parameters_mhe_std[:, 1]
        param['gamma'] = estimated_parameters_mhe_std[:, 2]
        param.to_csv(f'./data/mhe_std/parameters_{patient_index}.csv')
    return np.mean(time_p), np.mean(time_n), np.mean(time_mhe), np.mean(time_2), np.mean(time_mhe_std)


BIS_param_nominal = pas.BIS_model().hill_param

# Qp = np.load('data/cov_propo.npy')
# Qr = np.load('data/cov_remi.npy')
# Q = np.block([[Qp, np.zeros((4, 4))], [np.zeros((4, 4)), Qr]])
# R = np.load('data/R.npy')


mean_c50p = 4.47
mean_c50r = 19.3
mean_gamma = 1.13
cv_c50p = 0.182
cv_c50r = 0.888
cv_gamma = 0.304
# estimation of log normal standard deviation
w_c50p = np.sqrt(np.log(1+cv_c50p**2))
w_c50r = np.sqrt(np.log(1+cv_c50r**2))
w_gamma = np.sqrt(np.log(1+cv_gamma**2))

c50p_normal = scipy.stats.lognorm(scale=mean_c50p, s=w_c50p)
c50r_normal = scipy.stats.lognorm(scale=mean_c50r, s=w_c50r)
gamma_normal = scipy.stats.lognorm(scale=mean_gamma, s=w_gamma)

nb_points = 5
points = np.linspace(0, 1, nb_points+1)
points = [np.mean([points[i], points[i+1]]) for i in range(nb_points)]

c50p_list = c50p_normal.ppf(points)

nb_points = 6
points = np.linspace(0, 1, nb_points+1)
points = [np.mean([points[i], points[i+1]]) for i in range(nb_points)]

c50r_list = c50r_normal.ppf(points)
gamma_list = gamma_normal.ppf(points)

# c50p_list = BIS_param_nominal[0]*np.exp([-2.2*w_c50p, -w_c50p, -0.4*w_c50p, 0, w_c50p])  # , -w_c50p
# c50r_list = BIS_param_nominal[1]*np.exp([-2.2*w_c50r, -w_c50r, -0.4*w_c50r, 0, 0.6*w_c50r, w_c50r])
# gamma_list = BIS_param_nominal[2]*np.exp([-2.2*w_gamma, -w_gamma, -0.4*w_gamma, 0, 0.8*w_gamma, 1.5*w_gamma])  #
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

        proba_c50p = c50p_normal.cdf(c50p_set[1]) - c50p_normal.cdf(c50p_set[0])

        proba_c50r = c50r_normal.cdf(c50r_set[1]) - c50r_normal.cdf(c50r_set[0])

        proba_gamma = gamma_normal.cdf(gamma_set[1]) - gamma_normal.cdf(gamma_set[0])

        proba = proba_c50p * proba_c50r * proba_gamma
    elif method == 'uniform':
        proba = 1/(len(c50p_list))/(len(c50r_list))/(len(gamma_list))
    return proba


def init_proba(alpha):
    grid_vector = []
    eta0 = []
    for i, c50p in enumerate(c50p_list[1:-1]):
        for j, c50r in enumerate(c50r_list[1:-1]):
            for k, gamma in enumerate(gamma_list[1:-1]):
                grid_vector.append([c50p, c50r, gamma]+BIS_param_nominal[3:])
                c50p_set = [np.mean([c50p_list[i], c50p]),
                            np.mean([c50p_list[i+2], c50p])]

                c50r_set = [np.mean([c50r_list[j], c50r]),
                            np.mean([c50r_list[j+2], c50r])]

                gamma_set = [np.mean([gamma_list[k], gamma]),
                             np.mean([gamma_list[k+2], gamma])]

                eta0.append(alpha*(1-get_probability(c50p_set, c50r_set, gamma_set, 'proportional')))
    i_nom = np.argmin(np.sum(np.abs(np.array(grid_vector)-np.array(BIS_param_nominal))), axis=0)
    eta0[i_nom] = alpha
    return grid_vector, eta0

# %% define the design parameters


if __name__ == '__main__':
    import optuna
    study_petri = optuna.load_study(study_name="petri_final", storage="sqlite:///data/mekf.db")
    # print(study_petri.best_params)
    P0 = 1e-3 * np.eye(8)
    Q = study_petri.best_params['Q']
    Q_mat = Q * np.diag([0.1, 0.1, 0.05, 0.05, 1, 1, 10, 1])  # np.diag([1, 1/550, 1/550, 1, 1, 1/50, 1/750, 1])
    R = study_petri.best_params['R']
    alpha = study_petri.best_params['alpha']
    grid_vector, eta0 = init_proba(alpha)
    lambda_1 = 1
    lambda_2 = study_petri.best_params['lambda_2']
    nu = 1.e-5
    epsilon = study_petri.best_params['epsilon']
    design_parameters_p = [R, Q, P0, eta0, grid_vector, lambda_1, lambda_2, nu, epsilon]

    # ------------------------------------------------------------------------
    # MHE parameters
    # theta = [0.001, 800, 1e2, 0.015]*3
    # theta[4] = 0.0001
    # study_mhe = optuna.load_study(study_name="mhe_final_2", storage="sqlite:///data/mhe.db")
    # gamma = study_mhe.best_params['eta']
    # theta = [gamma, 1, 300, 0.005]*3
    # theta[4] = gamma
    # theta[8] = gamma*10
    # Q = np.diag([1, 550, 550, 1, 1, 50, 750, 1])
    # R = study_mhe.best_params['R']
    # N_mhe = study_mhe.best_params['N_mhe']
    # MHE_param = [R, Q, theta, N_mhe]
    MHE_param = None

    # MHE standard parameters
    study_mhe_std = optuna.load_study(study_name="mhe_std_final", storage="sqlite:///data/mhe.db")

    R = study_mhe_std.best_params['R']
    q = study_mhe_std.best_params['q']
    Q_std = np.diag([1, 550, 550, 1, 1, 50, 750, 1]+[1e3]*3)*q
    # R = 1/R_p
    # Q_std = np.linalg.inv(Q_p)
    # p = study_mhe_std.best_params['p']
    eta = study_mhe_std.best_params['eta']
    N_mhe_std = study_mhe_std.best_params['N_mhe']
    P = np.diag([1, 550, 550, 1, 1, 50, 750, 1])
    theta = [100, 0, 300, 0.005]*3
    theta[0] = eta
    theta[4] = theta[0]
    theta[8] = theta[0]*100
    MHE_std = [R, Q_std, theta, N_mhe_std, P]

    design_parameters = [design_parameters_p, None, MHE_param, MHE_std]

    # %% run the simulation using multiprocessing
    patient_index_list = np.arange(0, 500)
    # np.random.seed(2)
    # patient_index_list = np.random.randint(0, 500, 16)
    # patient_index_list = patient_index_list[[4]]
    start = time.perf_counter()
    ekf_P_ekf_N_MHE = [True, False, False, False, True]
    function = partial(simulation, design_param=design_parameters, run_bool=ekf_P_ekf_N_MHE)

    with mp.Pool(processes=mp.cpu_count()) as pool:
        r = list(tqdm.tqdm(pool.imap(function, patient_index_list), total=len(patient_index_list)))

    end = time.perf_counter()
    print(f'elapsed time: {end-start}')
    # print(f'average time per simulation: {(end-start)*mp.cpu_count/len(patient_index_list)}')
    time_p = []
    time_n = []
    time_mhe = []
    time_2 = []
    time_mhe_std = []
    for el in r:
        time_p.append(el[0])
        time_n.append(el[1])
        time_mhe.append(el[2])
        time_2.append(el[3])
        time_mhe_std.append(el[4])
    print(f'mean time p: {np.mean(time_p)}')
    print(f'mean time n: {np.mean(time_n)}')
    print(f'time mhe: {np.mean(time_mhe)}')
    print(f'time 2: {np.mean(time_2)}')
    print(f'time mhe std: {np.mean(time_mhe_std)}')

    # %% plot the results
    path = './data/mekf_p/'
    path_simu = './data/simulations/'
    if False:
        from metrics_function import one_line
        np.random.seed(2)
        patient_index_list = np.random.randint(0, 500, 16)
        patient_index_list = patient_index_list[[2]]
        for patient_index in patient_index_list:
            time_step = 2
            pred_time = 3*60
            stop_time_list = [i-1 for i in range(15, 15*60 - pred_time*time_step, 30)]

            r = one_line(patient_index, path, stop_time_list, pred_time, plot=True)

            print(f"patient {patient_index}: {np.sum(r.values)}")

            bis_estimated = pd.read_csv(path + f'bis_estimated_{patient_index}.csv', index_col=0).values
            bis_measured = pd.read_csv(f'{path_simu}simu_{patient_index}.csv', index_col=0)['BIS']
            parameters_estimated = pd.read_csv(path + f'parameters_{patient_index}.csv', index_col=0)
            true_parameters = pd.read_csv(f'{path_simu}parameters.csv',
                                          index_col=0).iloc[patient_index].values[-6:]
            # get the effect site concentration
            x_propo = pd.read_csv(f'{path_simu}simu_{patient_index}.csv', index_col=0)['x_propo_4']
            x_remi = pd.read_csv(f'{path_simu}simu_{patient_index}.csv', index_col=0)['x_remi_4']
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
