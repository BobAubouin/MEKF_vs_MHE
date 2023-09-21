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

from estimators import MEKF_Petri, MEKF_Narendra, MHE
# %% define simulation function

Patient_data = pd.read_csv(r"./scripts/info_clinic_vitalDB.csv")
caseid_list = list(np.loadtxt('./scripts/caseid_list.txt', dtype=int))
caseid_list.remove(104)
caseid_list.remove(859)

mean_c50p = 5
mean_c50r = 27
mean_gamma = 2.9


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
    Patient_info = [age, height, weight, sex]
    Patient_simu = pd.read_csv(f'./data/vital/case_{patient_index}.csv')
    BIS = Patient_simu['BIS/BIS'].to_numpy()
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

    mekf_p = MEKF_Petri(A, B, design_param_p[4], ts=2, Q=design_param_p[1], R=design_param_p[0],
                        P0=design_param_p[2], eta0=design_param_p[3], design_param=design_param_p[5:])
    mekf_n = MEKF_Narendra(A, B, design_param_n[4], ts=2, Q=design_param_n[1], R=design_param_n[0],
                           P0=design_param_n[2], eta0=design_param_n[3], design_param=design_param_n[5:])
    mhe = MHE(A, B, BIS_param_nominal, ts=2, Q=design_param_mhe[1], R=design_param_mhe[0],
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
        pd.DataFrame(x_p).to_csv(f'./data/vital/mekf_p/x_{patient_index}.csv')
        pd.DataFrame(estimated_parameters_p).to_csv(f'./data/vital/mekf_p/parameters_{patient_index}.csv')
    if run_bool[1]:
        pd.DataFrame(bis_estimated_n).to_csv(f'./data/vital/mekf_n/bis_estimated_{patient_index}.csv')
        pd.DataFrame(x_n).to_csv(f'./data/vital/mekf_n/x_{patient_index}.csv')
        pd.DataFrame(estimated_parameters_n).to_csv(f'./data/vital/mekf_n/parameters_{patient_index}.csv')
    if run_bool[2]:
        pd.DataFrame(bis_estimated_mhe).to_csv(f'./data/vital/mhe/bis_estimated_{patient_index}.csv')
        pd.DataFrame(x_mhe).to_csv(f'./data/vital/mhe/x_{patient_index}.csv')
        pd.DataFrame(estimated_parameters_mhe).to_csv(f'./data/vital/mhe/parameters_{patient_index}.csv')

    return time_max_p, time_max_n, time_max_mhe

# %% define the design parameters


# Petri parameters
P0 = 1e-3 * np.eye(8)
Q = 1e-2 * np.diag([0.01]*4+[1]*4)  # np.diag([1, 1/550, 1/550, 1, 1, 1/50, 1/750, 1])
R = 1e-2 * np.eye(1)

lambda_1 = 1
lambda_2 = 1
nu = 1e-4
epsilon = 0.3


# definition of the grid
BIS_param_nominal = [mean_c50r, mean_c50p, mean_gamma, 0, 97.4, 97.4]
cv_c50p = 0.182
cv_c50r = 0.2
cv_gamma = 0.304
# cv_c50p = 0.182
# cv_c50r = 0.888
# cv_gamma = 0.304
# estimation of log normal standard deviation
w_c50p = np.sqrt(np.log(1+cv_c50p**2))
w_c50r = np.sqrt(np.log(1+cv_c50r**2))
w_gamma = np.sqrt(np.log(1+cv_gamma**2))

c50p_list = BIS_param_nominal[0]*np.exp([-2*w_c50p, 0, w_c50p])  # , -w_c50p
c50r_list = BIS_param_nominal[1]*np.exp([-2*w_c50r, -1*w_c50r, 0, w_c50r, ])
gamma_list = BIS_param_nominal[2]*np.exp([-2*w_gamma, 0, w_gamma])  # , -w_gamma
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


grid_vector = []
eta0 = []
proba = []
alpha = 100
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
            # proba.append(get_probability(c50p_set, c50r_set, gamma_set, 'proportional'))


design_parameters_p = [R, Q, P0, eta0, grid_vector, lambda_1, lambda_2, nu, epsilon]
# MEKF_Narendra parameters

MMPC_param = [40, 1, 1, 0.05/2, 0.5]
alpha = MMPC_param[1]
beta = MMPC_param[2]
lambda_p = MMPC_param[3]
hysteresis = MMPC_param[4]
window_length = MMPC_param[0]

design_parameters_n = [R, Q, P0, eta0, grid_vector, alpha, beta, lambda_p, hysteresis, window_length]

# MHE parameters
theta = [1, 1, 300, 0.005]*3
theta[4] = 1/100
Q = np.diag([1, 550, 550, 1, 1, 50, 750, 1])
R = 10
N_mhe = 15
MHE_param = [R, Q, theta, N_mhe]

design_parameters = [design_parameters_p, design_parameters_n, MHE_param]

# %% run the simulation using multiprocessing
patient_index_list = np.arange(0, 100)
start = time.perf_counter()
ekf_N_ekf_P_MHE = [True, True, True]
function = partial(simulation, design_param=design_parameters, run_bool=ekf_N_ekf_P_MHE)
with mp.Pool(mp.cpu_count()-2) as p:
    r = list(tqdm.tqdm(p.imap(function, patient_index_list), total=len(patient_index_list)))

end = time.perf_counter()
print(f'elapsed time: {end-start}')
# print(f'average time per simulation: {(end-start)*mp.cpu_count/len(patient_index_list)}')
time_max_p = 0
time_max_n = 0
time_max_mhe = 0
for el in r:
    time_max_p = max(time_max_p, el[0])
    time_max_n = max(time_max_n, el[1])
    time_max_mhe = max(time_max_mhe, el[2])
print(f'time max p: {time_max_p}')
print(f'time max n: {time_max_n}')
print(f'time max mhe: {time_max_mhe}')


# %% plot the results
if False:
    patient_index_list = np.arange(5)
    for patient_index in patient_index_list:
        bis_estimated = pd.read_csv(f'./data/mekf/bis_estimated_{patient_index}.csv', index_col=0).values
        bis_measured = pd.read_csv(f'./data/simulations/simu_{patient_index}.csv', index_col=0)['BIS']
        parameters_estimated = pd.read_csv(f'./data/mekf/parameters_{patient_index}.csv', index_col=0).values
        true_parameters = pd.read_csv(f'./data/simulations/parameters.csv', index_col=0).iloc[patient_index].values[-6:]
        # get the effect site concentration
        x_propo = pd.read_csv(f'./data/simulations/simu_{patient_index}.csv', index_col=0)['x_propo_4']
        x_remi = pd.read_csv(f'./data/simulations/simu_{patient_index}.csv', index_col=0)['x_remi_4']
        x_estimated = pd.read_csv(f'./data/mekf/x_{patient_index}.csv', index_col=0).values

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
            plt.plot(np.ones(len(parameters_estimated[:, 0]))*value, 'k--', alpha=0.5)
        plt.plot(parameters_estimated[:, 0], label='estimated')
        plt.plot(np.ones(len(parameters_estimated[:, 0]))*true_parameters[0], label='true')
        plt.legend()
        plt.title(f'patient {patient_index}')
        plt.grid()
        plt.ylabel('C50p')

        plt.subplot(3, 1, 2)
        for value in c50r_list[1:-1]:
            plt.plot(np.ones(len(parameters_estimated[:, 0]))*value, 'k--', alpha=0.5)
        plt.plot(parameters_estimated[:, 1], label='estimated')
        plt.plot(np.ones(len(parameters_estimated[:, 1]))*true_parameters[1], label='true')
        plt.legend()
        plt.grid()
        plt.ylabel('C50r')

        plt.subplot(3, 1, 3)
        for value in gamma_list[1:-1]:
            plt.plot(np.ones(len(parameters_estimated[:, 0]))*value, 'k--', alpha=0.5)
        plt.plot(parameters_estimated[:, 2], label='estimated')
        plt.plot(np.ones(len(parameters_estimated[:, 2]))*true_parameters[2], label='true')
        plt.legend()
        plt.grid()
        plt.ylabel('gamma')
        plt.show()

        # plot the effect site concentration
        plt.figure()
        plt.plot(x_propo, 'r', label='propo')
        plt.plot(x_remi, 'b', label='remi')
        plt.plot(x_estimated[3, :], 'r--', label='propo estimated')
        plt.plot(x_estimated[7, :], 'b--', label='remi estimated')
        plt.legend()
        plt.title(f'patient {patient_index}')
        plt.grid()
        plt.show()

# %%
