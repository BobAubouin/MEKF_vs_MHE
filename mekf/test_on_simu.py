"""Test MEKF on the simulated data."""

# %% Import
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import python_anesthesia_simulator as pas
import scipy

from mekf import MEKF
# %% define simulation function


def simulation(patient_index: int, design_param: list) -> tuple[list]:
    """_summary_

    Parameters
    ----------
    patient_index : int
        index of the patient to simulate
    design_param : list
        list of the design parameters [R, Q, P0, eta0, grid_vector, lambda_1, lambda_2, nu, epsilon]

    Returns
    -------
    tuple[list]
        last parameter estimation.
    """

    # load the data
    Patient_info = pd.read_csv(
        './data/simulations/parameters.csv').iloc[patient_index][['age', 'height', 'weight', 'gender']].values
    Patient_simu = pd.read_csv(f'./data/simulations/simu_{patient_index}.csv')
    BIS = Patient_simu['BIS'].to_numpy()
    U_propo = Patient_simu['u_propo'].to_numpy()
    U_remi = Patient_simu['u_remi'].to_numpy()

    # define the model
    model = 'Eleveld'
    simluator = pas.Patient(Patient_info, model_propo=model, model_remi=model)
    A_p = simluator.propo_pk.continuous_sys.A[:4, :4]
    A_r = simluator.remi_pk.continuous_sys.A[:4, :4]
    B_p = simluator.propo_pk.continuous_sys.B[:4]
    B_r = simluator.remi_pk.continuous_sys.B[:4]

    A = np.block([[A_p, np.zeros((4, 4))], [np.zeros((4, 4)), A_r]])
    B = np.block([[B_p, np.zeros((4, 1))], [np.zeros((4, 1)), B_r]])

    mekf = MEKF(A, B, design_param[4], ts=1, Q=design_param[1], R=design_param[0],
                P0=design_param[2], eta0=design_param[3], design_param=design_param[5:])

    # run the simulation
    x = np.zeros((8, len(BIS)))
    bis_estimated = np.zeros((len(BIS), 1))
    best_index = np.zeros((len(BIS), 1))
    estimated_parameters = np.zeros((len(BIS), 3))
    for i, bis in enumerate(BIS):
        u = np.array([[U_propo[i]], [U_remi[i]]])
        x[:, [i]], bis_estimated[[i]], best_index[[i]] = mekf.one_step(u, bis)
        estimated_parameters[[i]] = mekf.EKF_list[best_index[[i]]].BIS_param[:3]

    # save bis_esttimated, x, and parameters in csv
    pd.DataFrame(bis_estimated).to_csv(f'./data/mekf/bis_estimated_{patient_index}.csv')
    pd.DataFrame(x).to_csv(f'./data/mekf/x_{patient_index}.csv')
    pd.DataFrame(estimated_parameters).to_csv(f'./data/mekf/parameters_{patient_index}.csv')

    return bis_estimated[-1]

# %% define the design parameters


P0 = 1e-1 * np.eye(8)
Q = 1e1 * np.eye(8)
R = 1e-1 * np.eye(1)

lambda_1 = 0.5
lambda_2 = 0.5
nu = 0.9
epsilon = 0.9


# definition of the grid
BIS_param_nominal = pas.BIS_model().hill_param

cv_c50p = 0.182
cv_c50r = 0.888
cv_gamma = 0.304
# estimation of log normal standard deviation
w_c50p = np.sqrt(np.log(1+cv_c50p**2))
w_c50r = np.sqrt(np.log(1+cv_c50r**2))
w_gamma = np.sqrt(np.log(1+cv_gamma**2))

c50p_list = BIS_param_nominal[0]*np.exp([-2*w_c50p, 0, w_c50p])
c50r_list = BIS_param_nominal[1]*np.exp([-3*w_c50r, -2*w_c50r, -1*w_c50r, 0, w_c50r])
gamma_list = BIS_param_nominal[2]*np.exp([-2*w_gamma, 0, w_gamma])
# surrender list by Inf value
c50p_list = np.concatenate(([-np.Inf], c50p_list, [np.Inf]))
c50r_list = np.concatenate(([-np.Inf], c50r_list, [np.Inf]))
gamma_list = np.concatenate(([-np.Inf], gamma_list, [np.Inf]))


def get_probability(c50p_set: list, c50r_set: list, gamma_set: list) -> float:
    """_summary_

    Parameters
    ----------
    c50p_set : float
        c50p set.
    c50r_set : float
        c50r set.
    gamma_set : float
        gamma set.

    Returns
    -------
    float
        propability of the parameter set.
    """
    mean_c50p = 4.47
    mean_c50r = 19.3
    mean_gamma = 1.13
    cv_c50p = 0.182
    cv_c50r = 0.888
    cv_gamma = 0.304
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

            eta0.append(alpha*(1-get_probability(c50p_set, c50r_set, gamma_set)))
            proba.append(get_probability(c50p_set, c50r_set, gamma_set))

design_parameters = [R, Q, P0, eta0, grid_vector, lambda_1, lambda_2, nu, epsilon]
# select 20 random patient index between 0 and 999

patient_index_list = [1]

# %% run the simulation
for patient_index in patient_index_list:
    simulation(patient_index, design_parameters)

# %% plot the results
for patient_index in patient_index_list:
    bis_estimated = pd.read_csv(f'./data/mekf/bis_estimated_{patient_index}.csv', index_col=0).values
    bis_measured = pd.read_csv(f'./data/simulations/simu_{patient_index}.csv', index_col=0)['BIS']
    parameters_estimated = pd.read_csv(f'../data/mekf/parameters_{patient_index}.csv', index_col=0).values
    true_parameters = pd.read_csv(f'./data/simulations/parameters.csv', index_col=0).iloc[patient_index].values[-6:]

    plt.figure()
    plt.plot(bis_estimated, label='estimated')
    plt.plot(bis_measured, label='measured')
    plt.legend()
    plt.title(f'patient {patient_index}')
    plt.grid()
    plt.show()

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.plot(parameters_estimated[:, 0], label='estimated')
    plt.plot(np.ones(len(parameters_estimated[:, 0]))*true_parameters[0], label='true')
    plt.legend()
    plt.title(f'patient {patient_index}')
    plt.grid()
    plt.ylabel('C50p')

    plt.subplot(3, 1, 2)
    plt.plot(parameters_estimated[:, 1], label='estimated')
    plt.plot(np.ones(len(parameters_estimated[:, 1]))*true_parameters[1], label='true')
    plt.legend()
    plt.grid()
    plt.ylabel('C50r')

    plt.subplot(3, 1, 3)
    plt.plot(parameters_estimated[:, 2], label='estimated')
    plt.plot(np.ones(len(parameters_estimated[:, 2]))*true_parameters[2], label='true')
    plt.legend()
    plt.grid()
    plt.ylabel('gamma')
    plt.show()
