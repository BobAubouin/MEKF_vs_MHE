# Description: grid search for the petri method
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
import optuna
from functools import partial
import scipy

from test_on_simu import simulation
from metrics_function import one_line
import python_anesthesia_simulator as pas

mekf_p_path = 'data/mekf_p/'
time_step = 2
pred_time = 3*60
stop_time_list = [i-1 for i in range(15, 15*60 - pred_time*time_step, 30)]


np.random.seed(2)
case_list = np.random.randint(0, 500, 16)


# define objective function

BIS_param_nominal = pas.BIS_model().hill_param

# Qp = np.load('data/cov_propo.npy')
# Qr = np.load('data/cov_remi.npy')
# Q = np.block([[Qp, np.zeros((4, 4))], [np.zeros((4, 4)), Qr]])
# R = np.load('data/R.npy')


cv_c50p = 0.182
cv_c50r = 0.888
cv_gamma = 0.304
# estimation of log normal standard deviation
w_c50p = np.sqrt(np.log(1+cv_c50p**2))
w_c50r = np.sqrt(np.log(1+cv_c50r**2))
w_gamma = np.sqrt(np.log(1+cv_gamma**2))

c50p_list = BIS_param_nominal[0]*np.exp([-2.2*w_c50p, -w_c50p, -0.4*w_c50p, 0, w_c50p])  # , -w_c50p
c50r_list = BIS_param_nominal[1]*np.exp([-2.2*w_c50r, -w_c50r, -0.4*w_c50r, 0, 0.6*w_c50r, w_c50r])
gamma_list = BIS_param_nominal[2]*np.exp([-2.2*w_gamma, -w_gamma, -0.4*w_gamma, 0, 0.8*w_gamma, 1.5*w_gamma])  #
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
    return grid_vector, eta0


def one_obj(case, petri_param):
    simulation(case, [petri_param, None, None, None], [True, False, False, False, False])
    plt.pause(2)
    r = one_line(case, mekf_p_path, stop_time_list, pred_time)
    return np.sum(r.values)


def objective_function(trial):
    # Petri parameters
    P0 = 1e-3 * np.eye(8)
    Q = trial.suggest_float('Q', 1e-3, 1e0, log=True)
    Q_mat = Q * np.diag([0.1, 0.1, 0.05, 0.05, 1, 1, 10, 1])  # np.diag([1, 1/550, 1/550, 1, 1, 1/50, 1/750, 1])
    R = trial.suggest_float('R', 5e1, 1e4, log=True)
    alpha = trial.suggest_float('alpha', 0.1, 1e3, log=True)
    grid_vector, eta0 = init_proba(alpha)
    lambda_1 = 1
    lambda_2 = trial.suggest_float('lambda_2', 1e-2, 1e4, log=True)
    nu = 1.e-5
    epsilon = trial.suggest_float('epsilon', 0.3, 1)
    petri_param = [R, Q, P0, eta0, grid_vector, lambda_1, lambda_2, nu, epsilon]
    with mp.Pool(16) as pool:
        res = list(pool.imap(partial(one_obj, petri_param=petri_param), case_list))
    return np.mean(res)


study = optuna.create_study(direction='minimize', study_name='petri_final_6',
                            storage='sqlite:///data/petri_2.db', load_if_exists=True)
study.optimize(objective_function, n_trials=100)

print(study.best_params)
