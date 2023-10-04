import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from tqdm import tqdm
from functools import partial
from itertools import product
import scipy

from test_on_simu import simulation
from metrics_function import one_line
import python_anesthesia_simulator as pas

mhe_path = 'data/mekf_n/'
time_step = 2
pred_time = 120//time_step
stop_time_list = [i//time_step for i in range(15, 15*60 - pred_time*time_step, 30)]


np.random.seed(1)
case_list = np.random.randint(0, 500, 16)


# define objective function


def one_obj(case, narendra_param):
    simulation(case, [None, narendra_param, None], [False, True, False])
    r = one_line(case, mhe_path, stop_time_list, pred_time)
    return np.sum(r.values)


def objective_function(narendra_param):
    with mp.Pool(16) as pool:
        res = list(pool.imap(partial(one_obj, narendra_param=narendra_param), case_list))
    return np.max(res)


# Petri parameters
P0 = 1e-3 * np.eye(8)
Q = 1e0 * np.diag([0.01]*4+[1]*4)  # np.diag([1, 1/550, 1/550, 1, 1, 1/50, 1/750, 1])
R_list = np.logspace(-2, 1, 4)

alpha = 0
beta = 1
lambda_list = np.logspace(-4, -2, 3)
N_list = [15, 20, 25]
epsilon_list = [0.5, 0.6, 0.7]

# definition of the grid
BIS_param_nominal = pas.BIS_model().hill_param

cv_c50p = 0.182
cv_c50r = 0.888
cv_gamma = 0.304
# estimation of log normal standard deviation
w_c50p = np.sqrt(np.log(1+cv_c50p**2))
w_c50r = np.sqrt(np.log(1+cv_c50r**2))
w_gamma = np.sqrt(np.log(1+cv_gamma**2))

c50p_list = BIS_param_nominal[0]*np.exp([-2*w_c50p, -w_c50p, 0, w_c50p])  # , -w_c50p
c50r_list = BIS_param_nominal[1]*np.exp([-2*w_c50r, -1*w_c50r, 0, w_c50r, ])
gamma_list = BIS_param_nominal[2]*np.exp([-2*w_gamma, -w_gamma, 0, w_gamma])  # , -w_gamma
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


# %% Grid search
results = pd.DataFrame(columns=['R', 'N', 'lambda', 'epsilon', 'objective_function'])

for R, lambdap, N, epsilon in tqdm(product(R_list, lambda_list, N_list, epsilon_list), desc='Grid search', total=len(R_list)*len(lambda_list)*len(N_list)*len(epsilon_list)):

    narendra_param = [R, Q, P0, eta0, grid_vector, alpha, beta, lambdap, epsilon, N]
    res = objective_function(narendra_param)
    results.loc[len(results)] = [R, N, lambdap, epsilon, res]

results.to_csv('data/grid_search_narendra.csv')

