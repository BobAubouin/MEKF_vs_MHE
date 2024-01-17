import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial
import optuna

from test_on_simu import simulation
from metrics_function import one_line


mhe_path = 'data/mhe_std/'
time_step = 2
pred_time = 3*60
stop_time_list = [i-1 for i in range(15, 15*60 - pred_time*time_step, 30)]


np.random.seed(2)
case_list = np.random.randint(0, 500, 16)


R_inv = np.load('data/R.npy')
R = 1/R_inv
Qp = np.load('data/cov_propo.npy')
Qr = np.load('data/cov_remi.npy')
Qinv = np.block([[Qp, np.zeros((4, 4))], [np.zeros((4, 4)), Qr]])
Q_8 = np.linalg.inv(Qinv)
Q = np.block([[Q, np.zeros((8, 3))], [np.zeros((3, 8)), np.diag([1e5]*3)]])


def one_obj(case, mhe_param):
    simulation(case, [None, None, None, mhe_param], [False, False, False, False, True])
    plt.pause(2)
    r = one_line(case, mhe_path, stop_time_list, pred_time)
    # print(f"patient {case} : {np.sum(r.values)}")
    return np.sum(r.values)


def objective_function(trial):
    p = trial.suggest_float('p', 1e0, 1e6, log=True)
    eta = trial.suggest_float('eta', 1e-4, 1e1, log=True)
    N_mhe = trial.suggest_int('N_mhe', 10, 30)
    P = Q_8*p
    theta = [100, 0, 300, 0.005]*3
    theta[0] = eta
    theta[4] = theta[0]/10
    theta[8] = theta[0]

    mhe_param = [R, Q, theta, N_mhe, P]

    with mp.Pool(16) as pool:
        res = list(pool.imap(partial(one_obj, mhe_param=mhe_param), case_list))
    return np.mean(res)


study = optuna.create_study(direction='minimize', study_name='mhe_std_final_4',
                            storage='sqlite:///data/mhe.db', load_if_exists=True)
study.optimize(objective_function, n_trials=200)

print(study.best_params)

# R = study.best_params['R']
# q = study.best_params['q']
# eta = study.best_params['eta']
# N_mhe = study.best_params['N_mhe']

# Q = np.diag([1, 550, 550, 1, 1, 50, 750, 1]+[1e3]*3)*q
# P = np.diag([1, 550, 550, 1, 1, 50, 750, 1])
# theta = [100, 0, 300, 0.005]*3
# theta[0] = eta
# theta[4] = theta[0]/10
# theta[8] = theta[0]

# mhe_param = [R, Q, theta, N_mhe, P]
# with mp.Pool(16) as pool:
#     res = list(pool.imap(partial(one_obj, mhe_param=mhe_param), case_list))
# print(np.mean(res))
