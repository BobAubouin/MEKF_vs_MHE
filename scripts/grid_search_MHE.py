import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from functools import partial
import optuna

from test_on_simu import simulation
from metrics_function import one_line


mhe_path = 'data/mhe/'
time_step = 2
pred_time = 3*60
stop_time_list = [i-1 for i in range(15, 15*60 - pred_time*time_step, 30)]


np.random.seed(2)
case_list = np.random.randint(0, 500, 16)


def one_obj(case, mhe_param):
    simulation(case, [None, None, mhe_param], [False, False, True])
    plt.pause(2)
    r = one_line(case, mhe_path, stop_time_list, pred_time)
    return np.sum(r.values)

def objective_function(trial):
    R = trial.suggest_float('R', 1e-4, 1e1, log=True)
    eta = trial.suggest_float('eta', 1e-4, 1e1, log=True)
    N_mhe = trial.suggest_int('N_mhe', 10, 30)
    Q = np.diag([1, 550, 550, 1, 1, 50, 750, 1])
    theta = [100, 1, 300, 0.005]*3
    theta[0] = eta
    theta[4] = theta[0]/10
    theta[8] = theta[0]

    mhe_param = [R, Q, theta, N_mhe]

    with mp.Pool(16) as pool:
        res = list(pool.imap(partial(one_obj, mhe_param=mhe_param), case_list))
    return np.mean(res)

study = optuna.create_study(direction='minimize', study_name='mhe_final_2', storage='sqlite:///data/mhe.db', load_if_exists=True)
study.optimize(objective_function, n_trials=200)

print(study.best_params)