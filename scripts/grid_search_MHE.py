import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import multiprocessing as mp
from tqdm import tqdm
from functools import partial
from itertools import product

from test_on_simu import simulation
from metrics_function import one_line


mhe_path = 'data/mhe/'
time_step = 2
pred_time = 120//time_step
stop_time_list = [i//time_step for i in range(15, 15*60 - pred_time*time_step, 30)]


np.random.seed(1)
case_list = np.random.randint(0, 500, 8)

# plot BIS for al those cases
# plt.figure()
# for case in case_list:
#     bis = pd.read_csv(f'data/simulations/simu_{case}.csv', index_col=0)['BIS']
#     plt.plot(bis, label=f'case {case}')
# plt.show()

# define objective function


def one_obj(case, mhe_param):
    simulation(case, [None, None, mhe_param], [False, False, True])
    r = one_line(case, mhe_path, stop_time_list, pred_time)
    return np.sum(r.values)


def objective_function(mhe_param):
    with mp.Pool(8) as pool:
        res = list(pool.imap(partial(one_obj, mhe_param=mhe_param), case_list))
    return max(res)


theta = [100, 1, 300, 0.005]*3
theta[4] = 1
Q = np.diag([1, 550, 550, 1, 1, 50, 750, 1])
R_list = np.logspace(-3, 0, 1)*5
N_mhe_list = [15]  # , 20, 25]
theta_list_1 = np.logspace(-2, 2, 1)

# %% Grid search
results = pd.DataFrame(columns=['R', 'theta_1', 'N_mhe', 'objective_function'])

for R_id, theta_1_id, N_mhe_id in tqdm(product(range(len(R_list)), range(len(theta_list_1)), range(len(N_mhe_list))), desc='Grid search'):
    print(R_id, theta_1_id, N_mhe_id)

    R = float(R_list[R_id])
    theta[0] = theta_list_1[theta_1_id]
    theta[4] = theta[0]/10
    theta[8] = theta[0]
    N_mhe = N_mhe_list[N_mhe_id]
    mhe_param = [R, Q, theta, N_mhe]
    res = objective_function(mhe_param)
    results.loc[len(results)] = [R, theta[0], N_mhe, res]

results.to_csv('data/grid_search_mhe.csv')
