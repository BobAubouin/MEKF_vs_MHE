from metrics_function import one_line
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import python_anesthesia_simulator as pas

# generate True data without uncertainties
id = 9999
ts = 2
Patient_info = [55, 180, 60, 0]
patient_sim = pas.Patient(patient_characteristic=Patient_info, model_propo='Eleveld', model_remi='Eleveld', ts=ts)

N_simu = 15*60//ts
u_propo = np.zeros(N_simu)
u_remi = np.zeros(N_simu)

u_propo[0: 50//ts] = 1.7
u_propo[4*60//ts:] = 0.15

u_remi[0: 90//ts] = 1.5
u_remi[(4*60+30)//ts:] = 0.2
x0_propo = np.zeros(6)
x0_remi = np.zeros(5)
res = patient_sim.full_sim(u_propo, u_remi, x0_propo=x0_propo, x0_remi=x0_remi)
bis_simu = res['BIS']
# time = np.arange(len(bis_simu))*ts/60
# plt.plot(time, bis_simu)
# plt.show()
# generate input

# save data
# simu side
param_data = pd.read_csv(f'./data/simulations/parameters.csv', index_col=0)
param_data.loc[id, ['age', 'height', 'weight', 'gender']] = Patient_info
param_data = param_data.fillna(0)
param_data.to_csv(f'data/simulations/parameters.csv')
res.to_csv(f'data/simulations/simu_{id}.csv')

# observer side
path = f'./data/observer_test/'

Time = res['Time'].to_numpy()
data_frame_param = pd.DataFrame(columns=['Time', 'c50p', 'c50r', 'gamma'])
data_frame_param['Time'] = Time
data_frame_param['c50p'] = patient_sim.hill_param[0]
data_frame_param['c50r'] = patient_sim.hill_param[1]
data_frame_param['gamma'] = patient_sim.hill_param[2]

data_frame_param.to_csv(path + f"parameters_{id}.csv")


states = res[['Time', 'x_propo_1', 'x_propo_2', 'x_propo_3',
              'x_propo_4', 'x_remi_1', 'x_remi_2', 'x_remi_3', 'x_remi_4']]
states.to_csv(path + f"x_{id}.csv")


# %% test the metric function

time_step = 2
pred_time = 120
stop_time_list = [i-1 for i in range(15, 15*60 - pred_time*time_step, 30)]

res = one_line(id, path, stop_time_list, pred_time)

assert res.shape == (1, len(stop_time_list))
assert np.linalg.norm(res.values) < 1e-5

print('test passed')
