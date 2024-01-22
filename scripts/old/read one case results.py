import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import python_anesthesia_simulator as pas
from metrics_function import one_line

path = './data/plus/'
patient_index = 4

BIS_param_nominal = pas.BIS_model().hill_param

cv_c50p = 0.182
cv_c50r = 0.888
cv_gamma = 0.304
w_c50p = np.sqrt(np.log(1+cv_c50p**2))
w_c50r = np.sqrt(np.log(1+cv_c50r**2))
w_gamma = np.sqrt(np.log(1+cv_gamma**2))

c50p_list = BIS_param_nominal[0]*np.exp([-2*w_c50p, -w_c50p, -0.5*w_c50p, 0, w_c50p])  # , -w_c50p
c50r_list = BIS_param_nominal[1]*np.exp([-2*w_c50r, -w_c50r, -0.5*w_c50r, 0, 0.5*w_c50r, w_c50r])
gamma_list = BIS_param_nominal[2]*np.exp([-2*w_gamma, -w_gamma, -0.5*w_gamma, 0, w_gamma, 2*w_gamma])  #
# surrender list by Inf value
c50p_list = np.concatenate(([-np.Inf], c50p_list, [np.Inf]))
c50r_list = np.concatenate(([-np.Inf], c50r_list, [np.Inf]))
gamma_list = np.concatenate(([-np.Inf], gamma_list, [np.Inf]))

bis_estimated = pd.read_csv(path + f'bis_estimated_{patient_index}.csv', index_col=0).values
bis_measured = pd.read_csv(f'./data/simulations/simu_{patient_index}.csv', index_col=0)['BIS']
parameters_estimated = pd.read_csv(path + f'parameters_{patient_index}.csv', index_col=0)
true_parameters = pd.read_csv(f'./data/simulations/parameters.csv',
                              index_col=0).iloc[patient_index].values[-6:]
# get the effect site concentration
x_propo = pd.read_csv(f'./data/simulations/simu_{patient_index}.csv', index_col=0)['x_propo_4']
x_remi = pd.read_csv(f'./data/simulations/simu_{patient_index}.csv', index_col=0)['x_remi_4']
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

time_step = 2
pred_time = 3*60
stop_time_list = [i-1 for i in range(15, 15*60 - pred_time*time_step, 120)]
r = one_line(patient_index, path, stop_time_list, pred_time, plot=True)


plt.plot(stop_time_list, r.values[0])
plt.grid()
plt.show()
