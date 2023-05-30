""" Plot the results of the MEKF."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

patient_number = 100
parameters_path = './data/simulations/parameters.csv'  # path to the parameters
results_path = './data/mekf/'  # path to the results

df_param = pd.read_csv(parameters_path)

# get the error between the last estimation and the true PD parameters for each patient
error = np.zeros((patient_number, 3))
for i in range(patient_number):
    # load the parameters
    param = df_param.iloc[i][['c50p', 'c50r', 'gamma']].values
    # load the results
    df = pd.read_csv(results_path + f'parameters_{i}.csv')
    # get the last estimation
    param_estimated = df.iloc[-1].values[-3:]
    # get the mean estimation
    param_estimated = np.mean(df.values[:, -3:], axis=0)
    # compute the error
    error[i] = param - param_estimated

    # evaluate the stability of the estimation


# plot the results
plt.figure()
plt.subplot(1, 3, 1)
plt.hist(error[:, 0], label='estimation error', bins=int(patient_number/5))
plt.hist(df_param['c50p'][:patient_number] - np.mean(df_param['c50p'][:patient_number]),
         label='true value', bins=int(patient_number/5), alpha=0.5)
plt.title('C50p')
plt.legend()
plt.subplot(1, 3, 2)
plt.hist(error[:, 1], bins=int(patient_number/5))
plt.hist(df_param['c50r'][:patient_number] - np.mean(df_param['c50r']
         [:patient_number]), bins=int(patient_number/5), alpha=0.5)
plt.title('C50r')
plt.subplot(1, 3, 3)
plt.hist(error[:, 2], bins=int(patient_number/5))
plt.hist(df_param['gamma'][:patient_number] - np.mean(df_param['gamma']
         [:patient_number]), bins=int(patient_number/5), alpha=0.5)
plt.title('gamma')
plt.show()
