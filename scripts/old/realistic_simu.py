""" Generate simulation files of anetshesia Induction with Propofol and Remifentanil. """


# %% Imports
import multiprocessing as mp
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import python_anesthesia_simulator as pas
from TCI_control import TCI
from tqdm import tqdm


# %% Parameters

nb_patient = 100
output_folder = './data/simulations/'
parameter_file = 'parameters.csv'

sim_duration = 60*15  # 10 mintues
sampling_time = 2  # 1 second
BIS_target = 50
first_propo_target = 3.5
first_remi_target = 4
control_time = 10
model_PK = 'Eleveld'
# %% Run simulation

parameter = pd.DataFrame(columns=['age', 'height', 'weight', 'gender',
                                  'c50p', 'c50r', 'gamma', 'beta', 'E0', 'Emax'])


def run_simulation(i):
    # Generate parameters
    np.random.seed(i)
    age = np.random.randint(low=18, high=70)
    height = np.random.randint(low=150, high=190)
    weight = np.random.randint(low=50, high=100)
    gender = np.random.randint(low=0, high=2)

    patient_info = [age, height, weight, gender]
    # Define patient and controller
    patient = pas.Patient(patient_info, ts=sampling_time,
                          model_propo=model_PK, model_remi=model_PK,
                          random_PD=True, random_PK=True)

    controller_propo = TCI(patient_info=patient_info, drug_name='Propofol',
                           sampling_time=sampling_time, model_used=model_PK, control_time=control_time)
    controller_remi = TCI(patient_info=patient_info, drug_name='Remifentanil',
                          sampling_time=sampling_time, model_used=model_PK, control_time=control_time)

    target_propo = first_propo_target
    target_remi = first_remi_target
    control_time_propo = np.random.randint(low=1*60, high=90)//sampling_time
    control_time_remi = np.random.randint(low=1*60, high=90)//sampling_time
    # run simulation
    for j in range(int(sim_duration/sampling_time)):
        if j % (control_time//sampling_time) == 0:
            u_propo = controller_propo.one_step(target_propo) * 10/3600
            u_remi = controller_remi.one_step(target_remi) * 10/3600
        patient.one_step(u_propo, u_remi, noise=True)
        # at each control time if we are not in the intervall [40,60]
        if j % control_time_propo == 0 and j > 90//sampling_time:
            if patient.dataframe['BIS'].iloc[-1] < BIS_target-20:
                target_propo -= 2
            elif patient.dataframe['BIS'].iloc[-1] < BIS_target-10:
                target_propo -= 1
            elif patient.dataframe['BIS'].iloc[-1] < BIS_target-5:
                target_propo -= .5
            elif patient.dataframe['BIS'].iloc[-1] > BIS_target+20:
                target_propo += 2
            elif patient.dataframe['BIS'].iloc[-1] > BIS_target+10:
                target_propo += 1
            elif patient.dataframe['BIS'].iloc[-1] > BIS_target+5:
                target_propo += .5
        if j % control_time_remi == 0 and j > 90//sampling_time:
            if patient.dataframe['BIS'].iloc[-1] < BIS_target-5:
                target_remi -= 1
            elif patient.dataframe['BIS'].iloc[-1] > BIS_target+5:
                target_remi += 1
        # Save simulation
    patient.dataframe.to_csv(output_folder + f'simu_{i}.csv')

    # add a line to the parameter dataframe
    return ([i] + patient_info + patient.hill_param)


start = time.perf_counter()
results = []
with mp.Pool(mp.cpu_count()) as pool:
    results = list(tqdm(pool.imap(run_simulation, range(nb_patient)), total=nb_patient))

for result in results:
    parameter.loc[result[0]] = result[1:]  # add a line to the parameter dataframe
end = time.perf_counter()
print(f'Elapsed time: {end-start} s')
print(f'Average time per simulation: {(end-start)/nb_patient} s')
parameter.to_csv(output_folder + parameter_file)


# %% plot all BIS results with mean value and standard deviation
# get data
BIS = np.zeros((nb_patient, int(sim_duration/sampling_time)))
u_propo = np.zeros((nb_patient, int(sim_duration/sampling_time)))
u_remi = np.zeros((nb_patient, int(sim_duration/sampling_time)))
x_propo = np.zeros((nb_patient, int(sim_duration/sampling_time)))
x_remi = np.zeros((nb_patient, int(sim_duration/sampling_time)))
for i in range(nb_patient):
    case = pd.read_csv(output_folder + f'simu_{i}.csv')
    BIS[i, :] = case['BIS']
    u_propo[i, :] = case['u_propo']
    u_remi[i, :] = case['u_remi']
    x_propo[i, :] = case['x_propo_4']
    x_remi[i, :] = case['x_remi_4']
# plot
plt.figure()
plt.plot(np.arange(0, sim_duration, sampling_time)/60, BIS.T, color='grey', alpha=0.5)
plt.plot(np.arange(0, sim_duration, sampling_time)/60, np.mean(BIS, axis=0), color='black', label='mean')
plt.plot(np.arange(0, sim_duration, sampling_time)/60, np.mean(BIS, axis=0) +
         np.std(BIS, axis=0), color='black', linestyle='--', label='std')
plt.plot(np.arange(0, sim_duration, sampling_time)/60, np.mean(BIS, axis=0) -
         np.std(BIS, axis=0), color='black', linestyle='--')
plt.plot(np.arange(0, sim_duration, sampling_time)/60,
         np.ones(int(sim_duration/sampling_time))*BIS_target, color='red', label='target')
plt.legend()
plt.grid()
plt.xlabel('Time (min)')
plt.ylabel('BIS')
# save figure as pdf
plt.savefig('report/images/BIS_data.pdf', bbox_inches='tight')
plt.show()


# plot the input values
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(np.arange(0, sim_duration, sampling_time)/60, u_propo.T, color='grey', alpha=0.5)
plt.plot(np.arange(0, sim_duration, sampling_time)/60, np.mean(u_propo, axis=0), color='black', label='mean')
plt.plot(np.arange(0, sim_duration, sampling_time)/60, np.mean(u_propo, axis=0) +
         np.std(u_propo, axis=0), color='black', linestyle='--', label='std')
plt.plot(np.arange(0, sim_duration, sampling_time)/60, np.mean(u_propo, axis=0) -
         np.std(u_propo, axis=0), color='black', linestyle='--')
plt.legend()
plt.grid()
plt.xlabel('Time (min)')
plt.ylabel('Propofol infusion rate (mg/min)')
plt.subplot(2, 1, 2)
plt.plot(np.arange(0, sim_duration, sampling_time)/60, u_remi.T, color='grey', alpha=0.5)
plt.plot(np.arange(0, sim_duration, sampling_time)/60, np.mean(u_remi, axis=0), color='black', label='mean')
plt.plot(np.arange(0, sim_duration, sampling_time)/60, np.mean(u_remi, axis=0) +
         np.std(u_remi, axis=0), color='black', linestyle='--', label='std')
plt.plot(np.arange(0, sim_duration, sampling_time)/60, np.mean(u_remi, axis=0) -
         np.std(u_remi, axis=0), color='black', linestyle='--')
plt.legend()
plt.grid()
plt.xlabel('Time (min)')
plt.ylabel('Remifentanil infusion rate (ng/min)')
plt.savefig('report/images/input_data.pdf', bbox_inches='tight')
plt.show()

# plot the concentration
plt.figure()
plt.subplot(2, 1, 1)
plt.plot(np.arange(0, sim_duration, sampling_time)/60, x_propo.T, color='grey', alpha=0.5)
plt.plot(np.arange(0, sim_duration, sampling_time)/60, np.mean(x_propo, axis=0), color='black', label='mean')
plt.plot(np.arange(0, sim_duration, sampling_time)/60, np.mean(x_propo, axis=0) +
         np.std(x_propo, axis=0), color='black', linestyle='--', label='std')
plt.plot(np.arange(0, sim_duration, sampling_time)/60, np.mean(x_propo, axis=0) -
         np.std(x_propo, axis=0), color='black', linestyle='--')
plt.legend()
plt.grid()
plt.xlabel('Time (min)')
plt.ylabel('Propofol concentration (µg/ml)')
plt.subplot(2, 1, 2)
plt.plot(np.arange(0, sim_duration, sampling_time)/60, x_remi.T, color='grey', alpha=0.5)
plt.plot(np.arange(0, sim_duration, sampling_time)/60, np.mean(x_remi, axis=0), color='black', label='mean')
plt.plot(np.arange(0, sim_duration, sampling_time)/60, np.mean(x_remi, axis=0) +
         np.std(x_remi, axis=0), color='black', linestyle='--', label='std')
plt.plot(np.arange(0, sim_duration, sampling_time)/60, np.mean(x_remi, axis=0) -
         np.std(x_remi, axis=0), color='black', linestyle='--')
plt.legend()
plt.grid()
plt.xlabel('Time (min)')
plt.ylabel('Remifentanil concentration (ng/ml)')
plt.savefig('report/images/concentration_data.pdf', bbox_inches='tight')
plt.show()


# %% plot for a specifc patient
patient = 3
plt.figure()
plt.subplot(2, 1, 1)
# plot the bis
plt.plot(np.arange(0, sim_duration, sampling_time)/60, BIS[patient, :], label='measure')
plt.plot(np.arange(0, sim_duration, sampling_time)/60,
         np.ones(int(sim_duration/sampling_time))*BIS_target, label='target')
plt.grid()
plt.xlabel('Time (min)')
plt.ylabel('BIS')

# plot the input values
plt.subplot(2, 1, 2)
plt.plot(np.arange(0, sim_duration, sampling_time)/60, u_propo[patient, :], label='Propofol')
plt.plot(np.arange(0, sim_duration, sampling_time)/60, u_remi[patient, :], label='Remifentanil')
plt.legend()
plt.grid()
plt.xlabel('Time (min)')
plt.ylabel('Infusion rate')
plt.show()
