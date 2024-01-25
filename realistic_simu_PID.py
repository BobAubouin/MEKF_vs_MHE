""" Generate simulation files of anetshesia Induction with Propofol and Remifentanil. """


# %% Imports
import multiprocessing as mp
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import python_anesthesia_simulator as pas
from tqdm import tqdm


class PID():
    """Implementation of a working PID with anti-windup.

    PID = Kp ( 1 + Te / (Ti - Ti z^-1) + Td (1-z^-1) / (Td/N (1-z^-1) + Te) )
    """

    def __init__(self, Kp: float, Ti: float, Td: float, N: int = 5,
                 Ts: float = 1, umax: float = 1e10, umin: float = -1e10):
        """
        Init the class.

        Parameters
        ----------
        Kp : float
            Gain.
        Ti : float
            Integrator time constant.
        Td : float
            Derivative time constant.
        N : int, optional
            Interger to filter the derivative part. The default is 5.
        Ts : float, optional
            Sampling time. The default is 1.
        umax : float, optional
            Upper saturation of the control input. The default is 1e10.
        umin : float, optional
            Lower saturation of the control input. The default is -1e10.

        Returns
        -------
        None.

        """
        self.Kp = Kp
        self.Ti = Ti
        self.Td = Td
        self.N = N
        self.Ts = Ts
        self.umax = umax
        self.umin = umin

        self.integral_part = 0
        self.derivative_part = 0
        self.last_BIS = 100

    def one_step(self, BIS: float, Bis_target: float) -> float:
        """Compute the next command for the PID controller.

        Parameters
        ----------
        BIS : float
            Last BIS measurement.
        Bis_target : float
            Current BIS target.

        Returns
        -------
        control_input: float
            control value computed by the PID.
        """
        error = -(Bis_target - BIS)
        self.integral_part += self.Ts / self.Ti * error

        self.derivative_part = (self.derivative_part * self.Td / self.N +
                                self.Td * (BIS - self.last_BIS)) / (self.Ts +
                                                                    self.Td / self.N)
        self.last_BIS = BIS

        control_input = self.Kp * (error + self.integral_part + self.derivative_part)

        # Anti windup Conditional Integration from
        # Visioli, A. (2006). Anti-windup strategies. Practical PID control, 35-60.
        if (control_input >= self.umax) and control_input * error <= 0:
            self.integral_part = self.umax / self.Kp - error - self.derivative_part
            control_input = np.array(self.umax)

        elif (control_input <= self.umin) and control_input * error <= 0:
            self.integral_part = self.umin / self.Kp - error - self.derivative_part
            control_input = np.array(self.umin)

        return control_input

# %% Parameters


nb_patient = 10
output_folder = './data/simulations_easy/'
parameter_file = 'parameters.csv'

sim_duration = 60*15  # 10 mintues
sampling_time = 2  # 1 second
BIS_target = 50
model_PK = 'Eleveld'
ratio = 2
Kp = 0.0354
Ti = 511.8
Td = 8.989
umax = 6.67

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
                          random_PD=True, random_PK=False)

    controller = PID(Kp, Ti, Td, Ts=sampling_time, umax=umax, umin=0)

    Up = 0
    # run simulation
    for j in range(int(sim_duration/sampling_time)):
        patient.one_step(Up, Up*ratio, noise=False)
        Up = controller.one_step(patient.dataframe['BIS'].iloc[-1], BIS_target)
        Up = np.clip(Up, 0, umax)
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
plt.savefig('figures/BIS_data.pdf', bbox_inches='tight')
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
plt.savefig('figures/input_data.pdf', bbox_inches='tight')
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
plt.savefig('figures/concentration_data.pdf', bbox_inches='tight')
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
# plt.xlabel('Time (min)')
plt.ylabel('BIS')

# plot the input values
plt.subplot(2, 1, 2)
plt.plot(np.arange(0, sim_duration, sampling_time)/60, u_propo[patient, :], label='Propofol')
plt.plot(np.arange(0, sim_duration, sampling_time)/60, u_remi[patient, :], label='Remifentanil')
plt.legend()
plt.grid()
plt.xlabel('Time (min)')
plt.ylabel('Infusion rate')
plt.savefig('figures/test_control.pdf', bbox_inches='tight')
plt.show()
