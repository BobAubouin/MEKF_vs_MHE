import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import python_anesthesia_simulator as pas
from tqdm import tqdm
import matplotlib.colors as mcolors
import matplotlib
import multiprocessing as mp
from functools import partial


# plot config
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

font = {'family': 'serif',
        'weight': 'normal',
        'size': 16}


plt.rc('text', usetex=True)
matplotlib.rc('font', **font)

# define the path to results files
mekf_n_path = 'data/vital/mekf_n/'
mekf_p_path = 'data/vital/mekf_p/'
mhe_path = 'data/vital/mhe/'
Patient_data = pd.read_csv(r"./scripts/info_clinic_vitalDB.csv")
caseid_list = list(np.loadtxt('./scripts/caseid_list.txt', dtype=int))
caseid_list.remove(104)
caseid_list.remove(859)

number_of_patients = 50

# %% Load the results


def metrics_function(path: str, patient_id: int, stop_time: int, pred_time: int = -1):
    """ This function compute the metrics for a given patient. The metrics is defined as the norm of the difference
    between the true BIS and the estimated BIS. When the estimated BIS is computed in an open loop fashion, with the states and parameters estimated at time stop_time.

    Parameters
    ----------
    path : string
        Path of the results files.
    patient_id : int
        index of the patient to compute the metrics.
    stop_time : int
        index of the moment to start the open loop simulation.

    Returns
    -------
    metrics : float
        metrics value.
    """
    patient_id = caseid_list[patient_id]
    parameters = pd.read_csv(path + 'parameters_' + str(patient_id) + '.csv', header=None).to_numpy()
    states = pd.read_csv(path + 'x_' + str(patient_id) + '.csv', header=None).to_numpy()

    age = int(Patient_data[Patient_data['caseid'] == str(patient_id)]['age'])
    height = float(Patient_data[Patient_data['caseid'] == str(patient_id)]['height'])
    weight = float(Patient_data[Patient_data['caseid'] == str(patient_id)]['weight'])
    sex = str(Patient_data[Patient_data['caseid'] == str(patient_id)]['sex'])
    if sex == "M":
        sex = 1  # Male (M)
    else:
        sex = 0  # Female (F)
    patient_param = [age, height, weight, sex]

    data_simu = pd.read_csv(f'data/vital/case_{patient_id}.csv')
    u_propro = data_simu['Orchestra/PPF20_RATE'][stop_time: stop_time + pred_time].to_numpy()
    u_remi = data_simu['Orchestra/RFTN20_RATE'][stop_time: stop_time + pred_time].to_numpy()
    true_bis = data_simu['BIS/BIS'][stop_time: stop_time + pred_time].to_numpy()
    hill_param = np.concatenate((parameters[stop_time, :], [0, 97.4, 97.4]))
    patient_sim = pas.Patient(patient_characteristic=patient_param, model_propo='Eleveld',
                              model_remi='Eleveld', hill_param=hill_param, ts=2)
    x0_propo = np.concatenate((states[:4, stop_time], [0, 0]))
    x0_remi = np.concatenate((states[4:8, stop_time], [0]))
    res = patient_sim.full_sim(u_propro, u_remi, x0_propo=x0_propo, x0_remi=x0_remi)
    bis_test = res['BIS']
    return np.linalg.norm(true_bis-bis_test)


time_step = 2
pred_time = 120//time_step
stop_time_list = [i//time_step for i in range(15, 15*60 - pred_time*time_step, 30)]


def one_line(i, path, stop_time_list, pred_time):
    metrics = pd.DataFrame()
    for stop_time in stop_time_list:
        metrics.loc[i, stop_time] = metrics_function(path, i, stop_time, pred_time)
    return metrics


try:
    metrics_MEKF_N = pd.read_csv('data/metrics_vital_MEKF_N.csv', index_col=0)
except FileNotFoundError:
    with mp.Pool(mp.cpu_count()) as pool:
        res = list(tqdm(pool.imap(partial(one_line, path=mekf_n_path, stop_time_list=stop_time_list,
                                          pred_time=pred_time), range(number_of_patients)), total=number_of_patients, desc='MEKF_N'))
    metrics_MEKF_N = pd.concat(res)
    metrics_MEKF_N.to_csv('data/metrics_vital_MEKF_N.csv')

try:
    metrics_MEKF_P = pd.read_csv('data/metrics_vital_MEKF_P.csv', index_col=0)
except FileNotFoundError:
    with mp.Pool(mp.cpu_count()) as pool:
        res = list(tqdm(pool.imap(partial(one_line, path=mekf_p_path, stop_time_list=stop_time_list,
                                          pred_time=pred_time), range(number_of_patients)), total=number_of_patients, desc='MEKF_P'))
    metrics_MEKF_P = pd.concat(res)
    metrics_MEKF_P.to_csv('data/metrics_vital_MEKF_P.csv')

try:
    metrics_MHE = pd.read_csv('data/metrics_vital_MHE.csv', index_col=0)
except FileNotFoundError:
    with mp.Pool(mp.cpu_count()) as pool:
        res = list(tqdm(pool.imap(partial(one_line, path=mhe_path, stop_time_list=stop_time_list,
                                          pred_time=pred_time), range(number_of_patients)), total=number_of_patients, desc='MHE'))
    metrics_MHE = pd.concat(res)
    metrics_MHE.to_csv('data/metrics_vital_MHE.csv')

# %% Plot the results


# plot a comparison between the metrics for the different stop time on the same figure

mean_MEKF_N = metrics_MEKF_N.mean()
mean_MEKF_P = metrics_MEKF_P.mean()
mean_MHE = metrics_MHE.mean()

std_MEKF_N = metrics_MEKF_N.std()
std_MEKF_P = metrics_MEKF_P.std()
std_MHE = metrics_MHE.std()


plt.figure()
transparency = 0.3
plt.fill_between(stop_time_list, mean_MEKF_N-std_MEKF_N,
                 mean_MEKF_N+std_MEKF_N, alpha=transparency,
                 facecolor=mcolors.TABLEAU_COLORS['tab:blue'])  # , hatch="\\\\")

plt.fill_between(stop_time_list, mean_MEKF_P-std_MEKF_P,
                 mean_MEKF_P+std_MEKF_P, alpha=transparency,
                 facecolor=mcolors.TABLEAU_COLORS['tab:orange'])

plt.fill_between(stop_time_list, mean_MHE-std_MHE, mean_MHE+std_MHE,
                 alpha=transparency, facecolor=mcolors.TABLEAU_COLORS['tab:green'])  # , hatch="////")


plt.plot(stop_time_list, mean_MEKF_N, label='MEKF_Narendra', color=mcolors.TABLEAU_COLORS['tab:blue'])
plt.plot(stop_time_list, mean_MEKF_P, label='MEKF_Petri', color=mcolors.TABLEAU_COLORS['tab:orange'])
plt.plot(stop_time_list, mean_MHE, label='MHE', color=mcolors.TABLEAU_COLORS['tab:green'])
# add the standard deviation to the plot

plt.legend()
plt.ylabel('metrics')
plt.xlabel('stop time')
plt.grid()
savepath = "figures/stats_comp_vital.pdf"
plt.savefig(savepath, bbox_inches='tight', format='pdf')
plt.show()

# plot the maximum value of the metrics
plt.figure()
plt.plot(stop_time_list, metrics_MEKF_N.max(), label='MEKF_Narendra')
plt.plot(stop_time_list, metrics_MEKF_P.max(), label='MEKF_Petri')
plt.plot(stop_time_list, metrics_MHE.max(), label='MHE')
plt.legend()
plt.ylabel('metrics')
plt.xlabel('stop time')
plt.grid()
savepath = "figures/stats_max_vital.pdf"
plt.savefig(savepath, bbox_inches='tight', format='pdf')
plt.show()
