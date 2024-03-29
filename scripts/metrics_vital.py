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
mekf_n_path = './data/vital/mekf_n/'
mekf_p_path = 'data/vital/mekf_p/'
mhe_path = 'data/vital/mhe_std/'
plus_path = 'data/vital/plus/'
Patient_data = pd.read_csv(r"./scripts/info_clinic_vitalDB.csv")
caseid_list = list(np.loadtxt('./scripts/caseid_list.txt', dtype=int))
caseid_list.remove(104)
caseid_list.remove(859)
caseid_list.remove(29)

number_of_patients = len(caseid_list)
bool_data = [True, True, False]

# %% Load the results


def metrics_function(path: str, patient_id: int, stop_time: int, pred_time: int = -1, plot: bool = False):
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
    parameters = pd.read_csv(path + 'parameters_' + str(patient_id) + '.csv', index_col=0)
    states = pd.read_csv(path + 'x_' + str(patient_id) + '.csv', index_col=0)

    age = int(Patient_data[Patient_data['caseid'] == str(patient_id)]['age'])
    height = float(Patient_data[Patient_data['caseid'] == str(patient_id)]['height'])
    weight = float(Patient_data[Patient_data['caseid'] == str(patient_id)]['weight'])
    sex = str(Patient_data[Patient_data['caseid'] == str(patient_id)]['sex'])
    if sex == "M":
        sex = 1  # Male (M)
    else:
        sex = 0  # Female (F)
    patient_param = [age, height, weight, sex]

    data_simu = pd.read_csv(f'./data/vital/case_{patient_id}.csv')
    u_propo = data_simu['Orchestra/PPF20_RATE'][stop_time//2: stop_time//2 + pred_time//2].to_numpy() * 20/3600
    u_remi = data_simu['Orchestra/RFTN20_RATE'][stop_time//2: stop_time//2 + pred_time//2].to_numpy() * 20/3600
    true_bis = data_simu['BIS/BIS'][stop_time//2: stop_time//2 + pred_time//2].to_numpy()
    hill_param = np.concatenate((parameters.loc[parameters['Time'] == stop_time, [
                                'c50p', 'c50r', 'gamma']].to_numpy()[0], [0, 97.4, 97.4]))
    patient_sim = pas.Patient(patient_characteristic=patient_param, model_propo='Eleveld',
                              model_remi='Eleveld', hill_param=hill_param, ts=2)
    x0_propo = np.concatenate(
        (states.loc[states['Time'] == stop_time, [f"x_propo_{i}" for i in range(1, 5)]].to_numpy()[0], [0, 0]))
    x0_remi = np.concatenate(
        (states.loc[states['Time'] == stop_time, [f"x_remi_{i}" for i in range(1, 5)]].to_numpy()[0], [0]))
    res = patient_sim.full_sim(u_propo, u_remi, x0_propo=x0_propo, x0_remi=x0_remi)
    bis_test = res['BIS']
    if plot:
        return np.linalg.norm(true_bis-bis_test), true_bis, bis_test
    return np.linalg.norm(true_bis-bis_test)


time_step = 2
pred_time = 3*60
stop_time_list = [i-1 for i in range(15, 15*60 - pred_time*time_step, 30)]


def one_line(i, path, stop_time_list, pred_time, plot: bool = False):
    metrics = pd.DataFrame()
    for stop_time in stop_time_list:
        if plot:
            metrics.loc[i, stop_time], true_bis, bis_test = metrics_function(path, i, stop_time, pred_time, plot)
            plt.plot(np.linspace(stop_time, stop_time + pred_time, len(true_bis)), true_bis, 'b')
            plt.plot(np.linspace(stop_time, stop_time + pred_time, len(true_bis)), bis_test, 'r')

        else:
            metrics.loc[i, stop_time] = metrics_function(path, i, stop_time, pred_time)
    if plot:
        plt.grid()
        plt.show()

    return metrics


# try:
#     metrics_MEKF_N = pd.read_csv('data/metrics_vital_MEKF_N.csv', index_col=0)
# except FileNotFoundError:
#     with mp.Pool(mp.cpu_count()) as pool:
#         res = list(tqdm(pool.imap(partial(one_line, path=mekf_n_path, stop_time_list=stop_time_list,
#                                           pred_time=pred_time), range(number_of_patients)), total=number_of_patients, desc='MEKF_N'))
#     metrics_MEKF_N = pd.concat(res)
#     metrics_MEKF_N.to_csv('data/metrics_vital_MEKF_N.csv')

if __name__ == '__main__':
    if bool_data[0]:
        try:
            metrics_MEKF_P = pd.read_csv('data/metrics_vital_MEKF_P.csv', index_col=0)
        except FileNotFoundError:
            with mp.Pool(mp.cpu_count()) as pool:
                res = list(tqdm(pool.imap(partial(one_line, path=mekf_p_path, stop_time_list=stop_time_list,
                                                  pred_time=pred_time), caseid_list), total=number_of_patients, desc='MEKF_P'))
            metrics_MEKF_P = pd.concat(res)
            metrics_MEKF_P.to_csv('data/metrics_vital_MEKF_P.csv')
    if bool_data[1]:
        try:
            metrics_MHE = pd.read_csv('data/metrics_vital_MHE_std.csv', index_col=0)
        except FileNotFoundError:
            with mp.Pool(mp.cpu_count()) as pool:
                res = list(tqdm(pool.imap(partial(one_line, path=mhe_path, stop_time_list=stop_time_list,
                                                  pred_time=pred_time), caseid_list), total=number_of_patients, desc='MHE'))
            metrics_MHE = pd.concat(res)
            metrics_MHE.to_csv('data/metrics_vital_MHE_std.csv')
    if bool_data[2]:
        try:
            metrics_plus = pd.read_csv('data/metrics_vital_plus.csv', index_col=0)
        except FileNotFoundError:
            with mp.Pool(mp.cpu_count()) as pool:
                res = list(tqdm(pool.imap(partial(one_line, path=plus_path, stop_time_list=stop_time_list,
                                                  pred_time=pred_time), caseid_list), total=number_of_patients, desc='Plus'))
            metrics_plus = pd.concat(res)
            metrics_plus.to_csv('data/metrics_vital_plus.csv')

    # %% Plot the results

    # plot a comparison between the metrics for the different stop time on the same figure
    stop_time_list = [el/60 for el in stop_time_list]
    if bool_data[0]:
        mean_MEKF_P = metrics_MEKF_P.mean()
        std_MEKF_P = metrics_MEKF_P.std()
    if bool_data[1]:
        mean_MHE = metrics_MHE.mean()
        std_MHE = metrics_MHE.std()
    if bool_data[2]:
        mean_plus = metrics_plus.mean()
        std_plus = metrics_plus.std()

    plt.figure()
    transparency = 0.3
    # plt.fill_between(stop_time_list, mean_MEKF_N-std_MEKF_N,
    #                  mean_MEKF_N+std_MEKF_N, alpha=transparency,
    #                  facecolor=mcolors.TABLEAU_COLORS['tab:blue'])  # , hatch="\\\\")
    if bool_data[0]:
        plt.fill_between(stop_time_list, mean_MEKF_P-std_MEKF_P,
                         mean_MEKF_P+std_MEKF_P, alpha=transparency,
                         facecolor=mcolors.TABLEAU_COLORS['tab:blue'])
    if bool_data[1]:
        plt.fill_between(stop_time_list, mean_MHE-std_MHE, mean_MHE+std_MHE,
                         alpha=transparency, facecolor=mcolors.TABLEAU_COLORS['tab:orange'])  # , hatch="////")
    if bool_data[2]:
        plt.fill_between(stop_time_list, mean_plus-std_plus, mean_plus+std_plus,
                         alpha=transparency, facecolor=mcolors.TABLEAU_COLORS['tab:green'])

    # plt.plot(stop_time_list, mean_MEKF_N, label='MEKF_Narendra', color=mcolors.TABLEAU_COLORS['tab:blue'])
    if bool_data[0]:
        plt.plot(stop_time_list, mean_MEKF_P, label='MEKF', color=mcolors.TABLEAU_COLORS['tab:blue'])
    if bool_data[1]:
        plt.plot(stop_time_list, mean_MHE, label='MHE', color=mcolors.TABLEAU_COLORS['tab:orange'])
    if bool_data[2]:
        plt.plot(stop_time_list, mean_plus, label='MM', color=mcolors.TABLEAU_COLORS['tab:green'])

    # add the standard deviation to the plot

    plt.legend()
    plt.ylabel('metrics')
    plt.xlabel('stop time (min)')
    plt.grid()
    savepath = "figures/stats_comp_vital.pdf"
    plt.savefig(savepath, bbox_inches='tight', format='pdf')
    plt.show()

    # plot the maximum value of the metrics
    plt.figure()
    if bool_data[0]:
        plt.plot(stop_time_list, metrics_MEKF_P.max(), label='MEKF')
    if bool_data[1]:
        plt.plot(stop_time_list, metrics_MHE.max(), label='MHE')
    if bool_data[2]:
        plt.plot(stop_time_list, metrics_plus.max(), label='MM')

    plt.legend()
    plt.ylabel('metrics')
    plt.xlabel('stop time (min)')
    plt.grid()
    savepath = "figures/stats_max_vital.pdf"
    plt.savefig(savepath, bbox_inches='tight', format='pdf')
    plt.show()
