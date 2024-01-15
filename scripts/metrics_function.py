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
mekf_n_path = 'data/mekf_n/'
mekf_p_path = 'data/mekf_p/'
mhe_path = 'data/mhe_std/'
plus_path = 'data/plus/'

number_of_patients = 100

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
    patient_param = pd.read_csv(f'data/simulations/parameters.csv', index_col=0)
    patient_param = patient_param.loc[patient_id, ['age', 'height', 'weight', 'gender']].to_numpy()
    data_simu = pd.read_csv(f'data/simulations/simu_{patient_id}.csv', index_col=0)
    # select the data in the time window
    data_bool = (data_simu.Time >= stop_time) & (data_simu.Time < stop_time + pred_time)
    u_propo = data_simu.u_propo[data_bool].to_numpy()
    u_remi = data_simu.u_remi[data_bool].to_numpy()
    true_bis = data_simu.BIS[data_bool].to_numpy()
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
    flag = True
    for stop_time in stop_time_list:
        if plot:
            metrics.loc[i, stop_time], true_bis, bis_test = metrics_function(path, i, stop_time, pred_time, plot)
            x = np.linspace(stop_time, stop_time + pred_time, len(true_bis))/60
            plt.fill_between(x, true_bis, bis_test.to_numpy(), color='r', alpha=0.5)
            if flag:
                plt.plot(x, true_bis, 'b', label='measure')
                plt.plot(x, bis_test, 'r', label='open-loop simulation')
                flag = False
            else:
                plt.plot(x, true_bis, 'b')
                plt.plot(x, bis_test, 'r')

        else:
            metrics.loc[i, stop_time] = metrics_function(path, i, stop_time, pred_time)
    if plot:
        plt.ylabel('BIS')
        plt.xlabel('stop time (min)')
        plt.legend()
        plt.grid()
        plt.show()

    return metrics


if __name__ == '__main__':

    # try:
    #     metrics_MEKF_N = pd.read_csv('data/metrics_MEKF_N.csv', index_col=0)
    # except FileNotFoundError:
    #     with mp.Pool(mp.cpu_count()) as pool:
    #         res = list(tqdm(pool.imap(partial(one_line, path=mekf_n_path, stop_time_list=stop_time_list,
    #                                           pred_time=pred_time), range(number_of_patients)), total=number_of_patients, desc='MEKF_N'))
    #     metrics_MEKF_N = pd.concat(res)
    #     metrics_MEKF_N.to_csv('data/metrics_MEKF_N.csv')

    try:
        metrics_MEKF_P = pd.read_csv('data/metrics_MEKF_P.csv', index_col=0)
    except FileNotFoundError:
        with mp.Pool(mp.cpu_count()) as pool:
            res = list(tqdm(pool.imap(partial(one_line, path=mekf_p_path, stop_time_list=stop_time_list,
                                              pred_time=pred_time), range(number_of_patients)), total=number_of_patients, desc='MEKF_P'))
        metrics_MEKF_P = pd.concat(res)
        metrics_MEKF_P.to_csv('data/metrics_MEKF_P.csv')

    try:
        metrics_MHE = pd.read_csv('data/metrics_MHE_std.csv', index_col=0)
    except FileNotFoundError:
        with mp.Pool(mp.cpu_count()) as pool:
            res = list(tqdm(pool.imap(partial(one_line, path=mhe_path, stop_time_list=stop_time_list,
                                              pred_time=pred_time), range(number_of_patients)), total=number_of_patients, desc='MHE'))
        metrics_MHE = pd.concat(res)
        metrics_MHE.to_csv('data/metrics_MHE.csv')
    try:
        metrics_plus = pd.read_csv('data/metrics_plus.csv', index_col=0)
    except FileNotFoundError:
        with mp.Pool(mp.cpu_count()) as pool:
            res = list(tqdm(pool.imap(partial(one_line, path=plus_path, stop_time_list=stop_time_list,
                                              pred_time=pred_time), range(number_of_patients)), total=number_of_patients, desc='Plus'))
        metrics_plus = pd.concat(res)
        metrics_plus.to_csv('data/metrics_plus.csv')

    # %% Plot the results

    # plot a comparison between the metrics for the different stop time on the same figure
    stop_time_list = [el/60 for el in stop_time_list]
    # mean_MEKF_N = metrics_MEKF_N.mean()
    mean_MEKF_P = metrics_MEKF_P.mean()
    mean_MHE = metrics_MHE.mean()
    mean_plus = metrics_plus.mean()

    # std_MEKF_N = metrics_MEKF_N.std()
    std_MEKF_P = metrics_MEKF_P.std()
    std_MHE = metrics_MHE.std()
    std_plus = metrics_plus.std()

    plt.figure()
    transparency = 0.3
    # plt.fill_between(stop_time_list, mean_MEKF_N-std_MEKF_N,
    #                  mean_MEKF_N+std_MEKF_N, alpha=transparency,
    #                  facecolor=mcolors.TABLEAU_COLORS['tab:blue'])  # , hatch="\\\\")

    plt.fill_between(stop_time_list, mean_MEKF_P-std_MEKF_P,
                     mean_MEKF_P+std_MEKF_P, alpha=transparency,
                     facecolor=mcolors.TABLEAU_COLORS['tab:orange'])

    plt.fill_between(stop_time_list, mean_MHE-std_MHE, mean_MHE+std_MHE,
                     alpha=transparency, facecolor=mcolors.TABLEAU_COLORS['tab:blue'])  # , hatch="////")
    plt.fill_between(stop_time_list, mean_plus-std_plus, mean_plus+std_plus,
                     alpha=transparency, facecolor=mcolors.TABLEAU_COLORS['tab:green'])

    # plt.plot(stop_time_list, mean_MEKF_N, label='MEKF_Narendra', color=mcolors.TABLEAU_COLORS['tab:blue'])
    plt.plot(stop_time_list, mean_MHE, label='MHE', color=mcolors.TABLEAU_COLORS['tab:blue'])
    plt.plot(stop_time_list, mean_MEKF_P, label='MEKF', color=mcolors.TABLEAU_COLORS['tab:orange'])
    plt.plot(stop_time_list, mean_plus, label='MEKF', color=mcolors.TABLEAU_COLORS['tab:green'])

    # add the standard deviation to the plot

    plt.legend()
    plt.ylabel('metrics')
    plt.xlabel('stop time (min)')
    plt.grid()
    savepath = "figures/stats_comp.pdf"
    plt.savefig(savepath, bbox_inches='tight', format='pdf')
    plt.show()

    # plot the maximum value of the metrics
    plt.figure()
    # plt.plot(stop_time_list, metrics_MEKF_N.max(), label='MEKF_Narendra')
    plt.plot(stop_time_list, metrics_MHE.max(), label='MHE')
    plt.plot(stop_time_list, metrics_MEKF_P.max(), label='MEKF')
    plt.plot(stop_time_list, metrics_plus.max(), label='Plus')

    plt.legend()
    plt.ylabel('metrics')
    plt.xlabel('stop time (min)')
    plt.grid()
    savepath = "figures/stats_max.pdf"
    plt.savefig(savepath, bbox_inches='tight', format='pdf')
    plt.show()
