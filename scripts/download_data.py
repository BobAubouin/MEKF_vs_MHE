import pandas as pd
import numpy as np
import vitaldb as vdb
import os
import multiprocessing as mp
from tqdm import tqdm

main_signals = ['BIS/BIS', 'Orchestra/PPF20_RATE', 'Orchestra/RFTN20_RATE']
sampling_time = 2
data_length = 15*60 // sampling_time
save_path = 'data/vital'
if not os.path.exists(save_path):
    os.makedirs(save_path)

caseid_list = list(np.loadtxt('./scripts/caseid_list.txt', dtype=int))
caseid_list.remove(104)
caseid_list.remove(859)


def download_case(integer: int):
    case_id = caseid_list[integer]
    if os.path.exists(os.path.join(save_path, f'case_{case_id}.csv')):
        return

    df = vdb.VitalFile(int(case_id), track_names=main_signals).to_pandas(
        track_names=main_signals, interval=sampling_time)
    df = df.dropna()

    for i in range(len(df)):
        if df[main_signals[2]].iloc[i] > 1 or df[main_signals[1]].iloc[i] > 1:
            start = i
            break

    df = df.iloc[start:start+data_length]
    df = df.reset_index(drop=True)
    df[main_signals[0]] = df[main_signals[0]].replace(0, np.nan).fillna(method='ffill').fillna(method='bfill')

    df.to_csv(os.path.join(save_path, f'case_{case_id}.csv'), index=False)
    return


# read case list


N_patient = len(caseid_list)

with mp.Pool(processes=mp.cpu_count()) as pool:
    for _ in tqdm(pool.imap_unordered(download_case, range(N_patient)), total=N_patient):
        pass
