import MHE_Class_SimData as MHE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from multiprocessing import Pool
import time

def process_case(Patient):
    Patient = int(Patient)
    mhe = MHE.MHE(nb_Patient=Patient)
    Par = mhe.BIS_MHE()[1]
    C50p, C50r, Gamma = Par

    C50p = C50p[len(C50p)//3:]
    C50r = C50r[len(C50r)//3:]
    Gamma = Gamma[len(Gamma)//3:]

    mean_C50p = np.mean(C50p)
    std_C50p = np.std(C50p)
    mean_C50r = np.mean(C50r)
    std_C50r = np.std(C50r)
    mean_Gamma = np.mean(Gamma)
    std_Gamma = np.std(Gamma)

    data = {
        'Patient': Patient,
        'mean(C50p)': mean_C50p,
        'std(C50p)': std_C50p,
        'mean(C50r)': mean_C50r,
        'std(C50r)': std_C50r,
        'mean(Gamma)': mean_Gamma,
        'std(Gamma)': std_Gamma,
    }

    return data

start_time = time.perf_counter()

if __name__ == '__main__':
    Patient = np.array([range(0, 999)]).flatten()

    Patient = Patient.astype(int)

    with Pool(processes=8) as pool:
        results = pool.map(process_case, Patient[0:1000])

    numsim = pd.DataFrame(results)
    numsim.to_csv('numsim.csv', index=False)

    for param in ['C50p', 'C50r', 'Gamma']:
        means = numsim[f'mean({param})']
        stds = numsim[f'std({param})']
        plt.plot(means, label=f'mean({param})')
        plt.fill_between(range(len(means)), means - stds, means + stds, alpha=0.2)
    plt.legend()

    end_time = time.perf_counter()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {round(elapsed_time/60, 2)} minutes")
    
    plt.show()