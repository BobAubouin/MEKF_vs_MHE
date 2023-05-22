import MHE_Class as MHE
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

numsim = []
for Patient in range(0, 20):
    mhe = MHE.MHE(nb_Patient=Patient)
    X_estimate = mhe.BIS_MHE()[1]
    Emax = X_estimate[:, 8]
    C50p = X_estimate[:, 9]
    C50r = X_estimate[:, 10]
    Gamma = X_estimate[:, 11]
    Beta = X_estimate[:, 12]

    Emax = Emax[len(Emax)//3:]
    C50p = C50p[len(C50p)//3:]
    C50r = C50r[len(C50r)//3:]
    Gamma = Gamma[len(Gamma)//3:]
    Beta = Beta[len(Beta)//3:]

    mean_Emax = np.mean(Emax)
    std_Emax = np.std(Emax)
    mean_C50p = np.mean(C50p)
    std_C50p = np.std(C50p)
    mean_C50r = np.mean(C50r)
    std_C50r = np.std(C50r)
    mean_Gamma = np.mean(Gamma)
    std_Gamma = np.std(Gamma)
    mean_Beta = np.mean(Beta)
    std_Beta = np.std(Beta)

    data = {
        'Patient': Patient,
        'mean(Emax)': mean_Emax,
        'std(Emax)': std_Emax,
        'mean(C50p)': mean_C50p,
        'std(C50p)': std_C50p,
        'mean(C50r)': mean_C50r,
        'std(C50r)': std_C50r,
        'mean(Gamma)': mean_Gamma,
        'std(Gamma)': std_Gamma,
        'mean(Beta)': mean_Beta,
        'std(Beta)': std_Beta
    }

    numsim.append(data)

numsim = pd.DataFrame(numsim)
numsim.to_csv('numsim.csv', index = False)

for param in ['Emax', 'C50p', 'C50r', 'Gamma', 'Beta']:
    means = numsim[f'mean({param})']
    stds = numsim[f'std({param})']
    plt.plot(means, label=f'mean({param})')
    plt.fill_between(range(len(means)), means - stds, means + stds, alpha=0.2)

plt.legend()
plt.show()