import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv(r"numsim.csv")
Patient = data.iloc[:, 0]
mean_Emax = data.iloc[:, 1]
std_Emax = data.iloc[:, 2]
mean_C50p = data.iloc[:, 3]
std_C50p = data.iloc[:, 4]
mean_C50r = data.iloc[:, 5]
std_C50r = data.iloc[:, 6]
mean_Gamma = data.iloc[:, 7]
std_Gamma = data.iloc[:, 8]
mean_Beta = data.iloc[:, 9]
std_Beta = data.iloc[:, 10]

plt.figure()
plt.bar(range(len(Patient)), mean_Emax, yerr=std_Emax, align='center', alpha=0.5, ecolor='green', capsize=2)
plt.xticks(range(len(Patient)), Patient, rotation=90, fontsize=5)
plt.xlabel('Patient')
plt.ylabel('Mean')
plt.title('Emax')

plt.figure()
plt.bar(range(len(Patient)), mean_C50p, yerr=std_C50p, align='center', alpha=0.5, ecolor='green', capsize=2)
plt.xticks(range(len(Patient)), Patient, rotation=90, fontsize=5)
plt.xlabel('Patient')
plt.ylabel('Mean')
plt.title('C50p')

plt.figure()
plt.bar(range(len(Patient)), mean_C50r, yerr=std_C50p, align='center', alpha=0.5, ecolor='green', capsize=2)
plt.xticks(range(len(Patient)), Patient, rotation=90, fontsize=5)
plt.xlabel('Patient')
plt.ylabel('Mean')
plt.title('C50r')

plt.figure()
plt.bar(range(len(Patient)), mean_Gamma, yerr=std_Gamma, align='center', alpha=0.5, ecolor='green', capsize=2)
plt.xticks(range(len(Patient)), Patient, rotation=90, fontsize=5)
plt.xlabel('Patient')
plt.ylabel('Mean')
plt.title('Gamma')

plt.figure()
plt.bar(range(len(Patient)), mean_Beta, yerr=std_Beta, align='center', alpha=0.5, ecolor='green', capsize=2)
plt.xticks(range(len(Patient)), Patient, rotation=90, fontsize=5)
plt.xlabel('Patient')
plt.ylabel('Mean')
plt.title('Beta')

fig, axs = plt.subplots(1, 5)
axs[0].hist(mean_Emax, bins=20)
axs[0].set_xlabel('Mean')
axs[0].set_ylabel('Frequency')
axs[0].set_title('Emax')

axs[1].hist(mean_C50p, bins=20)
axs[1].set_xlabel('Mean')
axs[1].set_ylabel('Frequency')
axs[1].set_title('C50p')

axs[2].hist(mean_C50r, bins=20)
axs[2].set_xlabel('Mean')
axs[2].set_ylabel('Frequency')
axs[2].set_title('C50r')

axs[3].hist(mean_Gamma, bins=20)
axs[3].set_xlabel('Mean')
axs[3].set_ylabel('Frequency')
axs[3].set_title('Gamma')

axs[4].hist(mean_Beta, bins=20)
axs[4].set_xlabel('Mean')
axs[4].set_ylabel('Frequency')
axs[4].set_title('Beta')


plt.tight_layout()
plt.show()