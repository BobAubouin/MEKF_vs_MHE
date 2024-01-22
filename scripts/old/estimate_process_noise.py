import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import multiprocessing as mp
import python_anesthesia_simulator as pas
from tqdm import tqdm


def process_error_from_case(case_id: int):
    # load the data
    Patient_info = pd.read_csv(
        './data/simulations/parameters.csv').iloc[case_id][['age', 'height', 'weight', 'gender']].values
    Patient_simu = pd.read_csv(f'./data/simulations/simu_{case_id}.csv')
    Time = Patient_simu['Time'].to_numpy()
    BIS = Patient_simu['BIS'].to_numpy()
    U_propo = Patient_simu['u_propo'].to_numpy()
    U_remi = Patient_simu['u_remi'].to_numpy()
    xp = Patient_simu[[f"x_propo_{i}" for i in range(1, 5)]].to_numpy()
    xr = Patient_simu[[f"x_remi_{i}" for i in range(1, 5)]].to_numpy()

    # define the model
    model = 'Eleveld'
    simulator = pas.Patient(Patient_info, model_propo=model, model_remi=model, ts=2)
    A_p = simulator.propo_pk.discretize_sys.A[:4, :4]
    A_r = simulator.remi_pk.discretize_sys.A[:4, :4]
    B_p = simulator.propo_pk.discretize_sys.B[:4]
    B_r = simulator.remi_pk.discretize_sys.B[:4]

    error_propo = np.zeros((len(Time), 4))
    error_remi = np.zeros((len(Time), 4))
    for i in range(len(Time)-1):
        error_propo[i] = xp[i+1] - A_p@xp[i] - (B_p*U_propo[i])[:, 0]
        error_remi[i] = xr[i+1] - A_r@xr[i] - (B_r*U_remi[i])[:, 0]
    return error_propo, error_remi


# get the error for each case and concatenate them
for i in tqdm(range(500)):
    error_propo, error_remi = process_error_from_case(i)
    if i == 0:
        error_propo_all = error_propo
        error_remi_all = error_remi
    else:
        error_propo_all = np.concatenate((error_propo_all, error_propo), axis=0)
        error_remi_all = np.concatenate((error_remi_all, error_remi), axis=0)

# compute the covariance matrix
cov_propo = np.cov(error_propo_all.T)
cov_remi = np.cov(error_remi_all.T)

# save the results
np.save('data/cov_propo.npy', cov_propo)
np.save('data/cov_remi.npy', cov_remi)

# plot the results
plt.figure()
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.hist(error_propo_all[:, i])
    plt.hist(error_remi_all[:, i], alpha=0.5)
    plt.legend(['propo', 'remi'])
    plt.title(f'error {i+1}')
plt.show()

plt.figure()
for i in range(4):
    plt.subplot(2, 2, i+1)
    plt.plot(error_propo_all[:, i])
    plt.plot(error_remi_all[:, i], alpha=0.5)
    plt.legend(['propo', 'remi'])
    plt.title(f'error {i+1}')
plt.show()

# plot the covariance matrix
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(cov_propo)
plt.colorbar()
plt.title('propo')
plt.subplot(1, 2, 2)
plt.imshow(cov_remi)
plt.colorbar()
plt.title('remi')
plt.show()

# get Noise data fro mthe simulator
output_noise = pas.Patient([50, 170, 77, 0], model_propo='Eleveld', model_remi='Eleveld', ts=2).bis_noise

R = np.cov(output_noise.T)
print(R)
np.save('data/R.npy', R)

plt.figure()
plt.hist(output_noise)
plt.show()
