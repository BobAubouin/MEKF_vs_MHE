import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import scipy
import matplotlib

# plot config
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

font = {'family': 'serif',
        'weight': 'normal',
        'size': 10}


plt.rc('text', usetex=True)
matplotlib.rc('font', **font)
param = pd.DataFrame(columns=['c50p', 'c50r', 'gamma'])

path = './data/vital/mhe/'

i = 0
for filename in os.listdir(path):
    if filename.startswith('parameters'):
        case = pd.read_csv(path + filename)
        # removed columns Unnamed:0
        param.loc[i] = [0, 0, 0]
        param['c50p'].loc[i] = case['0'].iloc[-1]
        param['c50r'].loc[i] = case['1'].iloc[-1]
        param['gamma'].loc[i] = case['2'].iloc[-1]
        i += 1

# plot the distribution for each column

mean_c50p_init = 4.47
mean_c50r_init = 19.3
mean_gamma_init = 1.13
cv_c50p_init = 0.182
cv_c50r_init = 0.888
cv_gamma_init = 0.304
w_c50p_init = np.sqrt(np.log(1+cv_c50p_init**2))
w_c50r_init = np.sqrt(np.log(1+cv_c50r_init**2))
w_gamma_init = np.sqrt(np.log(1+cv_gamma_init**2))

# define probability
mean_c50p = 5
mean_c50r = 27
mean_gamma = 2.9
cv_c50p = 0.182
cv_c50r = 0.2
cv_gamma = 0.304
w_c50p = np.sqrt(np.log(1+cv_c50p**2))
w_c50r = np.sqrt(np.log(1+cv_c50r**2))
w_gamma = np.sqrt(np.log(1+cv_gamma**2))


s, loc, scale = scipy.stats.lognorm.fit(data=param['c50p'].to_numpy())
mean_c50p = np.exp(loc + scale**2/2)
std_c50p = np.sqrt((np.exp(scale**2)-1)*np.exp(2*loc+scale**2))
print(f'c50p: mean = {mean_c50p}, std = {std_c50p}')
print(s)

c50p_normal = scipy.stats.lognorm(scale=scale, s=s, loc=loc)
c50p_normal_init = scipy.stats.lognorm(scale=mean_c50p_init, s=w_c50p_init)
x_c50p = np.linspace(2, 10, 100)
y_c50p = c50p_normal.pdf(x_c50p)
y_c50p_init = c50p_normal_init.pdf(x_c50p)

s, loc, scale = scipy.stats.lognorm.fit(data=param['c50r'].to_numpy())
mean_c50r = np.exp(loc + scale**2/2)
std_c50r = np.sqrt((np.exp(scale**2)-1)*np.exp(2*loc+scale**2))
print(f'c50r: mean = {mean_c50r}, std = {std_c50r}')
print(s)

c50r_normal = scipy.stats.lognorm(scale=scale, s=s, loc=loc)
c50r_normal_init = scipy.stats.lognorm(scale=mean_c50r_init, s=w_c50r_init)
x_c50r = np.linspace(0, 50, 100)
y_c50r = c50r_normal.pdf(x_c50r)
y_c50r_init = c50r_normal_init.pdf(x_c50r)

s, loc, scale = scipy.stats.lognorm.fit(data=param['gamma'].to_numpy())
mean_gamma = np.exp(loc + scale**2/2)
std_gamma = np.sqrt((np.exp(scale**2)-1)*np.exp(2*loc+scale**2))
print(f'gamma: mean = {mean_gamma}, std = {std_gamma}')
print(s)

gamma_normal = scipy.stats.lognorm(scale=scale, s=s, loc=loc)
gamma_normal_init = scipy.stats.lognorm(scale=mean_gamma_init, s=w_gamma_init)
x_gamma = np.linspace(0, 5, 100)
y_gamma = gamma_normal.pdf(x_gamma)
y_gamma_init = gamma_normal_init.pdf(x_gamma)


plt.figure()
plt.subplot(3, 1, 1)
plt.hist(param['c50p'], bins=20, density=True, label='MHE')
plt.plot(x_c50p, y_c50p_init, color='#ff7f0eff', label='Bouillon')
plt.plot(x_c50p, y_c50p, color='#2ca02cff', label='New')
plt.ylabel('C50p')
plt.legend()
plt.subplot(3, 1, 2)
plt.hist(param['c50r'], bins=20, density=True, label='MHE')
plt.plot(x_c50r, y_c50r_init, color='#ff7f0eff', label='Bouillon')
plt.plot(x_c50r, y_c50r, color='#2ca02cff', label='New')
plt.ylabel('C50r')
plt.legend()

plt.subplot(3, 1, 3)
plt.hist(param['gamma'], bins=20, density=True, label='MHE')
plt.plot(x_gamma, y_gamma_init, color='#ff7f0eff', label='Bouillon')
plt.plot(x_gamma, y_gamma, color='#2ca02cff', label='New')
plt.ylabel('$\gamma$')
plt.legend()
plt.savefig('figures/param_dist.pdf', bbox_inches='tight', format='pdf')
plt.show()
