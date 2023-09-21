import matplotlib.patches as patches
from matplotlib.path import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
# plot config
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

font = {'family': 'serif',
        'weight': 'normal',
        'size': 16}


plt.rc('text', usetex=True)
matplotlib.rc('font', **font)


method = 'petri'

grid_results = pd.read_csv(f'data/grid_search_{method}.csv', index_col=0)
grid_results.sort_values('objective_function', inplace=True)
print(grid_results.head(10))

if method == 'mhe':
    grid_results['$log_{10}(R)$'] = np.log10(grid_results['R'])
    grid_results["$log_{10}(\gamma)$"] = np.log10(grid_results['theta_1'])
    grid_results['$N_{mhe}$'] = grid_results['N_mhe']
elif method == 'petri':
    grid_results['$log_{10}(R)$'] = np.log10(grid_results['R'])
    grid_results['$log_{10}(\lambda_2)$'] = np.log10(grid_results['lambda_2'])
    grid_results['$log_{10}(\\nu)$'] = np.log10(grid_results['nu'])
    grid_results['$\epsilon$'] = grid_results['epsilon']

# %% Plot results

fig, host = plt.subplots(figsize=(10, 5))
grid_results = grid_results.dropna()
# create some dummy data
if method == 'mhe':
    ynames = ['$log_{10}(R)$', '$log_{10}(\gamma)$', '$N_{mhe}$', 'objective_function']
elif method == 'petri':
    ynames = ['$log_{10}(R)$', '$log_{10}(\lambda_2)$', '$log_{10}(\\nu)$', '$\epsilon$', 'objective_function']


# organize the data
ys = grid_results[ynames].to_numpy()
ymins = ys.min(axis=0)
ymaxs = ys.max(axis=0)
dys = ymaxs - ymins
ymins -= dys * 0.05  # add 5% padding below and above
ymaxs += dys * 0.05
dys = ymaxs - ymins

# transform all data to be compatible with the main axis
zs = np.zeros_like(ys)
zs[:, 0] = ys[:, 0]
zs[:, 1:] = (ys[:, 1:] - ymins[1:]) / dys[1:] * dys[0] + ymins[0]


axes = [host] + [host.twinx() for i in range(ys.shape[1] - 1)]
for i, ax in enumerate(axes):
    ax.set_ylim(ymins[i], ymaxs[i])
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    if ax != host:
        ax.spines['left'].set_visible(False)
        ax.yaxis.set_ticks_position('right')
        ax.spines["right"].set_position(("axes", i / (ys.shape[1] - 1)))
    # if i == 0:
    #     ax.set_yscale('log')

host.set_xlim(0, ys.shape[1] - 1)
host.set_xticks(range(ys.shape[1]))
host.set_xticklabels(ynames, fontsize=14)
host.tick_params(axis='x', which='major', pad=7)
host.spines['right'].set_visible(False)
host.xaxis.tick_top()
# host.set_title('Parallel Coordinates Plot', fontsize=18)

up = np.array([248, 123, 0])/255
down = np.array([0, 200, 14])/255
min = grid_results.objective_function.min()
max = grid_results.objective_function.max()
for j in range(len(ys)):
    # to just draw straight lines between the axes:
    # host.plot(range(ys.shape[1]), zs[j,:], c=colors[(category[j] - 1) % len(colors) ])

    # create bezier curves
    # for each axis, there will a control vertex at the point itself, one at 1/3rd towards the previous and one
    #   at one third towards the next axis; the first and last axis have one less control vertex
    # x-coordinate of the control vertices: at each integer (for the axes) and two inbetween
    # y-coordinate: repeat every point three times, except the first and last only twice
    verts = list(zip([x for x in np.linspace(0, len(ys) - 1, len(ys) * 3 - 2, endpoint=True)],
                     np.repeat(zs[j, :], 3)[1:-1]))
    # for x,y in verts: host.plot(x, y, 'go') # to show the control points of the beziers
    codes = [Path.MOVETO] + [Path.CURVE4 for _ in range(len(verts) - 1)]
    path = Path(verts, codes)
    alpha = (grid_results.objective_function.iloc[j] - min)/(max-min)
    if alpha == 0:
        patch = patches.PathPatch(path, facecolor='none', lw=2, edgecolor='#5d88f9ff')
        patch_min = patch
    else:
        patch = patches.PathPatch(path, facecolor='none', lw=1, edgecolor=tuple(up * alpha + down*(1-alpha)))
        host.add_patch(patch)
host.add_patch(patch_min)
plt.tight_layout()
plt.savefig(f'figures/parallel_coordinates_{method}.pdf', bbox_inches='tight', format='pdf')
plt.show()
