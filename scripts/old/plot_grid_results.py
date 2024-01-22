import matplotlib.patches as patches
from matplotlib.path import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import optuna
import matplotlib.colors as colors
import matplotlib.cm as cmx
# plot config
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

font = {'family': 'serif',
        'weight': 'normal',
        'size': 16}


plt.rc('text', usetex=True)
matplotlib.rc('font', **font)


method = 'petri'
if method == 'mhe':
    study_mhe = optuna.load_study(study_name="mhe_final_2", storage="sqlite:///data/mhe.db")
    print(study_mhe.best_params)
    grid_results = study_mhe.trials_dataframe()

elif method == 'petri':
    study_petri = optuna.load_study(study_name="petri_final_3", storage="sqlite:///data/petri_2.db")
    print(study_petri.best_params)
    grid_results = study_petri.trials_dataframe()


if method == 'mhe':
    grid_results['$log_{10}(R)$'] = np.log10(grid_results['params_R'])
    grid_results["$log_{10}(\\beta)$"] = np.log10(grid_results['params_eta'])
    grid_results['$N_{mhe}$'] = grid_results['params_N_mhe']
elif method == 'petri':
    grid_results['$log_{10}(R)$'] = np.log10(grid_results['params_R'])
    grid_results['$log_{10}(\lambda_2)$'] = np.log10(grid_results['params_lambda_2'])
    # grid_results['$log_{10}(\\nu)$'] = np.log10(grid_results['nu'])
    grid_results['$\epsilon$'] = grid_results['params_epsilon']
    grid_results['$log_{10}(Q)$'] = np.log10(grid_results['params_Q'])
elif method == 'narendra':
    grid_results['$log_{10}(R)$'] = np.log10(grid_results['R'])
    grid_results['$log_{10}(\lambda)$'] = np.log10(grid_results['lambda'])
    grid_results['$\epsilon$'] = grid_results['epsilon']
    grid_results['$N$'] = grid_results['N']
# %% Plot results

fig, host = plt.subplots(figsize=(10, 5))
grid_results = grid_results.dropna()
# create some dummy data
if method == 'mhe':
    ynames = ['$log_{10}(R)$', '$log_{10}(\\beta)$', '$N_{mhe}$', 'value']
elif method == 'petri':
    ynames = ['$log_{10}(R)$', '$log_{10}(\lambda_2)$', '$log_{10}(Q)$', 'value']
elif method == 'narendra':
    ynames = ['$log_{10}(R)$', '$log_{10}(\lambda)$', '$\epsilon$', '$N$', 'value']

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

cmap = plt.get_cmap('nipy_spectral').copy()
begin = 0.5
end = 0.8
cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=begin, b=end),
    cmap(np.linspace(begin, end, 255, endpoint=True)))
min = grid_results.value.min()
max = grid_results.value.max()
cNorm = colors.Normalize(vmin=min, vmax=max)
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cmap)

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
    if grid_results.value.iloc[j] == min:
        patch = patches.PathPatch(path, facecolor='none', lw=2, edgecolor='#263dffff')
        patch_min = patch
    else:

        color = scalarMap.to_rgba(grid_results.value.iloc[j])
        patch = patches.PathPatch(path, facecolor='none', lw=1, edgecolor=color, alpha=1)
        host.add_patch(patch)
host.add_patch(patch_min)
# for i, ax in enumerate(axes):
#     if i < len(ys[0])-1:
#         ax.yaxis.set_ticks(np.round(np.unique(ys[:, i]), 1))
#         ax.yaxis.set_ticklabels(np.round(np.unique(ys[:, i]), 1))
fig.colorbar(scalarMap, ax=ax, orientation="vertical", pad=0.1, aspect=40, ticks=[])
plt.tight_layout()
plt.savefig(f'figures/parallel_coordinates_{method}.pdf', bbox_inches='tight', format='pdf')
plt.show()


# for i in range(len(ynames)-1):
#     plt.subplot(len(ynames)-1, 1, i+1)
#     plt.plot(grid_results[ynames[i]], grid_results['value'], 'o', alpha=0.5)
#     plt.xlabel(ynames[i])
#     plt.grid()
#     # plt.ylabel('objective function')
# plt.tight_layout()

# plt.show()
# plt.savefig(f'figures/scatter_{method}.pdf', bbox_inches='tight', format='pdf')
