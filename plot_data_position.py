import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
from data_plot_class import DataPlotHelper

path_folder = 'data/icra2023/2-exp-ImpLoop3-NoAdaptation-PARTIAL-RESULTS-DONT-EDIT/'
data_info = pd.read_csv(path_folder+'data_info.csv')

dh = DataPlotHelper(path_folder)

# LOAD DATA
Z_AXIS = 2
STEP_XAXIS=0.25
COLORS = ['Green', 'Blue']
TRIALS_IDXS = [1, 2, 3]
HEIGHTS = [27]
LEGEND_SIZE = 14

params = {
    'empty': False,
    'vic': False,
    'color': 'Blue',
    'trial_idx': 1,
    'height': 27,
    'impedance_loop': 30,
    'online_adaptation': False,
    'i_initial': 0,
    'i_final': -1,
    'data_to_plot': 'EE_twist_d',
    }

params['color'] = 'Blue'
params['height'] = 27
params['trial_idx'] = 1
params['vic'] = False

idx_start = dh.get_idx_from_file(params, data_info, idx_name='idx_start')
idx_end = dh.get_idx_from_file(params, data_info, idx_name='idx_end')
params['i_initial'] = idx_start
params['i_final'] = idx_end

params['data_to_plot'] = 'time'
time = dh.get_data(params, axis=0)
time = time - time[0]

params['data_to_plot'] = 'EE_position_d'
EE_twist_d = dh.get_data(params, axis=Z_AXIS)

params['data_to_plot'] = 'EE_position'
EE_twist = dh.get_data(params, axis=Z_AXIS)

# -----------------

params['color'] = 'Blue'
params['height'] = 27
params['trial_idx'] = 1
params['vic'] = True

idx_start = dh.get_idx_from_file(params, data_info, idx_name='idx_start')
idx_end = dh.get_idx_from_file(params, data_info, idx_name='idx_end')
params['i_initial'] = idx_start+1
params['i_final'] = idx_end+1

params['data_to_plot'] = 'time'
time_vic = dh.get_data(params, axis=0)
time_vic = time_vic - time_vic[0]

params['data_to_plot'] = 'EE_position_d'
EE_twist_d_vic = dh.get_data(params, axis=Z_AXIS)

params['data_to_plot'] = 'EE_position'
EE_twist_vic = dh.get_data(params, axis=Z_AXIS)
# END LOAD DATA

xlim_plot = [time_vic[0], time_vic[-1]]
ylim_plot = [0, .6]
labels=['$x_{KMP}$', '$x_{KMP+VIC}$', '$x_d$']
ylabel = '$x~[m]$'
xlabel = '$time~[s]$'
xticks =      [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
xtickslabels = ['$0$', '$0.25$', '$0.5$', '$0.75$', '$1.0$', '$1.25$', '$1.5$']
yticks = None
ytickslabels = None
fig_size = [10, 4]  # width, height

fig, ax = dh.set_axis(xlim_plot=xlim_plot, xlabel=xlabel, xticks=xticks, xtickslabels=xtickslabels,
                      ylim_plot=ylim_plot, ylabel=ylabel, yticks=yticks, ytickslabels=ytickslabels,
                      fig_size=fig_size)

fig, ax = dh.plot_single(time=time, data=EE_twist, fig=fig, ax=ax)
fig, ax = dh.plot_single(time=time_vic, data=EE_twist_vic, fig=fig, ax=ax)
fig, ax = dh.plot_single(time=time, data=EE_twist_d, fig=fig, ax=ax, color_shape='k--')

# fig, ax = dh.set_axis(xlim_plot=xlim_plot, xlabel=xlabel, xticks=xticks, xtickslabels=xtickslabels,
#                       ylim_plot=ylim_plot, ylabel=ylabel, yticks=yticks, ytickslabels=ytickslabels,
#                       fig_size=fig_size)
# fig, ax = dh.plot_single(time=time_vic, data=EE_twist_d_vic, fig=fig, ax=ax, color_shape='g--')

labels=['$x_{KMP}$', '$x_{KMP+VIC}$', '$x_{d}$']
ax.legend(labels=labels, borderaxespad=0.1,
          handlelength=0.8, fontsize=LEGEND_SIZE)

plt.show()


# ------------------------------ MEAN
params['color'] = 'Blue'
params['height'] = 27
params['trial_idx'] = 1
params['vic'] = False

ee_pose = np.zeros((1500, 3))

for i in [1, 2, 3]:
    params['trial_idx'] = i

    if i == 1:
        offset = -14
    if i == 2:
        offset = 0
    if i==3:
        offset = 20

    idx_start = dh.get_idx_from_file(params, data_info, idx_name='idx_start')
    idx_end = dh.get_idx_from_file(params, data_info, idx_name='idx_end')
    params['i_initial'] = idx_start + offset
    params['i_final'] = idx_end + offset

    params['data_to_plot'] = 'time'
    time = dh.get_data(params, axis=0)
    time = time - time[0]

    params['data_to_plot'] = 'EE_position'
    ee_pose[:,i-1] = dh.get_data(params, axis=Z_AXIS)


params['color'] = 'Blue'
params['height'] = 27
params['trial_idx'] = 1
params['vic'] = True

ee_pose_vic = np.zeros((1500, 3))

vic_offset = 24

for i in [1, 2, 3]:
    params['trial_idx'] = i

    offset = 6 if i==2 else 0

    idx_start = dh.get_idx_from_file(params, data_info, idx_name='idx_start')
    idx_end = dh.get_idx_from_file(params, data_info, idx_name='idx_end')
    params['i_initial'] = idx_start + offset + vic_offset
    params['i_final'] = idx_end + offset + vic_offset

    params['data_to_plot'] = 'time'
    time = dh.get_data(params, axis=0)
    time = time - time[0]

    params['data_to_plot'] = 'EE_position'
    ee_pose_vic[:,i-1] = dh.get_data(params, axis=Z_AXIS)

EE = np.mean(ee_pose, axis=1)
EE_vic = np.mean(ee_pose_vic, axis=1)


file_name = path_folder + 'empty.mat'
idx_start = dh.get_idx_from_file(params, data_info, idx_name='idx_start', file_name=file_name)
idx_end = dh.get_idx_from_file(params, data_info, idx_name='idx_end', file_name=file_name)
params['i_initial'] = idx_start
params['i_final'] = idx_end
params['data_to_plot'] = 'time'
time_empty = dh.get_data(params, axis=0)
time_empty = time_empty - time_empty[0]
params['data_to_plot'] = 'EE_position'
EE_empty = dh.get_data(params, Z_AXIS, file_name)



fig, ax = dh.set_axis(xlim_plot=xlim_plot, xlabel=xlabel, xticks=xticks, xtickslabels=xtickslabels,
                      ylim_plot=ylim_plot, ylabel=ylabel, yticks=yticks, ytickslabels=ytickslabels,
                      fig_size=fig_size)
fig, ax = dh.plot_single(time=time, data=EE, fig=fig, ax=ax)
fig, ax = dh.plot_single(time=time, data=EE_vic, fig=fig, ax=ax)
fig, ax = dh.plot_single(time=time, data=EE_empty, fig=fig, ax=ax)
fig, ax = dh.plot_single(time=time, data=EE_twist_d, fig=fig, ax=ax, color_shape='k--')


# fig, ax = dh.plot_single(time=time_vic, data=EE_twist_d_vic, fig=fig, ax=ax, color_shape='g--')

labels=['$\overline{x}_{KMP}$', '$\overline{x}_{KMP+VIC}$', '$x_{empty}$', '$x_{d}$']
ax.legend(labels=labels, borderaxespad=0.1,
          handlelength=0.8, fontsize=LEGEND_SIZE)

plt.show()
# fig.savefig('position_average_comparison.png')