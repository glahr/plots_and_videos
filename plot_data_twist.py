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

params['data_to_plot'] = 'EE_twist_d'
EE_twist_d = dh.get_data(params, axis=Z_AXIS)

params['data_to_plot'] = 'EE_twist'
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

params['data_to_plot'] = 'EE_twist_d'
EE_twist_d_vic = dh.get_data(params, axis=Z_AXIS)

params['data_to_plot'] = 'EE_twist'
EE_twist_vic = dh.get_data(params, axis=Z_AXIS)
# END LOAD DATA

xlim_plot = [time_vic[0], time_vic[-1]]
ylim_plot = [-1.5, 0.5]
labels=['$\dot{x}_{KMP}$', '$\dot{x}_{KMP+VIC}$', '$\dot{x}_d$']
ylabel = '$\dot{x}~[m/s]$'
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

labels=['$\dot{x}_{KMP}$', '$\dot{x}_{KMP+VIC}$', '$\dot{x}_{d}$']
ax.legend(labels=labels, borderaxespad=0.1,
          handlelength=0.8, fontsize=LEGEND_SIZE)

plt.show()



