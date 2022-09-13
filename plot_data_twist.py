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
LEGEND_SIZE = 12
N_POINTS = 1000

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

xlim_plot = [time_vic[0], 500/1000]#N_POINTS/1000]
ylim_plot = [-1.3, 0.4]
labels=['$\dot{x}_{KMP}$', '$\dot{x}_{KMP+VIC}$', '$\dot{x}_d$']
ylabel = '$\dot{\\boldsymbol{x}}~[m/s]$'
xlabel = '$\\boldsymbol{t}~[s]$'
xticks =      [0, 0.25, 0.5, 0.75, 1.0]
xtickslabels = ['$0$', '$0.25$', '$0.5$', '$0.75$', '$1.0$']
# xticks =      [0, 0.1, 0.2, 0.3, 0.4, 0.5]#, 0.75, 1.0]
# xtickslabels = ['$0$', '$0.1$', '$0.2$', '$0.3$','$0.4$', '$0.5$']#, '$0.75$', '$1.0$']
yticks = None #[-1, -0.5, 0.0]
ytickslabels = None # ['$-1$', '$-0.5$', '$0$']
fig_size = [8, 3]  # width, height

# fig, ax = dh.set_axis(xlim_plot=xlim_plot, xlabel=xlabel, xticks=xticks, xtickslabels=xtickslabels,
#                       ylim_plot=ylim_plot, ylabel=ylabel, yticks=yticks, ytickslabels=ytickslabels,
#                       fig_size=fig_size)

# fig, ax = dh.plot_single(time=time, data=EE_twist, fig=fig, ax=ax)
# fig, ax = dh.plot_single(time=time_vic, data=EE_twist_vic, fig=fig, ax=ax)
# fig, ax = dh.plot_single(time=time, data=EE_twist_d, fig=fig, ax=ax, color_shape='k--')

# # fig, ax = dh.set_axis(xlim_plot=xlim_plot, xlabel=xlabel, xticks=xticks, xtickslabels=xtickslabels,
# #                       ylim_plot=ylim_plot, ylabel=ylabel, yticks=yticks, ytickslabels=ytickslabels,
# #                       fig_size=fig_size)
# # fig, ax = dh.plot_single(time=time_vic, data=EE_twist_d_vic, fig=fig, ax=ax, color_shape='g--')

# labels=['$\dot{x}_{KMP}$', '$\dot{x}_{KMP+VIC}$', '$\dot{x}_{d}$']
# ax.legend(labels=labels, borderaxespad=0.1,
#           handlelength=0.8, fontsize=LEGEND_SIZE)

# plt.show()


# ------------------------------ MEAN
params['color'] = 'Blue'
params['height'] = 27
params['trial_idx'] = 1
params['vic'] = False

ees = np.zeros((N_POINTS, 3))

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

    params['data_to_plot'] = 'EE_twist'
    ees[:,i-1] = dh.get_data(params, axis=Z_AXIS)


params['color'] = 'Blue'
params['height'] = 27
params['trial_idx'] = 1
params['vic'] = True

ees_vic = np.zeros((N_POINTS, 3))

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

    params['data_to_plot'] = 'EE_twist'
    ees_vic[:,i-1] = dh.get_data(params, axis=Z_AXIS)

EE = np.mean(ees, axis=1)
EE_vic = np.mean(ees_vic, axis=1)


file_name = path_folder + 'empty.mat'
idx_start = dh.get_idx_from_file(params, data_info, idx_name='idx_start', file_name=file_name)
idx_end = dh.get_idx_from_file(params, data_info, idx_name='idx_end', file_name=file_name)
params['i_initial'] = idx_start
params['i_final'] = idx_end
params['data_to_plot'] = 'time'
time_empty = dh.get_data(params, axis=0, file_name=file_name)
time_empty = time_empty - time_empty[0]
params['data_to_plot'] = 'EE_twist'
EE_empty = dh.get_data(params, Z_AXIS, file_name)


file_name = path_folder + 'const-imp.mat'
idx_start = dh.get_idx_from_file(params, data_info, idx_name='idx_start', file_name=file_name)
idx_end = dh.get_idx_from_file(params, data_info, idx_name='idx_end', file_name=file_name)
params['i_initial'] = idx_start
params['i_final'] = idx_end
params['data_to_plot'] = 'time'
time_const_imp = dh.get_data(params, axis=0, file_name=file_name)
time_const_imp = time_const_imp - time_const_imp[0]
params['data_to_plot'] = 'EE_twist'
EE_const_imp = dh.get_data(params, Z_AXIS, file_name)
EE_const_imp_tail = np.ones(N_POINTS-len(EE_const_imp))*EE_const_imp[-1]


fig, ax = dh.set_axis(xlim_plot=xlim_plot, xlabel=xlabel, xticks=xticks, xtickslabels=xtickslabels,
                      ylim_plot=ylim_plot, ylabel=ylabel, yticks=yticks, ytickslabels=ytickslabels,
                      fig_size=fig_size)
# ax.set_prop_cycle(color=['black', 'blue', 'blue', 'red', 'green', 'blue'])
fig, ax = dh.plot_single(time=time[:len(EE_const_imp)], data=EE_const_imp, fig=fig, ax=ax, color='#c1272d')
fig, ax = dh.plot_single(time=time, data=EE, fig=fig, ax=ax)
fig, ax = dh.plot_single(time=time, data=EE_vic, fig=fig, ax=ax)
# fig, ax = dh.plot_single(time=time_empty, data=EE_empty, fig=fig, ax=ax)
fig, ax = dh.plot_single(time=time, data=EE_twist_d, fig=fig, ax=ax, shape='--', color='k')
fig, ax = dh.plot_single(time=time[:len(EE_const_imp)][-1], data=EE_const_imp[-1], fig=fig, ax=ax, shape='x', color='#c1272d')
# fig, ax = dh.plot_single(time=time[len(EE_const_imp):N_POINTS], data=EE_const_imp_tail, fig=fig, ax=ax, shape='--', color='#c1272d')
ax.axvline(x = 0.08, linestyle='-', color = 'k', label = 'axvline - full height')
ax.text(x=0.065, y=.45, s='D')
# ax.axvline(x = 0.168, linestyle='--', color = 'k', label = 'axvline - full height')
ax.axvspan(0.08, .12, facecolor='b', alpha=0.1, label='_nolegend_')
# fig, ax = dh.plot_single(time=time_vic, data=EE_twist_d_vic, fig=fig, ax=ax, color_shape='g--')
# ax.text(x=0.018, y=0.15, s='P-C')
# ax.text(x=0.17, y=0.15, s='A-C')

# labels=['$\overline{\dot{x}}_{VM}$', '$\overline{\dot{x}}_{VM+VIC}$', '$\dot{x}_{empty}$', '$\dot{x}_{const-imp}$', '$\dot{x}_{d}$']
labels=['$\dot{x}_{FPIC}$', '$\overline{\dot{x}}_{VM}$', '$\overline{\dot{x}}_{VIC}$', '$\dot{x}_{d}$']
ax.legend(labels=labels, borderaxespad=0.1,
          handlelength=0.8, fontsize=LEGEND_SIZE, loc='lower right')#, bbox_to_anchor=(.9, 1))#, loc='upper right')

# plt.show()
fig.savefig('comparison_twist_average.png', bbox_inches='tight', pad_inches=0, dpi=300)