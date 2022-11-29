from cycler import cycler
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
from data_plot_class import DataPlotHelper

path_folder = 'data/icra2023/2-exp-ImpLoop3-NoAdaptation-PARTIAL-RESULTS-DONT-EDIT/'
data_info = pd.read_csv(path_folder+'data_info.csv')

dh = DataPlotHelper(path_folder)

n_subplots = 3
fig_size = [10, 12]
fig, ax = plt.subplots(n_subplots,figsize=fig_size)

####################### FORCES

# LOAD DATA
Z_AXIS = 2
STEP_XAXIS=0.25
COLORS = ['Green', 'Blue']
TRIALS_IDXS = [1, 2, 3]
HEIGHTS = [27]
LEGEND_SIZE = 14
N_POINTS=1000

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

params['data_to_plot'] = 'FT_ati'
FT_ati = dh.get_data(params, axis=Z_AXIS)

#----------------------

params['color'] = 'Blue'
params['height'] = 27
params['trial_idx'] = 1
params['vic'] = True

idx_start = dh.get_idx_from_file(params, data_info, idx_name='idx_start')
idx_end = dh.get_idx_from_file(params, data_info, idx_name='idx_end')
params['i_initial'] = idx_start + 38
params['i_final'] = idx_end + 38

params['data_to_plot'] = 'time'
time = dh.get_data(params, axis=0)
time = time - time[0]

params['data_to_plot'] = 'FT_ati'
FT_ati_vic = dh.get_data(params, axis=Z_AXIS)

# ----------------------
xlim_plot = [time[0], N_POINTS/1000]
ylim_plot = [-3.5, 32]
ylabel = '$F~[N]$'
xlabel = '$time~[s]$'
xticks =      [0, 0.25, 0.5, 0.75, 1.0]
xtickslabels = ['$0$', '$0.25$', '$0.5$', '$0.75$', '$1.0$']
yticks = None
ytickslabels = None
fig_size = [10, 4]  # width, height

#------------------ MEAN

params['color'] = 'Blue'
params['height'] = 27
params['trial_idx'] = 1
params['vic'] = False

fts = np.zeros((N_POINTS, 3))

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

    params['data_to_plot'] = 'FT_ati'
    # FT_ati = dh.get_data(params, axis=Z_AXIS
    # fts.append(dh.get_data(params, axis=Z_AXIS))
    fts[:,i-1] = dh.get_data(params, axis=Z_AXIS)


params['color'] = 'Blue'
params['height'] = 27
params['trial_idx'] = 1
params['vic'] = True

fts_vic = np.zeros((N_POINTS, 3))

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

    params['data_to_plot'] = 'FT_ati'
    # FT_ati = dh.get_data(params, axis=Z_AXIS
    # fts_vic.append(dh.get_data(params, axis=Z_AXIS))
    fts_vic[:,i-1] = dh.get_data(params, axis=Z_AXIS)

FT = np.mean(fts, axis=1)
FT_vic = np.mean(fts_vic, axis=1)

# plt.plot(FT)
# plt.plot(fts)
# plt.legend(['mean','1', '2', '3'])
# plt.show()

# plt.plot(FT_vic)
# plt.plot(fts_vic)

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

# EMPTY
file_name = path_folder + 'empty.mat'
idx_start = dh.get_idx_from_file(params, data_info, idx_name='idx_start', file_name=file_name)
idx_end = dh.get_idx_from_file(params, data_info, idx_name='idx_end', file_name=file_name)
params['i_initial'] = idx_start
params['i_final'] = idx_end
params['data_to_plot'] = 'time'
time_em = dh.get_data(params, axis=0)
time_em = time_em - time_em[0]
params['data_to_plot'] = 'FT_ati'
ft_emp = dh.get_data(params, axis=Z_AXIS, file_name=file_name)
# params['data_to_plot'] = 'EE_twist_d'
# ee_d_emp = dh.get_data(params, axis=Z_AXIS)

# CONST IMP
file_name = path_folder + 'const-imp.mat'
idx_start = dh.get_idx_from_file(params, data_info, idx_name='idx_start', file_name=file_name)
idx_end = dh.get_idx_from_file(params, data_info, idx_name='idx_end', file_name=file_name)
params['i_initial'] = idx_start
params['i_final'] = idx_end
params['data_to_plot'] = 'time'
time_const_imp = dh.get_data(params, axis=0, file_name=file_name)
time_const_imp = time_const_imp - time_const_imp[0]
params['data_to_plot'] = 'FT_ati'
ft_const_imp = dh.get_data(params, axis=Z_AXIS, file_name=file_name)
ft_const_imp_tail = np.ones(N_POINTS-len(ft_const_imp))*ft_const_imp[-1]
# params['data_to_plot'] = 'EE_twist_d'
# ee_d_emp = dh.get_data(params, axis=Z_AXIS)

fig, ax[0] = dh.set_axis(fig=fig, ax=ax[0], xlim_plot=xlim_plot, xlabel=xlabel, xticks=xticks, xtickslabels=xtickslabels,
                      ylim_plot=ylim_plot, ylabel=ylabel, yticks=yticks, ytickslabels=ytickslabels,
                      fig_size=fig_size, n_subplots=1)

fig, ax[0] = dh.plot_single(time=time, data=FT, fig=fig, ax=ax[0])
fig, ax[0] = dh.plot_single(time=time, data=FT_vic, fig=fig, ax=ax[0])
fig, ax[0] = dh.plot_single(time=time, data=ft_emp, fig=fig, ax=ax[0])
fig, ax[0] = dh.plot_single(time=time[:len(ft_const_imp)], data=ft_const_imp, fig=fig, ax=ax[0], color='#c1272d')
fig, ax[0] = dh.plot_single(time=time[len(ft_const_imp):N_POINTS], data=ft_const_imp_tail, fig=fig, ax=ax[0], shape='--', color='#c1272d')

labels=['$\overline{F}_{KMP}$', '$\overline{F}_{KMP+VIC}$', '$Empty$', '$Const.~Imp.$']
ax[0].legend(labels=labels, borderaxespad=0.1,
          handlelength=0.8, fontsize=LEGEND_SIZE)

plt.show()
# fig.savefig('forces_average_comparison.png')


###################################### TWIST

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

xlim_plot = [time_vic[0], N_POINTS/1000]
ylim_plot = [-1.5, 0.5]
labels=['$\dot{x}_{KMP}$', '$\dot{x}_{KMP+VIC}$', '$\dot{x}_d$']
ylabel = '$\dot{x}~[m/s]$'
xlabel = '$time~[s]$'
xticks =      [0, 0.25, 0.5, 0.75, 1.0]
xtickslabels = ['$0$', '$0.25$', '$0.5$', '$0.75$', '$1.0$']
yticks = None
ytickslabels = None
fig_size = [10, 4]  # width, height

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
fig, ax = dh.plot_single(time=time, data=EE, fig=fig, ax=ax)
fig, ax = dh.plot_single(time=time, data=EE_vic, fig=fig, ax=ax)
fig, ax = dh.plot_single(time=time_empty, data=EE_empty, fig=fig, ax=ax)
fig, ax = dh.plot_single(time=time[:len(EE_const_imp)], data=EE_const_imp, fig=fig, ax=ax, color='#c1272d')
fig, ax = dh.plot_single(time=time, data=EE_twist_d, fig=fig, ax=ax, shape='--', color='k')
fig, ax = dh.plot_single(time=time[len(EE_const_imp):N_POINTS], data=EE_const_imp_tail, fig=fig, ax=ax, shape='--', color='#c1272d')

# fig, ax = dh.plot_single(time=time_vic, data=EE_twist_d_vic, fig=fig, ax=ax, color_shape='g--')

labels=['$\overline{\dot{x}}_{KMP}$', '$\overline{\dot{x}}_{KMP+VIC}$', '$\dot{x}_{empty}$', '$\dot{x}_{const-imp}$', '$\dot{x}_{d}$']
ax.legend(labels=labels, borderaxespad=0.1,
          handlelength=0.8, fontsize=LEGEND_SIZE)

plt.show()
# fig.savefig('twist_average_comparison.png')


############################## POSITION

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
EE_position_d = dh.get_data(params, axis=Z_AXIS)

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
xticks =      [0, 0.25, 0.5, 0.75, 1.0]
xtickslabels = ['$0$', '$0.25$', '$0.5$', '$0.75$', '$1.0$']
yticks = None
ytickslabels = None
fig_size = [10, 4]  # width, height

# fig, ax = dh.set_axis(xlim_plot=xlim_plot, xlabel=xlabel, xticks=xticks, xtickslabels=xtickslabels,
#                       ylim_plot=ylim_plot, ylabel=ylabel, yticks=yticks, ytickslabels=ytickslabels,
#                       fig_size=fig_size)

# fig, ax = dh.plot_single(time=time, data=EE_twist, fig=fig, ax=ax)
# fig, ax = dh.plot_single(time=time_vic, data=EE_twist_vic, fig=fig, ax=ax)
# fig, ax = dh.plot_single(time=time, data=EE_position_d, fig=fig, ax=ax, color_shape='k--')

# # fig, ax = dh.set_axis(xlim_plot=xlim_plot, xlabel=xlabel, xticks=xticks, xtickslabels=xtickslabels,
# #                       ylim_plot=ylim_plot, ylabel=ylabel, yticks=yticks, ytickslabels=ytickslabels,
# #                       fig_size=fig_size)
# # fig, ax = dh.plot_single(time=time_vic, data=EE_twist_d_vic, fig=fig, ax=ax, color_shape='g--')

# labels=['$x_{KMP}$', '$x_{KMP+VIC}$', '$x_{d}$']
# ax.legend(labels=labels, borderaxespad=0.1,
#           handlelength=0.8, fontsize=LEGEND_SIZE)

# plt.show()


# ------------------------------ MEAN
params['color'] = 'Blue'
params['height'] = 27
params['trial_idx'] = 1
params['vic'] = False

ee_pose = np.zeros((N_POINTS, 3))

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

ee_pose_vic = np.zeros((N_POINTS, 3))

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
time_empty = dh.get_data(params, axis=0, file_name=file_name)
time_empty = time_empty - time_empty[0]
params['data_to_plot'] = 'EE_position'
EE_empty = dh.get_data(params, Z_AXIS, file_name)


file_name = path_folder + 'const-imp.mat'
idx_start = dh.get_idx_from_file(params, data_info, idx_name='idx_start', file_name=file_name)
idx_end = dh.get_idx_from_file(params, data_info, idx_name='idx_end', file_name=file_name)
params['i_initial'] = idx_start
params['i_final'] = idx_end
params['data_to_plot'] = 'time'
time_const_imp = dh.get_data(params, axis=0, file_name=file_name)
time_const_imp = time_const_imp - time_const_imp[0]
params['data_to_plot'] = 'EE_position'
EE_const_imp = dh.get_data(params, Z_AXIS, file_name)
EE_const_imp_tail = np.ones(N_POINTS-len(EE_const_imp))*EE_const_imp[-1]


fig, ax = dh.set_axis(xlim_plot=xlim_plot, xlabel=xlabel, xticks=xticks, xtickslabels=xtickslabels,
                      ylim_plot=ylim_plot, ylabel=ylabel, yticks=yticks, ytickslabels=ytickslabels,
                      fig_size=fig_size)
fig, ax = dh.plot_single(time=time, data=EE, fig=fig, ax=ax)
fig, ax = dh.plot_single(time=time, data=EE_vic, fig=fig, ax=ax)
fig, ax = dh.plot_single(time=time, data=EE_empty, fig=fig, ax=ax)
fig, ax = dh.plot_single(time=time[:len(EE_const_imp)], data=EE_const_imp, fig=fig, ax=ax, color='#c1272d')
fig, ax = dh.plot_single(time=time, data=EE_position_d, fig=fig, ax=ax, shape='--', color='k')
fig, ax = dh.plot_single(time=time[len(EE_const_imp):N_POINTS], data=EE_const_imp_tail, fig=fig, ax=ax, shape='--', color='#c1272d')


# fig, ax = dh.plot_single(time=time_vic, data=EE_twist_d_vic, fig=fig, ax=ax, color_shape='g--')

labels=['$\overline{x}_{KMP}$', '$\overline{x}_{KMP+VIC}$', '$x_{empty}$', '$x_{const-imp}$', '$x_{d}$']
ax.legend(labels=labels, borderaxespad=0.1,
          handlelength=0.8, fontsize=LEGEND_SIZE)

plt.show()
# fig.savefig('position_average_comparison.png')

