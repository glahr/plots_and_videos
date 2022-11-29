import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
from data_plot_class import DataPlotHelper
import matplotlib.animation as animation

# from plot_data_twist_and_pos import EE_const_imp

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

# ---------------------- plot
xlim_plot = [time[0], 1000/1000]#N_POINTS/1000]
ylim_plot = [-3.5, 32]
ylabel = '$\\boldsymbol{F}~[N]$'    
# xlabel = '$\\boldsymbol{t}~[s]$'
xlabel = ''
# xticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5]#, 0.75, 1.0]
# xtickslabels = ['$0$', '$0.1$', '$0.2$', '$0.3$','$0.4$', '$0.5$']#, '$0.75$', '$1.0$']
xticks = []
xtickslabels = []
yticks = [0, 10, 20, 30]
ytickslabels = ['$0$', '$10$', '$20$', '$30$']
fig_size = [8, 8]  # width, height

n_subplots=4
fig, ax = plt.subplots(n_subplots,figsize=fig_size)#, constrained_layout=True)

fig, ax[0] = dh.set_axis(fig=fig, ax=ax[0], xlim_plot=xlim_plot, xlabel=xlabel, xticks=xticks, xtickslabels=xtickslabels,
                      ylim_plot=ylim_plot, ylabel=ylabel, yticks=yticks, ytickslabels=ytickslabels)

# fig, ax[0] = dh.plot_single(time=time[:len(ft_const_imp)], data=ft_const_imp, fig=fig, ax=ax[0], color='#c1272d')
''' 
fig, ax[0] = dh.plot_single(time=time, data=FT, fig=fig, ax=ax[0])
fig, ax[0] = dh.plot_single(time=time, data=FT_vic, fig=fig, ax=ax[0])
fig, ax[0] = dh.plot_single(time=time[:len(ft_const_imp)][-1], data=ft_const_imp[-1], fig=fig, ax=ax[0], shape='x', color='#c1272d')
'''
# fig, ax = dh.plot_single(time=time, data=ft_emp, fig=fig, ax=ax)
# fig, ax = dh.plot_single(time=time[len(ft_const_imp):N_POINTS], data=ft_const_imp_tail, fig=fig, ax=ax, shape='--', color='#c1272d')
# ax[0].axvspan(0.08, .12, facecolor='b', alpha=0.1, label='_nolegend_')
# ax[0].text(x=0.075, y=33.25, s='D')
# ax.text(x=0.15, y=20, s='A-C')

print("max FPCI = ", max(ft_const_imp))
print("max VM = ", max(FT))
print("max VIC = ", max(FT_vic))

# labels=['$\overline{F}_{VM}$', '$\overline{F}_{VM+VIC}$', '$Empty$', '$Const.~Imp.$']
# labels=['$\\boldsymbol{F}_{FP-IC}$', '$\overline{\\boldsymbol{F}}_{VM-IC}$', '$\overline{\\boldsymbol{F}}_{VM-VIC}$']
# ax[0].legend(labels=labels, borderaxespad=0.1,
#           handlelength=0.8, fontsize=LEGEND_SIZE)

# plt.show()
# fig.savefig('comparison_forces_average.png', bbox_inches='tight', pad_inches=0, dpi=300)

###############################################

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

#------------------ MEAN FORCES

params['color'] = 'Blue'
params['height'] = 27
params['trial_idx'] = 1
params['vic'] = False

k = np.zeros((N_POINTS, 3))

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

    params['data_to_plot'] = 'K'
    # FT_ati = dh.get_data(params, axis=Z_AXIS
    # fts.append(dh.get_data(params, axis=Z_AXIS))
    k[:,i-1] = dh.get_data(params, axis=Z_AXIS)


params['color'] = 'Blue'
params['height'] = 27
params['trial_idx'] = 1
params['vic'] = True

k_vic = np.zeros((N_POINTS, 3))

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

    params['data_to_plot'] = 'K'
    # FT_ati = dh.get_data(params, axis=Z_AXIS
    # fts_vic.append(dh.get_data(params, axis=Z_AXIS))
    k_vic[:,i-1] = dh.get_data(params, axis=Z_AXIS)

K = np.mean(k, axis=1)
K_vic = np.mean(k_vic, axis=1)

# xlim_plot = [time[0], 500/1000]#time[-1]]
ylim_plot = [100, 800]
labels=['$K_{KMP}$', '$K_{KMP+VIC}$']
ylabel = '$\\boldsymbol{K}~[N/m]$'
# xlabel = '$\\boldsymbol{t}~[s]$'
xticks = [] #     [0, 0.1, 0.2, 0.3, 0.4, 0.5]#, 0.75, 1.0]
xtickslabels = [] #['$0$', '$0.1$', '$0.2$', '$0.3$','$0.4$', '$0.5$']#, '$0.75$', '$1.0$']
yticks = None
ytickslabels = None
# fig_size = [8, 6]  # width, height

fig, ax[1] = dh.set_axis(fig=fig, ax=ax[1], xlim_plot=xlim_plot, xlabel=xlabel, xticks=xticks, xtickslabels=xtickslabels,
                         ylim_plot=ylim_plot, ylabel=ylabel, yticks=yticks, ytickslabels=ytickslabels)
'''
fig, ax[1] = dh.plot_single(time=time, data=K, fig=fig, ax=ax[1])
'''

i_min = np.where(K_vic == np.min(K_vic))

n = 15
K_vic = pd.DataFrame(K_vic[i_min[0][0]+1:]).rolling(n).mean().values

# i = 0

# while np.isnan(K_vic[i]):
#     i += 1
# for idx in range(i):
#     K_vic[idx] = K_vic[i]

K_vic = np.concatenate((np.ones(i_min[0][0]-n-3).reshape(-1, 1)*750, K_vic[n-1:], np.ones(2*n+3).reshape(-1, 1)*750))

'''
fig, ax[1] = dh.plot_single(time=time, data=K_vic, fig=fig, ax=ax[1])
'''
# ax[1].axvline(x = 0.08, ymin=0, ymax=2.07, linestyle='-', linewidth=2, color = 'k', label = 'axvline - full height', clip_on=False)
# ax[1].text(x=0.072, y=825, s='D')
# ax[1].axvspan(0.08, .12, facecolor='b', alpha=0.1, label='_nolegend_')
# ax.text(x=0.018, y=450, s='P-C')
# ax.text(x=0.12, y=450, s='A-C')

# fig, ax = dh.plot_single(time=time_vic, data=EE_twist_d_vic, fig=fig, ax=ax[1], color_shape='g--')

# labels=['$\\boldsymbol{K}_{FP-IC/VM-IC}$', '$\overline{\\boldsymbol{K}}_{VM-VIC}$']
# ax[1].legend(labels=labels, borderaxespad=0.1,
#           handlelength=0.8, fontsize=LEGEND_SIZE, loc='lower right')

# fig.subplots_adjust(hspace=0.08)

# plt.show()
# fig.savefig('comparison_force_and_k.pdf', bbox_inches='tight', pad_inches=0, dpi=300)

###################################################
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

    params['data_to_plot'] = 'EE_twist_d'
    EE_twist_d = dh.get_data(params, axis=Z_AXIS)

EE_twist = np.mean(ees, axis=1)
EE_twist_vic = np.mean(ees_vic, axis=1)


file_name = path_folder + 'empty.mat'
idx_start = dh.get_idx_from_file(params, data_info, idx_name='idx_start', file_name=file_name)
idx_end = dh.get_idx_from_file(params, data_info, idx_name='idx_end', file_name=file_name)
params['i_initial'] = idx_start
params['i_final'] = idx_end
params['data_to_plot'] = 'time'
time_empty = dh.get_data(params, axis=0, file_name=file_name)
time_empty = time_empty - time_empty[0]
params['data_to_plot'] = 'EE_twist'
EE_twist_empty = dh.get_data(params, Z_AXIS, file_name)


file_name = path_folder + 'const-imp.mat'
idx_start = dh.get_idx_from_file(params, data_info, idx_name='idx_start', file_name=file_name)
idx_end = dh.get_idx_from_file(params, data_info, idx_name='idx_end', file_name=file_name)
params['i_initial'] = idx_start
params['i_final'] = idx_end
params['data_to_plot'] = 'time'
time_const_imp = dh.get_data(params, axis=0, file_name=file_name)
time_const_imp = time_const_imp - time_const_imp[0]
params['data_to_plot'] = 'EE_twist'
EE_twist_const_imp = dh.get_data(params, Z_AXIS, file_name)
EE_twist_const_imp_tail = np.ones(N_POINTS-len(EE_twist_const_imp))*EE_twist_const_imp[-1]

xlim_plot = [time[0], 1000/1000]#N_POINTS/1000]
ylim_plot = [-1.3, 0.4]
labels=['$\dot{\\boldsymbol{x}}_{KMP}$', '$\dot{\\boldsymbol{x}}_{KMP+VIC}$', '$\dot{\\boldsymbol{x}}_d$']
ylabel = '$\dot{\\boldsymbol{x}}~[m/s]$'
xlabel = ''#$\\boldsymbol{t}~[s]$'
xticks = [] #     [0, 0.25, 0.5, 0.75, 1.0]
xtickslabels = [] #['$0$', '$0.25$', '$0.5$', '$0.75$', '$1.0$']
# xticks =      [0, 0.1, 0.2, 0.3, 0.4, 0.5]#, 0.75, 1.0]
# xtickslabels = ['$0$', '$0.1$', '$0.2$', '$0.3$','$0.4$', '$0.5$']#, '$0.75$', '$1.0$']
yticks = None #[-1, -0.5, 0.0]
ytickslabels = None # ['$-1$', '$-0.5$', '$0$']
# fig_size = [8, 5]  # width, height

# n_subplots=2
# fig, ax = plt.subplots(n_subplots,figsize=fig_size)#, constrained_layout=True)

fig, ax[2] = dh.set_axis(fig=fig, ax=ax[2], xlim_plot=xlim_plot, xlabel=xlabel, xticks=xticks, xtickslabels=xtickslabels,
                      ylim_plot=ylim_plot, ylabel=ylabel, yticks=yticks, ytickslabels=ytickslabels,
                      fig_size=fig_size)
# ax.set_prop_cycle(color=['black', 'blue', 'blue', 'red', 'green', 'blue'])
'''
fig, ax[2] = dh.plot_single(time=time[:len(EE_twist_const_imp)], data=EE_twist_const_imp, fig=fig, ax=ax[2], color='#c1272d')
fig, ax[2] = dh.plot_single(time=time, data=EE_twist, fig=fig, ax=ax[2])
fig, ax[2] = dh.plot_single(time=time, data=EE_twist_vic, fig=fig, ax=ax[2])
# fig, ax = dh.plot_single(time=time_empty, data=EE_empty, fig=fig, ax=ax)
fig, ax[2] = dh.plot_single(time=time[:len(EE_twist_const_imp)][-1], data=EE_twist_const_imp[-1], fig=fig, ax=ax[2], shape='x', color='#c1272d')
# ax[2].text(x=0.068, y=.48, s='D')
# ax[2].axvspan(0.08, .12, facecolor='b', alpha=0.1, label='_nolegend_')
'''
fig, ax[2] = dh.plot_single(time=time, data=EE_twist_d, fig=fig, ax=ax[2], shape='--', color='k')

# labels=['$\overline{\dot{x}}_{VM}$', '$\overline{\dot{x}}_{VM+VIC}$', '$\dot{x}_{empty}$', '$\dot{x}_{const-imp}$', '$\dot{x}_{d}$']
# labels=['$\dot{\\boldsymbol{x}}_{FP-IC}$', '$\overline{\dot{\\boldsymbol{x}}}_{VM-IC}$', '$\overline{\dot{\\boldsymbol{x}}}_{VM-VIC}$', '$\dot{\\boldsymbol{x}}_{d}$']
# ax[2].legend(labels=labels, borderaxespad=0.1,
#           handlelength=0.8, fontsize=LEGEND_SIZE, loc='lower right')#, bbox_to_anchor=(.9, 1))#, loc='upper right')

# plt.show()


############################################### POSE

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
N_POINTS=1000

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

EE_pos = np.mean(ee_pose, axis=1)
EE_pos_vic = np.mean(ee_pose_vic, axis=1)


file_name = path_folder + 'empty.mat'
idx_start = dh.get_idx_from_file(params, data_info, idx_name='idx_start', file_name=file_name)
idx_end = dh.get_idx_from_file(params, data_info, idx_name='idx_end', file_name=file_name)
params['i_initial'] = idx_start
params['i_final'] = idx_end
params['data_to_plot'] = 'time'
time_empty = dh.get_data(params, axis=0, file_name=file_name)
time_empty = time_empty - time_empty[0]
params['data_to_plot'] = 'EE_position'
EE_pos_empty = dh.get_data(params, Z_AXIS, file_name)

params['data_to_plot'] = 'EE_position_d'
EE_position_d = dh.get_data(params, Z_AXIS, file_name)


file_name = path_folder + 'const-imp.mat'
idx_start = dh.get_idx_from_file(params, data_info, idx_name='idx_start', file_name=file_name)
idx_end = dh.get_idx_from_file(params, data_info, idx_name='idx_end', file_name=file_name)
params['i_initial'] = idx_start
params['i_final'] = idx_end
params['data_to_plot'] = 'time'
time_const_imp = dh.get_data(params, axis=0, file_name=file_name)
time_const_imp = time_const_imp - time_const_imp[0]
params['data_to_plot'] = 'EE_position'
EE_pos_const_imp = dh.get_data(params, Z_AXIS, file_name)
EE_pos_const_imp_tail = np.ones(N_POINTS-len(EE_pos_const_imp))*EE_pos_const_imp[-1]

xlim_plot = [time[0], time[-1]]
ylim_plot = [0, .6]
labels=['$x_{KMP}$', '$x_{KMP+VIC}$', '$x_d$']
ylabel = '$\\boldsymbol{x}~[m]$'
xlabel = '$\\boldsymbol{t}~[s]$'
xticks =      [0, 0.25, 0.5, 0.75, 1.0]
xtickslabels = ['$0$', '$0.25$', '$0.5$', '$0.75$', '$1.0$']
yticks = None # [0, 0.2, 0.4, 0.6]
ytickslabels = None #['$0$', '$0.2$', '$0.4$', '$0.6$']
# fig_size = [8, 3]  # width, height

fig, ax[3] = dh.set_axis(fig=fig, ax=ax[3], xlim_plot=xlim_plot, xlabel=xlabel, xticks=xticks, xtickslabels=xtickslabels,
                      ylim_plot=ylim_plot, ylabel=ylabel, yticks=yticks, ytickslabels=ytickslabels,
                      fig_size=fig_size)

'''
fig, ax[3] = dh.plot_single(time=time[:len(EE_pos_const_imp)], data=EE_pos_const_imp, fig=fig, ax=ax[3], color='#c1272d')
fig, ax[3] = dh.plot_single(time=time, data=EE_pos, fig=fig, ax=ax[3])
fig, ax[3] = dh.plot_single(time=time, data=EE_pos_vic, fig=fig, ax=ax[3])
# fig, ax = dh.plot_single(time=time, data=EE_empty, fig=fig, ax=ax)
fig, ax[3] = dh.plot_single(time=time[:len(EE_pos_const_imp)][-1], data=EE_pos_const_imp[-1], fig=fig, ax=ax[3], shape='x', color='#c1272d')
'''
fig, ax[3] = dh.plot_single(time=time, data=EE_position_d, fig=fig, ax=ax[3], shape='--', color='k')
# fig, ax = dh.plot_single(time=time[len(EE_const_imp):N_POINTS], data=EE_const_imp_tail, fig=fig, ax=ax, shape='--', color='#c1272d')
# ax.axvline(x = 0.079, linestyle='-', color = 'k', label = 'axvline - full height')
# ax.axvline(x = 0.12, linestyle='--', color = 'k', label = 'axvline - full height')
# ax.text(x=0.05, y=25, s='D')
# ax[3].axvline(x = 0.6, ymin=0, ymax=2.07, linestyle='-', linewidth=2, color = 'k', label = 'axvline - full height', clip_on=False)
# ax[3].axvspan(0.08, .12, facecolor='b', alpha=0.1, label='_nolegend_')
# ax[3].axvline(x = 0.08, ymin=0, ymax=2.07, linestyle='-', linewidth=2, color = 'k', label = 'axvline - full height', clip_on=False)
# ax[3].axvline(x = 0.168, ymin=0, ymax=2.07, linestyle='--', linewidth=1, color = 'k', label = 'axvline - full height', clip_on=False)
# ax[3].text(x=0.065, y=.625, s='D')
# fig, ax = dh.plot_single(time=time_vic, data=EE_twist_d_vic, fig=fig, ax=ax, color_shape='g--')

# labels=['$\overline{x}_{VM}$', '$\overline{x}_{VM+VIC}$', '$x_{empty}$', '$x_{const-imp}$', '$x_{d}$']
# labels=['$\\boldsymbol{x}_{FP-IC}$', '$\overline{\\boldsymbol{x}}_{VM-IC}$', '$\overline{\\boldsymbol{x}}_{VM-VIC}$', '$\\boldsymbol{x}_{d}$']
# ax[3].legend(labels=labels, borderaxespad=0.1,
#           handlelength=0.8, fontsize=LEGEND_SIZE)
fig.subplots_adjust(hspace=0.05)
# ax[0].plot(0.168, -1, 'ko', markersize=6, alpha=1.0)
# ax[3].plot(0.168, 0.41672, 'ko', markersize=6, alpha=1.0)
# plt.show()
# fig.savefig('comparison_twist_position.pdf', bbox_inches='tight', pad_inches=0, dpi=300)


# plt.show()

##########################################################################################

def animate(j):
    idx = j*STEP + idx_start

    if plot_now == 'vm':
        plots_force.set_data(time[:idx], FT[:idx])
        plots_k.set_data(time[:idx], K[:idx],)
        plots_pos.set_data(time[:idx], EE_pos[:idx])
        plots_twist.set_data(time[:idx], EE_twist[:idx])
    
    if plot_now == 'vic':
        plots_force.set_data(time[:idx], FT_vic[:idx])
        plots_k.set_data(time[:idx], K_vic[:idx],)
        plots_pos.set_data(time[:idx], EE_pos_vic[:idx])
        plots_twist.set_data(time[:idx], EE_twist_vic[:idx])
    
    if plot_now == 'const':
        if idx > len(ft_const_imp):
            idx = len(ft_const_imp)-1
            ax[0].plot(time[idx], ft_const_imp[idx], 'x', linewidth=dh.lw, color='#c1272d')
            ax[1].plot(time[idx], K[idx], 'x', linewidth=dh.lw, color='#c1272d')
            ax[2].plot(time[idx], EE_twist_const_imp[idx], 'x', linewidth=dh.lw, color='#c1272d')
            ax[3].plot(time[idx], EE_pos_const_imp[idx], 'x', linewidth=dh.lw, color='#c1272d')

            plots_force.set_data(time[:idx], ft_const_imp[:idx])
            plots_k.set_data(time[:idx], K[:idx])
            plots_twist.set_data(time[:idx], EE_twist_const_imp[:idx])
            plots_pos.set_data(time[:idx], EE_pos_const_imp[:idx])
            return None
        plots_force.set_data(time[:idx], ft_const_imp[:idx])
        plots_k.set_data(time[:idx], K[:idx])
        plots_twist.set_data(time[:idx], EE_twist_const_imp[:idx])
        plots_pos.set_data(time[:idx], EE_pos_const_imp[:idx])
    
    if plot_now == 'vm+vic':
        plots_force_vm.set_data(time[:idx], FT[:idx])
        plots_k_vm.set_data(time[:idx], K[:idx],)
        plots_pos_vm.set_data(time[:idx], EE_pos[:idx])
        plots_twist_vm.set_data(time[:idx], EE_twist[:idx])
    
        plots_force_vic.set_data(time[:idx], FT_vic[:idx])
        plots_k_vic.set_data(time[:idx], K_vic[:idx],)
        plots_pos_vic.set_data(time[:idx], EE_pos_vic[:idx])
        plots_twist_vic.set_data(time[:idx], EE_twist_vic[:idx])
    
    if plot_now == 'all':
        plots_force_vm.set_data(time[:idx], FT[:idx])
        plots_k_vm.set_data(time[:idx], K[:idx],)
        plots_pos_vm.set_data(time[:idx], EE_pos[:idx])
        plots_twist_vm.set_data(time[:idx], EE_twist[:idx])
    
        plots_force_vic.set_data(time[:idx], FT_vic[:idx])
        plots_k_vic.set_data(time[:idx], K_vic[:idx],)
        plots_pos_vic.set_data(time[:idx], EE_pos_vic[:idx])
        plots_twist_vic.set_data(time[:idx], EE_twist_vic[:idx])

        if idx > len(ft_const_imp):
            idx = len(ft_const_imp)-1
            ax[0].plot(time[idx], ft_const_imp[idx], 'x', linewidth=dh.lw, color='#c1272d')
            ax[1].plot(time[idx], K[idx], 'x', linewidth=dh.lw, color='#c1272d')
            ax[2].plot(time[idx], EE_twist_const_imp[idx], 'x', linewidth=dh.lw, color='#c1272d')
            ax[3].plot(time[idx], EE_pos_const_imp[idx], 'x', linewidth=dh.lw, color='#c1272d')
        plots_force_const.set_data(time[:idx], ft_const_imp[:idx])
        plots_k_const.set_data(time[:idx], K[:idx])
        plots_twist_const.set_data(time[:idx], EE_twist_const_imp[:idx])
        plots_pos_const.set_data(time[:idx], EE_pos_const_imp[:idx])
     
    return None 


########### PLOT CONFIG ##############
idx_start = 0
plot_now = 'all'

if plot_now == 'vm':
    plots_force, = ax[0].plot(time[0], FT[0], linewidth=dh.lw, color='#1f77b4')
    plots_k, = ax[1].plot(time[0], K[0], linewidth=dh.lw, color='#1f77b4')
    plots_twist, = ax[2].plot(time[0], EE_twist[0], linewidth=dh.lw, color='#1f77b4')
    plots_pos, = ax[3].plot(time[0], EE_pos[0], linewidth=dh.lw, color='#1f77b4')

    labels=['$\overline{\\boldsymbol{F}}_{VM-IC}$']
    ax[0].legend(labels=labels, borderaxespad=0.1, handlelength=0.8, fontsize=LEGEND_SIZE)

    labels=['$\\boldsymbol{K}_{VM-IC}$']
    ax[1].legend(labels=labels, borderaxespad=0.1, handlelength=0.8, fontsize=LEGEND_SIZE, loc='lower right')

    labels=['$\dot{\\boldsymbol{x}}_{d}$', '$\overline{\dot{\\boldsymbol{x}}}_{VM-IC}$']
    ax[2].legend(labels=labels, borderaxespad=0.1, handlelength=0.8, fontsize=LEGEND_SIZE, loc='lower right')

    labels=['$\\boldsymbol{x}_{d}$', '$\overline{\\boldsymbol{x}}_{VM-IC}$']
    ax[3].legend(labels=labels, borderaxespad=0.1, handlelength=0.8, fontsize=LEGEND_SIZE)

if plot_now == 'vic':
    plots_force, = ax[0].plot(time[0], FT_vic[0], linewidth=dh.lw, color='#ff7f0e')
    plots_k, = ax[1].plot(time[0], K_vic[0], linewidth=dh.lw, color='#ff7f0e')
    plots_twist, = ax[2].plot(time[0], EE_twist_vic[0], linewidth=dh.lw, color='#ff7f0e')
    plots_pos, = ax[3].plot(time[0], EE_pos_vic[0], linewidth=dh.lw, color='#ff7f0e')

    labels=['$\overline{\\boldsymbol{F}}_{VM-VIC}$']
    ax[0].legend(labels=labels, borderaxespad=0.1, handlelength=0.8, fontsize=LEGEND_SIZE)

    labels=['$\\boldsymbol{K}_{VM-VIC}$']
    ax[1].legend(labels=labels, borderaxespad=0.1, handlelength=0.8, fontsize=LEGEND_SIZE, loc='lower right')

    labels=['$\dot{\\boldsymbol{x}}_{d}$', '$\overline{\dot{\\boldsymbol{x}}}_{VM-VIC}$']
    ax[2].legend(labels=labels, borderaxespad=0.1, handlelength=0.8, fontsize=LEGEND_SIZE, loc='lower right')

    labels=['$\\boldsymbol{x}_{d}$', '$\overline{\\boldsymbol{x}}_{VM-VIC}$']
    ax[3].legend(labels=labels, borderaxespad=0.1, handlelength=0.8, fontsize=LEGEND_SIZE)

if plot_now == 'const':
    plots_force, = ax[0].plot(time[0], ft_const_imp[0], linewidth=dh.lw, color='#c1272d')
    plots_k, = ax[1].plot(time[0], K[0], linewidth=dh.lw, color='#c1272d')
    plots_twist, = ax[2].plot(time[0], EE_twist_const_imp[0], linewidth=dh.lw, color='#c1272d')
    plots_pos, = ax[3].plot(time[0], EE_pos_const_imp[0], linewidth=dh.lw, color='#c1272d')

    labels=['$\overline{\\boldsymbol{F}}_{FP-IC}$']
    ax[0].legend(labels=labels, borderaxespad=0.1, handlelength=0.8, fontsize=LEGEND_SIZE)

    labels=['$\\boldsymbol{K}_{FP-IC}$']
    ax[1].legend(labels=labels, borderaxespad=0.1, handlelength=0.8, fontsize=LEGEND_SIZE, loc='lower right')

    labels=['$\dot{\\boldsymbol{x}}_{d}$', '$\overline{\dot{\\boldsymbol{x}}}_{FP-IC}$']
    ax[2].legend(labels=labels, borderaxespad=0.1, handlelength=0.8, fontsize=LEGEND_SIZE, loc='lower right')

    labels=['$\\boldsymbol{x}_{d}$', '$\overline{\\boldsymbol{x}}_{FP-IC}$']
    ax[3].legend(labels=labels, borderaxespad=0.1, handlelength=0.8, fontsize=LEGEND_SIZE)

if plot_now == 'vm+vic':
    plots_force_vm, = ax[0].plot(time[0], FT[0], linewidth=dh.lw, color='#1f77b4')
    plots_k_vm, = ax[1].plot(time[0], K[0], linewidth=dh.lw, color='#1f77b4')
    plots_twist_vm, = ax[2].plot(time[0], EE_twist[0], linewidth=dh.lw, color='#1f77b4')
    plots_pos_vm, = ax[3].plot(time[0], EE_pos[0], linewidth=dh.lw, color='#1f77b4')

    plots_force_vic, = ax[0].plot(time[0], FT_vic[0], linewidth=dh.lw, color='#ff7f0e')
    plots_k_vic, = ax[1].plot(time[0], K_vic[0], linewidth=dh.lw, color='#ff7f0e')
    plots_twist_vic, = ax[2].plot(time[0], EE_twist_vic[0], linewidth=dh.lw, color='#ff7f0e')
    plots_pos_vic, = ax[3].plot(time[0], EE_pos_vic[0], linewidth=dh.lw, color='#ff7f0e')

    labels=['$\overline{\\boldsymbol{F}}_{VM-IC}$', '$\overline{\\boldsymbol{F}}_{VM-VIC}$']
    ax[0].legend(labels=labels, borderaxespad=0.1, handlelength=0.8, fontsize=LEGEND_SIZE)

    labels=['$\\boldsymbol{K}_{VM-IC}$', '$\\boldsymbol{K}_{VM-VIC}$']
    ax[1].legend(labels=labels, borderaxespad=0.1, handlelength=0.8, fontsize=LEGEND_SIZE, loc='lower right')

    labels=['$\dot{\\boldsymbol{x}}_{d}$', '$\overline{\dot{\\boldsymbol{x}}}_{VM-IC}$', '$\overline{\dot{\\boldsymbol{x}}}_{VM-VIC}$']
    ax[2].legend(labels=labels, borderaxespad=0.1, handlelength=0.8, fontsize=LEGEND_SIZE, loc='lower right')

    labels=['$\\boldsymbol{x}_{d}$', '$\overline{\\boldsymbol{x}}_{VM-IC}$', '$\overline{\\boldsymbol{x}}_{VM-VIC}$']
    ax[3].legend(labels=labels, borderaxespad=0.1, handlelength=0.8, fontsize=LEGEND_SIZE)

if plot_now == 'all':
    plots_force_const, = ax[0].plot(time[0], ft_const_imp[0], linewidth=dh.lw, color='#c1272d')
    plots_k_const, = ax[1].plot(time[0], K[0], linewidth=dh.lw, color='#c1272d')
    plots_twist_const, = ax[2].plot(time[0], EE_twist_const_imp[0], linewidth=dh.lw, color='#c1272d')
    plots_pos_const, = ax[3].plot(time[0], EE_pos_const_imp[0], linewidth=dh.lw, color='#c1272d')

    plots_force_vm, = ax[0].plot(time[0], FT[0], linewidth=dh.lw, color='#1f77b4')
    plots_k_vm, = ax[1].plot(time[0], K[0], linewidth=dh.lw, color='#1f77b4')
    plots_twist_vm, = ax[2].plot(time[0], EE_twist[0], linewidth=dh.lw, color='#1f77b4')
    plots_pos_vm, = ax[3].plot(time[0], EE_pos[0], linewidth=dh.lw, color='#1f77b4')

    plots_force_vic, = ax[0].plot(time[0], FT_vic[0], linewidth=dh.lw, color='#ff7f0e')
    plots_k_vic, = ax[1].plot(time[0], K_vic[0], linewidth=dh.lw, color='#ff7f0e')
    plots_twist_vic, = ax[2].plot(time[0], EE_twist_vic[0], linewidth=dh.lw, color='#ff7f0e')
    plots_pos_vic, = ax[3].plot(time[0], EE_pos_vic[0], linewidth=dh.lw, color='#ff7f0e')

    labels=['$\\boldsymbol{F}_{FP-IC}$','$\overline{\\boldsymbol{F}}_{VM-IC}$', '$\overline{\\boldsymbol{F}}_{VM-VIC}$']
    ax[0].legend(labels=labels, borderaxespad=0.1, handlelength=0.8, fontsize=LEGEND_SIZE)

    labels=['$\\boldsymbol{K}_{FP-IC}$','$\\boldsymbol{K}_{VM-IC}$', '$\\boldsymbol{K}_{VM-VIC}$']
    ax[1].legend(labels=labels, borderaxespad=0.1, handlelength=0.8, fontsize=LEGEND_SIZE, loc='lower right')

    labels=['$\dot{\\boldsymbol{x}}_{d}$', '$\dot{\\boldsymbol{x}}_{FP-IC}$', '$\overline{\dot{\\boldsymbol{x}}}_{VM-IC}$', '$\overline{\dot{\\boldsymbol{x}}}_{VM-VIC}$']
    ax[2].legend(labels=labels, borderaxespad=0.1, handlelength=0.8, fontsize=LEGEND_SIZE, loc='lower right')

    labels=['$\\boldsymbol{x}_{d}$', '$\\boldsymbol{x}_{FP-IC}$', '$\overline{\\boldsymbol{x}}_{VM-IC}$', '$\overline{\\boldsymbol{x}}_{VM-VIC}$']
    ax[3].legend(labels=labels, borderaxespad=0.1, handlelength=0.8, fontsize=LEGEND_SIZE)

STEP = 5

# if plot_now == 'const':
#     STEP = 1

n_frames = int((N_POINTS)/STEP)
print("n_frames = ", n_frames)
# print("video duration = +-", n_frames/FPS)

animation_1 = animation.FuncAnimation(plt.gcf(), animate, interval=1, repeat=False, frames=n_frames)

### visualization
# plt.show()

video_name = 'video_'+plot_now+'.mp4'
### creating and saving the video
writervideo = animation.FFMpegWriter(fps=50)
animation_1.save(path_folder + video_name, writer=writervideo)
