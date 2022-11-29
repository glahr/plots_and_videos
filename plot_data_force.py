from cycler import cycler
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
xlim_plot = [time[0], 500/1000]#N_POINTS/1000]
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
fig_size = [8, 2]  # width, height

# fig, ax = dh.set_axis(xlim_plot=xlim_plot, xlabel=xlabel, xticks=xticks, xtickslabels=xtickslabels,
#                       ylim_plot=ylim_plot, ylabel=ylabel, yticks=yticks, ytickslabels=ytickslabels,
#                       fig_size=fig_size)

# fig, ax = dh.plot_single(time=time, data=FT_ati, fig=fig, ax=ax)
# fig, ax = dh.plot_single(time=time, data=FT_ati_vic, fig=fig, ax=ax)

# labels=['$F_{KMP}$', '$F_{KMP+VIC}$']
# ax.legend(labels=labels, borderaxespad=0.1,
#           handlelength=0.8, fontsize=LEGEND_SIZE)

# plt.show()

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

fig, ax = dh.set_axis(xlim_plot=xlim_plot, xlabel=xlabel, xticks=xticks, xtickslabels=xtickslabels,
                      ylim_plot=ylim_plot, ylabel=ylabel, yticks=yticks, ytickslabels=ytickslabels,
                      fig_size=fig_size, n_subplots=1)

fig, ax = dh.plot_single(time=time[:len(ft_const_imp)], data=ft_const_imp, fig=fig, ax=ax, color='#c1272d')
fig, ax = dh.plot_single(time=time, data=FT, fig=fig, ax=ax)
fig, ax = dh.plot_single(time=time, data=FT_vic, fig=fig, ax=ax)
fig, ax = dh.plot_single(time=time[:len(ft_const_imp)][-1], data=ft_const_imp[-1], fig=fig, ax=ax, shape='x', color='#c1272d')
# fig, ax = dh.plot_single(time=time, data=ft_emp, fig=fig, ax=ax)
# fig, ax = dh.plot_single(time=time[len(ft_const_imp):N_POINTS], data=ft_const_imp_tail, fig=fig, ax=ax, shape='--', color='#c1272d')
ax.axvline(x = 0.08, linestyle='-', color = 'k', label = 'axvline - full height')
# ax.axvline(x = 0.12, linestyle='--', color = 'k', label = 'axvline - full height')
ax.axvspan(0.08, .12, facecolor='b', alpha=0.1, label='_nolegend_')
ax.text(x=0.072, y=33, s='D')
# ax.text(x=0.15, y=20, s='A-C')

print("max FPCI = ", max(ft_const_imp))
print("max VM = ", max(FT))
print("max VIC = ", max(FT_vic))

# labels=['$\overline{F}_{VM}$', '$\overline{F}_{VM+VIC}$', '$Empty$', '$Const.~Imp.$']
labels=['$FPIC$', '$\overline{F}_{VM}$', '$\overline{F}_{VIC}$']
ax.legend(labels=labels, borderaxespad=0.1,
          handlelength=0.8, fontsize=LEGEND_SIZE)

# plt.show()
fig.savefig('comparison_forces_average.png', bbox_inches='tight', pad_inches=0, dpi=300)