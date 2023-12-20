from cycler import cycler
import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
from data_plot_class import DataPlotHelper
import os
from scipy.io import loadmat

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
idx_torque_vm = 1

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

    if i == idx_torque_vm:
        params['data_to_plot'] = 'tau_measured'
        torque_vm = dh.get_data(params)

params['color'] = 'Blue'
params['height'] = 27
params['trial_idx'] = 1
params['vic'] = True

fts_vic = np.zeros((N_POINTS, 3))
vic_offset = 24

idx_torque_vic = 1

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

    if i == idx_torque_vic:
        params['data_to_plot'] = 'tau_measured'
        torque_vic = dh.get_data(params)

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
ft_const_imp = dh.get_data(params, axis=Z_AXIS, file_name=file_name)[:-190]
ft_const_imp_tail = np.ones(N_POINTS-len(ft_const_imp))*ft_const_imp[-1]
# params['data_to_plot'] = 'EE_twist_d'
# ee_d_emp = dh.get_data(params, axis=Z_AXIS)
params['data_to_plot'] = 'tau_measured'
torque_fp =  dh.get_data(params, file_name=file_name)


# ---------------------- plot
xlim_plot = [time[0], 1000/1000]#N_POINTS/1000]
ylim_plot = [-33, 4]
ylabel = '$\\boldsymbol{F}~[N]$'    
# xlabel = '$\\boldsymbol{t}~[s]$'
xlabel = ''
# xticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5]#, 0.75, 1.0]
# xtickslabels = ['$0$', '$0.1$', '$0.2$', '$0.3$','$0.4$', '$0.5$']#, '$0.75$', '$1.0$']
# xticks = []
# xtickslabels = []
# yticks = [0, -10, -20, -30]
# ytickslabels = ['$0$', '$-10$', '$-20$', '$-30$']
fig_size = [8, 6]  # width, height

n_subplots=2
fig, ax = plt.subplots(n_subplots,figsize=fig_size) #, constrained_layout=True)

# fig, ax[0] = dh.set_axis(fig=fig, ax=ax[0], xlim_plot=xlim_plot, xlabel=xlabel, xticks=xticks, xtickslabels=xtickslabels,
#                       ylim_plot=ylim_plot, ylabel=ylabel, yticks=yticks, ytickslabels=ytickslabels)
fig, ax[0] = dh.set_axis(fig=fig, ax=ax[0], xlim_plot=xlim_plot, xlabel=xlabel, ylim_plot=ylim_plot, ylabel=ylabel)

fig, ax[0] = dh.plot_single(time=time[:len(ft_const_imp)], data=-ft_const_imp, fig=fig, ax=ax[0], color='#c1272d')
fig, ax[0] = dh.plot_single(time=time, data=-FT, fig=fig, ax=ax[0])
fig, ax[0] = dh.plot_single(time=time, data=-FT_vic, fig=fig, ax=ax[0])
fig, ax[0] = dh.plot_single(time=time[:len(ft_const_imp)][-1], data=-ft_const_imp[-1], fig=fig, ax=ax[0], shape='x', color='#c1272d')
# fig, ax = dh.plot_single(time=time, data=ft_emp, fig=fig, ax=ax)
# fig, ax = dh.plot_single(time=time[len(ft_const_imp):N_POINTS], data=ft_const_imp_tail, fig=fig, ax=ax, shape='--', color='#c1272d')
# ax[0].axvline(x = 0.08, linestyle='-', color = 'k', label = 'axvline - full height')
# ax.axvline(x = 0.12, linestyle='--', color = 'k', label = 'axvline - full height')
ax[0].axvspan(0.08, .109, facecolor='b', alpha=0.1, label='_nolegend_')
ax[0].text(x=0.073, y=5.5, s='D')
ax[0].text(x=0.59, y=5.5, s='F')
ax[1].axvline(x = 0.6, ymin=0, ymax=2.07, linestyle='-', linewidth=2, color = 'k', label = '_nolegend_', clip_on=False)

print("max FPCI = ", max(ft_const_imp))
print("max VM = ", max(FT))
print("max VIC = ", max(FT_vic))

# labels=['$\overline{F}_{VM}$', '$\overline{F}_{VM+VIC}$', '$Empty$', '$Const.~Imp.$']
labels=['$FP-IC$', '$VM-IC$', '$VM-VIC$']
ax[0].legend(labels=labels, borderaxespad=0.1,
          handlelength=0.8, fontsize=LEGEND_SIZE)

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
    if i == 3:
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

xlim_plot = [time[0], 1000/1000]#time[-1]]
ylim_plot = [100, 800]
labels=['$K_{KMP}$', '$K_{KMP+VIC}$']
ylabel = '$\\boldsymbol{K}~[N/m]$'
xlabel = '$\\boldsymbol{t}~[s]$'
# xticks =      [0, 0.2, 0.4, 0.6, 0.8, 1.0]
# xtickslabels = ['$'+str(a)+'$' for a in xticks] #['$0$', '$0.1$', '$0.2$', '$0.3$','$0.4$', '$0.5$']#, '$0.75$', '$1.0$']
# yticks = None
# ytickslabels = None
# fig_size = [8, 3]  # width, height

# fig, ax[1] = dh.set_axis(fig=fig, ax=ax[1], xlim_plot=xlim_plot, xlabel=xlabel, xticks=xticks, xtickslabels=xtickslabels,
                        #  ylim_plot=ylim_plot, ylabel=ylabel, yticks=yticks, ytickslabels=ytickslabels)
fig, ax[1] = dh.set_axis(fig=fig, ax=ax[1], xlim_plot=xlim_plot, xlabel=xlabel, ylim_plot=ylim_plot, ylabel=ylabel)
labels=['$FP-IC/VM-IC$', '$VM-VIC$']
fig, ax[1] = dh.plot_single(time=time, data=K, fig=fig, ax=ax[1], label=labels[0])

i_min = np.where(K_vic == np.min(K_vic))

n = 15
K_vic = pd.DataFrame(K_vic[i_min[0][0]+1:]).rolling(n).mean().values

# i = 0

# while np.isnan(K_vic[i]):
#     i += 1
# for idx in range(i):
#     K_vic[idx] = K_vic[i]

K_vic = np.concatenate((np.ones(i_min[0][0]-n-3).reshape(-1, 1)*750, K_vic[n-1:], np.ones(2*n+3).reshape(-1, 1)*750))
fig, ax[1] = dh.plot_single(time=time, data=K_vic, fig=fig, ax=ax[1], label=labels[1])
ax[1].axvline(x = 0.08, ymin=0, ymax=2.07, linestyle='-', linewidth=2, color = 'k', label='_nolegend_', clip_on=False)
# ax[1].text(x=0.072, y=825, s='D')
ax[1].axvspan(0.08, .109, facecolor='b', alpha=0.1, label='_nolegend_')
# ax.text(x=0.018, y=450, s='P-C')
# ax.text(x=0.12, y=450, s='A-C')

# fig, ax = dh.plot_single(time=time_vic, data=EE_twist_d_vic, fig=fig, ax=ax[1], color_shape='g--')


ax[1].legend(borderaxespad=0.1,
          handlelength=0.8, fontsize=LEGEND_SIZE, loc='lower right')

fig.subplots_adjust(hspace=0.08)

alpha_grids = 0.12
y_grids_ft = [-30, -20, -10, 0]
y_grids_Kp = [250, 500, 750]
xticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

idx_ft = 0
idx_Kp = 1

fontsize_labels = 15
for j, e in enumerate(ax):
    [e.axvline(xg, color='k', alpha=alpha_grids) for xg in xticks]
    e.set_xticks(xticks)
    e.set_xticklabels(['$'+str(xt)+'$' for xt in xticks], fontsize=fontsize_labels)
    if idx_ft == j:
        [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_ft]
        e.set_yticks(y_grids_ft)
        e.set_yticklabels(['$'+str(a)+'$' for a in y_grids_ft], size=fontsize_labels)
        ax[idx_ft].set_xticks([])
    if idx_Kp == j:
        [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_Kp]
        e.set_yticks(y_grids_Kp)
        e.set_yticklabels(['$'+str(a)+'$' for a in y_grids_Kp], size=fontsize_labels)

fig.align_ylabels()
# plt.show()
fig.savefig('comparison_force_and_k.png', pad_inches=0, dpi=300)
fig.savefig('comparison_force_and_k.pdf', pad_inches=0, dpi=300)


##################################

fig_tau, ax_tau = plt.subplots(3,1,figsize=[7, 4.5])

fig_tau, ax_tau[0] = dh.set_axis(fig=fig_tau, ax=ax_tau[0])

tau_max = np.array([87, 87, 87, 87, 12, 12, 12])

torque_fp_normalized = np.zeros_like(torque_fp)
torque_vm_normalized = np.zeros_like(torque_vm)
torque_vic_normalized = np.zeros_like(torque_vic)

for i, t_fp in enumerate(torque_fp):
    torque_fp_normalized[i] = np.divide(t_fp, tau_max)

for i, t_vm in enumerate(torque_vm):
    torque_vm_normalized[i] = np.divide(t_vm, tau_max)

for i, t_vic in enumerate(torque_vic):
    torque_vic_normalized[i] = np.divide(t_vic, tau_max)

ax_tau[0].plot(time_const_imp[:-180], torque_fp_normalized[:-180], linewidth=dh.lw-1)
ax_tau[1].plot(time, torque_vm_normalized, linewidth=dh.lw-1)
ax_tau[2].plot(time, torque_vic_normalized, linewidth=dh.lw-1)

for ax_ in ax_tau:
    ax_.set_xlim([0, 1])

for ax_ in ax_tau:
    ax_.set_ylim([-1.1, 1.1])

ax_tau[0].set_xticks([])
ax_tau[1].set_xticks([])
ax_tau[2].set_xlabel('$\\boldsymbol{t}~[s]$')

ax_tau[0].set_ylabel('$\\hat{\\boldsymbol{\\tau}}_{FP}$')
ax_tau[1].set_ylabel('$\\hat{\\boldsymbol{\\tau}}_{VM+IC}$')
ax_tau[2].set_ylabel('$\\hat{\\boldsymbol{\\tau}}_{VM+VIC}$')

alpha_grids = 0.12
y_grids_tau = [-1, -0.5, 0, 0.5, 1]
xticks = [0, 0.2, 0.4, 0.6, 0.8, 1.0]

fontsize_labels = 14
for j, e in enumerate(ax_tau):
    [e.axvline(xg, color='k', alpha=alpha_grids) for xg in xticks]
    
    if j != 2:
        [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_tau]
        e.set_yticks(y_grids_tau)
        e.set_yticklabels(['$'+str(a)+'$' for a in y_grids_tau], size=fontsize_labels)
    else:
        [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_tau]
        e.set_yticks(y_grids_tau)
        e.set_yticklabels(['$'+str(a)+'$' for a in y_grids_tau], size=fontsize_labels)
        e.set_xticks(xticks)
        e.set_xticklabels(['$'+str(xt)+'$' for xt in xticks], fontsize=fontsize_labels)

labels = ['$\\hat{\\boldsymbol{\\tau}}_'+str(i)+'$' for i in range(1,8)]
ax_tau[0].legend(labels=labels, borderaxespad=0.1,
          handlelength=0.8, fontsize=LEGEND_SIZE+1, loc='lower right', ncol=2)

fig_tau.subplots_adjust(hspace=0.02)
# fig_tau.set_tight_layout(tight=True)
fig_tau.set_constrained_layout(constrained=True)
ax_tau[2].axvline(x = 0.08, ymin=0, ymax=3.27, linestyle='-', linewidth=1.4, color = 'k', label = '_nolegend_', clip_on=False)
ax_tau[0].text(x=0.073, y=1.24, s='D', fontsize=16)
ax_tau[0].text(x = 0.59, y=1.24, s='F', fontsize=16)
ax_tau[2].axvline(x = 0.6, ymin=0, ymax=3.27, linestyle='-', linewidth=1.4, color = 'k', label = '_nolegend_', clip_on=False)

# plt.show()
fig_tau.savefig('comparison_torques.pdf', pad_inches=0, dpi=300)
fig_tau.savefig('comparison_torques.png', pad_inches=0, dpi=300)

############################################
fig_size = [5, 2]  # width, height

file_name = 'acc_profile.mat'
print(file_name)
n_subplots=1
fig, ax = plt.subplots(n_subplots,figsize=fig_size, constrained_layout=True)
f = loadmat(file_name)
print(f.keys())
xc = np.array(f['xc'])
yc = np.array(f['yc'])
print(xc[0,0][0])
print(yc[0,0][0])
i = 0
for elemx, elemy in zip(xc, yc):
    # ax.plot(elemx[0][0],elemy[0][0], linewidth=3, alpha=0.5)
    # ax.plot(elemx[0][0],elemy[0][0]/np.max(elemy[0][0]), linewidth=3, alpha=0.5)
    ax.plot(np.concatenate([[0.0], elemx[0][0]]), np.concatenate([[elemy[0][0][0]], elemy[0][0]]), linewidth=3, label='$demo_'+str(i+1)+'$')
    i += 1
ax.set_xlabel('$Time~[s]$', size=15)
ax.set_ylabel('$ACC[-]$', size=15)
# ax.grid()
# ax.legend(fontsize=15, ncol=2)

ax.set_xticks([0, 0.1/2, 0.2/2, 0.3/2])
ax.set_xticklabels(['$0$', '$0.1$', '$0.2$', '$0.3$'], size=13)
ax.set_xlim([0, 0.32/2])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.set_yticks([0, 850, 1700])
ax.set_yticklabels(['$0$', '$850$', '$1700$'], size=13)
# ax.set_xticklabels([])
fig.savefig('gmmgmr_acc.png', dpi=400)
# plt.show()

############################################
fig_size = [5, 2.2]  # width, height

file_name = 'position_learning.mat'
print(file_name)
n_subplots=1
fig, ax = plt.subplots(n_subplots,figsize=fig_size, constrained_layout=True)
fig_gmr, ax_gmr = plt.subplots(n_subplots,figsize=fig_size, constrained_layout=True)
f = loadmat(file_name)
print(f.keys())
xc = np.array(f['xc'])
yc = np.array(f['yc'])
print(xc[0,0][0])
print(yc[0,0][0])
i = 0
for elemx, elemy in zip(xc, yc):
    if i != 0:
        ax.plot(np.concatenate([[0.0], elemx[0][0]]), np.concatenate([[elemy[0][0][0]], elemy[0][0]]), linewidth=3, label='$demo_'+str(i)+'$')
        ax_gmr.plot(np.concatenate([[0.0], elemx[0][0]]), np.concatenate([[elemy[0][0][0]], elemy[0][0]]), linewidth=3, label='$demo_'+str(i)+'$')
    i += 1

i = 0
for elemx, elemy in zip(xc, yc):
    if i == 0:
        delta_d = np.concatenate([[elemy[0][0][0]], elemy[0][0]])
        ax.plot(np.concatenate([[0.0], elemx[0][0]]), delta_d, 'k', linewidth=4, label='$GMR$')
        ax_gmr.plot(np.concatenate([[0.0], elemx[0][0]]), delta_d, 'k', linewidth=4, label='$GMR$')
    i += 1
ax.set_xlabel('$Time~[s]$', size=15)
ax.set_ylabel('$\Delta d_h~[m]$', size=15)
# ax.grid()
ax_gmr.legend(fontsize=15, ncol=2, fancybox=True, framealpha=1)
ax.set_xticks([0, 0.1, 0.2, 0.3])
ax.set_xticklabels(['$0$', '$0.1$', '$0.2$', '$0.3$'], size=13)
ax.set_xlim([0, 0.32])
ax.set_yticks([0, -0.15, -.266, -.3])
ax.set_yticklabels(['$0$', '$-0.15$', '$d_h$', '$-0.30$'], size=13)
ax.axhline(-0.266, color='k', linestyle='--', linewidth=2)
# ax.set_xticklabels([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax_gmr.set_ylabel('$\Delta d_h~[m]$')
ax_gmr.set_xlabel('$Time~[s]$', size=15)
ax_gmr.set_xlim([0, 0.32])
ax_gmr.set_xticks([0, 0.1, 0.2, 0.3])
ax_gmr.set_xticklabels(['$0$', '$0.1$', '$0.2$', '$0.3$'], size=13)
ax_gmr.set_yticks([0, -0.15, -.30])
ax_gmr.set_yticklabels(['$0$', '$-0.15$', '$-0.30$'], size=13)
# ax_gmr.set_xticklabels([])
ax_gmr.spines['top'].set_visible(False)
ax_gmr.spines['right'].set_visible(False)

fig.savefig('gmmgmr_position_demo.png', dpi=400)
fig_gmr.savefig('gmmgmr_position_gmr.png', dpi=400)
# plt.show()

#############################
xlim_plot = [time[0], 1000/1000]#N_POINTS/1000]
# ylim_plot = [-33, 4]
# ylabel = '$\\boldsymbol{F}~[N]$'    
# xlabel = '$\\boldsymbol{t}~[s]$'
xlabel = ''
# xticks = [0, 0.1, 0.2, 0.3, 0.4, 0.5]#, 0.75, 1.0]
# xtickslabels = ['$0$', '$0.1$', '$0.2$', '$0.3$','$0.4$', '$0.5$']#, '$0.75$', '$1.0$']
# xticks = []
# xtickslabels = []
# yticks = [0, -10, -20, -30]
# ytickslabels = ['$0$', '$-10$', '$-20$', '$-30$']
fig_size = [5, 2]  # width, height

file_name = 'impedance_z_profiles.mat'
print(file_name)
n_subplots=1
fig, ax = plt.subplots(n_subplots,figsize=fig_size, constrained_layout=True)
f = loadmat(file_name)
print(f.keys())
time = np.array(f['x2'])
z_imp = np.array(f['y2'])
print(time.shape)
print(z_imp.shape)
# ax.plot(time.T,z_imp.T, linewidth=3, color='r')
ax.plot(-delta_d, z_imp.T, linewidth=4, color='k')
# ax.plot(z_imp.T, z_impedance, linewidth=3, color='k')
ax.set_xlabel('$-\Delta d_h~[m]$', size=15)
ax.set_ylabel('$K_{HVS}~[N/m]$', size=15)
# ax.grid()
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.set_xticks([0, 0.1, 0.2, 0.266, 0.3])
ax.set_xticklabels(['$0$', '$0.1$', '$0.2$', '$d_h$', '$0.3$'], size=13)
ax.set_xlim([0, 0.3])
ax.axvline(0.266, ymin=0, ymax=0.95, color='k', linestyle='--', linewidth=2)

ax.set_yticks([0, 500, 1000])
ax.set_yticklabels(['$0$', '$500$', '$1000$'], size=13)

fig.savefig('human_stiffness.png', dpi=400, bbox_inches='tight')
plt.show()