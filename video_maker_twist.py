import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
from data_plot_class import DataPlotHelper

STEP = 5
FPS = 60
f_size = 20
legend_size = 14
ms = 8
lw = 2
plot_row = 0
plot_col = 0
# init = 4235
force_lim_plot = [-1.5, 0.5]
pos_lim_plot = [-1.1, 1.1]
stiff_lim_plot = [200, 800]
tank_lim_plot = [0, 10]
# xlim_plot = [0, 4]
dt = 1e-3
x = 0
y = 1
z = 2

plt.rcParams.update({'font.size': f_size})
plt.rcParams['text.usetex'] = True ## enable TeX style labels
plt.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"

def animate(j):
    # TODO: we need to deal with a problem related to arrays size. Supposed we have 10 points with step of 3,
    #       so we will plot [3, 3, 3] but one point will be left out. So we need to guarante to plot [3, 3, 3, 1]
    idx = j*STEP + idx_start

    # x_interval = np.arange(0, (np.array(data.get('log_desired_online_force'))[init:idx].shape[0] + 1) * dt, dt)

    ### force ###
    plots_ee_twist.set_data(time[idx_start:idx]-time[idx_start], EE_twist[idx_start:idx,2],linewidth=lw)
    plots_ee_twist_d.set_data(time[idx_start:idx]-time[idx_start], EE_twist_d[idx_start:idx,2],linewidth=lw- 0.5)
    # plots_fz_d.set_data(x_interval, np.array(data.get('log_desired_online_force'))[(init - 1):idx, z])

    # plots_fx.set_data(x_interval, np.array(data.get('compute_force'))[(init - 1):idx, x])
    # plots_fy.set_data(x_interval, np.array(data.get('compute_force'))[(init - 1):idx, y])
    # plots_fz.set_data(x_interval, np.array(data.get('compute_force'))[(init - 1):idx, z])

    # ### position ###
    # plots_x_d.set_data(x_interval, np.array(data.get('eeinw_desired_position'))[(init - 1):idx, x])
    # plots_x_d.set_data(x_interval, np.array(data.get('eeinw_desired_position'))[(init - 1):idx, y])
    # plots_x_d.set_data(x_interval, np.array(data.get('eeinw_desired_position'))[(init - 1):idx, z])

    # plots_x.set_data(x_interval, np.array(data.get('eeinw_feedback_position'))[(init - 1):idx, x])
    # plots_y.set_data(x_interval, np.array(data.get('eeinw_feedback_position'))[(init - 1):idx, y])
    # plots_z.set_data(x_interval, np.array(data.get('eeinw_feedback_position'))[(init - 1):idx, z])

    # ### stiffness ###
    # plots_kx.set_data(x_interval, np.array(data.get('generated_stiff'))[(init - 1):idx, x])
    # plots_ky.set_data(x_interval, np.array(data.get('generated_stiff'))[(init - 1):idx, y])
    # plots_kz.set_data(x_interval, np.array(data.get('generated_stiff'))[(init - 1):idx, z])
    
    return sub1



def create_axis():
    sub1.set_ylabel('$\dot{x}$ $[m/s]$', size=f_size)
    sub1.set_xlabel('$time$ $[s]$', size=f_size)
    sub1.set_xlim(xlim_plot)
    sub1.set_ylim(ylim_plot)
    sub1.set_xticks([0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5])
    sub1.set_xticklabels(['$0$', '$0.25$', '$0.5$', '$0.75$', '$1.0$', '$1.25$', '$1.5$'])
    sub1.grid(which='major', alpha=0.2, linestyle='--')
    # labels = ('$F^d_x$', '$\hat{F}_x$', '$F^d_y$', '$\hat{F}_y$', '$F^d_z$', '$\hat{F}_z$')
    labels = ['$\dot{x}_d$', '$\dot{x}$']
    # sub1.legend(label=labels, ncol=6, borderaxespad=0.1,
    #             handlelength=0.8, fontsize=legend_size)
    sub1.legend(labels=labels, borderaxespad=0.1,
                handlelength=0.8, fontsize=legend_size, loc='lower right')

    fig.tight_layout()

    # sub2.set_ylabel(r'$\boldsymbol{x}$ $[m]$', size=f_size)
    # sub2.set_xlim(xlim_plot)
    # sub2.set_ylim(pos_lim_plot)
    # sub2.set_xticks([])
    # sub2.set_title(' ', size=f_size, fontweight='light')
    # plot_list = [plots_x_d, plots_y_d, plots_z_d, plots_x, plots_y, plots_z]
    # new_order = [0, 3, 1, 4, 2, 5]
    # new_plots_list = [plot_list[i] for i in new_order]
    # labels = ('$x^d$', '$x$', '$y^d$', '$y$', '$z^d$', '$z$')
    # sub2.grid(which='major', alpha=0.2, linestyle='--')
    # sub2.legend(new_plots_list, labels, ncol=6, handlelength=0.6, borderaxespad=0.1,
    #             loc='lower left', fontsize=legend_size)


    # sub3.set_ylabel('$k_i$ $[N/m]$', size=f_size)
    # sub3.set_xlim(xlim_plot)
    # sub3.set_ylim(stiff_lim_plot)
    # sub3.set_xticks([])
    # sub3.set_title(' ', size=f_size, fontweight='light')
    # sub3.grid(which='major', alpha=0.2, linestyle='--')
    # new_plots_list = [plots_kx, plots_ky, plots_kz]
    # labels = ('$k_x$', '$k_y$', '$k_z$')
    # sub3.legend(new_plots_list, labels, ncol=3, borderaxespad=0.1, fontsize=legend_size)

    return sub1, #sub2, sub3


########### PLOT CONFIG ##############
fig = plt.figure(constrained_layout=False, figsize=[10, 4])
sub1 = fig.add_subplot(1, 1, 1)
# sub2 = fig.add_subplot(3, 1, 2)
# sub3 = fig.add_subplot(3, 1, 3)

path_folder = 'data/icra2023/2-exp-ImpLoop3-NoAdaptation-PARTIAL-RESULTS-DONT-EDIT/'
data_info = pd.read_csv(path_folder+'data_info.csv')

dh = DataPlotHelper(path_folder)

params = {  
    'vic': False,
    'color': 'Blue',
    'trial_idx': 1,
    'height': 27,
    'impedance_loop': 30,
    'online_adaptation': False,
    'i_initial': 0,
    'i_final': -1,
    'data_to_plot': 'FT_ati',
    }

idx_start = dh.get_idx_from_file(params, data_info, 'idx_start')[0]
idx_end = dh.get_idx_from_file(params, data_info, 'idx_end')[0]
print(idx_start)
print(idx_end)
init = idx_start

N_POINTS = idx_end - idx_start  # if idx_end != -1 else 

params['data_to_plot'] = 'time'
time = dh.get_data(params)

params['data_to_plot'] = 'EE_twist_d'
EE_twist_d = dh.get_data(params)

params['data_to_plot'] = 'EE_twist'
EE_twist = dh.get_data(params)

sub1.set_prop_cycle(color=['black', 'blue', 'blue', 'red', 'green', 'blue'])
plots_ee_twist_d, = sub1.plot(time[init:idx_start]-time[idx_start], EE_twist_d[init:idx_start, 2],linewidth=lw - 0.5, linestyle='dashed')
plots_ee_twist, = sub1.plot(time[init:idx_start]-time[idx_start], EE_twist[init:idx_start, 2],linewidth=lw)

xlim_plot = [time[idx_start]-time[idx_start], time[idx_end]-time[idx_start]]
ylim_plot = [-1.5, 0.5]

# plots_ee_twist_d, = sub1.plot([0], EE_twist_d[idx_start, 2])


# plots_fx, plots_fy, plots_fz, = sub1.plot(np.arange(0, (np.array(data.get('compute_force'))[init:idx].shape[0] + 1) * dt, dt),
#                                           np.array(data.get('compute_force'))[(init - 1):idx],
#                                           linewidth=lw)

# sub2.set_prop_cycle(color=['red', 'green', 'blue', 'red', 'green', 'blue'])
# plots_x_d, plots_y_d, plots_z_d, = sub2.plot(np.arange(0, (np.array(data.get('eeinw_desired_position'))[init:idx].shape[0]+1) * dt, dt),
#                                              np.array(data.get('eeinw_desired_position'))[(init - 1):idx],
#                                              linewidth=lw - 0.5, linestyle='dashed')
# plots_x, plots_y, plots_z, = sub2.plot(np.arange(0, (np.array(data.get('eeinw_feedback_position'))[init:idx].shape[0]+1) * dt, dt),
#                                        np.array(data.get('eeinw_feedback_position'))[(init - 1):idx],
#                                        linewidth=lw)

# sub3.set_prop_cycle(color=['red', 'green', 'blue'])
# plots_kx, plots_ky, plots_kz, = sub3.plot(np.arange(0, (np.array(data.get('generated_stiff'))[init:idx].shape[0] + 1) * dt, dt),
#                                           np.array(data.get('generated_stiff'))[(init - 1):idx],
#                                           linewidth=lw)

n_frames = int((N_POINTS)/STEP)
print("n_frames = ", n_frames)
print("video duration = +-", n_frames/FPS)

animation_1 = animation.FuncAnimation(plt.gcf(), animate, init_func=create_axis, interval=1, repeat=False, frames=n_frames)

### visualization
# plt.show()

video_name = params['color'] + '_opt_kmp_' 
video_name += 'vic_' if params['vic'] else ''
video_name += str(params['trial_idx']) + '_'
video_name += params['data_to_plot'] + '.mp4'

### creating and saving the video
writervideo = animation.FFMpegWriter(fps=50)
animation_1.save(path_folder + video_name, writer=writervideo)
