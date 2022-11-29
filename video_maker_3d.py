import numpy as np
import h5py
import os
import matplotlib.pyplot as plt
import matplotlib.animation as animation

N_POINTS = 90000
STEP = 500
FPS = 60
f_size = 20
legend_size = 14
ms = 8
lw = 2
plot_row = 0
plot_col = 0
init = 4235
force_lim_plot = [-80, 130]
pos_lim_plot = [-1.1, 1.1]
stiff_lim_plot = [200, 800]
tank_lim_plot = [0, 10]
xlim_plot = [0, 90]
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
    idx = j*STEP + init

    x_interval = np.arange(0, (np.array(data.get('log_desired_online_force'))[init:idx].shape[0] + 1) * dt, dt)

    ### force ###
    plots_fx_d.set_data(x_interval, np.array(data.get('log_desired_online_force'))[(init - 1):idx, x])
    plots_fy_d.set_data(x_interval, np.array(data.get('log_desired_online_force'))[(init - 1):idx, y])
    plots_fz_d.set_data(x_interval, np.array(data.get('log_desired_online_force'))[(init - 1):idx, z])

    plots_fx.set_data(x_interval, np.array(data.get('compute_force'))[(init - 1):idx, x])
    plots_fy.set_data(x_interval, np.array(data.get('compute_force'))[(init - 1):idx, y])
    plots_fz.set_data(x_interval, np.array(data.get('compute_force'))[(init - 1):idx, z])

    ### position ###
    plots_x_d.set_data(x_interval, np.array(data.get('eeinw_desired_position'))[(init - 1):idx, x])
    plots_x_d.set_data(x_interval, np.array(data.get('eeinw_desired_position'))[(init - 1):idx, y])
    plots_x_d.set_data(x_interval, np.array(data.get('eeinw_desired_position'))[(init - 1):idx, z])

    plots_x.set_data(x_interval, np.array(data.get('eeinw_feedback_position'))[(init - 1):idx, x])
    plots_y.set_data(x_interval, np.array(data.get('eeinw_feedback_position'))[(init - 1):idx, y])
    plots_z.set_data(x_interval, np.array(data.get('eeinw_feedback_position'))[(init - 1):idx, z])

    ### stiffness ###
    plots_kx.set_data(x_interval, np.array(data.get('generated_stiff'))[(init - 1):idx, x])
    plots_ky.set_data(x_interval, np.array(data.get('generated_stiff'))[(init - 1):idx, y])
    plots_kz.set_data(x_interval, np.array(data.get('generated_stiff'))[(init - 1):idx, z])
    
    return sub1, sub2, sub3



def load_data():
    ''' IMPORT DATA '''
    curr_dir = os.getcwd()
    results_dir = '/final results/Repeat/'
    os_data = h5py.File(curr_dir + results_dir + 'moca_red_qp_pbd_planner_1.mat', 'r')
    return os_data


def create_axis():
    sub1.set_ylabel('$F$ $[N]$', size=f_size)
    sub1.set_xlim(xlim_plot)
    sub1.set_ylim(force_lim_plot)
    sub1.set_xticks([])
    sub1.grid(which='major', alpha=0.2, linestyle='--')
    plot_list = [plots_fx_d, plots_fy_d, plots_fz_d, plots_fx, plots_fy, plots_fz]
    new_order = [0, 3, 1, 4, 2, 5]
    new_plots_list = [plot_list[i] for i in new_order]
    labels = ('$F^d_x$', '$\hat{F}_x$', '$F^d_y$', '$\hat{F}_y$', '$F^d_z$', '$\hat{F}_z$')
    sub1.legend(new_plots_list, labels, ncol=6, borderaxespad=0.1,
                handlelength=0.8, fontsize=legend_size)


    sub2.set_ylabel(r'$\boldsymbol{x}$ $[m]$', size=f_size)
    sub2.set_xlim(xlim_plot)
    sub2.set_ylim(pos_lim_plot)
    sub2.set_xticks([])
    sub2.set_title(' ', size=f_size, fontweight='light')
    plot_list = [plots_x_d, plots_y_d, plots_z_d, plots_x, plots_y, plots_z]
    new_order = [0, 3, 1, 4, 2, 5]
    new_plots_list = [plot_list[i] for i in new_order]
    labels = ('$x^d$', '$x$', '$y^d$', '$y$', '$z^d$', '$z$')
    sub2.grid(which='major', alpha=0.2, linestyle='--')
    sub2.legend(new_plots_list, labels, ncol=6, handlelength=0.6, borderaxespad=0.1,
                loc='lower left', fontsize=legend_size)


    sub3.set_ylabel('$k_i$ $[N/m]$', size=f_size)
    sub3.set_xlim(xlim_plot)
    sub3.set_ylim(stiff_lim_plot)
    sub3.set_xticks([])
    sub3.set_title(' ', size=f_size, fontweight='light')
    sub3.grid(which='major', alpha=0.2, linestyle='--')
    new_plots_list = [plots_kx, plots_ky, plots_kz]
    labels = ('$k_x$', '$k_y$', '$k_z$')
    sub3.legend(new_plots_list, labels, ncol=3, borderaxespad=0.1, fontsize=legend_size)

    return sub1, sub2, sub3


########### PLOT CONFIG ##############
fig = plt.figure(constrained_layout=False, figsize=[10, 8])
sub1 = fig.add_subplot(3, 1, 1)
sub2 = fig.add_subplot(3, 1, 2)
sub3 = fig.add_subplot(3, 1, 3)

###### OS #######
data = load_data()

idx = init

sub1.set_prop_cycle(color=['red', 'green', 'blue', 'red', 'green', 'blue'])
plots_fx_d, plots_fy_d, plots_fz_d, = sub1.plot(np.arange(0, (np.array(data.get('log_desired_online_force'))[init:idx].shape[0] + 1) * dt, dt),
                                                np.array(data.get('log_desired_online_force'))[(init - 1):idx],
                                                linewidth=lw - 0.5, linestyle='dashed')
plots_fx, plots_fy, plots_fz, = sub1.plot(np.arange(0, (np.array(data.get('compute_force'))[init:idx].shape[0] + 1) * dt, dt),
                                          np.array(data.get('compute_force'))[(init - 1):idx],
                                          linewidth=lw)

sub2.set_prop_cycle(color=['red', 'green', 'blue', 'red', 'green', 'blue'])
plots_x_d, plots_y_d, plots_z_d, = sub2.plot(np.arange(0, (np.array(data.get('eeinw_desired_position'))[init:idx].shape[0]+1) * dt, dt),
                                             np.array(data.get('eeinw_desired_position'))[(init - 1):idx],
                                             linewidth=lw - 0.5, linestyle='dashed')
plots_x, plots_y, plots_z, = sub2.plot(np.arange(0, (np.array(data.get('eeinw_feedback_position'))[init:idx].shape[0]+1) * dt, dt),
                                       np.array(data.get('eeinw_feedback_position'))[(init - 1):idx],
                                       linewidth=lw)

sub3.set_prop_cycle(color=['red', 'green', 'blue'])
plots_kx, plots_ky, plots_kz, = sub3.plot(np.arange(0, (np.array(data.get('generated_stiff'))[init:idx].shape[0] + 1) * dt, dt),
                                          np.array(data.get('generated_stiff'))[(init - 1):idx],
                                          linewidth=lw)

n_frames = int((N_POINTS-init)/STEP)
print("n_frames = ", n_frames)
print("video duration = +-", n_frames/FPS)

animation_1 = animation.FuncAnimation(plt.gcf(), animate, init_func=create_axis, interval=0, repeat=False, frames=n_frames)

### visualization
plt.show()

### creating and saving the video
# writervideo = animation.FFMpegWriter(fps=FPS)
# file_path = 'video.mp4'
# animation_1.save(file_path, writer=writervideo)
