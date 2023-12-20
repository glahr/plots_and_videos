import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib.gridspec import GridSpec
import matplotlib.colors as mcolors
import pandas as pd
from data_plot_class_tro import DataPlotHelper
import os
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise
from cycler import cycler
import palettable
from palettable.cartocolors.qualitative import Vivid_10 as myc
from scipy.signal import find_peaks as peaks
from data_class import DataClass
import matplotlib.animation as animation


z_catching = 0.35
lw = 2

ball_radius = (95/2/1000)
tau_limits = np.array([87, 87, 87, 87, 12, 12, 12])

chosen_exps = {'fp-kh': 26, 'fp-kl': 25, 'vm-kl': 7, 'vm-kh': 3, 'vm-sic': 9,'vm-vic': 14, 'vm-vic-dim': 17}
# chosen_exps_joints = [11, 13, 14, 17]
chosen_exps_joints = [11, 13, 14, 17]

p = os.getcwd()
path_folder = 'tro2023/data/'
files_names = os.listdir(path=path_folder)
files_names.sort()

dh = DataPlotHelper(path_folder=path_folder, files_names=files_names)
data_info = pd.read_csv(p+'/tro2023/data_info.csv')

params = {
    'idx': 1,
    'type': None,
    'rmm': False,
    'height': None,
    'poc': False,
    'gain': None,
    'idx_initial': 0,
    'idx_end': -1,
    'data_to_plot': None,
}

##### TESTING: OK
# file_name = '12-vm+sic+rmm-1.mat'
# params['data_to_plot'] = 'ft_'
# ft_ = dh.get_data(params, file_name=file_name)

# params['data_to_plot'] = 'time'
# time = dh.get_data(params, file_name=file_name)
# time -= time[0]

# params['data_to_plot'] = 'Kp'
# Kp = dh.get_data(params, file_name=file_name)

# plt.plot(time, Kp)
# plt.show()

xlimits = [0, 1]

def kalman_filtering_1d(time_, z_ball_, ft_, increased_height=False):
    dim_x = 2  # [z, z_dot]
    dim_z = 1
    dt = 1/500
    g = 9.81

    kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)

    # state extrapolation: x[n+1] = F x[n] + G u[n]

    if not increased_height:
        z_init = 0.7
    else:
        z_init = 0.8

    kf.x = np.array([[z_init],
                     [0]])   # x: state vector (nx x 1)
    kf.F = np.array([[1, dt],                                       # F: state transition matrix (nx x nx)
                     [0, 1]])
    kf.H = np.array([[1, 0]])                                       # H: observation matrix (nz x nx)
    kf.B = np.array([[0.5*dt*dt],
                     [dt]])                             # B: control matrix (nx x nu)
    kf.P *= 100.                                                   # P: estimate uncertainty
    kf.R = 0.5                                                        # R: measurement uncertainty
    kf.Q = Q_discrete_white_noise(dim=dim_x, dt=dt, var=1)       # Q: process noise uncertainty

    z_actual_hat = np.zeros_like(z_ball_)
    z_dot_actual_hat = np.zeros_like(z_ball_)
    time_intersec = np.zeros_like(z_ball_)
    z_dot_intersec = np.zeros_like(z_ball_)
    z_intersec = np.zeros_like(z_ball_)

    last_t = 0
    had_impact = False

    if not increased_height:
        z_threshold = 0.665
    else:
        z_threshold = 0.734

    for i, ti in enumerate(time_):
        # if vel_[i] != 0:
        if ti - last_t > dt:
            if z_ball_[i] < z_threshold and z_ball_[i] > 0:
                kf.predict(u=-9.81)                                         # u: input variable
                kf.update(z=z_ball_[i])                                           # z: output vector

                if not had_impact:
                    intersec_time = (kf.x[1] + np.sqrt(kf.x[1]*kf.x[1]+2*g*(kf.x[0]-z_catching)))/g
                    z_intersec[i] = z_catching
                    z_dot_intersec[i] = -g*intersec_time + kf.x[1]
                    z_actual_hat[i]            = kf.x[0]
                    z_dot_actual_hat[i]        = kf.x[1]
                    time_intersec[i]           = intersec_time
                    if abs(ft_[i]) > 3:
                        had_impact = True
                last_t = ti
    

    return z_actual_hat[np.nonzero(z_actual_hat)], \
            z_dot_actual_hat[np.nonzero(z_dot_actual_hat)], \
            z_intersec[np.nonzero(z_intersec)], \
            z_dot_intersec[np.nonzero(z_dot_intersec)], \
            time_intersec[np.nonzero(time_intersec)], \
            time_[np.nonzero(time_intersec)]

def kalman_filtering_2d(time_, z_ball_, y_ball_, ft_):
    dim_x = 4  # [z, z_dot, y, y_dot]
    dim_z = 2
    dt = 1/500
    g = 9.81

    kf = KalmanFilter(dim_x=dim_x, dim_z=dim_z)

    # state extrapolation: x[n+1] = F x[n] + G u[n]

    kf.x = np.array([[0.3],     # z_init     
                     [1],       # z_dot_init 
                     [-1.6],    # y_init
                     [2.5]])    # y_dot_init

    kf.F = np.array([[1, dt, 0, 0],     # [z]
                     [0, 1, 0, 0],      # [z_dot]
                     [0, 0, 1, dt],     # [y]
                     [0, 0, 0, 1]])     # [y_dot]

    kf.H = np.array([[1, 0, 0, 0],
                     [0, 0, 1, 0]])

    kf.B = np.array([[0.5*dt*dt],   # [z]
                     [dt],          # [z_dot]
                     [0],           # [y]
                     [0]])          # [y_dot]

    kf.P *= 100.                                                   # P: estimate uncertainty
    kf.R = 0.5                                                        # R: measurement uncertainty
    kf.Q = Q_discrete_white_noise(dim=dim_x, dt=dt, var=0.1)       # Q: process noise uncertainty

    z_actual_hat = np.zeros_like(z_ball_)
    z_dot_actual_hat = np.zeros_like(z_ball_)
    z_dot_intersec = np.zeros_like(z_ball_)
    z_intersec = np.zeros_like(z_ball_)
    time_intersec = np.zeros_like(z_ball_)
    y_actual_hat = np.zeros_like(z_ball_)
    y_dot_actual_hat = np.zeros_like(z_ball_)
    y_dot_intersec = np.zeros_like(z_ball_)
    y_intersec = np.zeros_like(z_ball_)

    last_t = 0
    had_impact = False

    for i, ti in enumerate(time_):
        # if vel_[i] != 0:
        if ti - last_t > dt:
            if z_ball_[i] > 0.05:
                kf.predict(u=-9.81)                                         # u: input variable
                kf.update(z=np.array([[z_ball_[i]],
                                      [y_ball_[i]]]))                                           # z: output vector

                if not had_impact:
                    intersec_time = (kf.x[1] + np.sqrt(kf.x[1]*kf.x[1]+2*g*(kf.x[0]-z_ball_[i])))/g
                    z_intersec[i] = z_catching
                    z_dot_intersec[i] = -g*intersec_time + kf.x[1]
                    z_actual_hat[i]            = kf.x[0]
                    z_dot_actual_hat[i]        = kf.x[1]
                    y_intersec[i] = kf.x[2] + kf.x[3]*intersec_time
                    y_dot_intersec[i] = kf.x[3]
                    y_actual_hat[i]            = kf.x[2]
                    y_dot_actual_hat[i]        = kf.x[3]
                    time_intersec[i]           = intersec_time
                    if np.linalg.norm(ft_[i]) > 3:
                        had_impact = True
                last_t = ti
    

    return z_actual_hat[np.nonzero(z_actual_hat)], \
            z_dot_actual_hat[np.nonzero(z_dot_actual_hat)], \
            z_intersec[np.nonzero(z_intersec)], \
            z_dot_intersec[np.nonzero(z_dot_intersec)], \
            y_actual_hat[np.nonzero(y_actual_hat)], \
            y_dot_actual_hat[np.nonzero(y_dot_actual_hat)], \
            y_intersec[np.nonzero(y_intersec)], \
            y_dot_intersec[np.nonzero(y_dot_intersec)], \
            time_intersec[np.nonzero(time_intersec)], \
            time_[np.nonzero(time_intersec)]

##### initial plots
def get_legend(file_name):
    leg = ''
    leg += 'vm_'  if 'vm' in file_name else ''
    leg += 'fp_'  if 'fp' in file_name else ''
    # gains
    leg += 'sic_' if 'sic'  in file_name else ''
    leg += 'low_' if 'low'  in file_name else ''
    leg += 'high_' if 'high' in file_name else ''
    leg += 'poc_'  if 'poc'  in file_name else ''
    # rmm
    leg += 'rmm_'  if 'rmm'  in file_name else ''
    # idx
    leg += file_name[-5]
    return leg

def plot_all_vanilla():
    
    idx_exps = [17, 18]
    all_groups = [[1, 2],
                  [3, 4, 5],
                  [6, 7, 8],
                  [9, 10, 11],
                  [12, 13],
                  [14, 15, 16],
                  [17, 18],
                  ]

    colors = ['b', 'r', 'g']

    # # dh.set_axis(xlim_plot=xlim_plot, xlabel=xlabel, xticks=xticks, xtickslabels=xtickslabels,
    # #             ylim_plot=ylim_plot, ylabel=ylabel, yticks=yticks, ytickslabels=ytickslabels,
    # #             fig_size=fig_size)

    # xlimits = [0, 2]
    # ylimits_ft_ = []
    # ylimits_Kp = []
    # ylimits_pos = []
    # ylimits_vel = []

    for idx_exps in all_groups:
        i = 0
        fig_ft, ax_ft = plt.subplots()
        fig_Kp, ax_Kp  = plt.subplots()
        fig_pos, ax_pos  = plt.subplots()
        fig_vel, ax_vel  = plt.subplots()

        legend_ft = []
        legend_Kp = []
        legend_pos = []
        legend_vel = []
        for idx_exp in idx_exps:
            file_name = dh._get_name(idx_exp)
            print(file_name)
            params['idx_initial'] = dh.get_idx_from_file(idx_exp, data_info, idx_name='idx_start')
            params['idx_end'] = dh.get_idx_from_file(idx_exp, data_info, idx_name='idx_end')

            time = dh.get_data(file_name=file_name,params=params, data_to_plot='time')
            time -= time[0]

            ft_ = dh.get_data(file_name=file_name, params=params, data_to_plot='ft_')[:,2]
            ax_ft.plot(time, ft_, linewidth=lw)
            ax_ft.set_xlabel('$Time~[s]$')
            ax_ft.set_ylabel('$F_z~[N]$')
            ax_ft.set_ylim([-12, 5])
            legend_ft.append(get_legend(file_name))

            Kp = dh.get_data(params, file_name=file_name, data_to_plot='Kp')[:,2]
            ax_Kp.plot(time, Kp, linewidth=lw)
            ax_Kp.set_xlabel('$Time~[s]$')
            ax_Kp.set_ylabel('$Kp~[-]$')
            ax_Kp.set_ylim([10, 50])
            legend_Kp.append(get_legend(file_name))

            pos = dh.get_data(params, file_name=file_name, data_to_plot='EE_position')[:,2]
            pos_d = dh.get_data(params, file_name=file_name, data_to_plot='EE_position_d')[:,2]
            ax_pos.plot(time, pos, colors[i], linewidth=lw)
            ax_pos.plot(time, pos_d, colors[i]+'--', linewidth=lw-0.5)
            ax_pos.set_xlabel('$Time~[s]$')
            ax_pos.set_ylabel('$z~[m]$')
            ax_pos.set_ylim([0.1, 0.5])
            legend_pos.append(get_legend(file_name))
            legend_pos.append(get_legend(file_name)+'_d')

            vel = dh.get_data(params, file_name=file_name, data_to_plot='EE_twist')[:,2]
            vel_d = dh.get_data(params, file_name=file_name, data_to_plot='EE_twist_d')[:,2]
            ax_vel.plot(time, vel, colors[i], linewidth=lw)
            ax_vel.plot(time, vel_d, colors[i]+'--', linewidth=lw-0.5)
            ax_vel.set_xlabel('$Time~[s]$')
            ax_vel.set_ylabel('$\dot{z}~[m/s]$')
            ax_vel.set_ylim([-1.5, 0.7])
            legend_vel.append(get_legend(file_name))
            legend_vel.append(get_legend(file_name)+'_d')

            i += 1


        fig_ft.legend(legend_ft)
        fig_Kp.legend(legend_Kp)
        fig_pos.legend(legend_pos)
        fig_vel.legend(legend_vel)

        fig_ft.tight_layout()
        fig_Kp.tight_layout()
        fig_pos.tight_layout()
        fig_vel.tight_layout()

        ax_ft.grid()
        ax_Kp.grid()
        ax_pos.grid()
        ax_vel.grid()
    plt.show()
        # fig_ft.savefig('images/'+file_name[3:-6]+'_ft.png')
        # fig_Kp.savefig('images/'+file_name[3:-6]+'_Kp.png')
        # fig_pos.savefig('images/'+file_name[3:-6]+'_pos.png')
        # fig_vel.savefig('images/'+file_name[3:-6]+'_vel.png')

def plot_vanilla_best_all_in_one():
    colors = ['b', 'r', 'g', 'y']
    chosen_exps = [3, 8, 11, 16]
    i = 0
    fig_ft, ax_ft = plt.subplots()
    fig_Kp, ax_Kp  = plt.subplots()
    fig_pos, ax_pos  = plt.subplots()
    fig_vel, ax_vel  = plt.subplots()

    legend_ft = []
    legend_Kp = []
    legend_pos = []
    legend_vel = []

    for idx_exp in chosen_exps:
        file_name = dh._get_name(idx_exp)
        print(file_name)
        params['idx_initial'] = dh.get_idx_from_file(idx_exp, data_info, idx_name='idx_start')
        params['idx_end'] = dh.get_idx_from_file(idx_exp, data_info, idx_name='idx_end')

        time = dh.get_data(file_name=file_name,params=params, data_to_plot='time')
        time -= time[0]

        ft_ = dh.get_data(file_name=file_name, params=params, data_to_plot='ft_')[:,2]
        ax_ft.plot(time, ft_, linewidth=lw)
        ax_ft.set_xlabel('$Time~[s]$')
        ax_ft.set_ylabel('$F_z~[N]$')
        ax_ft.set_xlim(xlimits)
        ax_ft.set_ylim([-12, 5])
        legend_ft.append(get_legend(file_name))

        Kp = dh.get_data(params, file_name=file_name, data_to_plot='Kp')[:,2]
        ax_Kp.plot(time, Kp, linewidth=lw)
        ax_Kp.set_xlabel('$Time~[s]$')
        ax_Kp.set_ylabel('$Kp~[-]$')
        ax_ft.set_xlim(xlimits)
        ax_Kp.set_ylim([10, 50])
        legend_Kp.append(get_legend(file_name))

        pos = dh.get_data(params, file_name=file_name, data_to_plot='EE_position')[:,2]
        pos_d = dh.get_data(params, file_name=file_name, data_to_plot='EE_position_d')[:,2]
        ax_pos.plot(time, pos, colors[i], linewidth=lw)
        ax_pos.plot(time, pos_d, colors[i]+'--', linewidth=lw-0.5)
        ax_pos.set_xlabel('$Time~[s]$')
        ax_pos.set_ylabel('$z~[m]$')
        ax_ft.set_xlim(xlimits)
        ax_pos.set_ylim([0.1, 0.5])
        legend_pos.append(get_legend(file_name))
        legend_pos.append(get_legend(file_name)+'_d')

        vel = dh.get_data(params, file_name=file_name, data_to_plot='EE_twist')[:,2]
        vel_d = dh.get_data(params, file_name=file_name, data_to_plot='EE_twist_d')[:,2]
        ax_vel.plot(time, vel, colors[i], linewidth=lw)
        ax_vel.plot(time, vel_d, colors[i]+'--', linewidth=lw-0.5)
        ax_vel.set_xlabel('$Time~[s]$')
        ax_vel.set_ylabel('$\dot{z}~[m/s]$')
        ax_ft.set_xlim(xlimits)
        ax_vel.set_ylim([-1.5, 0.7])
        legend_vel.append(get_legend(file_name))
        legend_vel.append(get_legend(file_name)+'_d')

        i += 1

    fig_ft.legend(legend_ft)
    fig_Kp.legend(legend_Kp)
    fig_pos.legend(legend_pos)
    fig_vel.legend(legend_vel)

    fig_ft.tight_layout()
    fig_Kp.tight_layout()
    fig_pos.tight_layout()
    fig_vel.tight_layout()

    ax_ft.grid()
    ax_Kp.grid()
    ax_pos.grid()
    ax_vel.grid()
    plt.show()
    # fig_ft.savefig('images/vanilla_best_ft.png')
    # fig_Kp.savefig('images/vanilla_best_Kp.png')
    # fig_pos.savefig('images/vanilla_best_pos.png')
    # fig_vel.savefig('images/vanilla_best_vel.png')

def plot_vanilla_best_4subplots_4figs():
    colors = ['b', 'r', 'g', 'y']
    styles = ['solid', 'dotted', 'dashdot', 'on-off-dash-seq']
    chosen_exps = [3, 8, 11, 16]
    i = 0
    fig_ft, ax_ft = plt.subplots(4,1)
    fig_Kp, ax_Kp  = plt.subplots(4,1)
    fig_pos, ax_pos  = plt.subplots(4,1)
    fig_vel, ax_vel  = plt.subplots(4,1)

    legend_ft = [r'$F^{VM+K_H}_z$', r'$F^{VM+K_L}_z$', r'$F^{VM+SIC}_z$', r'$F^{VM+POC}_z$']
    legend_Kp = [r'$K_H$', r'$K_L$', r'$SIC$', r'$POC$']
    legend_pos = [r'$z^{VM+K_H}$', r'$z^{VM+K_L}$', r'$z^{VM+SIC}$', r'$z^{VM+POC}$']
    legend_vel = [r'$\dot{z}^{VM+K_H}$', r'$\dot{z}^{VM+K_L}$', r'$\dot{z}^{VM+SIC}$', r'$\dot{z}^{VM+POC}$']

    for j, idx_exp in enumerate(chosen_exps):
        file_name = dh._get_name(idx_exp)
        print(file_name)
        params['idx_initial'] = dh.get_idx_from_file(idx_exp, data_info, idx_name='idx_start')
        params['idx_end'] = dh.get_idx_from_file(idx_exp, data_info, idx_name='idx_end')

        time = dh.get_data(file_name=file_name,params=params, data_to_plot='time')
        time -= time[0]

        ft_ = dh.get_data(file_name=file_name, params=params, data_to_plot='ft_')[:,2]
        ax_ft[j].plot(time, ft_, colors[i], linewidth=lw, label=legend_ft[i])
        # ax_ft[j].set_xlabel('$Time~[s]$')
        # ax_ft[j].set_ylabel('$F_z~[N]$')
        ax_ft[j].set_xlim(xlimits)
        ax_ft[j].set_ylim([-10, 5])
        # ax_ft[j].legend(prop={'size': 8}, loc='lower right')
        

        Kp = dh.get_data(params, file_name=file_name, data_to_plot='Kp')[:,2]
        ax_Kp[j].plot(time, Kp, colors[i], linewidth=lw, label=legend_Kp[i])
        # ax_Kp[j].set_xlabel('$Time~[s]$')
        # ax_Kp[j].set_ylabel('$Kp~[-]$')
        ax_Kp[j].set_xlim(xlimits)
        ax_Kp[j].set_ylim([0, 50])
        # ax_Kp[j].legend(prop={'size': 8}, loc='lower right')

        pos = dh.get_data(params, file_name=file_name, data_to_plot='EE_position')[:,2]
        pos_d = dh.get_data(params, file_name=file_name, data_to_plot='EE_position_d')[:,2]
        ax_pos[j].plot(time, pos, colors[i], linewidth=lw, label=legend_pos[i])
        ax_pos[j].plot(time, pos_d, colors[i]+'--', linewidth=lw-0.5, label=legend_pos[i][:-1]+'_d$')
        # ax_pos[j].set_xlabel('$Time~[s]$')
        # ax_pos[j].set_ylabel('$z~[m]$')
        ax_pos[j].set_xlim(xlimits)
        ax_pos[j].set_ylim([0.1, 0.5])
        # ax_pos[j].legend(prop={'size': 8}, loc='lower right')


        vel = dh.get_data(params, file_name=file_name, data_to_plot='EE_twist')[:,2]
        vel_d = dh.get_data(params, file_name=file_name, data_to_plot='EE_twist_d')[:,2]
        ax_vel[j].plot(time, vel, colors[i], linewidth=lw, label=legend_vel[i])
        ax_vel[j].plot(time, vel_d, colors[i]+'--', linewidth=lw-0.5, label=legend_vel[i][:-1]+'_d$')
        # ax_vel[j].set_xlabel('$Time~[s]$')
        # ax_vel[j].set_ylabel('$\dot{z}~[m/s]$')
        ax_vel[j].set_xlim(xlimits)
        ax_vel[j].set_ylim([-1.5, 1.0])
        # ax_vel[j].legend(prop={'size': 8}, loc='lower right')


        i += 1

    # fig_ft.legend(legend_ft)
    # fig_Kp.legend(legend_Kp)
    # fig_pos.legend(legend_pos)
    # fig_vel.legend(legend_vel)

    fig_ft.suptitle("$F_z~[N]$")
    fig_Kp.suptitle("$K_p~[-]$")
    fig_pos.suptitle("$z~[m]$")
    fig_vel.suptitle("$\dot{z}~[m/s]$")

    # fig_ft.tight_layout()
    # fig_Kp.tight_layout()
    # fig_pos.tight_layout()
    # fig_vel.tight_layout()
    fig_ft.set_constrained_layout(constrained=True)
    fig_Kp.set_constrained_layout(constrained=True)
    fig_pos.set_constrained_layout(constrained=True)
    fig_vel.set_constrained_layout(constrained=True)

    for j, _ in enumerate(chosen_exps):
        ax_ft[j].grid()
        ax_Kp[j].grid()
        ax_pos[j].grid()
        ax_vel[j].grid()
    plt.show()
    # fig_ft.savefig('images/vanilla_best_ft.png')
    # fig_Kp.savefig('images/vanilla_best_Kp.png')
    # fig_pos.savefig('images/vanilla_best_pos.png')
    # fig_vel.savefig('images/vanilla_best_vel.png')

def plot_vanilla_best_1subplot_4figs():
    colors = ['b', 'r', 'g', 'y']
    styles = ['solid', 'dashed', 'dotted', 'dashdot']
    chosen_exps = [3, 8, 11, 16]
    i = 0
    fig_ft, ax_ft = plt.subplots()
    fig_Kp, ax_Kp  = plt.subplots()
    fig_pos, ax_pos  = plt.subplots()
    fig_vel, ax_vel  = plt.subplots()

    legend_ft = [r'$F^{VM+K_H}_z$', r'$F^{VM+K_L}_z$', r'$F^{VM+SIC}_z$', r'$F^{VM+POC}_z$']
    legend_Kp = [r'$K_H$', r'$K_L$', r'$SIC$', r'$POC$']
    legend_pos = [r'$z^{VM+K_H}$', r'$z^{VM+K_L}$', r'$z^{VM+SIC}$', r'$z^{VM+POC}$']
    legend_vel = [r'$\dot{z}^{VM+K_H}$', r'$\dot{z}^{VM+K_L}$', r'$\dot{z}^{VM+SIC}$', r'$\dot{z}^{VM+POC}$']

    for j, idx_exp in enumerate(chosen_exps):
        file_name = dh._get_name(idx_exp)
        print(file_name)
        params['idx_initial'] = dh.get_idx_from_file(idx_exp, data_info, idx_name='idx_start')
        params['idx_end'] = dh.get_idx_from_file(idx_exp, data_info, idx_name='idx_end')

        time = dh.get_data(file_name=file_name,params=params, data_to_plot='time')
        time -= time[0]

        ft_ = dh.get_data(file_name=file_name, params=params, data_to_plot='ft_')[:,2]
        ax_ft.plot(time, ft_, colors[i], linewidth=lw, label=legend_ft[i], linestyle=styles[i])
        # ax_ft.set_xlabel('$Time~[s]$')
        # ax_ft.set_ylabel('$F_z~[N]$')
        ax_ft.set_xlim(xlimits)
        ax_ft.set_ylim([-10, 5])
        # ax_ft.legend(prop={'size': 8}, loc='lower right')

        Kp = dh.get_data(params, file_name=file_name, data_to_plot='Kp')[:,2]
        ax_Kp.plot(time, Kp, colors[i], linewidth=lw, label=legend_Kp[i])
        # ax_Kp.set_xlabel('$Time~[s]$')
        # ax_Kp.set_ylabel('$Kp~[-]$')
        ax_Kp.set_xlim(xlimits)
        ax_Kp.set_ylim([0, 50])
        # ax_Kp.legend(prop={'size': 8}, loc='lower right')

        pos = dh.get_data(params, file_name=file_name, data_to_plot='EE_position')[:,2]
        pos_d = dh.get_data(params, file_name=file_name, data_to_plot='EE_position_d')[:,2]
        ax_pos.plot(time, pos, colors[i], linewidth=lw, label=legend_pos[i])
        ax_pos.plot(time, pos_d, colors[i]+'--', linewidth=lw-0.5, label=legend_pos[i][:-1]+'_d$')
        # ax_pos.set_xlabel('$Time~[s]$')
        # ax_pos.set_ylabel('$z~[m]$')
        ax_pos.set_xlim(xlimits)
        ax_pos.set_ylim([0.1, 0.5])
        # ax_pos.legend(prop={'size': 8}, loc='lower right')

        vel = dh.get_data(params, file_name=file_name, data_to_plot='EE_twist')[:,2]
        vel_d = dh.get_data(params, file_name=file_name, data_to_plot='EE_twist_d')[:,2]
        ax_vel.plot(time, vel, colors[i], linewidth=lw, label=legend_vel[i])
        ax_vel.plot(time, vel_d, colors[i]+'--', linewidth=lw-0.5, label=legend_vel[i][:-1]+'_d$')
        # ax_vel.set_xlabel('$Time~[s]$')
        # ax_vel.set_ylabel('$\dot{z}~[m/s]$')
        ax_vel.set_xlim(xlimits)
        ax_vel.set_ylim([-1.5, 1.0])
        # ax_vel.legend(prop={'size': 8}, loc='lower right')

        i += 1

    # fig_ft.legend(legend_ft)
    # fig_Kp.legend(legend_Kp)
    # fig_pos.legend(legend_pos)
    # fig_vel.legend(legend_vel)

    fig_ft.suptitle("$F_z~[N]$")
    fig_Kp.suptitle("$K_p~[-]$")
    fig_pos.suptitle("$z~[m]$")
    fig_vel.suptitle("$\dot{z}~[m/s]$")

    # fig_ft.tight_layout()
    # fig_Kp.tight_layout()
    # fig_pos.tight_layout()
    # fig_vel.tight_layout()
    fig_ft.set_constrained_layout(constrained=True)
    fig_Kp.set_constrained_layout(constrained=True)
    fig_pos.set_constrained_layout(constrained=True)
    fig_vel.set_constrained_layout(constrained=True)

    ax_ft.grid()
    ax_Kp.grid()
    ax_pos.grid()
    ax_vel.grid()
    plt.show()
    # fig_ft.savefig('images/vanilla_best_ft.png')
    # fig_Kp.savefig('images/vanilla_best_Kp.png')
    # fig_pos.savefig('images/vanilla_best_pos.png')
    # fig_vel.savefig('images/vanilla_best_vel.png')

def plot_best_4subplots_1fig():
    # colors = ['b', 'r', 'g', 'y']
    colors = ['b', 'b', 'b', 'b']
    chosen_exps = [3, 8, 11, 14]
    styles = ['solid', 'dotted', 'dashdot', 'on-off-dash-seq']
    i = 0
    ylimits_ft = [-10, 5]
    ylimits_Kp = [0, 60]
    ylimits_pos = [0, 0.8]
    ylimits_vel = [-2.75, 1.0]
    # fig_ft, ax_ft = plt.subplots(1, 4)
    # fig_Kp, ax_Kp  = plt.subplots(1, 4)
    # fig_pos, ax_pos  = plt.subplots(1, 4)
    # fig_vel, ax_vel  = plt.subplots(1, 4)

    # idx_exp = chosen_exps[1]
    # file_name = dh._get_name(idx_exp)
    # params['idx_initial'] = dh.get_idx_from_file(idx_exp, data_info, idx_name='idx_start')
    # params['idx_end'] = dh.get_idx_from_file(idx_exp, data_info, idx_name='idx_end')
    # time = dh.get_data(file_name=file_name,params=params, data_to_plot='time')
    # time -= time[0]
    # pos_ball = dh.get_data(params, file_name=file_name, data_to_plot='ball_pose_')[:,2]
    # vel_d = dh.get_data(params, file_name=file_name, data_to_plot='EE_twist_d')[:,2]
    # z_actual_hat, z_dot_actual_hat, z_intersec, z_dot_intersec, time_intersec, time_f = kalman_filtering_1d(time, pos_ball, vel_d)
    # plt.plot(time, pos_ball, 'k')
    # plt.plot(time_f, z_actual_hat, 'r--')
    # plt.legend(['$z$', '$\hat{z}$'])
    # plt.show()

    # exit()

    fig, ax  = plt.subplots(len(chosen_exps), 4, figsize=(20, 5))
    # fig_b, ax_b = plt.subplots(1,1)

    idx_pos = 0
    idx_vel = 1
    idx_ft = 2
    idx_Kp = 3

    # legend_ft = [r'$F^{VM+K_H}_z$', r'$F^{VM+K_L}_z$', r'$F^{VM+SIC}_z$', r'$F^{VM+POC}_z$']
    legend_ft = [r'$F_z$', r'$F_z$', r'$F_z$', r'$F_z$']
    legend_Kp = [r'$K_H$', r'$K_L$', r'$SIC$', r'$POC$']
    # legend_pos = [r'$z^{VM+K_H}$', r'$z^{VM+K_L}$', r'$z^{VM+SIC}$', r'$z^{VM+POC}$']
    legend_pos = [r'$z$', r'$z$', r'$z$', r'$z$']
    # legend_vel = [r'$\dot{z}^{VM+K_H}$', r'$\dot{z}^{VM+K_L}$', r'$\dot{z}^{VM+SIC}$', r'$\dot{z}^{VM+POC}$']
    legend_vel = [r'$\dot{z}$', r'$\dot{z}$', r'$\dot{z}$', r'$\dot{z}$']

    for j, idx_exp in enumerate(chosen_exps):
        file_name = dh._get_name(idx_exp)
        print(file_name)
        params['idx_initial'] = dh.get_idx_from_file(idx_exp, data_info, idx_name='idx_start')
        params['idx_end'] = dh.get_idx_from_file(idx_exp, data_info, idx_name='idx_end')

        time = dh.get_data(file_name=file_name,params=params, data_to_plot='time')
        time -= time[0]

        ft_ = dh.get_data(file_name=file_name, params=params, data_to_plot='ft_')[:,2]
        ax[idx_ft][j].plot(time, ft_, colors[i], linewidth=lw, label=legend_ft[i])
        # ax[idx_ft][j].set_xlabel('$Time~[s]$')
        # ax[idx_ft][j].set_ylabel('$F_z~[N]$')
        ax[idx_ft][j].set_xlim(xlimits)
        ax[idx_ft][j].set_ylim(ylimits_ft)
        # ax[idx_ft][j].legend(prop={'size': 8}, loc='lower right')
        

        Kp = dh.get_data(params, file_name=file_name, data_to_plot='Kp')[:,2]
        # ax[idx_Kp][j].plot(time, Kp, colors[i], linewidth=lw, label=legend_Kp[i])
        ax[idx_Kp][j].plot(time, Kp, colors[i], linewidth=lw)
        # ax[idx_Kp][j].set_xlabel('$Time~[s]$')
        # ax[idx_Kp][j].set_ylabel('$Kp~[-]$')
        ax[idx_Kp][j].set_xlim(xlimits)
        ax[idx_Kp][j].set_ylim(ylimits_Kp)
        # ax[idx_Kp][j].legend(prop={'size': 8}, loc='lower right')

        pos = dh.get_data(params, file_name=file_name, data_to_plot='EE_position')[:,2]
        pos_d = dh.get_data(params, file_name=file_name, data_to_plot='EE_position_d')[:,2]
        ax[idx_pos][j].plot(time, pos, colors[i], linewidth=lw, label=legend_pos[i])
        ax[idx_pos][j].plot(time, pos_d, 'k--', linewidth=lw-0.5, label=legend_pos[i][:-1]+'_d$')
        # ax[idx_pos][j].set_xlabel('$Time~[s]$')
        # ax[idx_pos][j].set_ylabel('$z~[m]$')
        pos_ball = dh.get_data(params, file_name=file_name, data_to_plot='ball_pose_')[:,2]
        ax[idx_pos][j].set_xlim(xlimits)
        ax[idx_pos][j].set_ylim(ylimits_pos)
        # ax[idx_pos][j].legend(prop={'size': 8}, loc='lower right')


        vel = dh.get_data(params, file_name=file_name, data_to_plot='EE_twist')[:,2]
        vel_d = dh.get_data(params, file_name=file_name, data_to_plot='EE_twist_d')[:,2]
        ax[idx_vel][j].plot(time, vel, colors[i], linewidth=lw, label=legend_vel[i])
        ax[idx_vel][j].plot(time, vel_d, 'k--', linewidth=lw-0.5, label=legend_vel[i][:-1]+'_d$')
        # ax[idx_vel][j].set_xlabel('$Time~[s]$')
        # ax[idx_vel][j].set_ylabel('$\dot{z}~[m/s]$')
        ax[idx_vel][j].set_xlim(xlimits)
        ax[idx_vel][j].set_ylim(ylimits_vel)
        # ax[idx_vel][j].legend(prop={'size': 8}, loc='lower right')

        # apply KF
        z_actual_hat, z_dot_actual_hat, z_intersec, z_dot_intersec, time_intersec, time_f = kalman_filtering_1d(time, pos_ball, ft_)
        # ax[idx_pos][j].plot(time, pos_ball, 'k', linewidth=lw, label=legend_pos[i])
        ax[idx_pos][j].plot(time_f, z_actual_hat-(95/2/1000), color='m', linestyle='dashdot', linewidth=lw-0.5, label='$z_b$')
        ax[idx_vel][j].plot(time_f, z_dot_actual_hat, color='m', linestyle='dashdot', linewidth=lw-0.5, label='$\dot{z}_b$')

        i += 1

    # fig_ft.legend(legend_ft)
    # fig_Kp.legend(legend_Kp)
    # fig_pos.legend(legend_pos)
    # fig_vel.legend(legend_vel)

    # fig[idx_ft].title("$F_z~[N]$")
    # fig[idx_Kp].title("$K_p~[-]$")
    # fig[idx_pos].title("$z~[m]$")
    # fig[idx_vel].title("$\dot{z}~[m/s]$")

    # fig[0].tight_layout()
    # fig[0].tight_layout()
    # fig[0].tight_layout()
    # fig[0].tight_layout()
    # fig[0].sett_constrained_layout(constrained=True)

    ax[idx_pos,0].set_ylabel('$z~[m]$')
    ax[idx_vel,0].set_ylabel('$\dot{z}~[m]$')
    ax[idx_Kp,0].set_ylabel('$K_p$')
    ax[idx_ft,0].set_ylabel('$F_z~[N]$')
    fig.align_ylabels()

    # ax[0,0].set_title('$VM+K_{P_H}$')
    # ax[0,1].set_title('$VM+K_{P_L}$')
    ax[0,0].set_title('$VM+K_{H}$')
    ax[0,1].set_title('$VM+K_{L}$')
    ax[0,2].set_title('$VM+SIC$')
    ax[0,3].set_title('$VM+POC$')
    
    for j, _ in enumerate(chosen_exps):
        if j < 3:
            ax[j][0].set_xticks([])
            ax[j][1].set_xticks([])
            ax[j][2].set_xticks([])
            ax[j][3].set_xticks([])
        if j > 0:
            ax[0][j].set_yticks([])
            ax[1][j].set_yticks([])
            ax[2][j].set_yticks([])
            ax[3][j].set_yticks([])

    # grids creation
    x_grids = list(np.arange(0,2,0.25))
    n_divisions = 5
    alpha_grids = 0.12
    y_grids_ft = list(np.arange(ylimits_ft[0], ylimits_ft[-1], (ylimits_ft[-1]-ylimits_ft[0])/n_divisions))
    y_grids_Kp = list(np.arange(ylimits_Kp[0], ylimits_Kp[-1], (ylimits_Kp[-1]-ylimits_Kp[0])/n_divisions))
    y_grids_pos = list(np.arange(ylimits_pos[0], ylimits_pos[-1], (ylimits_pos[-1]-ylimits_pos[0])/n_divisions))
    y_grids_vel = list(np.arange(ylimits_vel[0], ylimits_vel[-1], (ylimits_vel[-1]-ylimits_vel[0])/n_divisions))
    for j, row in enumerate(ax):
        for e in row:
            [e.axvline(xg, color='k', alpha=alpha_grids) for xg in x_grids]
            if idx_ft == j:
                [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_ft]
                e.axhline(0, color='k')
            if idx_Kp == j:
                [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_Kp]
            if idx_pos == j:
                [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_pos]
            if idx_vel == j:
                [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_vel]
    
    # ft 
    for j in range(4):
        ax[idx_ft, j].axhline(-0.1*9.81, color='m', linestyle='dashdot', linewidth=lw-0.5, label='$W_b$')

    # plt.gca().grid(True)
    # plt.tight_layout()

    # legends
    # for j, row in enumerate(ax):
    #     for e in row:
    #         e.legend(prop={'size': 10}, loc='lower right')
    for j in range(4):
        if j == 1 or j == 2:
            ax[j, 3].legend(prop={'size': 10}, loc='lower right')
        if j == 0:
            ax[j, 3].legend(prop={'size': 10}, loc='upper right')

    plt.subplots_adjust(wspace=.025, hspace=0.045)
    plt.show()
    # fig_ft.savefig('images/vanilla_best_ft.png')
    # fig_Kp.savefig('images/vanilla_best_Kp.png')
    # fig_pos.savefig('images/vanilla_best_pos.png')
    # fig_vel.savefig('images/vanilla_best_vel.png')

def plot_best_4subplots_1fig_transpose():
    # colors = ['b', 'r', 'g', 'y']
    # chosen_exps = [25, 26, 7, 3, 11, 14]
    n_exps = len(chosen_exps)
    # colors = ['b', 'b', 'b', 'b']
    colors = ['b' for _ in range(n_exps)]
    styles = ['solid', 'dotted', 'dashdot', 'on-off-dash-seq']
    gray_cycler = (cycler(color=["#000000", "#333333", "#666666", "#999999", "#cccccc"]) +
                    cycler(linestyle=["-", "--", "-.", ":", "-"]))
    plt.rc("axes", prop_cycle=gray_cycler)
    i = 0
    ylimits_ft = [-20, 5]
    ylimits_Kp = [10, 50]
    ylimits_pos = [0, 0.8]
    ylimits_vel = [-2.75, 1]

    fig, ax  = plt.subplots(len(chosen_exps), 4, figsize=(20, 8), layout="constrained")
    # fig_b, ax_b = plt.subplots(1,1)

    idx_pos = 0
    idx_vel = 1
    idx_ft = 2
    idx_Kp = 3

    # legend_ft = [r'$F^{VM+K_H}_z$', r'$F^{VM+K_L}_z$', r'$F^{VM+SIC}_z$', r'$F^{VM+POC}_z$']
    # legend_ft = [r'$F_z$', r'$F_z$', r'$F_z$', r'$F_z$']
    legend_ft = [r'$F_z$' for _ in range(n_exps)]
    # legend_Kp = [r'$K_H$', r'$K_L$', r'$SIC$', r'$POC$']
    legend_Kp = [r'$K_H$' for _ in range(n_exps)]
    # legend_pos = [r'$z^{VM+K_H}$', r'$z^{VM+K_L}$', r'$z^{VM+SIC}$', r'$z^{VM+POC}$']
    # legend_pos = [r'$z$', r'$z$', r'$z$', r'$z$']
    legend_pos = [r'$z$' for _ in range(n_exps)]
    # legend_vel = [r'$\dot{z}^{VM+K_H}$', r'$\dot{z}^{VM+K_L}$', r'$\dot{z}^{VM+SIC}$', r'$\dot{z}^{VM+POC}$']
    legend_vel = [r'$\dot{z}$' for _ in range(n_exps)]
    add_lw = 1

    for j, idx_exp in enumerate(chosen_exps):
        file_name = dh._get_name(idx_exp)
        offset_final = 195 if 'fp' in file_name else 0
        params['idx_initial'] = dh.get_idx_from_file(idx_exp, data_info, idx_name='idx_start')+154
        params['idx_end'] = dh.get_idx_from_file(idx_exp, data_info, idx_name='idx_end')-offset_final

        time = dh.get_data(file_name=file_name,params=params, data_to_plot='time')
        time -= time[0]

        ft_ = dh.get_data(file_name=file_name, params=params, data_to_plot='ft_')[:,2]
        # ax[j][idx_ft].plot(time, ft_, colors[i], linewidth=lw, label=legend_ft[i])
        ax[j][idx_ft].plot(time, ft_, linewidth=lw+add_lw, label=legend_ft[i])

        idx_impact = np.where(ft_ < -3)[0][0]-3
        idx_finish_analysis = 1000 - idx_impact
        
        # n = 50
        # mystd = np.zeros_like(ft_)
        # for k, ft_i in enumerate(ft_):
        #     if k>n:
        #         mystd[k] = np.std(ft_[k-n:k])
        #     if mystd[k] < 0.15 and ft_i > -1:
        #         ax[j][idx_ft].plot(time[k], mystd[k], 'rx', linewidth=lw+add_lw, label='_nolegend_')
        #     else:
        #         ax[j][idx_ft].plot(time[k], mystd[k], 'bx', linewidth=lw+add_lw, label='_nolegend_')

        # for k, ft_i in enumerate(ft_):
        #     if np.abs(ft_i + 0.1*9.81) < 0.15:
        #         ax[j][idx_ft].plot(time[k], ft_i, 'rx', linewidth=lw+add_lw, label='_nolegend_')

        ax[j][idx_ft].set_xlim(xlimits)
        ax[j][idx_ft].set_ylim(ylimits_ft)
        # ax[idx_ft][j].legend(prop={'size': 8}, loc='lower right')
        

        Kp = dh.get_data(params, file_name=file_name, data_to_plot='Kp')[:,2]
        # ax[j][idx_Kp].plot(time, Kp, colors[i], linewidth=lw, label=legend_Kp[i])
        if idx_exp == 2:
            # ax[j][idx_Kp].plot(time, Kp[0]*np.ones_like(Kp), colors[i], linewidth=lw)
            ax[j][idx_Kp].plot(time, Kp[0]*np.ones_like(Kp), linewidth=lw+add_lw)
            ax[j][idx_Kp].set_xlim(xlimits)
            ax[j][idx_Kp].set_ylim(ylimits_Kp)
        else:
            # ax[j][idx_Kp].plot(time, Kp, colors[i], linewidth=lw)
            ax[j][idx_Kp].plot(time, Kp, linewidth=lw+add_lw)
            ax[j][idx_Kp].set_xlim(xlimits)
            ax[j][idx_Kp].set_ylim(ylimits_Kp)

        pos = dh.get_data(params, file_name=file_name, data_to_plot='EE_position')[:,2]
        pos_d = dh.get_data(params, file_name=file_name, data_to_plot='EE_position_d')[:,2]
        # ax[j][idx_pos].plot(time, pos, colors[i], linewidth=lw, label=legend_pos[i])
        ax[j][idx_pos].plot(time, pos, linewidth=lw+add_lw, label=legend_pos[i])
        if 'fp' in file_name:
            ax[j][idx_pos].plot(np.arange(0, 1, 1/1000), pos_d[0]*np.ones(1000), 'k--', linewidth=lw-0.5, label=legend_pos[i][:-1]+'_d$', alpha=0.6)
            ax[j][idx_pos].plot(time[-1], pos[-1], 'kx', ms=7)
            ax[j][idx_ft].plot(time[-1], ft_[-1], 'kx', ms=7)
        else:
            ax[j][idx_pos].plot(time, pos_d, 'k--', linewidth=lw-0.5+add_lw, label=legend_pos[i][:-1]+'_d$', alpha=0.6)
        # ax[j][idx_pos].plot(time[idx_impact], pos[idx_impact], 'ko')
        pos_ball = dh.get_data(params, file_name=file_name, data_to_plot='ball_pose_')[:,2]
        ax[j][idx_pos].set_xlim(xlimits)
        ax[j][idx_pos].set_ylim(ylimits_pos)

        vel = dh.get_data(params, file_name=file_name, data_to_plot='EE_twist')[:,2]
        vel_d = dh.get_data(params, file_name=file_name, data_to_plot='EE_twist_d')[:,2]
        # ax[j][idx_vel].plot(time, vel, colors[i], linewidth=lw, label=legend_vel[i])
        ax[j][idx_vel].plot(time, vel, linewidth=lw+add_lw, label=legend_vel[i])
        if 'fp' in file_name:
            ax[j][idx_vel].plot(np.arange(0, 1, 1/1000), vel_d[0]*np.ones(1000), 'k--', linewidth=lw-0.5+add_lw, label=legend_vel[i][:-1]+'_d$', alpha=0.6)
            ax[j][idx_vel].plot(time[-1], vel[-1], 'kx', ms=7)
            ax[j][idx_Kp].plot(time[-1], Kp[-1], 'kx', ms=7)
        else:
            ax[j][idx_vel].plot(time, vel_d, 'k--', linewidth=lw-0.5+add_lw, label=legend_vel[i][:-1]+'_d$', alpha=0.6)
        ax[j][idx_vel].set_xlim(xlimits)
        ax[j][idx_vel].set_ylim(ylimits_vel)

        # apply KF
        z_actual_hat, z_dot_actual_hat, z_intersec, z_dot_intersec, time_intersec, time_f = kalman_filtering_1d(time, pos_ball, ft_)
        # ax[j][idx_pos].plot(time, pos_ball, 'k', linewidth=lw, label=legend_pos[i])
        # ax[j][idx_pos].plot(time_f[:-2], z_actual_hat[:-2]-(95/2/1000), color='b', linestyle='dashdot', linewidth=lw-0.5+add_lw, label='$z_O$')
        ax[j][idx_pos].plot(time_f[:-2], z_actual_hat[:-2], color='b', linestyle='dashdot', linewidth=lw-0.5+add_lw, label='$z_O$')
        ax[j][idx_vel].plot(time_f[:-2], z_dot_actual_hat[:-2], color='b', linestyle='dashdot', linewidth=lw-0.5+add_lw, label='$\dot{z}_O$')
        if 'fp' not in file_name:
            if 'vm+low' in file_name:
                get_ = -6
            else:
                get_ = -4
        else:
            get_ = -2
        ax[j][idx_pos].add_patch(Ellipse((time_f[get_], z_actual_hat[get_]), 2*ball_radius/5+0.015, 2*ball_radius, color='b', fill=False))

        i += 1

        # print(file_name, '\tF_max = ', np.max(np.abs(ft_)), , 'mm', '\tEE_pos_impact = ', pos[idx_impact], 'mm')

        loi = LOI(time, ft_, idx_impact, idx_finish_analysis, file_name)
        Fmax = -np.min(ft_)

        try:
            idx_impact_ = np.where(time == time_f[-1])[0][0]
        except IndexError:
            idx_impact_ = len(time)
        vm_error = vel[idx_impact_] - z_dot_actual_hat[-1]

        dri = DRI(time, ft_, file_name)

        bti = BTI(time, ft_)

        print(file_name,  '\tLOI =',        loi,
                          '\tDRI = ',       dri,
                          '\tBTI = ',       bti,
                          '\tF_max = ',     Fmax,
                          '\tVM-error = ',  vm_error,
                          '\tE_pos_impact = ', (pos[idx_impact] - 0.35)*1000)

    fig.set_constrained_layout(constrained=True)    
    ax[0,idx_pos].set_title('$z~[m]$')
    ax[0,idx_vel].set_title('$\dot{z}~[m/s]$')
    ax[0,idx_Kp].set_title('$K_p$')
    ax[0,idx_ft].set_title('$F_z~[N]$')
    fig.align_ylabels()

    # ax[0,0].set_title('$VM+K_{P_H}$')
    # ax[0,1].set_title('$VM+K_{P_L}$')
    k = 0
    # ax[k,0].set_ylabel('$FP_{17}$'+'\n'+'$K_{L}$'); k+=1
    # ax[k,0].set_ylabel('$FP_{17}$'+'\n'+'$K_{H}$'); k+=1
    ax[k,0].set_ylabel('$FP$'+'\n'+'$K_{L}$'); k+=1
    ax[k,0].set_ylabel('$FP$'+'\n'+'$K_{H}$'); k+=1
    ax[k,0].set_ylabel('$VM$'+'\n'+'$K_{L}$'); k+=1
    ax[k,0].set_ylabel('$VM$'+'\n'+'$K_{H}$'); k+=1
    ax[k,0].set_ylabel('$VM$'+'\n'+'$SIC$'); k+=1
    ax[k,0].set_ylabel('$VM$'+'\n'+'$VIC$'); k+=1
    
    for j, _ in enumerate(chosen_exps):
        if j < n_exps-1:
            ax[j][0].set_xticks([])
            ax[j][1].set_xticks([])
            ax[j][2].set_xticks([])
            ax[j][3].set_xticks([])
    
    for j in range(4):
        ax[n_exps-1,j].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax[n_exps-1,j].set_xticklabels(['$0$', '$0.25$', '$0.5$', '$0.75$', '$1$'], size=13)
        
    #     # ax[j][0].set_yticks([])``
    #     ax[j][1].set_yticks([])
    #     ax[j][2].set_yticks([])
    #     ax[j][3].set_yticks([])

    # grids creation
    x_grids = list(np.arange(0,2,0.25))
    n_divisions = 5
    alpha_grids = 0.12
    # y_grids_ft = list(np.arange(ylimits_ft[0], ylimits_ft[-1], (ylimits_ft[-1]-ylimits_ft[0])/n_divisions))
    y_grids_ft = [-15, -10, -5, 0, 5]
    # y_grids_Kp = list(np.arange(ylimits_Kp[0], ylimits_Kp[-1], (ylimits_Kp[-1]-ylimits_Kp[0])/n_divisions))
    y_grids_Kp = [0, 10, 20, 30, 40, 50]
    # y_grids_pos = list(np.arange(ylimits_pos[0], ylimits_pos[-1], (ylimits_pos[-1]-ylimits_pos[0])/n_divisions))
    y_grids_pos = [i for i in list(np.arange(0, 0.81, 0.1))]
    # y_grids_vel = list(np.arange(ylimits_vel[0], ylimits_vel[-1], (ylimits_vel[-1]-ylimits_vel[0])/n_divisions))
    y_grids_vel = [-2, -1, 0, 0.5]
    for j, row in enumerate(ax):
        for i, e in enumerate(row):
            [e.axvline(xg, color='k', alpha=alpha_grids) for xg in x_grids]
            if idx_ft == i:
                [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_ft]
                # e.axhline(0, color='k')
                # e.set_yticks([0, -5, -15])
                # e.set_yticklabels(['$0$', '$-5$', '$-15$'], size=13)
                e.set_yticks([5, 0, -5, -10, -15, -20])
                e.set_yticklabels(['$5$', '$0$', "$-5$", '$-10$', '$-15$', '$-20$'], size=13)
            if idx_Kp == i:
                [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_Kp]
                e.set_yticks([10, 20, 30, 40, 50])
                e.set_yticklabels(['$10$', '$20$', '$30$', '$40$', '$50$'], size=13)
            if idx_pos == i:
                [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_pos]
                aux = [0, 0.2, 0.4, 0.6, 0.8]
                e.set_yticks(aux)
                e.set_yticklabels(['$'+str(a)+'$' for a in aux], size=13)
            if idx_vel == i:
                e.set_yticks([-2, -1, 0])
                e.set_yticklabels(['$-2$', '$-1$', '$0$'], size=13)
                [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_vel]
    
    
    # ft 
    for j in range(n_exps):
        ax[j, idx_ft].axhline(-0.1*9.81, color='#000000', linestyle='dashdot', linewidth=lw-0.5, label='$W_O$')
        # ax[j, idx_ft].axhline(-0.1*9.81*.7, color='#000000', linestyle='dashdot', linewidth=lw-1, label='_nolegend_')
        # ax[j, idx_ft].axhline(-0.1*9.81*1.3, color='#000000', linestyle='dashdot', linewidth=lw-1, label='_nolegend_')

    # plt.gca().grid(True)
    # plt.tight_layout()

    # legends
    # for j, row in enumerate(ax):
    #     for e in row:
    #         e.legend(prop={'size': 10}, loc='lower right')
    for j in range(4):
        if j < 3:
            leg = '_nolegend_'
        else:
            leg = '$t_c$'
        
        if j == 0:
            loc = 'upper right'
        else:
            loc = 'lower right'
        
        ax[-1, j].axvline(x = 0.34, ymin=0, ymax=6.96, linestyle='--', linewidth=1.1, color = 'r', label = leg, clip_on=False)
        ax[-1, j].legend(prop={'size': 14}, loc=loc)
        

    # ax[n_exps-1, 2].axvline(x = 0.34, ymin=0, ymax=7.06, linestyle='-', linewidth=1.3, color = 'r', label = 'axvline - full height', clip_on=False)

    # plt.subplots_adjust(hspace=0.045)
    fig.supxlabel('$Time~[s]$', size=20)
    plt.show()
    fig.savefig('images/1d-comparison-plots.png', pad_inches=0, dpi=400)
    # fig_Kp.savefig('images/vanilla_best_Kp.png')
    # fig_pos.savefig('images/vanilla_best_pos.png')
    # fig_vel.savefig('images/vanilla_best_vel.png')

def plot_zoom_forces():
    line_cycler   = (cycler(color=["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]) +
                    cycler(linestyle=["-", "--", "-.", "-", "--", "-.", "-"]))
    standard_cycler = cycler("color", ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"])
    marker_cycler = (cycler(color=["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]) +
                 cycler(linestyle=["none", "none", "none", "none", "none", "none", "none"]) +
                 cycler(marker=["4", "2", "3", "1", "+", "x", "."]))
    gray_cycler = (cycler(color=["#000000", "#333333", "#666666", "#999999"]) +
                    cycler(linestyle=["-", "--", "-", "-"]))
    # my_cyc = 
    plt.rc("axes", prop_cycle=gray_cycler)
    # plt.rc("axes", prop_cycle=cycler(color=myc.mpl_colors))
    # colors = ['b', 'r', 'g', 'y']
    # chosen_exps = [26, 25, 7, 3, 11, 14]
    n_exps = len(chosen_exps)
    colors = ['b']*4 + ['r']*4
    alphas = [1.0, 0.75, 0.5, 0.25] * 2
    styles = ['solid', 'dashed', 'dotted', 'dashdot']*2
    # colors = ['b' for _ in range(n_exps)]
    i = 0
    ylimits_ft = [-17.5, 5]

    fig, ax = plt.subplots(2, 1, figsize=(8,8))
    # fig_b, ax_b = plt.subplots(1,1)

    idx_ft = 1
    idx_init = 491
    idx_final = idx_init + 100

    legend_ft = [r'$F_z$' for _ in range(n_exps)]

    for j, idx_exp in enumerate(chosen_exps):
        file_name = dh._get_name(idx_exp)
        print(file_name)
        params['idx_initial'] = dh.get_idx_from_file(idx_exp, data_info, idx_name='idx_start')
        params['idx_end'] = dh.get_idx_from_file(idx_exp, data_info, idx_name='idx_end')

        time = dh.get_data(file_name=file_name,params=params, data_to_plot='time')
        time -= time[0]

        ft_ = dh.get_data(file_name=file_name, params=params, data_to_plot='ft_')[:,2]
        
        i = 0 if 'fp' in file_name else 1
        idx_final_fp = idx_init+24 if 'fp' in file_name else -1

        if 'poc' in file_name:
            ax[i].plot(time[idx_init:idx_final_fp], ft_[idx_init:idx_final_fp], linewidth=lw, label=legend_ft[i], color='r', alpha=0.7)#, linestyle=styles[i])
        else:
            ax[i].plot(time[idx_init:idx_final_fp], ft_[idx_init:idx_final_fp], linewidth=lw, label=legend_ft[i])#, linestyle=styles[i])
        xlimits = [time[idx_init], time[idx_final]]
        ax[i].set_xlim(xlimits)
        ax[i].set_ylim(ylimits_ft)
        # ax[idx_ft][j].legend(prop={'size': 8}, loc='lower right')
    
    # chosen_exps = [1, 2, 26, 25, 3, 8, 11, 14]
    # chosen_exps = [26, 25, 3, 8, 11, 14]

    legend_fp = []
    legend_vm = []
    # legend_.append('$FP_{17}$'+'\n'+'$K_{L}$')
    # legend_.append('$FP_{17}$'+'\n'+'$K_{H}$')
    legend_fp.append('$FP$'+'\n'+'$K_{L}$')
    legend_fp.append('$FP$'+'\n'+'$K_{H}$')
    legend_vm.append('$VM$'+'\n'+'$K_{L}$')
    legend_vm.append('$VM$'+'\n'+'$K_{H}$')
    legend_vm.append('$VM$'+'\n'+'$SIC$')
    legend_vm.append('$VM$'+'\n'+'$VIC$')
    ax[0].legend(legend_fp, prop={'size': 8}, fancybox=True, framealpha=0.5, loc='lower right')
    ax[1].legend(legend_vm, prop={'size': 8}, fancybox=True, framealpha=0.5, ncol=2, loc='lower right')
    [ax_.axhline(0, color='k', linestyle='dashed') for ax_ in ax]
    # [ax_.set_ylabel("$F_z~[N]$") for ax_ in ax]
    fig.supylabel("$F_z~[N]$")
    ax[1].set_xlabel('$Time~[s]$')
    # fig.supxlabel('$Time~[s]$')
    [ax_.grid() for ax_ in ax]    
    # fig.set_tight_layout(tight=True)

    plt.subplots_adjust(hspace=0.045)
    fig.set_constrained_layout(constrained=True)
    plt.show()
    fig.savefig('force_comparison.png')

def plot_zoom_forces_single_plot():
    line_cycler   = (cycler(color=["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]) +
                    cycler(linestyle=["-", "--", "-.", "-", "--", "-.", "-"]))
    standard_cycler = cycler("color", ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"])
    marker_cycler = (cycler(color=["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]) +
                 cycler(linestyle=["none", "none", "none", "none", "none", "none", "none"]) +
                 cycler(marker=["4", "2", "3", "1", "+", "x", "."]))
    # gray_cycler = (cycler(color=['greys']) + cycler(linestyle=["-", "--", "-"]))
    # colors_i_want = [0, 0, 1, 1, 2, 3, -1]
    colors_i_want = [0, 0, 1, 1, 2, 3, -1]
    my_cyc = cycler(color=[myc.mpl_colors[i] for i in colors_i_want]) + cycler(linestyle=["-", '-', '-', "-", "--", "-", "--"])
    # my_cyc = cycler(color=myc.mpl_colors[:5]) + cycler(linestyle=["-", "-", "--", "-", "--"])
    print(myc.mpl_colors)
    plt.rc("axes", prop_cycle=my_cyc)
    # plt.rc("axes", prop_cycle=cycler(color=myc.mpl_colors))
    # colors = ['b', 'r', 'g', 'y']
    # chosen_exps = [26, 25, 7, 3, 11, 14]
    n_exps = len(chosen_exps)
    colors = ['b']*4 + ['r']*4
    alphas = [1.0, 0.75, 0.5, 0.25] * 2
    styles = ['solid', 'dashed', 'dotted', 'dashdot']*2
    # colors = ['b' for _ in range(n_exps)]
    i = 0
    ylimits_ft = [-17.5, 5]

    fig, ax = plt.subplots(1, 1, figsize=(8,4))
    # fig_b, ax_b = plt.subplots(1,1)

    idx_ft = 1

    legend_ft = [r'$F_z$' for _ in range(n_exps)]

    for j, idx_exp in enumerate(['fp-kl', 'fp-kh', 'vm-kl', 'vm-kh', 'vm-sic','vm-vic']):
        file_name = dh._get_name(chosen_exps[idx_exp])
        print(file_name)

        if idx_exp == 'vm-sic' or idx_exp == 'vm-kh' or idx_exp == 'fp-kl':
            idx_init = 490
            idx_final = idx_init + 150
        elif idx_exp == 'fp-kh':
            idx_init = 491
            idx_final = idx_init + 150
        else:
            idx_init = 488
            idx_final = idx_init + 150
        
        params['idx_initial'] = dh.get_idx_from_file(chosen_exps[idx_exp], data_info, idx_name='idx_start')
        params['idx_end'] = dh.get_idx_from_file(chosen_exps[idx_exp], data_info, idx_name='idx_end')

        time = dh.get_data(file_name=file_name,params=params, data_to_plot='time')
        time -= time[0]

        ft_ = dh.get_data(file_name=file_name, params=params, data_to_plot='ft_')[:,2]
        
        i = 0 if 'fp' in file_name else 1
        idx_final_fp = idx_init+19 if 'fp' in file_name else -1

        time = time - time[idx_init] + 0.333

        if 'fp' in file_name:
            ax.plot(time[idx_final_fp-1], ft_[idx_final_fp-1], 'x', ms=7, label='_nolegend_')

        if 'poc' in file_name:
            ax.plot(time[idx_init:idx_final_fp], ft_[idx_init:idx_final_fp], linewidth=lw+1, label=legend_ft[i], color='k', alpha=0.7)#, linestyle=styles[i])
        else:
            ax.plot(time[idx_init:idx_final_fp], ft_[idx_init:idx_final_fp], linewidth=lw, label=legend_ft[i])#, linestyle=styles[i])
        xlimits = [time[idx_init], time[idx_final]]
        ax.set_xlim(xlimits)
        ax.set_ylim(ylimits_ft)
        # ax[idx_ft][j].legend(prop={'size': 8}, loc='lower right')
    
    # chosen_exps = [1, 2, 26, 25, 3, 8, 11, 14]
    # chosen_exps = [26, 25, 3, 8, 11, 14]

    legend_vm = []
    # legend_.append('$FP_{17}$'+'\n'+'$K_{L}$')
    # legend_.append('$FP_{17}$'+'\n'+'$K_{H}$')
    legend_vm.append('$FP$'+'\n'+'$K_{L}$')
    legend_vm.append('$FP$'+'\n'+'$K_{H}$')
    legend_vm.append('$VM$'+'\n'+'$K_{L}$')
    legend_vm.append('$VM$'+'\n'+'$K_{H}$')
    legend_vm.append('$VM$'+'\n'+'$SIC$')
    legend_vm.append('$VM$'+'\n'+'$VIC$')
    # ax[0].legend(legend_fp, prop={'size': 8}, fancybox=True, framealpha=0.5, loc='lower right')
    leg = ax.legend(legend_vm, prop={'size': 10}, fancybox=True, framealpha=0.5, ncol=2, loc='lower right')
    leg.get_frame().set_alpha(None)
    leg.get_frame().set_facecolor((1.0, 1.0, 1, 1))
    ax.axhline(0, color='k', linestyle='dashed')
    # [ax_.set_ylabel("$F_z~[N]$") for ax_ in ax]
    fig.supylabel("$F_z~[N]$")
    ax.set_xlabel('$Time~[s]$')
    # fig.supxlabel('$Time~[s]$')
    ax.grid()
    # fig.set_tight_layout(tight=True)

    ax.axvline(0.34, color='r', linestyle='dashed')

    plt.subplots_adjust(hspace=0.045)
    fig.set_constrained_layout(constrained=True)
    plt.show()
    fig.savefig('images/1d_force_comparison.png', dpi=400, bbox_inches='tight')

def plot_zoom_position_single_plot():
    line_cycler   = (cycler(color=["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]) +
                    cycler(linestyle=["-", "--", "-.", "-", "--", "-.", "-"]))
    standard_cycler = cycler("color", ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"])
    marker_cycler = (cycler(color=["#E69F00", "#56B4E9", "#009E73", "#0072B2", "#D55E00", "#CC79A7", "#F0E442"]) +
                 cycler(linestyle=["none", "none", "none", "none", "none", "none", "none"]) +
                 cycler(marker=["4", "2", "3", "1", "+", "x", "."]))
    # gray_cycler = (cycler(color=['greys']) + cycler(linestyle=["-", "--", "-"]))
    # colors_i_want = [0, 0, 1, 1, 2, 3, -1]
    colors_i_want = [2, 3, -1, -1]
    my_cyc = cycler(color=[myc.mpl_colors[i] for i in colors_i_want]) + cycler(linestyle=["--", '-', '--', "-"])
    # my_cyc = cycler(color=myc.mpl_colors[:5]) + cycler(linestyle=["-", "-", "--", "-", "--"])
    print(myc.mpl_colors)
    plt.rc("axes", prop_cycle=my_cyc)
    # plt.rc("axes", prop_cycle=cycler(color=myc.mpl_colors))
    # colors = ['b', 'r', 'g', 'y']
    # chosen_exps = [26, 25, 7, 3, 11, 14]
    n_exps = len(chosen_exps)
    colors = ['b']*4 + ['r']*4
    alphas = [1.0, 0.75, 0.5, 0.25] * 2
    styles = ['solid', 'dashed', 'dotted', 'dashdot']*2
    # colors = ['b' for _ in range(n_exps)]
    colors = ['b', 'y', 'g', 'r']
    i = 0
    ylimits_pos = [0.1, .55]

    fig, ax = plt.subplots(1, 1, figsize=(8,4))
    # fig_b, ax_b = plt.subplots(1,1)

    idx_ft = 1
    idx_init = 488
    idx_final = idx_init + 75

    legend_ft = [r'$z~[m]$' for _ in range(n_exps)]

    for j, idx_exp in enumerate(['vm-kl', 'vm-kh', 'vm-sic', 'vm-vic']):
        file_name = dh._get_name(chosen_exps[idx_exp])
        print(file_name)
        
        params['idx_initial'] = dh.get_idx_from_file(chosen_exps[idx_exp], data_info, idx_name='idx_start')
        params['idx_end'] = dh.get_idx_from_file(chosen_exps[idx_exp], data_info, idx_name='idx_end')

        time = dh.get_data(file_name=file_name,params=params, data_to_plot='time')
        time -= time[0]

        pos = dh.get_data(file_name=file_name, params=params, data_to_plot='EE_position')[:,2]
        ft_ = dh.get_data(file_name=file_name, params=params, data_to_plot='ft_')[:,2]
        
        # idx_final = -1
        time = time - time[idx_init] + 0.333

        offset_plot = 75
        if 'poc' in file_name:
            ax.plot(time[idx_init-offset_plot:idx_final], pos[idx_init-offset_plot:idx_final], linewidth=lw+1, label=legend_ft[i], color='k')#, linestyle=styles[i])
            # ax.plot(time[idx_init-offset_plot:idx_final], ft_[idx_init-offset_plot:idx_final], linewidth=lw+1, label=legend_ft[i], color='k', alpha=0.7)#, linestyle=styles[i])
        else:
            ax.plot(time[idx_init-offset_plot:idx_final], pos[idx_init-offset_plot:idx_final], linewidth=lw, label=legend_ft[i])#, color=colors[j])#, linestyle=styles[i])
            # ax.plot(time[idx_init-offset_plot:idx_final], ft_[idx_init-offset_plot:idx_final], linewidth=lw, label=legend_ft[i])#, linestyle=styles[i])
        xlimits = [0.34-0.06, 0.34+0.06]
        ax.set_xlim(xlimits)
        # ax.set_ylim(ylimits_pos)
        # ax[idx_ft][j].legend(prop={'size': 8}, loc='lower right')
    
    # chosen_exps = [1, 2, 26, 25, 3, 8, 11, 14]
    # chosen_exps = [26, 25, 3, 8, 11, 14]

    ax.plot(time, np.ones_like(pos)*0.35, 'k', linewidth=2, alpha=0.6)
    ax.text(time[idx_init-51], 0.356, '$x_o^{t_c}$', size=15)
    legend_vm = []
    # legend_.append('$FP_{17}$'+'\n'+'$K_{L}$')
    # legend_.append('$FP_{17}$'+'\n'+'$K_{H}$')
    # legend_vm.append('$FP$'+'\n'+'$K_{L}$')
    # legend_vm.append('$FP$'+'\n'+'$K_{H}$')
    legend_vm.append('$VM$'+'\n'+'$K_{L}$')
    legend_vm.append('$VM$'+'\n'+'$K_{H}$')
    legend_vm.append('$VM$'+'\n'+'$SIC$')
    legend_vm.append('$VM$'+'\n'+'$VIC$')
    # ax[0].legend(legend_fp, prop={'size': 8}, fancybox=True, framealpha=0.5, loc='lower right')
    leg = ax.legend(legend_vm, prop={'size': 10}, fancybox=True, framealpha=0.5, ncol=2, loc='lower left')
    leg.get_frame().set_alpha(None)
    leg.get_frame().set_facecolor((1.0, 1.0, 1, 1))

    ax.axvline(0.34, color='r', linestyle='dashed')
    # [ax_.set_ylabel("$F_z~[N]$") for ax_ in ax]
    fig.supylabel("$z~[m]$")
    ax.set_xlabel('$Time~[s]$')
    # fig.supxlabel('$Time~[s]$')
    ax.grid()
    # fig.set_tight_layout(tight=True)

    plt.subplots_adjust(hspace=0.045)
    fig.set_constrained_layout(constrained=True)
    plt.show()
    fig.savefig('images/1d_pos_comparison.png', dpi=400, bbox_inches='tight')

def plot_best_2d():
    # chosen_exps = [19, 21, 22, 23]
    chosen_exps = {'2d-1': 19, '2d-2': 21, '2d-rmm-1': 22, '2d-rmm-2': 23}
    data_2d = {k: DataClass() for k in chosen_exps.keys()}

    n_exps = len(chosen_exps)
    gray_cycler = (cycler(color=["#000000", "#333333", "#666666", "#999999", "#cccccc"]) +
                    cycler(linestyle=["-", "--", "-.", ":", "-"]))
    plt.rc("axes", prop_cycle=gray_cycler)

    ylimits_ft = [-20, 2]
    ylimits_Kp = [10, 50]
    ylimits_pos_z = [0.25, 1.]
    ylimits_pos_y = [-1.75, 0]
    ylimits_vel_z = [-2.75, 2.75]
    ylimits_vel_y = [-0.8, 4]
    xlimits_2d = [0, 1.5]

    # fig, ax  = plt.subplots(6, len(chosen_exps), figsize=(20,10), layout="constrained", 
    #                         gridspec_kw={'wspace': 0.0, 'hspace': 0.0}, sharey='row')
    fig, ax  = plt.subplots(8, len(chosen_exps), figsize=(22,22), #layout="constrained", 
                            gridspec_kw={'wspace': 0.05, 'hspace': 0.16}, sharey='row')
    # fig_yz, ax_yz  = plt.subplots(len(chosen_exps), 1)
    # fig_b, ax_b = plt.subplots(1,1)

    idx_pos_z = 0+1
    idx_vel_z = 1+1
    idx_pos_y = 2+1
    idx_vel_y = 3+1
    idx_ft = 4+1
    idx_Kp = 5+1
    idx_tau = 6+1
    idx_pos_yz = 0
    i=0

    # legend_ft = [r'$F^{VM+K_H}_z$', r'$F^{VM+K_L}_z$', r'$F^{VM+SIC}_z$', r'$F^{VM+POC}_z$']
    # legend_ft = [r'$F_z$', r'$F_z$', r'$F_z$', r'$F_z$']
    legend_ft = [r'$F_z$' for _ in range(n_exps)]
    # legend_Kp = [r'$K_H$', r'$K_L$', r'$SIC$', r'$POC$']
    legend_Kp = [r'$K_H$' for _ in range(n_exps)]
    # legend_pos = [r'$z^{VM+K_H}$', r'$z^{VM+K_L}$', r'$z^{VM+SIC}$', r'$z^{VM+POC}$']
    # legend_pos = [r'$z$', r'$z$', r'$z$', r'$z$']
    legend_pos = [r'$z$', r'$y$']*2
    # legend_vel = [r'$\dot{z}^{VM+K_H}$', r'$\dot{z}^{VM+K_L}$', r'$\dot{z}^{VM+SIC}$', r'$\dot{z}^{VM+POC}$']
    legend_vel = [r'$\dot{z}$', r'$\dot{y}$']*2

    offset_2d = 0
    n_joints = 7

    for j, idx_exp in enumerate(chosen_exps):
        data_2d[idx_exp].file_name = dh._get_name(chosen_exps[idx_exp])
        print(data_2d[idx_exp].file_name)
        if j == 0:
            offset_2d = -38
        if j == 1:
            offset_2d = 0
        if j == 2:
            offset_2d = -20
        if j == 3:
            offset_2d = -10
        
        data_2d[idx_exp].params = params
        data_2d[idx_exp].params['idx_initial'] = dh.get_idx_from_file(chosen_exps[idx_exp], data_info, idx_name='idx_start')-100+offset_2d
        data_2d[idx_exp].params['idx_end'] = dh.get_idx_from_file(chosen_exps[idx_exp], data_info, idx_name='idx_end')+200

        data_2d[idx_exp].time = dh.get_data(file_name=data_2d[idx_exp].file_name,params=data_2d[idx_exp].params, data_to_plot='time')
        data_2d[idx_exp].time -= data_2d[idx_exp].time[0]

        data_2d[idx_exp].ft_ = dh.get_data(file_name=data_2d[idx_exp].file_name, params=data_2d[idx_exp].params, data_to_plot='ft_')
        # ax[idx_ft][j].plot(time, ft_, colors[i], linewidth=lw, label=legend_ft[i])
        ax[idx_ft][j].plot(data_2d[idx_exp].time, data_2d[idx_exp].ft_[:,2], linewidth=lw, label=legend_ft[i])
        # ax[idx_ft][j].plot(time, -np.linalg.norm(ft_[:,:3], axis=1), linewidth=lw, label=legend_ft[i])
        ax[idx_ft][j].set_xlim(xlimits_2d)
        ax[idx_ft][j].set_ylim(ylimits_ft)
        # ax[idx_ft][j].legend(prop={'size': 8}, loc='lower right')

        idx_impact = np.where(data_2d[idx_exp].ft_ < -3)[0][0]-3
        idx_finish_analysis = np.where(data_2d[idx_exp].time > 1.5)[0][0]
        
        data_2d[idx_exp].Kp = dh.get_data(data_2d[idx_exp].params, file_name=data_2d[idx_exp].file_name, data_to_plot='Kp')
        # ax[j][idx_Kp].plot(time, Kp, colors[i], linewidth=lw, label=legend_Kp[i])
        # ax[j][idx_Kp].plot(time, Kp, colors[i], linewidth=lw)
        offset = 0 if j != 3 else 10
        offset_kp = 0
        if j == 0:
            offset_kp = 90
        if j == 1:
            offset_kp = 40
        if j == 2:
            offset_kp = 70
        if j == 3:
            offset_kp = 40
        ax[idx_Kp][j].plot(data_2d[idx_exp].time[offset_kp:]-data_2d[idx_exp].time[offset_kp], data_2d[idx_exp].Kp[offset_kp:,2]-offset+10, linewidth=lw, label='$K_{p_z}$')
        ax[idx_Kp][j].plot(data_2d[idx_exp].time[offset_kp:]-data_2d[idx_exp].time[offset_kp], data_2d[idx_exp].Kp[offset_kp:,1]-offset+10, linewidth=lw, label='$K_{p_y}$')
        ax[idx_Kp][j].set_xlim(xlimits_2d)
        ax[idx_Kp][j].set_ylim(ylimits_Kp)

        data_2d[idx_exp].pos = dh.get_data(data_2d[idx_exp].params, file_name=data_2d[idx_exp].file_name, data_to_plot='EE_position')
        data_2d[idx_exp].pos_d = dh.get_data(data_2d[idx_exp].params, file_name=data_2d[idx_exp].file_name, data_to_plot='EE_position_d')
        # ax[j][idx_pos].plot(time, pos, colors[i], linewidth=lw, label=legend_pos[i])
        ax[idx_pos_z][j].plot(data_2d[idx_exp].time, data_2d[idx_exp].pos[:,2], linewidth=lw, label=legend_pos[i])
        ax[idx_pos_z][j].plot(data_2d[idx_exp].time, data_2d[idx_exp].pos_d[:,2], 'k--', linewidth=lw-0.5, label=legend_pos[i][:-1]+'_d$', alpha=0.6)
        ax[idx_pos_y][j].plot(data_2d[idx_exp].time, data_2d[idx_exp].pos[:,1], linewidth=lw, label=legend_pos[i+1])
        ax[idx_pos_y][j].plot(data_2d[idx_exp].time, data_2d[idx_exp].pos_d[:,1], 'k--', linewidth=lw-0.5, label=legend_pos[i+1][:-1]+'_d$', alpha=0.6)
        ax[idx_pos_z][j].set_xlim(xlimits_2d)
        ax[idx_pos_z][j].set_ylim(ylimits_pos_z)
        ax[idx_pos_y][j].set_xlim(xlimits_2d)
        ax[idx_pos_y][j].set_ylim(ylimits_pos_y)
        
        data_2d[idx_exp].vel = dh.get_data(data_2d[idx_exp].params, file_name=data_2d[idx_exp].file_name, data_to_plot='EE_twist')
        data_2d[idx_exp].vel_d = dh.get_data(data_2d[idx_exp].params, file_name=data_2d[idx_exp].file_name, data_to_plot='EE_twist_d')
        # ax[j][idx_vel].plot(time, vel, colors[i], linewidth=lw, label=legend_vel[i])
        ax[idx_vel_z][j].plot(data_2d[idx_exp].time, data_2d[idx_exp].vel[:,2], linewidth=lw, label=legend_vel[i])
        ax[idx_vel_z][j].plot(data_2d[idx_exp].time, data_2d[idx_exp].vel_d[:,2], 'k--', linewidth=lw-0.5, label=legend_vel[i][:-1]+'_d$', alpha=0.6)
        ax[idx_vel_y][j].plot(data_2d[idx_exp].time, data_2d[idx_exp].vel[:,1], linewidth=lw, label=legend_vel[i+1])
        ax[idx_vel_y][j].plot(data_2d[idx_exp].time, data_2d[idx_exp].vel_d[:,1], 'k--', linewidth=lw-0.5, label=legend_vel[i+1][:-1]+'_d$', alpha=0.6)
        ax[idx_vel_z][j].set_xlim(xlimits_2d)
        ax[idx_vel_z][j].set_ylim(ylimits_vel_z)
        ax[idx_vel_y][j].set_xlim(xlimits_2d)
        ax[idx_vel_y][j].set_ylim(ylimits_vel_y)

        data_2d[idx_exp].pos_ball = dh.get_data(data_2d[idx_exp].params, file_name=data_2d[idx_exp].file_name, data_to_plot='ball_pose_')
        # apply KF
        data_2d[idx_exp].z_actual_hat, data_2d[idx_exp].z_dot_actual_hat, data_2d[idx_exp].z_intersec, data_2d[idx_exp].z_dot_intersec, \
            data_2d[idx_exp].y_actual_hat, data_2d[idx_exp].y_dot_actual_hat, data_2d[idx_exp].y_intersec, data_2d[idx_exp].y_dot_intersec,\
                data_2d[idx_exp].time_intersec, data_2d[idx_exp].time_f = kalman_filtering_2d(data_2d[idx_exp].time, data_2d[idx_exp].pos_ball[:,2], data_2d[idx_exp].pos_ball[:,1], data_2d[idx_exp].ft_)
        # ax[idx_pos][j].plot(time, pos_ball[:,2], 'r', linewidth=lw, label=legend_pos[i])
        if j == 0:
            final_ball_traj = -5
        if j == 1:
            final_ball_traj = -3
        if j == 2:
            final_ball_traj = -4
        if j == 3:
            final_ball_traj = -3
        ax[idx_pos_z][j].plot(data_2d[idx_exp].time_f[:final_ball_traj], data_2d[idx_exp].z_actual_hat[:final_ball_traj]-ball_radius, color='b', linestyle='dashdot', linewidth=lw-0.5, label='$z_b$')
        ax[idx_vel_z][j].plot(data_2d[idx_exp].time_f[:final_ball_traj], data_2d[idx_exp].z_dot_actual_hat[:final_ball_traj], color='b', linestyle='dashdot', linewidth=lw-0.5, label='$\dot{z}_b$')
        ax[idx_pos_y][j].plot(data_2d[idx_exp].time_f[:final_ball_traj], data_2d[idx_exp].y_actual_hat[:final_ball_traj]-ball_radius, color='b', linestyle='dashdot', linewidth=lw-0.5, label='$y_b$')
        ax[idx_vel_y][j].plot(data_2d[idx_exp].time_f[:final_ball_traj], data_2d[idx_exp].y_dot_actual_hat[:final_ball_traj], color='b', linestyle='dashdot', linewidth=lw-0.5, label='$\dot{y}_b$')

        
        # ax_yz[j].plot(y_actual_hat-(95/2/1000), z_actual_hat-(95/2/1000), color='#000000', linestyle='dashdot', linewidth=lw-0.5)
        # ax_yz[j].plot(pos[:,1], pos[:,2], linewidth=lw)
        
        ax[idx_pos_yz][j].plot(data_2d[idx_exp].y_actual_hat[:final_ball_traj]-ball_radius, data_2d[idx_exp].z_actual_hat[:final_ball_traj]-ball_radius, color='b', linestyle='dashdot', linewidth=lw-.5)
        ax[idx_pos_yz][j].plot(data_2d[idx_exp].pos[:,1], data_2d[idx_exp].pos[:,2], linewidth=lw)

        ax[idx_pos_yz][j].add_patch(Ellipse((data_2d[idx_exp].y_actual_hat[final_ball_traj]-ball_radius, data_2d[idx_exp].z_actual_hat[final_ball_traj]-ball_radius), 2*ball_radius/5+0.025, 2*ball_radius, color='b', fill=False))
        ax[idx_pos_z][j].add_patch(Ellipse((data_2d[idx_exp].time_f[final_ball_traj], data_2d[idx_exp].z_actual_hat[final_ball_traj]-ball_radius), 2*ball_radius/5+0.025, 2*ball_radius, color='b', fill=False))
        ax[idx_pos_y][j].add_patch(Ellipse((data_2d[idx_exp].time_f[final_ball_traj], data_2d[idx_exp].y_actual_hat[final_ball_traj]-ball_radius), 2*ball_radius/5+0.025, 4*ball_radius, color='b', fill=False))

        data_2d[idx_exp].tau_m = dh.get_data(params, file_name=data_2d[idx_exp].file_name, data_to_plot='tau_measured')
        data_2d[idx_exp].tau_m_norm = np.divide(data_2d[idx_exp].tau_m, tau_limits)
        # ax[idx_tau][j].plot(time, tau_m_norm, label=['$\\tau_'+str(i+1)+'$' for i in range(7)])
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]#, "#7f7f7f", "#bcbd22", "#17becf"]        
        joint_i = 0
        for tau_i, c  in zip(data_2d[idx_exp].tau_m_norm.T, colors):
            ax[idx_tau][j].plot(data_2d[idx_exp].time, tau_i, label='$\\tau_'+str(joint_i+1)+'$', color=c, linestyle='-')
            joint_i +=1 
        ax[idx_tau][j].set_xlim(xlimits_2d)
        ax[idx_tau][j].set_ylim([-1, 1])      

        # JOINT MEASUREMENTS
        data_2d[idx_exp].tau_adim = 0
        data_2d[idx_exp].tau_max = np.zeros((n_joints))
        data_2d[idx_exp].tau_rms = np.zeros((n_joints))
        data_2d[idx_exp].tau_d_rms = np.zeros((n_joints))
        data_2d[idx_exp].tau_m = dh.get_data(file_name=data_2d[idx_exp].file_name, params=data_2d[idx_exp].params, data_to_plot='tau_measured')[idx_impact:idx_finish_analysis]
        data_2d[idx_exp].tau_d = dh.get_data(file_name=data_2d[idx_exp].file_name, params=data_2d[idx_exp].params, data_to_plot='tau')[idx_impact:idx_finish_analysis]
        data_2d[idx_exp].q_dot = dh.get_data(file_name=data_2d[idx_exp].file_name, params=data_2d[idx_exp].params, data_to_plot='dq')[idx_impact:idx_finish_analysis]
        data_2d[idx_exp].tau_norm = np.zeros_like(data_2d[idx_exp].tau_m)
        data_2d[idx_exp].tau_norm = np.divide(data_2d[idx_exp].tau_m, tau_limits)
        data_2d[idx_exp].tau_max = np.max(np.abs(data_2d[idx_exp].tau_norm.T), axis=1)
        data_2d[idx_exp].tau_d_norm = np.zeros_like(data_2d[idx_exp].tau_d)
        data_2d[idx_exp].tau_d_norm = np.divide(data_2d[idx_exp].tau_d, tau_limits)

        data_2d[idx_exp].tau_rms = np.sqrt(np.mean(data_2d[idx_exp].tau_norm**2, axis=0)/(data_2d[idx_exp].time[idx_finish_analysis]-data_2d[idx_exp].time[idx_impact]))
        data_2d[idx_exp].tau_d_rms = np.sqrt(np.mean(data_2d[idx_exp].tau_d_norm**2, axis=0)/(data_2d[idx_exp].time[idx_finish_analysis]-data_2d[idx_exp].time[idx_impact]))
        data_2d[idx_exp].tau_rms_sum = np.sum(data_2d[idx_exp].tau_rms)

        # ADIM
        data_2d[idx_exp].w_fd = dh.get_data(file_name=data_2d[idx_exp].file_name, params=data_2d[idx_exp].params, data_to_plot='m_arm')
        T_ = data_2d[idx_exp].time[idx_finish_analysis]-data_2d[idx_exp].time[idx_impact]
        idx_step = 0
        time_aux = data_2d[idx_exp].time[0:idx_finish_analysis]
        data_2d[idx_exp].w_fd = data_2d[idx_exp].w_fd[0:idx_finish_analysis]
        while time_aux[idx_step] <= T_:
            if idx_step > 1:
                data_2d[idx_exp].tau_adim += (data_2d[idx_exp].w_fd[idx_step] + data_2d[idx_exp].w_fd[idx_step-1])*(time_aux[idx_step] - time_aux[idx_step-1])/2
            idx_step += 1
            if idx_step+1 >= len(data_2d[idx_exp].w_fd):
                break
        data_2d[idx_exp].tau_adim = data_2d[idx_exp].tau_adim/T_

        # METRICS
        data_2d[idx_exp].ft_ = data_2d[idx_exp].ft_[:,2]
        # pos = pos[:,2]
        # vel = vel[:,2]
        data_2d[idx_exp].loi = LOI(data_2d[idx_exp].time, data_2d[idx_exp].ft_, idx_impact, idx_finish_analysis, data_2d[idx_exp].file_name)
        data_2d[idx_exp].Fmax = -np.min(data_2d[idx_exp].ft_)
        # Fmax = np.max(np.linalg.norm(ft_[:,:3], axis=1))
        # vme = vel[idx_impact] - z_dot_actual_hat[-1]
        data_2d[idx_exp].vme = np.linalg.norm(np.array([data_2d[idx_exp].y_dot_actual_hat[-1], data_2d[idx_exp].z_dot_actual_hat[-1]])-data_2d[idx_exp].vel[idx_impact][1:3])
        # pos_error = np.linalg.norm(np.array([pos_d[idx_impact][1], 0.35]) - pos[idx_impact][1:3])
        data_2d[idx_exp].pos_error = np.linalg.norm(data_2d[idx_exp].pos_d[idx_impact][1:3] - data_2d[idx_exp].pos[idx_impact][1:3])
        data_2d[idx_exp].dri = DRI(data_2d[idx_exp].time, data_2d[idx_exp].ft_, data_2d[idx_exp].file_name)
        data_2d[idx_exp].bti = BTI(data_2d[idx_exp].time, data_2d[idx_exp].ft_, data_2d[idx_exp].file_name)

        print(data_2d[idx_exp].file_name,   '\tLOI =',           data_2d[idx_exp].loi,
                                            '\tDRI = ',          data_2d[idx_exp].dri,
                                            '\tBTI = ',          data_2d[idx_exp].bti,
                                            '\tF_max = ',        data_2d[idx_exp].Fmax,
                                            '\tVM-error = ',     data_2d[idx_exp].vme,
                                            '\tE_pos_impact = ', data_2d[idx_exp].pos_error,
                                            '\tball_max_pos = ', np.max(data_2d[idx_exp].z_actual_hat),
                                            '\tball_delta_y = ', data_2d[idx_exp].y_actual_hat[-1]-data_2d[idx_exp].y_actual_hat[0],
                                            "\tTAU ADIM = ",     data_2d[idx_exp].tau_adim,
                                            "\tTAU MAX = ",      np.max(data_2d[idx_exp].tau_max),
                                            "\tTAU RMS SUM = ",  data_2d[idx_exp].tau_rms_sum,
                                            "\n")
        

    for j, ax_ in enumerate(ax):
        if j < 7 and j > 0:
            for ax_i in ax_:
                ax_i.set_xticks([])
            # ax_i[1].set_xticks([])
    [ax_[1].set_yticks([]) for ax_ in ax]
    for j in range(len(chosen_exps)):
        aux = [0] + list(np.arange(0.25, 1.55, 0.25))
        ax[idx_tau][j].set_xticks(aux)
        ax[idx_tau][j].set_xticklabels(['$'+str(a)+'$' for a in aux], size=13)
    # ax[n_exps-1, 2].axvline(x = 0.685, ymin=0, ymax=2.2, linestyle='-', linewidth=1.3, color = 'r', label = 'axvline - full height', clip_on=False)

    label_size = 15
    ax[idx_pos_yz,0].set_ylabel('$z~[m]$', size=label_size)
    ax[idx_pos_z,0].set_ylabel('$z~[m]$', size=label_size)
    ax[idx_vel_z,0].set_ylabel('$\dot{z}~[m/s]$', size=label_size)
    ax[idx_pos_y,0].set_ylabel('$y~[m]$', size=label_size)
    ax[idx_vel_y,0].set_ylabel('$\dot{y}~[m/s]$', size=label_size)
    ax[idx_ft,0].set_ylabel('$F_z~[N]$', size=label_size)
    ax[idx_Kp,0].set_ylabel('$K_{p_{yz}}$', size=label_size)
    ax[idx_tau,0].set_ylabel('$\hat{\\boldsymbol{\\tau}}$', size=label_size)

    # grids creation
    x_grids = list(np.arange(0,2,0.25))
    n_divisions = 5
    alpha_grids = 0.12
    y_grids_ft = [-20, -15, -10, -5, 0, 5]
    y_grids_Kp = [10, 20, 30, 40, 50]
    
    y_grids_pos_z = [i for i in list(np.arange(0, 1.1, 0.2))]
    y_grids_vel_z = [-2, -1, 0, 1, 2.0, 3.0]
    
    y_grids_pos_y = [i for i in list(np.arange(-1.5, -0.2, 0.25))]
    y_grids_vel_y = [-2, -1, 0, 1, 2.0, 3.0]
    y_grids_tau = [-1,-0.5, 0, 0.5, 1]
    for j, row in enumerate(ax):
        for i, e in enumerate(row):
            [e.axvline(xg, color='k', alpha=alpha_grids) for xg in x_grids]
            if idx_ft == j:
                [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_ft]
                if i == 0:

                    aux = [5, 0, -5, -10, -15]
                    e.set_yticks(aux)
                    e.set_yticklabels(['$'+str(a)+'$' for a in aux], size=13)
            if idx_Kp == j:
                [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_Kp]
                if i == 0:
                    aux = [10, 20, 30, 40, 50]
                    e.set_yticks(aux)
                    e.set_yticklabels(['$'+str(a)+'$' for a in aux], size=13)
            if idx_pos_z == j:
                [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_pos_z]
                if i == 0:
                    aux = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                    e.set_yticks(aux)
                    e.set_yticklabels(['$'+str(a)+'$' for a in aux], size=13)
            if idx_vel_z == j:
                if i == 0:
                    aux = [-2, -1, 0, 1, 2]
                    e.set_yticks(aux)
                    e.set_yticklabels(['$'+str(a)+'$' for a in aux], size=13)
                [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_vel_z]
            if idx_pos_y == j:
                [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_pos_y]
                if i == 0:
                    aux = [0, -0.5, -1.0, -1.5]
                    e.set_yticks(aux)
                    e.set_yticklabels(['$'+str(a)+'$' for a in aux], size=13)
            if idx_vel_y == j:
                if i == 0:
                    aux = [0, 1, 2, 3]
                    e.set_yticks(aux)
                    e.set_yticklabels(['$'+str(a)+'$' for a in aux], size=13)
                [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_vel_y]
            if idx_pos_yz == j:
                if i == 0:
                    aux = y_grids_pos_z
                    e.set_yticks(aux)
                    e.set_yticklabels(['$'+str(a.round(decimals=2))+'$' for a in aux], size=13)
                aux = y_grids_pos_y
                e.set_xticks(aux[:-1])
                e.set_xticklabels(['$'+str(a.round(decimals=2))+'$' for a in aux[:-1]], size=13)
                [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_pos_y]
                [e.axvline(zg, color='k', alpha=alpha_grids) for zg in y_grids_pos_z]
            if idx_tau == j:
                if i == 0:
                    aux = [-1, -0.5, 0, 0.5, 1]
                    e.set_yticks(aux)
                    e.set_yticklabels(['$'+str(a)+'$' for a in aux], size=13)
                [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_tau]
                

    fig.align_ylabels()

    names = ['$VM-VIC_1$', '$VM-VIC_2$','$VM-VIC-DIM_1$', '$VM-VIC-DIM_2$']
    for ax_, name in zip(ax[0], names):
        ax_.set_title(name, size=20)
    
    # names = [r'$VM-VIC_1$', r'$VM-VIC_2$',r'$VM-VIC-RMM_1$', r'$VM-VIC-RMM_2$']
    # for ax_, name in zip(ax_yz, names):
    #     ax_aux = ax_.twinx()
    #     ax_aux.set_ylabel(name, rotation=-90, fontsize=10)
    
    # ax_yz[1].invert_xaxis()
    # ax_yz[1].grid()
    # ax_yz[0].set_ylabel('$z~[m]$')
    # ax_yz[1].set_ylabel('$z~[m]$')
    # fig_yz.supylabel('$z~[m]$')
    
    # ax_yz[-1].set_xlabel('$y~[m]$')
    ax[idx_pos_yz][1].set_xlabel('$y~[m]$', size=20)
    ax[idx_pos_yz][1].xaxis.set_label_coords(1, -0.25)

    # ax_yz[0].set_xlim([-1.75, -0.3])
    # ax_yz[1].set_xlim([-1.75, -0.3])
    # ax_yz[0].set_ylim([0.25, 1.0])
    # ax_yz[1].set_ylim([0.25, 1.0])

    # fig_yz.set_tight_layout(tight=True)

    # fig.supxlabel('$Time~[s]$', size=20)
    ax[idx_tau][1].set_xlabel('$Time~[s]$', size=16)
    ax[idx_tau][1].xaxis.set_label_coords(1, -0.35)


    # for ax_ in ax_yz:
    for ax_ in ax[idx_pos_yz]:
        ax_.set_xlim([-1.75, -0.25])
        ax_.set_ylim([0.2, 1.0])
        ax_.invert_xaxis()
        # ax_.grid()
        [ax_.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_pos_z]
        [ax_.axvline(zg, color='k', alpha=alpha_grids) for zg in y_grids_pos_y]
    
    for j in range(4):
        ax[idx_tau,j].axvline(x = 0.63, ymin=0, ymax=7.95, linestyle='--', linewidth=1.1, color = 'r', label = '_nolegend_', clip_on=False)  

    # ax[5, 0].legend(prop={'size': 14}, loc='top right')
    # ax[6, 0].legend(prop={'size': 14}, loc='top right')


    # for j in range(4):
    #     if j == 1 or j == 2:
    #         ax[j, 3].legend(prop={'size': 10}, loc='lower right')
    #     if j == 0:
    #         ax[j, 3].legend(prop={'size': 10}, loc='upper right')

    # fig.subplots_adjust(hspace=0., wspace=0.)
    # fig.set_constrained_layout(constrained=True)
    # fig.set_constrained_layout_pads(hspace=0., wspace=0.035)
    # fig.get_layout_engine.set(wspace=0, hspace=0)
    # fig_yz.set_constrained_layout(constrained=True)

    for j in range(4):
        bbox = ax[0][j].get_position()
        ax[0][j].set_position([bbox.x0, bbox.y0+0.04, bbox.x1-bbox.x0, bbox.y1 - bbox.y0], which='original')
        bbox = ax[-1][j].get_position()
        ax[-1][j].set_position([bbox.x0, bbox.y0, bbox.x1-bbox.x0, bbox.y1 - bbox.y0], which='original')

    size_leg = 9
    ax[1, 0].legend(prop={'size': size_leg}, loc='upper right', ncol=2)
    ax[2, 0].legend(prop={'size': size_leg}, loc='upper right', ncol=2)
    ax[3, 0].legend(prop={'size': size_leg}, loc='lower right', ncol=2)
    ax[4, 0].legend(prop={'size': size_leg}, loc='upper right', ncol=2)
    leg = ax[idx_tau, 0].legend(prop={'size': 8}, loc='lower right', bbox_to_anchor=(1, 0.65), ncol=3)
    leg.get_frame().set_alpha(None)
    leg.get_frame().set_facecolor((1.0, 1.0, 1, 1))


    ax[1][0].axhline(0.35, linestyle='--', linewidth=1.1, color = 'r', label = '_nolegend_')
    ax[1][1].axhline(0.35, linestyle='--', linewidth=1.1, color = 'r', label = '_nolegend_')
    ax[1][0].axvline(0.63, linestyle='--', linewidth=1.1, color = 'r', label = '_nolegend_')
    ax[1][1].axvline(0.63, linestyle='--', linewidth=1.1, color = 'r', label = '_nolegend_')

    ax[1][2].axhline(0.35, linestyle='--', linewidth=1.1, color = 'r', label = '_nolegend_')
    ax[1][3].axhline(0.35, linestyle='--', linewidth=1.1, color = 'r', label = '_nolegend_')
    ax[1][2].axvline(0.63, linestyle='--', linewidth=1.1, color = 'r', label = '_nolegend_')
    ax[1][3].axvline(0.63, linestyle='--', linewidth=1.1, color = 'r', label = '_nolegend_')

    ax[-1, 1].set_zorder(0)

    # print('VM-VIC1 metrics:','\tCatching time delay = ', .6865-0.6545)
    # print('\nVM-VIC2 metrics:','\tCatching time delay = ', .6821-0.6187)
    # print('\nVM-VIC-DIM1 metrics:','\tCatching time delay = ', 0.750-.631)
    # print('\nVM-VIC-DIM2 metrics:','\tCatching time delay = ', 0.6979-.631)

    # plt.show()
    fig.savefig('images/2d-time-plots.png', dpi=400, bbox_inches='tight')
    # fig_yz.savefig('images/2d_spatial.png')
    return data_2d

def plot_rmm_1d_all_in_one_4_plots():
    # colors = ['b', 'r', 'g', 'y']
    colors = ['b', 'b', 'b', 'b']
    chosen_exps = [12, 13, 17, 18]
    styles = ['solid', 'dotted', 'dashdot', 'on-off-dash-seq']
    i = 0
    ylimits_ft = [-10, 5]
    ylimits_Kp = [0, 60]
    ylimits_pos = [0, 0.8]
    ylimits_vel = [-2.75, 1.0]
    # fig_ft, ax_ft = plt.subplots(1, 4)
    # fig_Kp, ax_Kp  = plt.subplots(1, 4)
    # fig_pos, ax_pos  = plt.subplots(1, 4)
    # fig_vel, ax_vel  = plt.subplots(1, 4)

    # idx_exp = chosen_exps[1]
    # file_name = dh._get_name(idx_exp)
    # params['idx_initial'] = dh.get_idx_from_file(idx_exp, data_info, idx_name='idx_start')
    # params['idx_end'] = dh.get_idx_from_file(idx_exp, data_info, idx_name='idx_end')
    # time = dh.get_data(file_name=file_name,params=params, data_to_plot='time')
    # time -= time[0]
    # pos_ball = dh.get_data(params, file_name=file_name, data_to_plot='ball_pose_')[:,2]
    # vel_d = dh.get_data(params, file_name=file_name, data_to_plot='EE_twist_d')[:,2]
    # z_actual_hat, z_dot_actual_hat, z_intersec, z_dot_intersec, time_intersec, time_f = kalman_filtering_1d(time, pos_ball, vel_d)
    # plt.plot(time, pos_ball, 'k')
    # plt.plot(time_f, z_actual_hat, 'r--')
    # plt.legend(['$z$', '$\hat{z}$'])
    # plt.show()

    # exit()

    fig, ax  = plt.subplots(4, len(chosen_exps), figsize=(20, 5), layout="constrained", sharex=True)
    # fig_b, ax_b = plt.subplots(1,1)

    idx_pos = 0
    idx_vel = 1
    idx_ft = 2
    idx_Kp = 3

    # legend_ft = [r'$F^{VM+K_H}_z$', r'$F^{VM+K_L}_z$', r'$F^{VM+SIC}_z$', r'$F^{VM+POC}_z$']
    legend_ft = [r'$F_z$', r'$F_z$', r'$F_z$', r'$F_z$']
    legend_Kp = [r'$K_H$', r'$K_L$', r'$SIC$', r'$POC$']
    # legend_pos = [r'$z^{VM+K_H}$', r'$z^{VM+K_L}$', r'$z^{VM+SIC}$', r'$z^{VM+POC}$']
    legend_pos = [r'$z$', r'$z$', r'$z$', r'$z$']
    # legend_vel = [r'$\dot{z}^{VM+K_H}$', r'$\dot{z}^{VM+K_L}$', r'$\dot{z}^{VM+SIC}$', r'$\dot{z}^{VM+POC}$']
    legend_vel = [r'$\dot{z}$', r'$\dot{z}$', r'$\dot{z}$', r'$\dot{z}$']

    for j, idx_exp in enumerate(chosen_exps):
        file_name = dh._get_name(idx_exp)
        print(file_name)
        params['idx_initial'] = dh.get_idx_from_file(idx_exp, data_info, idx_name='idx_start')+200
        params['idx_end'] = dh.get_idx_from_file(idx_exp, data_info, idx_name='idx_end')

        time = dh.get_data(file_name=file_name,params=params, data_to_plot='time')
        time -= time[0]

        ft_ = dh.get_data(file_name=file_name, params=params, data_to_plot='ft_')[:,2]
        ax[idx_ft][j].plot(time, ft_, colors[i], linewidth=lw, label=legend_ft[i])
        # ax[idx_ft][j].set_xlabel('$Time~[s]$')
        # ax[idx_ft][j].set_ylabel('$F_z~[N]$')
        ax[idx_ft][j].set_xlim(xlimits)
        ax[idx_ft][j].set_ylim(ylimits_ft)
        # ax[idx_ft][j].legend(prop={'size': 8}, loc='lower right')
        

        Kp = dh.get_data(params, file_name=file_name, data_to_plot='Kp')[:,2]
        # ax[idx_Kp][j].plot(time, Kp, colors[i], linewidth=lw, label=legend_Kp[i])
        ax[idx_Kp][j].plot(time, Kp, colors[i], linewidth=lw)
        # ax[idx_Kp][j].set_xlabel('$Time~[s]$')
        # ax[idx_Kp][j].set_ylabel('$Kp~[-]$')
        ax[idx_Kp][j].set_xlim(xlimits)
        ax[idx_Kp][j].set_ylim(ylimits_Kp)
        # ax[idx_Kp][j].legend(prop={'size': 8}, loc='lower right')

        pos = dh.get_data(params, file_name=file_name, data_to_plot='EE_position')[:,2]
        pos_d = dh.get_data(params, file_name=file_name, data_to_plot='EE_position_d')[:,2]
        ax[idx_pos][j].plot(time, pos, colors[i], linewidth=lw, label=legend_pos[i])
        ax[idx_pos][j].plot(time, pos_d, 'k--', linewidth=lw-0.5, label=legend_pos[i][:-1]+'_d$')
        # ax[idx_pos][j].set_xlabel('$Time~[s]$')
        # ax[idx_pos][j].set_ylabel('$z~[m]$')
        pos_ball = dh.get_data(params, file_name=file_name, data_to_plot='ball_pose_')[:,2]
        ax[idx_pos][j].set_xlim(xlimits)
        ax[idx_pos][j].set_ylim(ylimits_pos)
        # ax[idx_pos][j].legend(prop={'size': 8}, loc='lower right')


        vel = dh.get_data(params, file_name=file_name, data_to_plot='EE_twist')[:,2]
        vel_d = dh.get_data(params, file_name=file_name, data_to_plot='EE_twist_d')[:,2]
        ax[idx_vel][j].plot(time, vel, colors[i], linewidth=lw, label=legend_vel[i])
        ax[idx_vel][j].plot(time, vel_d, 'k--', linewidth=lw-0.5, label=legend_vel[i][:-1]+'_d$')
        # ax[idx_vel][j].set_xlabel('$Time~[s]$')
        # ax[idx_vel][j].set_ylabel('$\dot{z}~[m/s]$')
        ax[idx_vel][j].set_xlim(xlimits)
        ax[idx_vel][j].set_ylim(ylimits_vel)
        # ax[idx_vel][j].legend(prop={'size': 8}, loc='lower right')

        # apply KF
        z_actual_hat, z_dot_actual_hat, z_intersec, z_dot_intersec, time_intersec, time_f = kalman_filtering_1d(time, pos_ball, ft_)
        # ax[idx_pos][j].plot(time, pos_ball, 'k', linewidth=lw, label=legend_pos[i])
        ax[idx_pos][j].plot(time_f, z_actual_hat-(95/2/1000), color='m', linestyle='dashdot', linewidth=lw-0.5, label='$z_b$')
        ax[idx_vel][j].plot(time_f, z_dot_actual_hat, color='m', linestyle='dashdot', linewidth=lw-0.5, label='$\dot{z}_b$')

        i += 1

    # fig_ft.legend(legend_ft)
    # fig_Kp.legend(legend_Kp)
    # fig_pos.legend(legend_pos)
    # fig_vel.legend(legend_vel)

    # fig[idx_ft].title("$F_z~[N]$")
    # fig[idx_Kp].title("$K_p~[-]$")
    # fig[idx_pos].title("$z~[m]$")
    # fig[idx_vel].title("$\dot{z}~[m/s]$")

    # fig[0].tight_layout()
    # fig[0].tight_layout()
    # fig[0].tight_layout()
    # fig[0].tight_layout()
    # fig[0].sett_constrained_layout(constrained=True)

    ax[idx_pos,0].set_ylabel('$z~[m]$')
    ax[idx_vel,0].set_ylabel('$\dot{z}~[m]$')
    ax[idx_Kp,0].set_ylabel('$K_p$')
    ax[idx_ft,0].set_ylabel('$F_z~[N]$')
    # fig.align_ylabels()

    # ax[0,0].set_title('$VM+K_{P_H}$')
    # ax[0,1].set_title('$VM+K_{P_L}$')
    # ax[0,0].set_title('$VM+K_{H}$')
    # ax[1,0].set_title('$VM+K_{L}$')
    # ax[2,0].set_title('$VM+SIC$')
    # ax[3,0].set_title('$VM+POC$')
    
    for j, _ in enumerate(chosen_exps):
        if j < 3:
            ax[0][j].set_xticks([])
            ax[1][j].set_xticks([])
            ax[2][j].set_xticks([])
            ax[3][j].set_xticks([])
        if j > 0:
            ax[j][0].set_yticks([])
            ax[j][1].set_yticks([])
            # ax[j][2].set_yticks([])
            # ax[j][3].set_yticks([])

    # grids creation
    x_grids = list(np.arange(0,2,0.25))
    n_divisions = 5
    alpha_grids = 0.12
    y_grids_ft = list(np.arange(ylimits_ft[0], ylimits_ft[-1], (ylimits_ft[-1]-ylimits_ft[0])/n_divisions))
    y_grids_Kp = list(np.arange(ylimits_Kp[0], ylimits_Kp[-1], (ylimits_Kp[-1]-ylimits_Kp[0])/n_divisions))
    y_grids_pos = list(np.arange(ylimits_pos[0], ylimits_pos[-1], (ylimits_pos[-1]-ylimits_pos[0])/n_divisions))
    y_grids_vel = list(np.arange(ylimits_vel[0], ylimits_vel[-1], (ylimits_vel[-1]-ylimits_vel[0])/n_divisions))
    for j, row in enumerate(ax):
        for e in row:
            [e.axvline(xg, color='k', alpha=alpha_grids) for xg in x_grids]
            if idx_ft == j:
                [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_ft]
                e.axhline(0, color='k')
            if idx_Kp == j:
                [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_Kp]
            if idx_pos == j:
                [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_pos]
            if idx_vel == j:
                [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_vel]
    
    # ft 
    # for j in range(4):
    #     ax[idx_ft, j].axhline(-0.1*9.81, color='m', linestyle='dashdot', linewidth=lw-0.5, label='$W_b$')

    # plt.gca().grid(True)
    # plt.tight_layout()

    # legends
    # for j, row in enumerate(ax):
    #     for e in row:
    #         e.legend(prop={'size': 10}, loc='lower right')
    for j in range(4):
        if j == 1 or j == 2:
            ax[j, 1].legend(prop={'size': 10}, loc='lower right')
        if j == 0:
            ax[j, 1].legend(prop={'size': 10}, loc='upper right')

    plt.subplots_adjust(wspace=.025, hspace=0.045)
    plt.show()
    # fig_ft.savefig('images/vanilla_best_ft.png')
    # fig_Kp.savefig('images/vanilla_best_Kp.png')
    # fig_pos.savefig('images/vanilla_best_pos.png')
    # fig_vel.savefig('images/vanilla_best_vel.png')

def plot_rmm_1d_1_plot():
    # colors = ['b', 'r', 'g', 'y']
    # colors = [myc.mpl_colors[3], myc.mpl_colors[-1]]
    colors = ['k']
    chosen_exps = [17]
    i = 0
    ylimits_ft = [-20, 5]
    ylimits_Kp = [10, 50]
    ylimits_pos = [0, 0.8]
    ylimits_vel = [-2.75, 1]

    fig, ax  = plt.subplots(5, 1, figsize=(6, 6),# layout="constrained",
                             gridspec_kw={'wspace': 0.0, 'hspace': 0.25})
    # fig_b, ax_b = plt.subplots(1,1)

    idx_pos = 0
    idx_vel = 1
    idx_ft = 2
    idx_Kp = 3
    idx_tau = 4

    # legend_ft = [r'$F^{VM+K_H}_z$', r'$F^{VM+K_L}_z$', r'$F^{VM+SIC}_z$', r'$F^{VM+POC}_z$']
    legend_ft = [r'$F_z$', r'$F_z$', r'$F_z$', r'$F_z$']*2
    legend_Kp = [r'$K_H$', r'$K_L$', r'$SIC$', r'$POC$']*2
    # legend_pos = [r'$z^{VM+K_H}$', r'$z^{VM+K_L}$', r'$z^{VM+SIC}$', r'$z^{VM+POC}$']
    legend_pos = [r'$z$', r'$z$', r'$z$', r'$z$']*2
    # legend_vel = [r'$\dot{z}^{VM+K_H}$', r'$\dot{z}^{VM+K_L}$', r'$\dot{z}^{VM+SIC}$', r'$\dot{z}^{VM+POC}$']
    legend_vel = [r'$\dot{z}$', r'$\dot{z}$', r'$\dot{z}$', r'$\dot{z}$']*2


    # params['idx_initial'] = 0
    # params['idx_end'] = -1
    # file_name = dh._get_name(chosen_exps[0])
    # time = dh.get_data(file_name=file_name,params=params, data_to_plot='time')
    # Kp = dh.get_data(params, file_name=file_name, data_to_plot='Kp')[:,2]

    # plt.plot(time, Kp)
    # plt.show()

    labels = ['$VM-VIC-DIM$']


    for j, idx_exp in enumerate(chosen_exps):
        file_name = dh._get_name(idx_exp)
        print(file_name)
        idx_start = 200
        params['idx_initial'] = dh.get_idx_from_file(idx_exp, data_info, idx_name='idx_start')+idx_start
        params['idx_end'] = dh.get_idx_from_file(idx_exp, data_info, idx_name='idx_end')

        time = dh.get_data(file_name=file_name,params=params, data_to_plot='time')
        time -= time[0]

        ft_ = dh.get_data(file_name=file_name, params=params, data_to_plot='ft_')[:,2]
        ax[idx_ft].plot(time, ft_, color=colors[i], linewidth=lw, label=legend_ft[i])
        ax[idx_ft].set_xlim(xlimits)
        ax[idx_ft].set_ylim(ylimits_ft)

        Kp = dh.get_data(params, file_name=file_name, data_to_plot='Kp')[:,2]
        ax[idx_Kp].plot(time, Kp, color=colors[i], linewidth=lw, label=labels[i])
        ax[idx_Kp].set_xlim(xlimits)
        ax[idx_Kp].set_ylim(ylimits_Kp)

        pos = dh.get_data(params, file_name=file_name, data_to_plot='EE_position')[:,2]
        pos_d = dh.get_data(params, file_name=file_name, data_to_plot='EE_position_d')[:,2]
        ax[idx_pos].plot(time, pos, color=colors[i], linewidth=lw, label=legend_pos[i])
        # if i == 0:
        ax[idx_pos].plot(time, pos_d, color=colors[i], linestyle='--', linewidth=lw-0.5, label=legend_pos[i][:-1]+'_d$', alpha=0.6)
        pos_ball = dh.get_data(params, file_name=file_name, data_to_plot='ball_pose_')[:,2]
        ax[idx_pos].set_xlim(xlimits)
        ax[idx_pos].set_ylim(ylimits_pos)


        vel = dh.get_data(params, file_name=file_name, data_to_plot='EE_twist')[:,2]
        if i == 0:
            vel_d = dh.get_data(params, file_name=file_name, data_to_plot='EE_twist_d')[:,2]
            # idx_impact = np.where(pos_d < 0.382)[0][0]
            # vel_d[idx_impact:] = np.zeros_like(vel_d[idx_impact:])
        else:
            vel_d = dh.get_data(params, file_name=file_name, data_to_plot='EE_twist_d')[:,2]
        ax[idx_vel].plot(time, vel, color=colors[i], linewidth=lw, label=legend_vel[i])
        ax[idx_vel].plot(time, vel_d, color=colors[i], linestyle='--', linewidth=lw-0.5, label=legend_vel[i][:-1]+'_d$', alpha=0.6)
        ax[idx_vel].set_xlim(xlimits)
        ax[idx_vel].set_ylim(ylimits_vel)

        # apply KF
        
        z_actual_hat, z_dot_actual_hat, z_intersec, z_dot_intersec, time_intersec, time_f = kalman_filtering_1d(time, pos_ball, ft_)
        # ax[idx_pos].plot(time_f, z_actual_hat-(95/2/1000), color='b', linestyle='dashdot', linewidth=lw-0.5, label='$z_b$')
        if i == 0:
            ax[idx_pos].plot(time_f[:-2], z_actual_hat[:-2], color='b', linestyle='dashdot', linewidth=lw-0.5, label='$z_b$')
            ax[idx_vel].plot(time_f[:-2], z_dot_actual_hat[:-2], color='b', linestyle='dashdot', linewidth=lw-0.5, label='$\dot{z}_b$')

        tau_m = dh.get_data(params, file_name=file_name, data_to_plot='tau_measured')
        tau_m_norm = np.divide(tau_m, tau_limits)
        if j == 1:
            idx_tau += 1
        # labels_tau = []
        # for i in range(7):
        #     if i == 5:
        #         labels_tau.append('$\\tau_'+str(i+1)+'$')
        #     else:
        #         labels_tau.append('_nolegend_')
        ax[idx_tau].plot(time, tau_m_norm, label=['$\\tau_'+str(i+1)+'$' for i in range(7)])
        ax[idx_tau].set_xlim(xlimits)
        ax[idx_tau].set_ylim([-1, 1])
        # ADJUST THIS GUYS

        i += 1

    # fig.set_constrained_layout(constrained=True)
    ax[idx_pos].add_patch(Ellipse((time_f[-2], z_actual_hat[-2]), 2*ball_radius/5+0.01, 2*ball_radius, color='b', fill=False))
    ax[idx_pos].set_ylabel('$z~[m]$')
    ax[idx_vel].set_ylabel('$\dot{z}~[m]$')
    ax[idx_Kp].set_ylabel('$K_p$')
    ax[idx_ft].set_ylabel('$F_z~[N]$')
    # ax[idx_tau-1].set_ylabel(r'$\hat{\boldsymbol{\tau}}_{K_H}$')
    ax[idx_tau].set_ylabel(r'$\hat{\boldsymbol{\tau}}$')
    ax[idx_tau].set_xlabel('$Time~[s]$', size=15)
    fig.align_ylabels()

    for ax_ in ax[:-1]:
        ax_.set_xticks([])

    ax[-1].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax[-1].set_xticklabels(['$0$', '$0.25$', '$0.5$', '$0.75$', '$1$'], size=13)

    # grids creation
    x_grids = list(np.arange(0,2,0.25))
    alpha_grids = 0.12
    y_grids_ft = [-15, -10, -5, 0, 5]
    y_grids_Kp = [0, 10, 20, 30, 40, 50]
    y_grids_pos = [i for i in list(np.arange(0, 0.81, 0.1))]
    y_grids_vel = [-2, -1, 0, 0.5]
    y_grids_tau = [-1, -0.5, 0, 0.5, 1]
    size_labels = 12
    for j, e in enumerate(ax):
        [e.axvline(xg, color='k', alpha=alpha_grids) for xg in x_grids]
        if idx_ft == j:
            e.set_yticks([5, 0, -5, -10, -15, -20])
            e.set_yticklabels(['$5$', '$0$', "$-5$", '$-10$', '$-15$', '$-20$'], size=size_labels)
            [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_ft]
            # e.axhline(0, color='k')
        if idx_Kp == j:
            e.set_yticks([10, 20, 30, 40, 50])
            e.set_yticklabels(['$10$', '$20$', '$30$', '$40$', '$50$'], size=size_labels)
            [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_Kp]
        if idx_pos == j:
            aux = [0, 0.2, 0.4, 0.6, 0.8]
            e.set_yticks(aux)
            e.set_yticklabels(['$'+str(a)+'$' for a in aux], size=size_labels)
            [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_pos]
        if idx_vel == j:
            e.set_yticks([-2, -1, 0])
            e.set_yticklabels(['$-2$', '$-1$', '$0$'], size=size_labels)
            [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_vel]
        if idx_tau == j:
            e.set_yticks(y_grids_tau)
            e.set_yticklabels(['$'+str(a)+'$' for a in y_grids_tau], size=size_labels)
            [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_tau]

    # for j in range(4):
    #     if j == 1:
    #         ax[j].legend(prop={'size': 10}, loc='lower right')
    #     if j == 0:
    leg = ax[-1].legend(prop={'size': 8}, loc='lower right', bbox_to_anchor=(1.01, 0.65), ncol=4)
    leg.get_frame().set_alpha(None)
    leg.get_frame().set_facecolor((1.0, 1.0, 1, 1))

    for j in range(5):
            if j == 0:
                ax[j].legend(prop={'size': 9}, loc='upper right', ncol=2)
            if j == 1:
                ax[j].legend(prop={'size': 9}, loc='lower right', ncol=2)
            if j == idx_tau:
                leg = ax[j].legend(prop={'size': 8}, loc='lower right', ncol=4, bbox_to_anchor=(1.01, 0.65))
                leg.get_frame().set_alpha(None)
                leg.get_frame().set_facecolor((1.0, 1.0, 1, 1))

    # ax[-2].legend(prop={'size': 10}, loc='lower right')
    # ax[-3].legend(prop={'size': 10}, loc='lower right')
    ax[-1].axvline(x = 0.293, ymin=0, ymax=6, linestyle='--', linewidth=1.1, color = 'r', label = '_nolegend_', clip_on=False)

    plt.show()
    fig.savefig('tro2023/images/1d_plots_dim.png', dpi=400, bbox_inches='tight')

def plot_rmm_1d_1_plot_with_without_dim():
    # colors = ['b', 'r', 'g', 'y']
    # colors = [myc.mpl_colors[3], myc.mpl_colors[-1]]
    colors = ['k', myc.mpl_colors[3]]
    chosen_exps = [17, 14]
    i = 0
    ylimits_ft = [-20, 5]
    ylimits_Kp = [10, 50]
    ylimits_pos = [0, 0.8]
    ylimits_vel = [-2.75, 1]

    fig_dim, ax_dim  = plt.subplots(6, 1, figsize=(6, 6),# layout="constrained",
                             gridspec_kw={'wspace': 0.0, 'hspace': 0.25})
    # fig_b, ax_b = plt.subplots(1,1)

    idx_pos = 0
    idx_vel = 1
    idx_ft = 2
    idx_Kp = 3
    idx_tau = 4

    # legend_ft = [r'$F^{VM+K_H}_z$', r'$F^{VM+K_L}_z$', r'$F^{VM+SIC}_z$', r'$F^{VM+POC}_z$']
    legend_ft = [r'$F_z$', r'$F_z$', r'$F_z$', r'$F_z$']*2
    legend_Kp = [r'$K_H$', r'$K_L$', r'$SIC$', r'$POC$']*2
    # legend_pos = [r'$z^{VM+K_H}$', r'$z^{VM+K_L}$', r'$z^{VM+SIC}$', r'$z^{VM+POC}$']
    legend_pos = [r'$z$', r'$z$', r'$z$', r'$z$']*2
    # legend_vel = [r'$\dot{z}^{VM+K_H}$', r'$\dot{z}^{VM+K_L}$', r'$\dot{z}^{VM+SIC}$', r'$\dot{z}^{VM+POC}$']
    legend_vel = [r'$\dot{z}$', r'$\dot{z}$', r'$\dot{z}$', r'$\dot{z}$']*2


    # params['idx_initial'] = 0
    # params['idx_end'] = -1
    # file_name = dh._get_name(chosen_exps[0])
    # time = dh.get_data(file_name=file_name,params=params, data_to_plot='time')
    # Kp = dh.get_data(params, file_name=file_name, data_to_plot='Kp')[:,2]

    # plt.plot(time, Kp)
    # plt.show()

    labels = ['$VM-VIC-DIM$', '$VM-VIC$']


    for j, idx_exp in enumerate(chosen_exps):
        file_name = dh._get_name(idx_exp)
        print(file_name)
        idx_start = 200
        params['idx_initial'] = dh.get_idx_from_file(idx_exp, data_info, idx_name='idx_start')+idx_start
        params['idx_end'] = dh.get_idx_from_file(idx_exp, data_info, idx_name='idx_end')

        time = dh.get_data(file_name=file_name,params=params, data_to_plot='time')
        time -= time[0]

        ft_ = dh.get_data(file_name=file_name, params=params, data_to_plot='ft_')[:,2]
        ax_dim[idx_ft].plot(time, ft_, color=colors[i], linewidth=lw, label=legend_ft[i])
        ax_dim[idx_ft].set_xlim(xlimits)
        ax_dim[idx_ft].set_ylim(ylimits_ft)

        Kp = dh.get_data(params, file_name=file_name, data_to_plot='Kp')[:,2]
        ax_dim[idx_Kp].plot(time, Kp, color=colors[i], linewidth=lw, label=labels[i])
        ax_dim[idx_Kp].set_xlim(xlimits)
        ax_dim[idx_Kp].set_ylim(ylimits_Kp)

        pos = dh.get_data(params, file_name=file_name, data_to_plot='EE_position')[:,2]
        pos_d = dh.get_data(params, file_name=file_name, data_to_plot='EE_position_d')[:,2]
        ax_dim[idx_pos].plot(time, pos, color=colors[i], linewidth=lw, label=legend_pos[i])
        # if i == 0:
        ax_dim[idx_pos].plot(time, pos_d, color=colors[i], linestyle='--', linewidth=lw-0.5, label=legend_pos[i][:-1]+'_d$', alpha=0.6)
        pos_ball = dh.get_data(params, file_name=file_name, data_to_plot='ball_pose_')[:,2]
        ax_dim[idx_pos].set_xlim(xlimits)
        ax_dim[idx_pos].set_ylim(ylimits_pos)


        vel = dh.get_data(params, file_name=file_name, data_to_plot='EE_twist')[:,2]
        if i == 0:
            vel_d = dh.get_data(params, file_name=file_name, data_to_plot='EE_twist_d')[:,2]
            idx_impact = np.where(pos_d < 0.382)[0][0]
            vel_d[idx_impact:] = np.zeros_like(vel_d[idx_impact:])
        else:
            vel_d = dh.get_data(params, file_name=file_name, data_to_plot='EE_twist_d')[:,2]
        ax_dim[idx_vel].plot(time, vel, color=colors[i], linewidth=lw, label=legend_vel[i])
        ax_dim[idx_vel].plot(time, vel_d, color=colors[i], linestyle='--', linewidth=lw-0.5, label=legend_vel[i][:-1]+'_d$', alpha=0.6)
        ax_dim[idx_vel].set_xlim(xlimits)
        ax_dim[idx_vel].set_ylim(ylimits_vel)

        # apply KF
        
        z_actual_hat, z_dot_actual_hat, z_intersec, z_dot_intersec, time_intersec, time_f = kalman_filtering_1d(time, pos_ball, ft_)
        # ax[idx_pos].plot(time_f, z_actual_hat-(95/2/1000), color='b', linestyle='dashdot', linewidth=lw-0.5, label='$z_b$')
        if i == 0:
            ax_dim[idx_pos].plot(time_f[:-2], z_actual_hat[:-2], color='b', linestyle='dashdot', linewidth=lw-0.5, label='$z_b$')
            ax_dim[idx_vel].plot(time_f[:-2], z_dot_actual_hat[:-2], color='b', linestyle='dashdot', linewidth=lw-0.5, label='$\dot{z}_b$')

        tau_m = dh.get_data(params, file_name=file_name, data_to_plot='tau_measured')
        tau_m_norm = np.divide(tau_m, tau_limits)
        if j == 1:
            idx_tau += 1
        if j==0:
            ax_dim[idx_pos].add_patch(Ellipse((time_f[-2], z_actual_hat[-2]), 2*ball_radius/5+0.01, 2*ball_radius, color='b', fill=False))

        # labels_tau = []
        # for i in range(7):
        #     if i == 5:
        #         labels_tau.append('$\\tau_'+str(i+1)+'$')
        #     else:
        #         labels_tau.append('_nolegend_')
        ax_dim[idx_tau].plot(time, tau_m_norm, label=['$\\tau_'+str(i+1)+'$' for i in range(7)])
        ax_dim[idx_tau].set_xlim(xlimits)
        ax_dim[idx_tau].set_ylim([-1, 1])
        # ADJUST THIS GUYS

        i += 1

    # fig.set_constrained_layout(constrained=True)
    ax_dim[idx_pos].set_ylabel('$z~[m]$')
    ax_dim[idx_vel].set_ylabel('$\dot{z}~[m]$')
    ax_dim[idx_Kp].set_ylabel('$K_p$')
    ax_dim[idx_ft].set_ylabel('$F_z~[N]$')
    ax_dim[idx_tau-1].set_ylabel(r'$\hat{\boldsymbol{\tau}}_{DIM}$')
    ax_dim[idx_tau].set_ylabel(r'$\hat{\boldsymbol{\tau}}$')
    ax_dim[idx_tau].set_xlabel('$Time~[s]$', size=15)
    fig_dim.align_ylabels()

    for ax_ in ax_dim[:-1]:
        ax_.set_xticks([])

    ax_dim[-1].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_dim[-1].set_xticklabels(['$0$', '$0.25$', '$0.5$', '$0.75$', '$1$'], size=13)

    # grids creation
    x_grids = list(np.arange(0,2,0.25))
    alpha_grids = 0.12
    y_grids_ft = [-15, -10, -5, 0, 5]
    y_grids_Kp = [0, 10, 20, 30, 40, 50]
    y_grids_pos = [i for i in list(np.arange(0, 0.81, 0.1))]
    y_grids_vel = [-2, -1, 0, 0.5]
    y_grids_tau = [-1, -0.5, 0, 0.5, 1]
    size_labels = 12
    for j, e in enumerate(ax_dim):
        [e.axvline(xg, color='k', alpha=alpha_grids) for xg in x_grids]
        if idx_ft == j:
            e.set_yticks([5, 0, -5, -10, -15, -20])
            e.set_yticklabels(['$5$', '$0$', "$-5$", '$-10$', '$-15$', '$-20$'], size=size_labels)
            [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_ft]
            # e.axhline(0, color='k')
        if idx_Kp == j:
            e.set_yticks([10, 20, 30, 40, 50])
            e.set_yticklabels(['$10$', '$20$', '$30$', '$40$', '$50$'], size=size_labels)
            [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_Kp]
        if idx_pos == j:
            aux = [0, 0.2, 0.4, 0.6, 0.8]
            e.set_yticks(aux)
            e.set_yticklabels(['$'+str(a)+'$' for a in aux], size=size_labels)
            [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_pos]
        if idx_vel == j:
            e.set_yticks([-2, -1, 0])
            e.set_yticklabels(['$-2$', '$-1$', '$0$'], size=size_labels)
            [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_vel]
        if idx_tau == j or idx_tau-1 == j:
            e.set_yticks(y_grids_tau)
            e.set_yticklabels(['$'+str(a)+'$' for a in y_grids_tau], size=size_labels)
            [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_tau]

    # for j in range(4):
    #     if j == 1:
    #         ax[j].legend(prop={'size': 10}, loc='lower right')
    #     if j == 0:
    leg = ax_dim[-1].legend(prop={'size': 8}, loc='lower right', bbox_to_anchor=(1.01, 0.75), ncol=4)
    leg.get_frame().set_alpha(None)
    leg.get_frame().set_facecolor((1.0, 1.0, 1, 1))
    # ax[-2].legend(prop={'size': 10}, loc='lower right')
    ax_dim[-3].legend(prop={'size': 10}, loc='lower right')
    ax_dim[-1].axvline(x = 0.293, ymin=0, ymax=7.25, linestyle='--', linewidth=1.1, color = 'r', label = '_nolegend_', clip_on=False)

    plt.show()
    fig_dim.savefig('images/1d_plots_dim.png', dpi=400, bbox_inches='tight')

def plot_rmm_2d_all_in_one():
    chosen_exps = [22, 23]
    n_exps = len(chosen_exps)
    gray_cycler = (cycler(color=["#000000", "#333333", "#666666", "#999999", "#cccccc"]) +
                    cycler(linestyle=["-", "--", "-.", ":", "-"]))
    plt.rc("axes", prop_cycle=gray_cycler)

    ylimits_ft = [-20, 5]
    ylimits_Kp = [5, 45]
    ylimits_pos_z = [0.25, 1.]
    ylimits_pos_y = [-1.75, 0]
    ylimits_vel_z = [-2.75, 2.75]
    ylimits_vel_y = [-0.8, 4]
    xlimits_2d = [0, 1.5]

    fig, ax  = plt.subplots(6, len(chosen_exps), figsize=(20,10))
    fig_yz, ax_yz  = plt.subplots(len(chosen_exps), 1)
    # fig_b, ax_b = plt.subplots(1,1)

    idx_pos_z = 0
    idx_vel_z = 1
    idx_pos_y = 2
    idx_vel_y = 3
    idx_ft = 4
    idx_Kp = 5
    idx_yz = 0
    i=0

    # legend_ft = [r'$F^{VM+K_H}_z$', r'$F^{VM+K_L}_z$', r'$F^{VM+SIC}_z$', r'$F^{VM+POC}_z$']
    # legend_ft = [r'$F_z$', r'$F_z$', r'$F_z$', r'$F_z$']
    legend_ft = [r'$F_z$' for _ in range(n_exps)]
    # legend_Kp = [r'$K_H$', r'$K_L$', r'$SIC$', r'$POC$']
    legend_Kp = [r'$K_H$' for _ in range(n_exps)]
    # legend_pos = [r'$z^{VM+K_H}$', r'$z^{VM+K_L}$', r'$z^{VM+SIC}$', r'$z^{VM+POC}$']
    # legend_pos = [r'$z$', r'$z$', r'$z$', r'$z$']
    legend_pos = [r'$z$' for _ in range(n_exps)]
    # legend_vel = [r'$\dot{z}^{VM+K_H}$', r'$\dot{z}^{VM+K_L}$', r'$\dot{z}^{VM+SIC}$', r'$\dot{z}^{VM+POC}$']
    legend_vel = [r'$\dot{z}$' for _ in range(n_exps)]

    for j, idx_exp in enumerate(chosen_exps):
        
        file_name = dh._get_name(idx_exp)
        print(file_name)
        params['idx_initial'] = dh.get_idx_from_file(idx_exp, data_info, idx_name='idx_start')-100
        params['idx_end'] = dh.get_idx_from_file(idx_exp, data_info, idx_name='idx_end')

        time = dh.get_data(file_name=file_name,params=params, data_to_plot='time')
        time -= time[0]

        ft_ = dh.get_data(file_name=file_name, params=params, data_to_plot='ft_')
        # ax[idx_ft][j].plot(time, ft_, colors[i], linewidth=lw, label=legend_ft[i])
        # ax[idx_ft][j].plot(time, ft_[:,2], linewidth=lw, label=legend_ft[i])
        ax[idx_ft][j].plot(time, -np.linalg.norm(ft_[:,:3], axis=1), linewidth=lw, label=legend_ft[i])
        ax[idx_ft][j].set_xlim(xlimits_2d)
        ax[idx_ft][j].set_ylim(ylimits_ft)
        # ax[idx_ft][j].legend(prop={'size': 8}, loc='lower right')
        
        Kp = dh.get_data(params, file_name=file_name, data_to_plot='Kp')
        # ax[j][idx_Kp].plot(time, Kp, colors[i], linewidth=lw, label=legend_Kp[i])
        # ax[j][idx_Kp].plot(time, Kp, colors[i], linewidth=lw)
        ax[idx_Kp][j].plot(time, Kp[:,2], linewidth=lw, label='$K_{p_z}$')
        ax[idx_Kp][j].plot(time, Kp[:,1], linewidth=lw, label='$K_{p_y}$')
        ax[idx_Kp][j].set_xlim(xlimits_2d)
        ax[idx_Kp][j].set_ylim(ylimits_Kp)

        pos = dh.get_data(params, file_name=file_name, data_to_plot='EE_position')
        pos_d = dh.get_data(params, file_name=file_name, data_to_plot='EE_position_d')
        # ax[j][idx_pos].plot(time, pos, colors[i], linewidth=lw, label=legend_pos[i])
        ax[idx_pos_z][j].plot(time, pos[:,2], linewidth=lw, label=legend_pos[i])
        ax[idx_pos_z][j].plot(time, pos_d[:,2], 'k--', linewidth=lw-0.5, label=legend_pos[i][:-1]+'_d$', alpha=0.6)
        ax[idx_pos_y][j].plot(time, pos[:,1], linewidth=lw, label=legend_pos[i])
        ax[idx_pos_y][j].plot(time, pos_d[:,1], 'k--', linewidth=lw-0.5, label=legend_pos[i][:-1]+'_d$', alpha=0.6)
        ax[idx_pos_z][j].set_xlim(xlimits_2d)
        ax[idx_pos_z][j].set_ylim(ylimits_pos_z)
        ax[idx_pos_y][j].set_xlim(xlimits_2d)
        ax[idx_pos_y][j].set_ylim(ylimits_pos_y)
        

        vel = dh.get_data(params, file_name=file_name, data_to_plot='EE_twist')
        vel_d = dh.get_data(params, file_name=file_name, data_to_plot='EE_twist_d')
        # ax[j][idx_vel].plot(time, vel, colors[i], linewidth=lw, label=legend_vel[i])
        ax[idx_vel_z][j].plot(time, vel[:,2], linewidth=lw, label=legend_vel[i])
        ax[idx_vel_z][j].plot(time, vel_d[:,2], 'k--', linewidth=lw-0.5, label=legend_vel[i][:-1]+'_d$', alpha=0.6)
        ax[idx_vel_y][j].plot(time, vel[:,1], linewidth=lw, label=legend_vel[i])
        ax[idx_vel_y][j].plot(time, vel_d[:,1], 'k--', linewidth=lw-0.5, label=legend_vel[i][:-1]+'_d$', alpha=0.6)
        ax[idx_vel_z][j].set_xlim(xlimits_2d)
        ax[idx_vel_z][j].set_ylim(ylimits_vel_z)
        ax[idx_vel_y][j].set_xlim(xlimits_2d)
        ax[idx_vel_y][j].set_ylim(ylimits_vel_y)

        pos_ball = dh.get_data(params, file_name=file_name, data_to_plot='ball_pose_')
        # apply KF
        z_actual_hat, z_dot_actual_hat, z_intersec, z_dot_intersec, \
            y_actual_hat, y_dot_actual_hat, y_intersec, y_dot_intersec,\
                time_intersec, time_f = kalman_filtering_2d(time, pos_ball[:,2], pos_ball[:,1], ft_)
        # ax[idx_pos][j].plot(time, pos_ball[:,2], 'r', linewidth=lw, label=legend_pos[i])
        ax[idx_pos_z][j].plot(time_f, z_actual_hat-(95/2/1000), color='#000000', linestyle='dashdot', linewidth=lw-0.5, label='$z_b$')
        ax[idx_vel_z][j].plot(time_f, z_dot_actual_hat, color='#000000', linestyle='dashdot', linewidth=lw-0.5, label='$\dot{z}_b$')
        ax[idx_pos_y][j].plot(time_f, y_actual_hat-(95/2/1000), color='#000000', linestyle='dashdot', linewidth=lw-0.5, label='$y_b$')
        ax[idx_vel_y][j].plot(time_f, y_dot_actual_hat, color='#000000', linestyle='dashdot', linewidth=lw-0.5, label='$\dot{y}_b$')

        
        ax_yz[j].plot(y_actual_hat-(95/2/1000), z_actual_hat-(95/2/1000), color='#000000', linestyle='dashdot', linewidth=lw-0.5, label='$yz_b$')
        ax_yz[j].plot(pos[:,1], pos[:,2], linewidth=lw, label=legend_pos[i])
        

    for j, ax_ in enumerate(ax):
        if j < 5:
            ax_[0].set_xticks([])
            ax_[1].set_xticks([])
    [ax_[1].set_yticks([]) for ax_ in ax]
    for j in range(len(chosen_exps)):
        aux = [0.25*i for i in range(7)]
        ax[5][j].set_xticks(aux)
        ax[5][j].set_xticklabels(['$'+str(a)+'$' for a in aux])
    # ax[n_exps-1, 2].axvline(x = 0.685, ymin=0, ymax=2.2, linestyle='-', linewidth=1.3, color = 'r', label = 'axvline - full height', clip_on=False)

    k = 0
    ax[k,0].set_ylabel('$z~[m]$'); k+=1
    ax[k,0].set_ylabel('$\dot{z}~[m/s]$'); k+=1
    ax[k,0].set_ylabel('$y~[m]$'); k+=1
    ax[k,0].set_ylabel('$\dot{y}~[m/s]$'); k+=1
    ax[k,0].set_ylabel('$||F||~[N]$'); k+=1
    ax[k,0].set_ylabel('$K_p$'); k+=1

    # grids creation
    x_grids = list(np.arange(0,2,0.25))
    n_divisions = 5
    alpha_grids = 0.12
    y_grids_ft = [-15, -10, -5, 0, 5]
    y_grids_Kp = [0, 10, 20, 30, 40, 50]
    
    y_grids_pos_z = [i for i in list(np.arange(0, 1.0, 0.2))]
    y_grids_vel_z = [-2, -1, 0, 1, 2.0, 3.0]
    
    y_grids_pos_y = [i for i in list(np.arange(-1.5, 0.1, 0.5))]
    y_grids_vel_y = [-2, -1, 0, 1, 2.0, 3.0]
    for j, row in enumerate(ax):
        for i, e in enumerate(row):
            [e.axvline(xg, color='k', alpha=alpha_grids) for xg in x_grids]
            if idx_ft == j:
                [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_ft]
                if i == 0:
                    aux = [5, 0, -5, -10, -15]
                    e.set_yticks(aux)
                    e.set_yticklabels(['$'+str(a)+'$' for a in aux], size=13)
            if idx_Kp == j:
                [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_Kp]
                if i == 0:
                    aux = [10, 20, 30, 40]
                    e.set_yticks(aux)
                    e.set_yticklabels(['$'+str(a)+'$' for a in aux], size=13)
            if idx_pos_z == j:
                [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_pos_z]
                if i == 0:
                    aux = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
                    e.set_yticks(aux)
                    e.set_yticklabels(['$'+str(a)+'$' for a in aux], size=13)
            if idx_vel_z == j:
                if i == 0:
                    aux = [-2, -1, 0, 1, 2]
                    e.set_yticks(aux)
                    e.set_yticklabels(['$'+str(a)+'$' for a in aux], size=13)
                [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_vel_z]
            if idx_pos_y == j:
                [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_pos_y]
                if i == 0:
                    aux = [0, -0.5, -1.0, -1.5]
                    e.set_yticks(aux)
                    e.set_yticklabels(['$'+str(a)+'$' for a in aux], size=13)
            if idx_vel_y == j:
                if i == 0:
                    aux = [0, 1, 2, 3]
                    e.set_yticks(aux)
                    e.set_yticklabels(['$'+str(a)+'$' for a in aux], size=13)
                [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_vel_y]

    fig.align_ylabels()

    ax_yz[0].grid()
    ax_yz[1].grid()
    ax_yz[0].set_ylabel('$z~[m]$')
    ax_yz[1].set_ylabel('$z~[m]$')
    ax_yz[1].set_xlabel('$y~[m]$')

    ax_yz[0].set_xlim([-1.75, -0.3])
    ax_yz[1].set_xlim([-1.75, -0.3])
    ax_yz[0].set_ylim([0.25, 1.0])
    ax_yz[1].set_ylim([0.25, 1.0])

    fig_yz.set_tight_layout(tight=True)

    fig.supxlabel('$Time~[s]$', size=20)

    # for j in range(4):
    #     if j == 1 or j == 2:
    #         ax[j, 3].legend(prop={'size': 10}, loc='lower right')
    #     if j == 0:
    #         ax[j, 3].legend(prop={'size': 10}, loc='upper right')

    ax_yz[0].invert_xaxis()
    ax_yz[1].invert_xaxis()

    plt.subplots_adjust(hspace=0.045, wspace=0.0075)
    fig.set_constrained_layout(constrained=True)
    fig_yz.set_constrained_layout(constrained=True)
    plt.show()
    # fig.savefig('images/2d_time_plots_rmm.png')
    # fig_yz.savefig('images/2d_spatial_rmm.png')

def plots_joints_max_torque():
    
    # fig, ax = plt.subplots(2, 2, figsize=(6, 6), subplot_kw=dict(polar=True))
    fig_1d_joints, ax_1d_joints = plt.subplots(2, 2, figsize=(6, 6), layout='constrained', subplot_kw=dict(polar=True))

    # chosen_exps_joints_ = [11, 13, 14, 18]
    chosen_exps_joints_ = chosen_exps_joints[:]

    n_joints = 7
    tau_max = np.zeros((n_joints, len(chosen_exps_joints_)))
    tau_energy = np.zeros((n_joints, len(chosen_exps_joints_)))
    tau_energy_max = np.zeros((n_joints, len(chosen_exps_joints_)))
    tau_rms = np.zeros((n_joints, len(chosen_exps_joints_)))
    tau_d_rms = np.zeros((n_joints, len(chosen_exps_joints_)))
    tau_adim = np.zeros((len(chosen_exps_joints_)))
    q_dot_limits = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])

    # legends = []
    legends = ['$VM-K_H$', '$VM-K_H-DIM$', '$VM-VIC$', '$VM-VIC-DIM$']

    for j, idx_exp in enumerate(chosen_exps_joints_):
        file_name = dh._get_name(idx_exp)
        print('\n', file_name)
        legends.append(file_name)
        params['idx_initial'] = dh.get_idx_from_file(idx_exp, data_info, idx_name='idx_start')+154
        params['idx_end'] = dh.get_idx_from_file(idx_exp, data_info, idx_name='idx_end')

        time = dh.get_data(file_name=file_name,params=params, data_to_plot='time')
        time -= time[0]

        idx_impact = np.where(time > 0.34)[0][0]
        idx_finish_analysis = np.where(time > 1)[0][0]

        tau_m = dh.get_data(file_name=file_name, params=params, data_to_plot='tau_measured')[idx_impact:idx_finish_analysis]
        tau_d = dh.get_data(file_name=file_name, params=params, data_to_plot='tau')[idx_impact:idx_finish_analysis]
        q_dot = dh.get_data(file_name=file_name, params=params, data_to_plot='dq')[idx_impact:idx_finish_analysis]

        # print(file_name, '\tF_max = ', np.max(np.abs(ft_)))
        tau_norm = np.zeros_like(tau_m)
        tau_norm = np.divide(tau_m, tau_limits)
        tau_max[:,j] = np.max(np.abs(tau_norm.T), axis=1)

        tau_d_norm = np.zeros_like(tau_d)
        tau_d_norm = np.divide(tau_d, tau_limits)

        q_dot_norm = np.divide(q_dot, q_dot_limits)

        # ENERGY
        previous_power = np.zeros(7)
        i = 0
        for tau, dq in zip(tau_m[idx_impact:], q_dot[idx_impact:]):
            # print(tau_energy[:,j])
            current_power = np.multiply(tau, dq)
            if i > 0:
                tau_energy[:, j] += (current_power + previous_power)*(time[idx_impact+i]-time[idx_impact+i-1])/2
                tau_energy_max[:, j] += (2*np.multiply(tau_limits, q_dot_limits))*(time[idx_impact+i]-time[idx_impact+i-1])/2
                if i > 200:
                    break

            # tau_energy_max[:, j] += np.multiply(tau_limits, q_dot_limits)
            previous_power = current_power
            i += 1
        print('TAU E = ', np.sum(np.abs(tau_energy[:, j]), axis=0))

        # print("TAU ENERGY = ", tau_energy)
        # TAU RMS
        tau_rms[:,j] = np.sqrt(np.mean(tau_norm**2, axis=0)/(time[idx_finish_analysis]-time[idx_impact]))
        tau_d_rms[:,j] = np.sqrt(np.mean(tau_d_norm**2, axis=0)/(time[idx_finish_analysis]-time[idx_impact]))

        # tau_rms[:,j] = np.divide(np.sqrt(np.mean(tau_m**2, axis=0)), tau_limits)
        print("TAU RMS SUM = ", np.sum(tau_rms[:,j]))
        print("TAU RMS NORM = ", np.divide(tau_rms[:,j],tau_d_rms[:,j]))


        # ADIM
        w_fd = dh.get_data(file_name=file_name, params=params, data_to_plot='m_arm')[idx_impact:idx_finish_analysis]
        T_ = time[idx_finish_analysis]-time[idx_impact]
        idx_step = 0
        time_aux = time[idx_impact:idx_finish_analysis]
        while time_aux[idx_step] <= T_:
            if idx_step > 1:
                tau_adim[j] += (w_fd[idx_step] + w_fd[idx_step-1])*(time_aux[idx_step] - time_aux[idx_step-1])/2
            idx_step += 1
            if idx_step+1 >= len(w_fd):
                break
        print("TAU ADIM = ", tau_adim[j]/(time[idx_finish_analysis]-time[idx_impact]))

    # Generating the X and Y axis data points

    # tau_energy = np.abs(np.divide(tau_energy, tau_energy_max))
    tau_energy = np.abs(tau_energy)
    print(np.sum(tau_energy, axis=0))
    

    # for j in range(len(chosen_exps)):
    #     tau_rms[:,j] = np.divide(tau_rms[:,j], tau_limits)

    # r=[8,8,8,8,8,8,8,8,8]
    theta = np.deg2rad(np.arange(0,360+360/7,360/7))

    # plotting the polar coordinates on the system
    # mycolors = list(mcolors.BASE_COLORS.keys())[:4]

    colors_i_want = [4, 0, -1, 1]
    mycolors = [myc.mpl_colors[i] for i in colors_i_want]# + cycler(linestyle=["-", "-", "--", "-", "--"])
    
    for j, c in enumerate(mycolors): #range(len(chosen_exps)):
        one_tau = np.concatenate([tau_max[:,j].reshape(n_joints,1), tau_max[0,j].reshape(1,1)],axis=0)
        one_energy = np.concatenate([tau_energy[:,j].reshape(n_joints,1), tau_energy[0,j].reshape(1,1)],axis=0)
        one_rms = np.concatenate([tau_rms[:,j].reshape(n_joints,1), tau_rms[0,j].reshape(1,1)],axis=0)
        one_rms_norm = np.concatenate([np.divide(tau_rms[:,j],tau_d_rms[:,j]).reshape(n_joints,1),
                                       np.divide(tau_rms[0,j],tau_d_rms[0,j]).reshape(1,1)],axis=0)
        # one_energy = np.abs(np.concatenate([tau_energy[:,j].reshape(n_joints,1), tau_energy[0,j].reshape(1,1)],axis=0))
        
        # ax.fill_between(theta, 0, one_tau[i], alpha = 0.3)
        fs = 8
        if j < 2:
            aux = ax_1d_joints[0][0].plot(theta,one_tau,marker='o',color=c)[0]
            x = aux.get_xdata()
            y = aux.get_ydata()
            ax_1d_joints[0][0].fill_betweenx(y, 0, x, alpha=0.15,color=c, label='_nolegend_')
            ax_1d_joints[0][0].set_yticks([0.25, 0.5, 0.75, 1.0])
            ax_1d_joints[0][0].set_yticklabels(['', '$0.5$', '', '$1$'], fontsize=fs)
            ax_1d_joints[0][0].set_xticks(theta)
            ax_1d_joints[0][0].set_xticklabels(['$\\tau_'+str(q+1)+'$' for q in [0, 1, 2, 3, 4, 5, 6, 0]])

            # aux = ax[1][0].plot(theta,one_energy,marker='o', label='_nolegend_')[0]
            aux = ax_1d_joints[1][0].plot(theta,one_rms,marker='o',color=c)[0]
            x = aux.get_xdata()
            y = aux.get_ydata()
            ax_1d_joints[1][0].fill_betweenx(y, 0, x, alpha=0.15,color=c, label='_nolegend_')
            ax_1d_joints[1][0].set_yticks([0.15, 0.3, 0.4])
            ax_1d_joints[1][0].set_yticklabels(['$0.15$', '$0.3$',''], fontsize=fs)
            ax_1d_joints[1][0].set_xticks(theta)
            ax_1d_joints[1][0].set_xticklabels(['$\\tau_'+str(q+1)+'$' for q in [0, 1, 2, 3, 4, 5, 6, 0]])

            # aux = ax[1][0].plot(theta,one_rms_norm,marker='o',color=c)[0]
            # x = aux.get_xdata()
            # y = aux.get_ydata()
            # ax[1][0].fill_betweenx(y, 0, x, alpha=0.15,color=c, label='_nolegend_')
            # ax[1][0].set_yticks([0.15, 0.3, 0.4])
            # ax[1][0].set_yticklabels(['$0.15$', '$0.3$',''], fontsize=fs)
            # ax[1][0].set_xticks(theta)
            # ax[1][0].set_xticklabels(['$\\tau_'+str(q+1)+'$' for q in [0, 1, 2, 3, 4, 5, 6, 0]])
        else:
            aux = ax_1d_joints[0][1].plot(theta,one_tau,marker='o',color=c)[0]
            x = aux.get_xdata()
            y = aux.get_ydata()
            ax_1d_joints[0][1].fill_betweenx(y, 0, x, alpha=0.15,color=c, label='_nolegend_')
            ax_1d_joints[0][1].set_yticks([0.25, 0.5, 0.75, 1.0])
            ax_1d_joints[0][1].set_yticklabels(['', '$0.5$', '', '$1$'], fontsize=fs)
            ax_1d_joints[0][1].set_xticks(theta)
            ax_1d_joints[0][1].set_xticklabels(['$\\tau_'+str(q+1)+'$' for q in [0, 1, 2, 3, 4, 5, 6, 0]])

            # aux = ax[1][1].plot(theta,one_energy,marker='o', label='_nolegend_')[0]
            aux = ax_1d_joints[1][1].plot(theta,one_rms,marker='o',color=c)[0]
            x = aux.get_xdata()
            y = aux.get_ydata()
            ax_1d_joints[1][1].fill_betweenx(y, 0, x, alpha=0.15,color=c, label='_nolegend_')
            ax_1d_joints[1][1].set_yticks([0.15, 0.3, 0.4])
            ax_1d_joints[1][1].set_yticklabels(['$0.15$', '$0.3$',''], fontsize=fs)
            ax_1d_joints[1][1].set_xticks(theta)
            ax_1d_joints[1][1].set_xticklabels(['$\\tau_'+str(q+1)+'$' for q in [0, 1, 2, 3, 4, 5, 6, 0]])

            # aux = ax[1][1].plot(theta,one_rms_norm,marker='o',color=c)[0]
            # x = aux.get_xdata()
            # y = aux.get_ydata()
            # ax[1][1].fill_betweenx(y, 0, x, alpha=0.15,color=c, label='_nolegend_')
            # ax[1][1].set_yticks([0.15, 0.3, 0.4])
            # ax[1][1].set_yticklabels(['$0.15$', '$0.3$',''], fontsize=fs)
            # ax[1][1].set_xticks(theta)
            # ax[1][1].set_xticklabels(['$\\tau_'+str(q+1)+'$' for q in [0, 1, 2, 3, 4, 5, 6, 0]])

    ax_1d_joints[1][0].legend(legends[:2], bbox_to_anchor=(.95, -.15), fontsize=10)
    ax_1d_joints[1][1].legend(legends[2:], bbox_to_anchor=(1, -.15), fontsize=10)

    # for ax_, leg in zip(ax[1], [legends[:2], legends[2:]]):
    #     ax_.legend(leg, bbox_to_anchor=(1, -.15), fontsize=10)
    
    # for ax_ in ax:
    #     for ax__ in ax_:
    #         ax__.grid()

    ax_1d_joints[0][0].set_ylabel('$\\hat{\\boldsymbol{\\tau}}_{max}$')
    ax_1d_joints[0][0].yaxis.set_label_coords(-0.125, 0.5)
    ax_1d_joints[1][0].set_ylabel('$\\hat{\\boldsymbol{\\tau}}_{RMS}$')
    # ax[1][0].set_ylabel('$\\hat{\\tau}_{RMS}/\\hat{\\tau}^{d}_{RMS}$')
    ax_1d_joints[1][0].yaxis.set_label_coords(-0.125, 0.5)

    # Setting the axis limit
    # for ax_ in ax:
    #     for ax__ in ax_:
    #         ax__.set_ylim(0,1)
    ax_1d_joints[0][0].set_ylim(0,1.15)
    ax_1d_joints[0][1].set_ylim(0,1.15)

    # fig.legend(legends[:2], loc='upper center',bbox_to_anchor=(.5,-0.1), bbox_transform=fig.transFigure)

    plt.show()
    fig_1d_joints.savefig('images/1d-dim-joint-polar-plots.png', pad_inches=0, dpi=400)

def plots_joints_max_torque_without_KH():
    
    # fig, ax = plt.subplots(2, 2, figsize=(6, 6), subplot_kw=dict(polar=True))
    fig_1d_joints, ax_1d_joints = plt.subplots(1, 2, figsize=(8, 6), subplot_kw=dict(polar=True))

    # chosen_exps = [11, 13, 15, 17]
    # chosen_exps_joints = [14, 18]
    chosen_exps_joints_ = chosen_exps_joints[2:]

    n_joints = 7
    tau_max = np.zeros((n_joints, len(chosen_exps_joints_)))
    tau_energy = np.zeros((n_joints, len(chosen_exps_joints_)))
    tau_energy_max = np.zeros((n_joints, len(chosen_exps_joints_)))
    tau_rms = np.zeros((n_joints, len(chosen_exps_joints_)))
    tau_d_rms = np.zeros((n_joints, len(chosen_exps_joints_)))
    tau_adim = np.zeros((len(chosen_exps_joints_)))
    q_dot_limits = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])

    # legends = []
    legends = ['$VM-VIC$', '$VM-VIC-DIM$']

    for j, idx_exp in enumerate(chosen_exps_joints_):
        file_name = dh._get_name(idx_exp)
        print('\n', file_name)
        legends.append(file_name)
        params['idx_initial'] = dh.get_idx_from_file(idx_exp, data_info, idx_name='idx_start')+154
        params['idx_end'] = dh.get_idx_from_file(idx_exp, data_info, idx_name='idx_end')

        time = dh.get_data(file_name=file_name,params=params, data_to_plot='time')
        time -= time[0]

        idx_impact = np.where(time > 0.294)[0][0]
        idx_finish_analysis = np.where(time > 1)[0][0]
        # idx_finish_analysis = idx_impact + 600

        tau_m = dh.get_data(file_name=file_name, params=params, data_to_plot='tau_measured')[idx_impact:idx_finish_analysis]
        tau_d = dh.get_data(file_name=file_name, params=params, data_to_plot='tau')[idx_impact:idx_finish_analysis]
        q_dot = dh.get_data(file_name=file_name, params=params, data_to_plot='dq')[idx_impact:idx_finish_analysis]

        # print(file_name, '\tF_max = ', np.max(np.abs(ft_)))
        tau_norm = np.zeros_like(tau_m)
        # tau_norm = np.divide(tau_m, tau_limits)
        tau_norm = tau_m
        tau_max[:,j] = np.max(np.abs(tau_norm.T), axis=1)
        # TAU MAX
        # print("TAU MAX = ", np.max(np.abs(tau_max)))
        print("TAU MAX = ", np.max(np.abs(tau_m)))

        tau_d_norm = np.zeros_like(tau_d)
        tau_d_norm = np.divide(tau_d, tau_limits)

        q_dot_norm = np.divide(q_dot, q_dot_limits)

        # ENERGY
        previous_power = np.zeros(7)
        i = 0
        for tau, dq in zip(tau_m[idx_impact:idx_finish_analysis], q_dot[idx_impact:idx_finish_analysis]):
            # print(tau_energy[:,j])
            current_power = np.multiply(tau, dq)
            if i > 0:
                tau_energy[:, j] += (current_power + previous_power)*(time[idx_impact+i]-time[idx_impact+i-1])/2
                tau_energy_max[:, j] += (2*np.multiply(tau_limits, q_dot_limits))*(time[idx_impact+i]-time[idx_impact+i-1])/2
                if i > 200:
                    break

            # tau_energy_max[:, j] += np.multiply(tau_limits, q_dot_limits)
            previous_power = current_power
            i += 1
        print('TAU E = ', np.sum(np.abs(tau_energy[:, j]), axis=0))

        
        # TAU RMS, 
        tau_rms[:,j] = np.sqrt(np.mean(tau_norm**2, axis=0)/(time[idx_finish_analysis]-time[idx_impact]))
        tau_d_rms[:,j] = np.sqrt(np.mean(tau_d_norm**2, axis=0)/(time[idx_finish_analysis]-time[idx_impact]))

        # tau_rms[:,j] = np.divide(np.sqrt(np.mean(tau_m**2, axis=0)), tau_limits)
        print("TAU RMS SUM = ", np.sum(tau_rms[:,j]))
        # print("TAU RMS NORM = ", np.divide(tau_rms[:,j],tau_d_rms[:,j]))


        # ADIM
        w_fd = dh.get_data(file_name=file_name, params=params, data_to_plot='m_arm')
        T_ = time[idx_finish_analysis]-time[idx_impact]
        idx_step = 0
        time_aux = time[0:idx_finish_analysis]
        w_fd = w_fd[0:idx_finish_analysis]
        while time_aux[idx_step] <= T_:
            if idx_step > 1:
                tau_adim[j] += (w_fd[idx_step] + w_fd[idx_step-1])*(time_aux[idx_step] - time_aux[idx_step-1])/2
            idx_step += 1
            if idx_step+1 >= len(w_fd):
                break
        print("TAU ADIM = ", tau_adim[j]/T_)


        
        ft_ = dh.get_data(file_name=file_name, params=params, data_to_plot='ft_')[:,2]
        Fmax = -np.min(ft_)

        loi = LOI(time, ft_[:], idx_impact, idx_finish_analysis, file_name=file_name)
        dri = DRI(time, ft_[:], file_name)
        bti = BTI(time, ft_[:], file_name=file_name)

        pos_ball = dh.get_data(params, file_name=file_name, data_to_plot='ball_pose_')[:,2]
        vel = dh.get_data(params, file_name=file_name, data_to_plot='EE_twist')[:,2]
        z_actual_hat, z_dot_actual_hat, z_intersec, z_dot_intersec, time_intersec, time_f = kalman_filtering_1d(time, pos_ball, ft_)
        # vm_error = 0
        vm_error = vel[idx_impact] - z_dot_actual_hat[-1]

        print(file_name,  '\tLOI =',        loi,
                          '\tDRI = ',       dri,
                          '\tBTI = ',       bti,
                          '\tF_max = ',     Fmax,
                          '\tVM-error = ',  vm_error)

    # Generating the X and Y axis data points

    # tau_energy = np.abs(np.divide(tau_energy, tau_energy_max))
    tau_energy = np.abs(tau_energy)
    print(np.sum(tau_energy, axis=0))
    

    # for j in range(len(chosen_exps)):
    #     tau_rms[:,j] = np.divide(tau_rms[:,j], tau_limits)

    # r=[8,8,8,8,8,8,8,8,8]
    theta = np.deg2rad(np.arange(0,360+360/7,360/7))

    # plotting the polar coordinates on the system
    # mycolors = list(mcolors.BASE_COLORS.keys())[:4]

    colors_i_want = [4, 0, -1, 1]
    mycolors = [myc.mpl_colors[i] for i in colors_i_want]# + cycler(linestyle=["-", "-", "--", "-", "--"])
    
    for j, c in enumerate(mycolors[2:]): #range(len(chosen_exps)):
        one_tau_max = np.concatenate([tau_max[:,j].reshape(n_joints,1), tau_max[0,j].reshape(1,1)],axis=0)
        one_energy = np.concatenate([tau_energy[:,j].reshape(n_joints,1), tau_energy[0,j].reshape(1,1)],axis=0)
        one_rms = np.concatenate([tau_rms[:,j].reshape(n_joints,1), tau_rms[0,j].reshape(1,1)],axis=0)
        one_rms_norm = np.concatenate([np.divide(tau_rms[:,j],tau_d_rms[:,j]).reshape(n_joints,1),
                                       np.divide(tau_rms[0,j],tau_d_rms[0,j]).reshape(1,1)],axis=0)
        # one_energy = np.abs(np.concatenate([tau_energy[:,j].reshape(n_joints,1), tau_energy[0,j].reshape(1,1)],axis=0))
        # ax.fill_between(theta, 0, one_tau[i], alpha = 0.3)
        fs = 12
        if j == 0: # tau_max
            aux = ax_1d_joints[0].plot(theta,one_tau_max,marker='o',color=c)[0]
            x = aux.get_xdata()
            y = aux.get_ydata()
            ax_1d_joints[0].fill_betweenx(y, 0, x, alpha=0.15,color=c, label='_nolegend_')
            # ax_1d_joints[0].set_yticks([0.25, 0.5, 0.75, 1.0])
            ax_1d_joints[0].set_yticks([0, 5, 10, 15, 20, 25, 30])
            # ax_1d_joints[0].set_yticklabels(['', '$0.5$', '', '$1$'], fontsize=fs)
            ax_1d_joints[0].set_yticklabels(['', '', '$10$', '', '$20$', '', '$30$'], fontsize=fs)
            ax_1d_joints[0].set_xticks(theta)
            ax_1d_joints[0].set_xticklabels(['$\\tau_'+str(q+1)+'$' for q in [0, 1, 2, 3, 4, 5, 6, 0]])

            # aux = ax[1][0].plot(theta,one_energy,marker='o', label='_nolegend_')[0]
            aux = ax_1d_joints[1].plot(theta,one_rms,marker='o',color=c)[0]
            x = aux.get_xdata()
            y = aux.get_ydata()
            ax_1d_joints[1].fill_betweenx(y, 0, x, alpha=0.15,color=c, label='_nolegend_')
            ax_1d_joints[1].set_yticks([0.15, 0.3, 0.4])
            ax_1d_joints[1].set_yticklabels(['$0.15$', '$0.3$',''], fontsize=fs)
            ax_1d_joints[1].set_xticks(theta)
            ax_1d_joints[1].set_xticklabels(['$\\tau_'+str(q+1)+'$' for q in [0, 1, 2, 3, 4, 5, 6, 0]])

            # aux = ax[1][0].plot(theta,one_rms_norm,marker='o',color=c)[0]
            # x = aux.get_xdata()
            # y = aux.get_ydata()
            # ax[1][0].fill_betweenx(y, 0, x, alpha=0.15,color=c, label='_nolegend_')
            # ax[1][0].set_yticks([0.15, 0.3, 0.4])
            # ax[1][0].set_yticklabels(['$0.15$', '$0.3$',''], fontsize=fs)
            # ax[1][0].set_xticks(theta)
            # ax[1][0].set_xticklabels(['$\\tau_'+str(q+1)+'$' for q in [0, 1, 2, 3, 4, 5, 6, 0]])
        else: # tau rms
            aux = ax_1d_joints[0].plot(theta,one_tau_max,marker='o',color=c)[0]
            x = aux.get_xdata()
            y = aux.get_ydata()
            ax_1d_joints[0].fill_betweenx(y, 0, x, alpha=0.15,color=c, label='_nolegend_')
            # ax_1d_joints[0].set_yticks([0.25, 0.5, 0.75, 1.0])
            # ax_1d_joints[0].set_yticks([0, 10, 15, 20])
            # ax_1d_joints[0].set_yticklabels(['', '$10$', '', '$20$'], fontsize=fs)
            ax_1d_joints[0].set_xticks(theta)
            ax_1d_joints[0].set_xticklabels(['$\\tau_'+str(q+1)+'$' for q in [0, 1, 2, 3, 4, 5, 6, 0]])

            # aux = ax[1][1].plot(theta,one_energy,marker='o', label='_nolegend_')[0]
            aux = ax_1d_joints[1].plot(theta,one_rms,marker='o',color=c)[0]
            x = aux.get_xdata()
            y = aux.get_ydata()
            ax_1d_joints[1].fill_betweenx(y, 0, x, alpha=0.15,color=c, label='_nolegend_')
            # ax_1d_joints[1].set_yticks([0.15, 0.3, 0.4])
            # ax_1d_joints[1].set_yticklabels(['$0.15$', '$0.3$',''], fontsize=fs)
            # ax_1d_joints[1].set_yticks([0, 5, 10, 15, 20])
            # ax_1d_joints[1].set_yticklabels(['', '', '$10$', '', '$20$'], fontsize=fs)
            # ax_1d_joints[0].set_yticks([0.25, 0.5, 0.75, 1.0])
            ax_1d_joints[1].set_yticks([0, 5, 10, 15, 20, 25, 30])
            # ax_1d_joints[0].set_yticklabels(['', '$0.5$', '', '$1$'], fontsize=fs)
            ax_1d_joints[1].set_yticklabels(['', '', '$10$', '', '$20$', '', '$30$'], fontsize=fs)
            ax_1d_joints[1].set_xticks(theta)
            ax_1d_joints[1].set_xticklabels(['$\\tau_'+str(q+1)+'$' for q in [0, 1, 2, 3, 4, 5, 6, 0]])

            # aux = ax[1][1].plot(theta,one_rms_norm,marker='o',color=c)[0]
            # x = aux.get_xdata()
            # y = aux.get_ydata()
            # ax[1][1].fill_betweenx(y, 0, x, alpha=0.15,color=c, label='_nolegend_')
            # ax[1][1].set_yticks([0.15, 0.3, 0.4])
            # ax[1][1].set_yticklabels(['$0.15$', '$0.3$',''], fontsize=fs)
            # ax[1][1].set_xticks(theta)
            # ax[1][1].set_xticklabels(['$\\tau_'+str(q+1)+'$' for q in [0, 1, 2, 3, 4, 5, 6, 0]])

    ax_1d_joints[0].legend(legends, bbox_to_anchor=(1.45, -.15), fontsize=10)
    # ax[1][1].legend(legends[2:], bbox_to_anchor=(1, -.15), fontsize=10)

    # for ax_, leg in zip(ax[1], [legends[:2], legends[2:]]):
    #     ax_.legend(leg, bbox_to_anchor=(1, -.15), fontsize=10)
    
    # for ax_ in ax:
    #     for ax__ in ax_:
    #         ax__.grid()

    ax_1d_joints[0].set_title('${\\boldsymbol{\\tau}}_{max}$')
    ax_1d_joints[0].yaxis.set_label_coords(-0.125, 0.5)
    ax_1d_joints[1].set_title('${\\boldsymbol{\\tau}}_{RMS}$')
    # ax[1][0].set_ylabel('$\\hat{\\tau}_{RMS}/\\hat{\\tau}^{d}_{RMS}$')
    ax_1d_joints[1].yaxis.set_label_coords(-0.125, 0.5)

    # Setting the axis limit
    # for ax_ in ax:
    #     for ax__ in ax_:
    #         ax__.set_ylim(0,1)
    ax_1d_joints[0].set_ylim(0,30)
    ax_1d_joints[1].set_ylim(0,30)

    ax_1d_joints[0].set_rlabel_position(5)
    ax_1d_joints[1].set_rlabel_position(5)
    # ax[1].set_ylim(0,1.15)

    # fig.legend(legends[:2], loc='upper center',bbox_to_anchor=(.5,-0.1), bbox_transform=fig.transFigure)

    plt.show()
    fig_1d_joints.savefig('images/1d-dim-joint-polar-plots.png', pad_inches=0, dpi=400, bbox_inches='tight')

def polar_plots_metrics():
    
    fig, ax = plt.subplots(1, figsize=(6, 6), subplot_kw=dict(polar=True))

    chosen_exps = [14, 17]

    n_joints = 7
    tau_max = np.zeros((n_joints, len(chosen_exps)))
    tau_max_idx = np.zeros((len(chosen_exps)))
    tau_rms = np.zeros((n_joints, len(chosen_exps)))
    tau_adim = np.zeros(len(chosen_exps))
    tau_energy = np.zeros((len(chosen_exps)))
    tau_limits = np.array([87, 87, 87, 87, 12, 12, 12])

    xlim = [0, 1]

    legends = []

    for j, idx_exp in enumerate(chosen_exps):
        file_name = dh._get_name(idx_exp)
        print('\n', file_name)
        legends.append(file_name)
        offset_final = 190 if 'fp' in file_name else 0
        params['idx_initial'] = dh.get_idx_from_file(idx_exp, data_info, idx_name='idx_start')+154
        params['idx_end'] = dh.get_idx_from_file(idx_exp, data_info, idx_name='idx_end')-offset_final

        time = dh.get_data(file_name=file_name,params=params, data_to_plot='time')
        time -= time[0]

        idx_impact = np.where(time > 0.34)[0][0]
        idx_finish_analysis = np.where(time > xlim[-1])[0][0]

        tau_m = dh.get_data(file_name=file_name, params=params, data_to_plot='tau_measured')[idx_impact:idx_finish_analysis]
        q_dot = dh.get_data(file_name=file_name, params=params, data_to_plot='dq')[idx_impact:idx_finish_analysis]

        # print(file_name, '\tF_max = ', np.max(np.abs(ft_)))
        # TAU MAX
        tau_norm = np.zeros_like(tau_m)
        tau_norm = np.divide(tau_m, tau_limits)
        tau_max[:,j] = np.max(np.abs(tau_norm.T), axis=1)
        tau_max_idx[j] = np.argmax(tau_max[:,j])+1
        print("TAU MAX FULL = ", tau_max[:,j], "\tTAU MAX MAX = ", np.max(tau_max[:,j]))
        print("TAU MAX IDX = ", tau_max_idx[j])

        # # TAU_RMS
        tau_rms[:,j] = np.divide(np.sqrt(np.mean(tau_m**2, axis=0)), tau_limits)
        print("TAU RMS = ", tau_rms[:,j], "\tTAU RMS MAX = ", np.max(tau_rms[:,j]))

        # ADIM
        w_fd = dh.get_data(file_name=file_name, params=params, data_to_plot='m_arm')[idx_impact:idx_finish_analysis]
        T_ = xlim[-1]
        idx_step = 0
        time_aux = time[idx_impact:idx_finish_analysis]
        while time_aux[idx_step] <= T_:
            if idx_step > 1:
                tau_adim[j] += (w_fd[idx_step] + w_fd[idx_step-1])*(time_aux[idx_step] - time_aux[idx_step-1])/2
            tau_energy[j] += tau_m[idx_step].T.dot(q_dot[idx_step])
            idx_step += 1
            if idx_step+1 >= len(w_fd):
                break
        print('last time = ', time_aux[idx_step])
        tau_adim[j] = tau_adim[j]/T_
        print("ADIM = ", tau_adim[j])

        # ENERGY
        # print(q_dot.shape)
        # print(tau_m.shape)
        # for tau, dq in zip(tau_m, q_dot):
        #     tau_energy[j] += tau.T.dot(dq)
        # print("TAU ENERGY = ", tau_energy[j])

    # Generating the X and Y axis data points
    tau_energy = tau_energy/np.min(tau_energy)
    # r=[8,8,8,8,8,8,8,8,8]
    theta = np.deg2rad(np.arange(0,360+360/4,360/4))

    metrics_labels = ['$ADIM$', '$\\hat{\\tau}_{max}$', '$\\hat{\\tau}_{RMS}$', '$E$', '$ADIM$']
    print(tau_energy)
    # plotting the polar coordinates on the system
    for j in range(len(chosen_exps)):
        # one_tau = np.concatenate([tau_max[:,j].reshape(n_joints,1), tau_max[0,j].reshape(1,1)],axis=0)
        metrics = np.array([tau_adim[j], np.max(tau_max[5, j]), np.sum(tau_rms[j]), tau_energy[j], tau_adim[j]])

        print("TAU RMS SUM = ", np.sum(tau_rms[j]))

        aux = ax.plot(theta,metrics,marker='o', label='_nolegend_')[0]
        x = aux.get_xdata()
        y = aux.get_ydata()
        ax.fill_betweenx(y, 0, x, alpha=0.15)
        ax.set_yticklabels([])
        ax.set_xticks(theta)
        ax.set_xticklabels(metrics_labels)
        
        # if j < 2:
        #     aux = ax[0].plot(theta,one_tau,marker='o', label='_nolegend_')[0]
        #     x = aux.get_xdata()
        #     y = aux.get_ydata()
        #     ax[0].fill_betweenx(y, 0, x, alpha=0.15)
        #     ax[0].set_yticklabels([])
        #     ax[0].set_xticks(theta)
        #     ax[0].set_xticklabels(['$\\tau_'+str(q+1)+'$' for q in [0, 1, 2, 3, 4, 5, 6, 0]])
        # else:
        #     aux = ax[1].plot(theta,one_tau,marker='o', label='_nolegend_')[0]
        #     x = aux.get_xdata()
        #     y = aux.get_ydata()
        #     ax[1].fill_betweenx(y, 0, x, alpha=0.15)
        #     ax[1].set_yticklabels([])
        #     ax[1].set_xticks(theta)
        #     ax[1].set_xticklabels(['$\\tau_'+str(q+1)+'$' for q in [0, 1, 2, 3, 4, 5, 6, 0]])

    
    ax.legend(legends, bbox_to_anchor=(0.5, 1), fontsize=10)

    # Setting the axis limit
    ax.set_ylim(0,1.65)

    # Displaying the plot
    plt.show()

def plot_joint_torques_vanilla():
    chosen_exps = [11, 14, 13, 17]    
    fig, ax = plt.subplots(len(chosen_exps))
    n_joints = 7
    tau_max = np.zeros((n_joints, len(chosen_exps)))
    tau_limits = np.array([87, 87, 87, 87, 12, 12, 12])

    legends = []

    for j, idx_exp in enumerate(chosen_exps):
        file_name = dh._get_name(idx_exp)
        legends.append(file_name)
        offset_final = 190 if 'fp' in file_name else 0
        params['idx_initial'] = dh.get_idx_from_file(idx_exp, data_info, idx_name='idx_start')+154
        params['idx_end'] = dh.get_idx_from_file(idx_exp, data_info, idx_name='idx_end')-offset_final

        time = dh.get_data(file_name=file_name,params=params, data_to_plot='time')
        time -= time[0]

        tau_m = dh.get_data(file_name=file_name, params=params, data_to_plot='tau_measured')
        # print(file_name, '\tF_max = ', np.max(np.abs(ft_)))
        tau_norm = np.zeros_like(tau_m)
        for i in range(len(tau_m)):
            tau_norm[i] = np.divide(tau_m[i], tau_limits)
        print(tau_m.shape)
        # tau_max[:,j] = np.max(np.abs(tau_norm.T), axis=1)

        ax[j].set_ylim([-30, 30])
        # ax[j].set_ylim([-1, 1])
        ax[j].set_xlim([0, 1.5])
        ax[j].plot(time, tau_m)
        
    # Generating the X and Y axis data points
    plt.show()

def plot_1d_increased_height():
    # colors = ['b', 'r', 'g', 'y']
    colors = ['k']
    chosen_exps = [24]
    i = 0
    ylimits_ft = [-20, 5]
    ylimits_Kp = [10, 50]
    ylimits_pos = [0, 0.8]
    ylimits_vel = [-2.75, 1]

    fig, ax  = plt.subplots(5, len(chosen_exps), figsize=(6, 6),# layout="constrained",
                             gridspec_kw={'wspace': 0.0, 'hspace': 0.25})
    # fig_b, ax_b = plt.subplots(1,1)

    idx_pos = 0
    idx_vel = 1
    idx_ft = 2
    idx_Kp = 3
    idx_tau = 4

    # legend_ft = [r'$F^{VM+K_H}_z$', r'$F^{VM+K_L}_z$', r'$F^{VM+SIC}_z$', r'$F^{VM+POC}_z$']
    legend_ft = [r'$F_z$', r'$F_z$', r'$F_z$', r'$F_z$']
    legend_Kp = [r'$K_H$', r'$K_L$', r'$SIC$', r'$VIC$']
    # legend_pos = [r'$z^{VM+K_H}$', r'$z^{VM+K_L}$', r'$z^{VM+SIC}$', r'$z^{VM+POC}$']
    legend_pos = [r'$z$', r'$z$', r'$z$', r'$z$']
    # legend_vel = [r'$\dot{z}^{VM+K_H}$', r'$\dot{z}^{VM+K_L}$', r'$\dot{z}^{VM+SIC}$', r'$\dot{z}^{VM+POC}$']
    legend_vel = [r'$\dot{z}$', r'$\dot{z}$', r'$\dot{z}$', r'$\dot{z}$']

    for j, idx_exp in enumerate(chosen_exps):
        file_name = dh._get_name(idx_exp)
        print(file_name)
        params['idx_initial'] = dh.get_idx_from_file(idx_exp, data_info, idx_name='idx_start')+200
        params['idx_end'] = dh.get_idx_from_file(idx_exp, data_info, idx_name='idx_end')

        time = dh.get_data(file_name=file_name,params=params, data_to_plot='time')
        time -= time[0]

        ft_ = dh.get_data(file_name=file_name, params=params, data_to_plot='ft_')[:,2]
        ax[idx_ft].plot(time, ft_, colors[i], linewidth=lw, label=legend_ft[i])
        ax[idx_ft].set_xlim(xlimits)
        ax[idx_ft].set_ylim(ylimits_ft)
        

        Kp = dh.get_data(params, file_name=file_name, data_to_plot='Kp')[:,2]
        ax[idx_Kp].plot(time, Kp, colors[i], linewidth=lw)
        ax[idx_Kp].set_xlim(xlimits)
        ax[idx_Kp].set_ylim(ylimits_Kp)

        pos = dh.get_data(params, file_name=file_name, data_to_plot='EE_position')[:,2]
        pos_d = dh.get_data(params, file_name=file_name, data_to_plot='EE_position_d')[:,2]
        ax[idx_pos].plot(time, pos, colors[i], linewidth=lw, label=legend_pos[i])
        ax[idx_pos].plot(time, pos_d, 'k--', linewidth=lw-0.5, label=legend_pos[i][:-1]+'_d$', alpha=0.6)
        # params['idx_initial'] += 100
        # params['idx_end'] += 100
        pos_ball = dh.get_data(params, file_name=file_name, data_to_plot='ball_pose_')[:,2]
        ax[idx_pos].set_xlim(xlimits)
        ax[idx_pos].set_ylim(ylimits_pos)


        vel = dh.get_data(params, file_name=file_name, data_to_plot='EE_twist')[:,2]
        vel_d = dh.get_data(params, file_name=file_name, data_to_plot='EE_twist_d')[:,2]
        ax[idx_vel].plot(time, vel, colors[i], linewidth=lw, label=legend_vel[i])
        ax[idx_vel].plot(time, vel_d, 'k--', linewidth=lw-0.5, label=legend_vel[i][:-1]+'_d$', alpha=0.6)
        ax[idx_vel].set_xlim(xlimits)
        ax[idx_vel].set_ylim(ylimits_vel)

        # apply KF        
        z_actual_hat, z_dot_actual_hat, z_intersec, z_dot_intersec, time_intersec, time_f = kalman_filtering_1d(time, pos_ball, ft_, increased_height=True)
        # ax[idx_pos].plot(time_f, z_actual_hat-(95/2/1000), color='b', linestyle='dashdot', linewidth=lw-0.5, label='$z_b$')
        ax[idx_pos].plot(time_f[:-2], z_actual_hat[:-2], color='b', linestyle='dashdot', linewidth=lw-0.5, label='$z_b$')
        ax[idx_vel].plot(time_f[:-2], z_dot_actual_hat[:-2], color='b', linestyle='dashdot', linewidth=lw-0.5, label='$\dot{z}_b$')

        i += 1

        tau_m = dh.get_data(params, file_name=file_name, data_to_plot='tau_measured')
        tau_m_norm = np.divide(tau_m, tau_limits)
        ax[idx_tau].plot(time, tau_m_norm, label=['$\\tau_'+str(i+1)+'$' for i in range(7)])
        ax[idx_tau].set_xlim(xlimits)
        ax[idx_tau].set_ylim([-1, 1])


        idx_impact = np.where(time > 0.294)[0][0]
        idx_finish_analysis = np.where(time > 1)[0][0]
        Fmax = -np.min(ft_)

        loi = LOI(time, ft_, idx_impact, idx_finish_analysis, file_name)
        dri = DRI(time, ft_, file_name)
        bti = BTI(time, ft_, file_name)

        vm_error = vel[idx_impact] - z_dot_actual_hat[-1]

        print(file_name,  '\tLOI =',        loi,
                          '\tDRI = ',       dri,
                          '\tBTI = ',       bti,
                          '\tF_max = ',     Fmax,
                          '\tVM-error = ',  vm_error)

    # fig.set_constrained_layout(constrained=True)
    ax[idx_pos].add_patch(Ellipse((time_f[-3], z_actual_hat[-3]), 2*ball_radius/5+0.01, 2*ball_radius, color='b', fill=False))
    ax[idx_pos].set_ylabel('$z~[m]$')
    ax[idx_vel].set_ylabel('$\dot{z}~[m]$')
    ax[idx_Kp].set_ylabel('$K_p$')
    ax[idx_ft].set_ylabel('$F_z~[N]$')
    ax[idx_tau].set_xlabel('$Time~[s]$', size=15)
    ax[idx_tau].set_ylabel(r'$\hat{\boldsymbol{\tau}}$')
    fig.align_ylabels()

    for ax_ in ax[:-1]:
        ax_.set_xticks([])

    ax[-1].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax[-1].set_xticklabels(['$0$', '$0.25$', '$0.5$', '$0.75$', '$1$'], size=13)

    # grids creation
    x_grids = list(np.arange(0,2,0.25))
    alpha_grids = 0.12
    y_grids_ft = [-15, -10, -5, 0, 5]
    y_grids_Kp = [0, 10, 20, 30, 40, 50]
    y_grids_pos = [i for i in list(np.arange(0, 0.81, 0.1))]
    y_grids_vel = [-2, -1, 0, 0.5]
    y_grids_tau = [-1, -0.5, 0, 0.5, 1]
    size_labels = 12
    for j, e in enumerate(ax):
        [e.axvline(xg, color='k', alpha=alpha_grids) for xg in x_grids]
        if idx_ft == j:
            e.set_yticks([5, 0, -5, -10, -15, -20])
            e.set_yticklabels(['$5$', '$0$', "$-5$", '$-10$', '$-15$', '$-20$'], size=13)
            [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_ft]
            # e.axhline(0, color='k')
        if idx_Kp == j:
            e.set_yticks([10, 20, 30, 40, 50])
            e.set_yticklabels(['$10$', '$20$', '$30$', '$40$', '$50$'], size=13)
            [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_Kp]
        if idx_pos == j:
            aux = [0.0, 0.2, 0.4, 0.6, 0.8]
            e.set_yticks(aux)
            e.set_yticklabels(['$'+str(a)+'$' for a in aux], size=13)
            [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_pos]
        if idx_vel == j:
            e.set_yticks([-2, -1, 0])
            e.set_yticklabels(['$-2$', '$-1$', '$0$'], size=13)
            [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_vel]
        if idx_tau == j:
            e.set_yticks(y_grids_tau)
            e.set_yticklabels(['$'+str(a)+'$' for a in y_grids_tau], size=size_labels)
            [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_tau]

    for j in range(5):
        if j == 0:
            ax[j].legend(prop={'size': 9}, loc='upper right', ncol=2)
        if j == 1:
            ax[j].legend(prop={'size': 9}, loc='lower right', ncol=2)
        if j == idx_tau:
            leg = ax[j].legend(prop={'size': 8}, loc='lower right', ncol=4, bbox_to_anchor=(1.01, 0.65))
            leg.get_frame().set_alpha(None)
            leg.get_frame().set_facecolor((1.0, 1.0, 1, 1))
    
    ax[-1].axvline(x = 0.294, ymin=0, ymax=6, linestyle='--', linewidth=1.1, color = 'r', label = '_nolegend_', clip_on=False)
    # ax[0].axvline(x = 0.294, ymin=0, ymax=6, linestyle='--', linewidth=1.1, color = 'r', label = '_nolegend_')
    # ax[0].axhline(0.35, linestyle='--', linewidth=1.1, color = 'r', label = '_nolegend_')
    print('time delay catching = ', 0.341-0.294)

    plt.show()
    fig.savefig('images/1d_increased_height.png', dpi=400, bbox_inches='tight')

def get_metrics_tables():
    # colors = ['b', 'r', 'g', 'y']
    n_exps = len(chosen_exps)
    # colors = ['b', 'b', 'b', 'b']
    i = 0
    # fig_b, ax_b = plt.subplots(1,1)

    idx_pos = 0
    idx_vel = 1
    idx_ft = 2
    idx_Kp = 3

    # legend_ft = [r'$F^{VM+K_H}_z$', r'$F^{VM+K_L}_z$', r'$F^{VM+SIC}_z$', r'$F^{VM+POC}_z$']
    # legend_ft = [r'$F_z$', r'$F_z$', r'$F_z$', r'$F_z$']
    legend_ft = [r'$F_z$' for _ in range(n_exps)]
    # legend_Kp = [r'$K_H$', r'$K_L$', r'$SIC$', r'$POC$']
    legend_Kp = [r'$K_H$' for _ in range(n_exps)]
    # legend_pos = [r'$z^{VM+K_H}$', r'$z^{VM+K_L}$', r'$z^{VM+SIC}$', r'$z^{VM+POC}$']
    # legend_pos = [r'$z$', r'$z$', r'$z$', r'$z$']
    legend_pos = [r'$z$' for _ in range(n_exps)]
    # legend_vel = [r'$\dot{z}^{VM+K_H}$', r'$\dot{z}^{VM+K_L}$', r'$\dot{z}^{VM+SIC}$', r'$\dot{z}^{VM+POC}$']
    legend_vel = [r'$\dot{z}$' for _ in range(n_exps)]
    add_lw = 1


    for j, idx_exp in enumerate(chosen_exps):
        file_name = dh._get_name(idx_exp)
        print(file_name)
        offset_final = 190 if 'fp' in file_name else 0
        params['idx_initial'] = dh.get_idx_from_file(idx_exp, data_info, idx_name='idx_start')+154
        params['idx_end'] = dh.get_idx_from_file(idx_exp, data_info, idx_name='idx_end')-offset_final

        time = dh.get_data(file_name=file_name,params=params, data_to_plot='time')
        time -= time[0]

        idx_impact = np.where(time > 0.34)[0][0] - 3

        try:
            idx_finish_analysis = np.where(time > 1)[0][0]
        except IndexError:
            idx_finish_analysis = len(time)

        ft_ = dh.get_data(file_name=file_name, params=params, data_to_plot='ft_')[:,2]
        # ax[idx_ft][j].legend(prop={'size': 8}, loc='lower right')
        

        Kp = dh.get_data(params, file_name=file_name, data_to_plot='Kp')[:,2]

        pos = dh.get_data(params, file_name=file_name, data_to_plot='EE_position')[:,2]
        pos_d = dh.get_data(params, file_name=file_name, data_to_plot='EE_position_d')[:,2]
        pos_ball = dh.get_data(params, file_name=file_name, data_to_plot='ball_pose_')[:,2]

        vel = dh.get_data(params, file_name=file_name, data_to_plot='EE_twist')[:,2]
        vel_d = dh.get_data(params, file_name=file_name, data_to_plot='EE_twist_d')[:,2]

        # apply KF
        z_actual_hat, z_dot_actual_hat, z_intersec, z_dot_intersec, time_intersec, time_f = kalman_filtering_1d(time, pos_ball, ft_)
        # ax[j][idx_pos].plot(time, pos_ball, 'k', linewidth=lw, label=legend_pos[i])
        # ax[j][idx_pos].plot(time_f[:-2], z_actual_hat[:-2]-(95/2/1000), color='b', linestyle='dashdot', linewidth=lw-0.5+add_lw, label='$z_O$')

        loi = LOI(time, ft_, idx_impact, idx_finish_analysis)
        
        Fmax = -np.min(ft_)

        try:
            idx_impact_ = np.where(time == time_f[-1])[0][0]
        except IndexError:
            idx_impact_ = len(time)
        vm_error = vel[idx_impact_] - z_dot_actual_hat[-1]

        dri = DRI(time, ft_, file_name)

        bti = BTI(time, ft_)

        previous_power = 0
        k = 0
        Energy = 0
        for fz_i, vel_i in zip(ft_[idx_impact:idx_finish_analysis], vel[idx_impact:idx_finish_analysis]):
            current_power = fz_i*vel_i
            if k > 0:
                Energy += (current_power+previous_power)*0.001/2
            previous_power = current_power
            k += 1


        print(file_name,  '\tLOI =',        loi,
                          '\tDRI = ',       dri,
                          '\tBTI = ',       bti,
                          '\tE = ',         Energy,
                          '\tF_max = ',     Fmax,
                          '\tVM-error = ',  vm_error)


        if 'fp' not in file_name:
            if 'vm+low' in file_name:
                get_ = -6
            else:
                get_ = -4
        else:
            get_ = -2

        i += 1

def LOI(time, ft, idx_impact, idx_ss, file_name):
    loi = 0
    fw = ft[-1]
    int_term_prev = 0
    if 'fp' in file_name:
        return 0
    for i in range(idx_impact, idx_ss):
        int_term = np.abs(ft[i] - fw)
        loi += np.abs((int_term + int_term_prev)*(time[i]-time[i-1])/2)
        int_term_prev = int_term
    return loi

def DRI(time, ft, exp=''):
    if 'fp' in exp:
        return 0
    # plt.plot(time, ft)
    # ft = np.abs(ft)
    # plt.plot(ft)
    i1 = np.argmin(ft)
    i2 = np.argmin(ft[i1+15:i1+25])+i1+15
    # plt.plot(ft[i1:])
    # finding i2
    # i2 = i1
    # for j, ft_i in enumerate(ft[i1:]):
    #     if j > 0:
    #         i2 += 1
    #         if ft_i > ft[i1:][j-1]:
    #             i2 -= 1
    #             break
    # # i2 = np.argmin(ft[i1:])
    # # plt.plot(ft[i2:])
    # aux = ft[i2:]
    # for j, ft_i in enumerate(aux):
    #     if j > 0:
    #         i2 += 1
    #         if ft_i < aux[j-1]:
    #             i2 -= 1
    #             break
    
    # plt.plot(ft[i2:])
    # plt.show()
    # print('i1 = ', i1)
    wb = -0.1*9.81
    delta = np.log((ft[i1]-wb)/(ft[i2]-wb))
    dri = 1/(np.sqrt(1+(2*np.pi/delta)**2))
    # plt.plot(time, ft)
    # plt.plot(time[i1], ft[i1], 'ko')
    # plt.plot(time[i2], ft[i2], 'rx')
    # plt.show()
    return dri

def BTI(time, ft, file_name):
    bti = 0
    idx_peak = np.argmin(ft)
    idx_positive_forces = np.where(ft[idx_peak:] > 0)[0]
    # plt.plot(time[idx_peak:], ft[idx_peak:])
    # plt.plot(time[idx_peak:][idx_positive_forces], ft[idx_peak:][idx_positive_forces], 'ro')
    # plt.plot(time[idx_peak:], np.zeros_like(ft[idx_peak:]), 'k--')
    # print(idx_positive_forces)
    # plt.show()

    offset = 2 if 'sic' in file_name else 1

    lists_above_zero = []
    list_aux = []
    if idx_positive_forces.size > 0:
        list_aux.append(idx_positive_forces[0])
    for i, e in enumerate(idx_positive_forces):
        if i > 0:
            if e - idx_positive_forces[i-1] > 1:
                lists_above_zero.append(list_aux)
                list_aux = []
            list_aux.append(e)
    else:
        if len(list_aux)>1:
            lists_above_zero.append(list_aux)
    
    for list_idxs in lists_above_zero:
        bti += time[list_idxs[-1]+offset] - time[list_idxs[0]-offset]

    # print(lists_above_zero)
    return bti

def robots_ratio():
    class robot():
        def __init__(self, payload, max_vel) -> None:
            self.payload = payload
            self.max_vel = max_vel
        
    franka = robot(3, 1.)
    iiwa = robot(14, 2)

    print('iiwa_billard = ', (0.5*6)/(iiwa.payload*iiwa.max_vel))
    print('franka = ', (0.1*3.25)/(franka.payload*franka.max_vel))

def cartesian_plots_and_metrics_all():
    # big pic 1d

    if True:
        n_exps = len(chosen_exps)-1  # remove -1 to add the plot for VM-VIC-DIM in the 1d comparison
        data_1d = {k: DataClass() for k in chosen_exps.keys()}
        # colors = ['b', 'b', 'b', 'b']
        colors = ['b' for _ in range(n_exps)]
        styles = ['solid', 'dotted', 'dashdot', 'on-off-dash-seq']
        gray_cycler = (cycler(color=["#000000", "#333333", "#666666", "#999999", "#cccccc"]) +
                        cycler(linestyle=["-", "--", "-.", ":", "-"]))
        plt.rc("axes", prop_cycle=gray_cycler)
        i = 0
        ylimits_ft = [-20, 5]
        ylimits_Kp = [10, 50]
        ylimits_pos = [0, 0.8]
        ylimits_vel = [-2.75, 1]

        fig_1d, ax_1d  = plt.subplots(n_exps, 4, figsize=(20, 8), layout="constrained")

        idx_pos = 0
        idx_vel = 1
        idx_ft = 2
        idx_Kp = 3

        legend_ft = [r'$F_z$' for _ in range(n_exps)]
        legend_Kp = [r'$K_H$' for _ in range(n_exps)]
        legend_pos = [r'$z$' for _ in range(n_exps)]
        legend_vel = [r'$\dot{z}$' for _ in range(n_exps)]
        add_lw = 1

        n_joints = 7

        for j, key_exp in enumerate(chosen_exps):
            data_1d[key_exp].file_name = dh._get_name(chosen_exps[key_exp])
            # IDXS
            params['idx_initial'] = dh.get_idx_from_file(chosen_exps[key_exp], data_info, idx_name='idx_start')+154
            if 'fp' in data_1d[key_exp].file_name:
                offset_final = 195
                params['idx_end'] = dh.get_idx_from_file(chosen_exps[key_exp], data_info, idx_name='idx_end')-offset_final
            else:
                params['idx_end'] = params['idx_initial'] + 1000*xlimits[-1]
            data_1d[key_exp].params = params

            # TIME
            data_1d[key_exp].time = dh.get_data(file_name=data_1d[key_exp].file_name,params=params, data_to_plot='time')
            data_1d[key_exp].time -= data_1d[key_exp].time[0]

            # FT AND IMPACT IDX
            data_1d[key_exp].ft_ = dh.get_data(file_name=data_1d[key_exp].file_name, params=params, data_to_plot='ft_')[:,2]
            data_1d[key_exp].idx_impact = np.where(data_1d[key_exp].ft_ < -3)[0][0]-3
            if 'fp' in data_1d[key_exp].file_name:
                data_1d[key_exp].idx_finish_analysis = -1
            else:
                data_1d[key_exp].idx_finish_analysis = np.where(data_1d[key_exp].time > xlimits[-1])[0][0]

            # KP
            data_1d[key_exp].Kp = dh.get_data(params, file_name=data_1d[key_exp].file_name, data_to_plot='Kp')[:,2]

            # POS 
            data_1d[key_exp].pos = dh.get_data(params, file_name=data_1d[key_exp].file_name, data_to_plot='EE_position')[:,2]
            data_1d[key_exp].pos_d = dh.get_data(params, file_name=data_1d[key_exp].file_name, data_to_plot='EE_position_d')[:,2]
            data_1d[key_exp].pos_ball = dh.get_data(params, file_name=data_1d[key_exp].file_name, data_to_plot='ball_pose_')[:,2]

            # VEL
            data_1d[key_exp].vel = dh.get_data(params, file_name=data_1d[key_exp].file_name, data_to_plot='EE_twist')[:,2]
            data_1d[key_exp].vel_d = dh.get_data(params, file_name=data_1d[key_exp].file_name, data_to_plot='EE_twist_d')[:,2]

            # apply KF
            data_1d[key_exp].z_actual_hat, data_1d[key_exp].z_dot_actual_hat, \
                data_1d[key_exp].z_intersec, data_1d[key_exp].z_dot_intersec, \
                    data_1d[key_exp].time_intersec, data_1d[key_exp].time_f = kalman_filtering_1d(data_1d[key_exp].time, data_1d[key_exp].pos_ball, data_1d[key_exp].ft_)

            # JOINT MEASUREMENTS
            data_1d[key_exp].tau_max = np.zeros((n_joints))
            data_1d[key_exp].tau_rms = np.zeros((n_joints))
            data_1d[key_exp].tau_d_rms = np.zeros((n_joints))

            data_1d[key_exp].tau_m = dh.get_data(file_name=data_1d[key_exp].file_name, params=params, data_to_plot='tau_measured')#[data_1d[key_exp].idx_impact:data_1d[key_exp].idx_finish_analysis]
            data_1d[key_exp].tau_d = dh.get_data(file_name=data_1d[key_exp].file_name, params=params, data_to_plot='tau')#[data_1d[key_exp].idx_impact:data_1d[key_exp].idx_finish_analysis]
            data_1d[key_exp].q_dot = dh.get_data(file_name=data_1d[key_exp].file_name, params=params, data_to_plot='dq')#[data_1d[key_exp].idx_impact:data_1d[key_exp].idx_finish_analysis]
            
            data_1d[key_exp].tau_norm = np.zeros_like(data_1d[key_exp].tau_m)
            data_1d[key_exp].tau_norm = np.divide(data_1d[key_exp].tau_m, tau_limits)
            data_1d[key_exp].tau_max = np.max(np.abs(data_1d[key_exp].tau_norm.T), axis=1)

            data_1d[key_exp].tau_d_norm = np.zeros_like(data_1d[key_exp].tau_d)
            data_1d[key_exp].tau_d_norm = np.divide(data_1d[key_exp].tau_d, tau_limits)
            
            # TAU RMS, 
            data_1d[key_exp].tau_rms = np.sqrt(np.mean(data_1d[key_exp].tau_norm**2, axis=0)/(data_1d[key_exp].time[data_1d[key_exp].idx_finish_analysis]-data_1d[key_exp].time[data_1d[key_exp].idx_impact]))
            data_1d[key_exp].tau_d_rms = np.sqrt(np.mean(data_1d[key_exp].tau_d_norm**2, axis=0)/(data_1d[key_exp].time[data_1d[key_exp].idx_finish_analysis]-data_1d[key_exp].time[data_1d[key_exp].idx_impact]))
            data_1d[key_exp].tau_rms_sum = np.sum(data_1d[key_exp].tau_rms)

            # ADIM
            w_fd = dh.get_data(file_name=data_1d[key_exp].file_name, params=params, data_to_plot='m_arm')
            T_ = data_1d[key_exp].time[data_1d[key_exp].idx_finish_analysis]-data_1d[key_exp].time[data_1d[key_exp].idx_impact]
            idx_step = 0
            time_aux = data_1d[key_exp].time[0:data_1d[key_exp].idx_finish_analysis]
            w_fd = w_fd[0:data_1d[key_exp].idx_finish_analysis]
            while time_aux[idx_step] <= T_:
                if idx_step > 1:
                    data_1d[key_exp].tau_adim += (w_fd[idx_step] + w_fd[idx_step-1])*(time_aux[idx_step] - time_aux[idx_step-1])/2
                idx_step += 1
                if idx_step+1 >= len(w_fd):
                    break
            data_1d[key_exp].tau_adim = data_1d[key_exp].tau_adim/T_

            # METRICS
            data_1d[key_exp].loi = LOI(data_1d[key_exp].time, data_1d[key_exp].ft_, data_1d[key_exp].idx_impact, data_1d[key_exp].idx_finish_analysis, data_1d[key_exp].file_name)
            data_1d[key_exp].Fmax = -np.min(data_1d[key_exp].ft_)
            data_1d[key_exp].vme = data_1d[key_exp].vel[data_1d[key_exp].idx_impact] - data_1d[key_exp].z_dot_actual_hat[-1]
            data_1d[key_exp].dri = DRI(data_1d[key_exp].time, data_1d[key_exp].ft_, data_1d[key_exp].file_name)
            data_1d[key_exp].bti = BTI(data_1d[key_exp].time, data_1d[key_exp].ft_, data_1d[key_exp].file_name)

            print(data_1d[key_exp].file_name,   '\tLOI =',        data_1d[key_exp].loi,
                                                '\tDRI = ',       data_1d[key_exp].dri,
                                                '\tBTI = ',       data_1d[key_exp].bti,
                                                '\tF_max = ',     data_1d[key_exp].Fmax,
                                                '\tVM-error = ',  data_1d[key_exp].vme,
                                                '\tE_pos_impact = ', (data_1d[key_exp].pos[data_1d[key_exp].idx_impact] - 0.35)*1000,
                                                '\tball_initial_pos = ',  data_1d[key_exp].pos_ball[0],
                                                '\trobot_initial_pos = ',  data_1d[key_exp].pos[0],
                                                "\tTAU ADIM = ", data_1d[key_exp].tau_adim,
                                                "\tTAU MAX = ", np.max(data_1d[key_exp].tau_max),
                                                "\tTAU RMS SUM = ", data_1d[key_exp].tau_rms_sum,
                                                "\n")

        
        
        data_1d[key_exp].tau_rms

        print('\n\nREGARDING REDUCTION OF TAU_RMS:')
        for key_vm_vic, key_vm_vic_dim in zip(['vm-vic'], ['vm-vic-dim']):
            for j in range(n_joints):
                print('JOINT ', str(j+1), ' changed to ', (data_1d[key_vm_vic_dim].tau_rms[j]-data_1d[key_vm_vic].tau_rms[j])/(data_1d[key_vm_vic].tau_rms[j])*100, '%')
        
        print('\n\nREGARDING REDUCTION OF TAU_MAX:')
        for key_vm_vic, key_vm_vic_dim in zip(['vm-vic'], ['vm-vic-dim']):
            for j in range(n_joints):
                print('JOINT ', str(j+1), ' changed to ', (data_1d[key_vm_vic_dim].tau_max[j]-data_1d[key_vm_vic].tau_max[j])/(data_1d[key_vm_vic].tau_max[j])*100, '%')


        # JUST PLOT
        for j, key_exp in enumerate(chosen_exps):
            if j < n_exps:
                ax_1d[j][idx_ft].plot(data_1d[key_exp].time, data_1d[key_exp].ft_, linewidth=lw+add_lw, label=legend_ft[i])
                ax_1d[j][idx_ft].set_xlim(xlimits)
                ax_1d[j][idx_ft].set_ylim(ylimits_ft)

                ax_1d[j][idx_Kp].plot(data_1d[key_exp].time, data_1d[key_exp].Kp, linewidth=lw+add_lw)
                ax_1d[j][idx_Kp].set_xlim(xlimits)
                ax_1d[j][idx_Kp].set_ylim(ylimits_Kp)

                ax_1d[j][idx_pos].plot(data_1d[key_exp].time, data_1d[key_exp].pos, linewidth=lw+add_lw, label=legend_pos[i])
                if 'fp' in data_1d[key_exp].file_name:
                    ax_1d[j][idx_pos].plot(np.arange(0, 1, 1/1000), data_1d[key_exp].pos_d[0]*np.ones(1000), 'k--', linewidth=lw-0.5, label=legend_pos[i][:-1]+'_d$', alpha=0.6)
                    ax_1d[j][idx_pos].plot(data_1d[key_exp].time[-1], data_1d[key_exp].pos[-1], 'kx', ms=7)
                    ax_1d[j][idx_ft].plot(data_1d[key_exp].time[-1], data_1d[key_exp].ft_[-1], 'kx', ms=7)
                else:
                    ax_1d[j][idx_pos].plot(data_1d[key_exp].time, data_1d[key_exp].pos_d, 'k--', linewidth=lw-0.5+add_lw, label=legend_pos[i][:-1]+'_d$', alpha=0.6)
                ax_1d[j][idx_pos].set_xlim(xlimits)
                ax_1d[j][idx_pos].set_ylim(ylimits_pos)

                ax_1d[j][idx_vel].plot(data_1d[key_exp].time, data_1d[key_exp].vel, linewidth=lw+add_lw, label=legend_vel[i])
                if 'fp' in data_1d[key_exp].file_name:
                    ax_1d[j][idx_vel].plot(np.arange(0, 1, 1/1000), data_1d[key_exp].vel_d[0]*np.ones(1000), 'k--', linewidth=lw-0.5+add_lw, label=legend_vel[i][:-1]+'_d$', alpha=0.6)
                    ax_1d[j][idx_vel].plot(data_1d[key_exp].time[-1], data_1d[key_exp].vel[-1], 'kx', ms=7)
                    ax_1d[j][idx_Kp].plot(data_1d[key_exp].time[-1], data_1d[key_exp].Kp[-1], 'kx', ms=7)
                else:
                    ax_1d[j][idx_vel].plot(data_1d[key_exp].time, data_1d[key_exp].vel_d, 'k--', linewidth=lw-0.5+add_lw, label=legend_vel[i][:-1]+'_d$', alpha=0.6)
                ax_1d[j][idx_vel].set_xlim(xlimits)
                ax_1d[j][idx_vel].set_ylim(ylimits_vel)

                ax_1d[j][idx_pos].plot(data_1d[key_exp].time_f[:-2], data_1d[key_exp].z_actual_hat[:-2], color='b', linestyle='dashdot', linewidth=lw-0.5+add_lw, label='$z_O$')
                ax_1d[j][idx_vel].plot(data_1d[key_exp].time_f[:-2], data_1d[key_exp].z_dot_actual_hat[:-2], color='b', linestyle='dashdot', linewidth=lw-0.5+add_lw, label='$\dot{z}_O$')
                if 'fp' not in data_1d[key_exp].file_name:
                    if 'vm+low' in data_1d[key_exp].file_name:
                        get_ = -6
                    else:
                        get_ = -4
                else:
                    get_ = -2

                ax_1d[j][idx_pos].add_patch(Ellipse((data_1d[key_exp].time_f[get_], data_1d[key_exp].z_actual_hat[get_]), 2*ball_radius/5+0.015, 2*ball_radius, color='b', fill=False))
                i += 1

        
        fig_1d.set_constrained_layout(constrained=True)
        ax_1d[0,idx_pos].set_title('$z~[m]$')
        ax_1d[0,idx_vel].set_title('$\dot{z}~[m/s]$')
        ax_1d[0,idx_Kp].set_title('$K_p$')
        ax_1d[0,idx_ft].set_title('$F_z~[N]$')
        fig_1d.align_ylabels()

        # ax[0,0].set_title('$VM+K_{P_H}$')
        # ax[0,1].set_title('$VM+K_{P_L}$')
        k = 0
        # ax[k,0].set_ylabel('$FP_{17}$'+'\n'+'$K_{L}$'); k+=1
        # ax[k,0].set_ylabel('$FP_{17}$'+'\n'+'$K_{H}$'); k+=1
        ax_1d[k,0].set_ylabel('$FP$'+'\n'+'$K_{L}$'); k+=1
        ax_1d[k,0].set_ylabel('$FP$'+'\n'+'$K_{H}$'); k+=1
        ax_1d[k,0].set_ylabel('$VM$'+'\n'+'$K_{L}$'); k+=1
        ax_1d[k,0].set_ylabel('$VM$'+'\n'+'$K_{H}$'); k+=1
        ax_1d[k,0].set_ylabel('$VM$'+'\n'+'$SIC$'); k+=1
        ax_1d[k,0].set_ylabel('$VM$'+'\n'+'$VIC$'); k+=1
        if n_exps == 7:
            ax_1d[k,0].set_ylabel('$VM$'+'\n'+'$VIC-DIM$'); k+=1
        
        for j, _ in enumerate(chosen_exps):
            if j < n_exps-1:
                ax_1d[j][0].set_xticks([])
                ax_1d[j][1].set_xticks([])
                ax_1d[j][2].set_xticks([])
                ax_1d[j][3].set_xticks([])
        
        for j in range(4):
            ax_1d[n_exps-1,j].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
            ax_1d[n_exps-1,j].set_xticklabels(['$0$', '$0.25$', '$0.5$', '$0.75$', '$1$'], size=13)
            
        #     # ax[j][0].set_yticks([])``
        #     ax[j][1].set_yticks([])
        #     ax[j][2].set_yticks([])
        #     ax[j][3].set_yticks([])

        # grids creation
        x_grids = list(np.arange(0,2,0.25))
        n_divisions = 5
        alpha_grids = 0.12
        # y_grids_ft = list(np.arange(ylimits_ft[0], ylimits_ft[-1], (ylimits_ft[-1]-ylimits_ft[0])/n_divisions))
        y_grids_ft = [-15, -10, -5, 0, 5]
        # y_grids_Kp = list(np.arange(ylimits_Kp[0], ylimits_Kp[-1], (ylimits_Kp[-1]-ylimits_Kp[0])/n_divisions))
        y_grids_Kp = [0, 10, 20, 30, 40, 50]
        # y_grids_pos = list(np.arange(ylimits_pos[0], ylimits_pos[-1], (ylimits_pos[-1]-ylimits_pos[0])/n_divisions))
        y_grids_pos = [i for i in list(np.arange(0, 0.81, 0.1))]
        # y_grids_vel = list(np.arange(ylimits_vel[0], ylimits_vel[-1], (ylimits_vel[-1]-ylimits_vel[0])/n_divisions))
        y_grids_vel = [-2, -1, 0, 0.5]
        for j, row in enumerate(ax_1d):
            for i, e in enumerate(row):
                [e.axvline(xg, color='k', alpha=alpha_grids) for xg in x_grids]
                if idx_ft == i:
                    [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_ft]
                    # e.axhline(0, color='k')
                    # e.set_yticks([0, -5, -15])
                    # e.set_yticklabels(['$0$', '$-5$', '$-15$'], size=13)
                    e.set_yticks([5, 0, -5, -10, -15, -20])
                    e.set_yticklabels(['$5$', '$0$', "$-5$", '$-10$', '$-15$', '$-20$'], size=13)
                if idx_Kp == i:
                    [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_Kp]
                    e.set_yticks([10, 20, 30, 40, 50])
                    e.set_yticklabels(['$10$', '$20$', '$30$', '$40$', '$50$'], size=13)
                if idx_pos == i:
                    [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_pos]
                    aux = [0, 0.2, 0.4, 0.6, 0.8]
                    e.set_yticks(aux)
                    e.set_yticklabels(['$'+str(a)+'$' for a in aux], size=13)
                if idx_vel == i:
                    e.set_yticks([-2, -1, 0])
                    e.set_yticklabels(['$-2$', '$-1$', '$0$'], size=13)
                    [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_vel]
        
        
        # ft
        for j in range(n_exps):
            ax_1d[j, idx_ft].axhline(-0.1*9.81, color='#000000', linestyle='dashdot', linewidth=lw-0.5, label='$F_w$')
            # ax[j, idx_ft].axhline(-0.1*9.81*.7, color='#000000', linestyle='dashdot', linewidth=lw-1, label='_nolegend_')
            # ax[j, idx_ft].axhline(-0.1*9.81*1.3, color='#000000', linestyle='dashdot', linewidth=lw-1, label='_nolegend_')

        # plt.gca().grid(True)
        # plt.tight_layout()

        # legends
        # for j, row in enumerate(ax):
        #     for e in row:
        #         e.legend(prop={'size': 10}, loc='lower right')
        for j in range(4):
            if j < 3:
                leg = '_nolegend_'
            else:
                leg = '$t_c$'
            
            if j == 0:
                loc = 'upper right'
            else:
                loc = 'lower right'
            
            ax_1d[-1, j].axvline(x = 0.34, ymin=0, ymax=6.96, linestyle='--', linewidth=1.1, color = 'r', label = leg, clip_on=False)
            ax_1d[-1, j].legend(prop={'size': 14}, loc=loc)
            

        # ax[n_exps-1, 2].axvline(x = 0.34, ymin=0, ymax=7.06, linestyle='-', linewidth=1.3, color = 'r', label = 'axvline - full height', clip_on=False)

        # plt.subplots_adjust(hspace=0.045)
        fig_1d.supxlabel('$Time~[s]$', size=20)
        # plt.show()
        fig_1d.savefig('images/1d-comparison-plots.png', pad_inches=0, dpi=400)

    ###############################JOINT ANALYSIS##################################
    # joint plots
    if True:
        fig_1d_joints, ax_1d_joints = plt.subplots(1, 2, figsize=(8, 6), subplot_kw=dict(polar=True))

        theta = np.deg2rad(np.arange(0,360+360/7,360/7))

        colors_i_want = [1, -1]
        mycolors = [myc.mpl_colors[i] for i in colors_i_want]
        
        mini_chosen_exps = {}
        mini_chosen_exps['vm-vic'] = chosen_exps['vm-vic']
        mini_chosen_exps['vm-vic-dim'] = chosen_exps['vm-vic-dim']

        i_colors = 0

        for j, key_exp in enumerate(mini_chosen_exps):
            one_tau_max = np.concatenate([data_1d[key_exp].tau_max.reshape(n_joints,1), data_1d[key_exp].tau_max[0].reshape(1,1)],axis=0)
            one_rms = np.concatenate([data_1d[key_exp].tau_rms.reshape(n_joints,1), data_1d[key_exp].tau_rms[0].reshape(1,1)],axis=0)

            fs = 12
            if j == 0: # tau_max
                aux = ax_1d_joints[0].plot(theta,one_tau_max,marker='o', linestyle='-',color=mycolors[i_colors])[0]
                x = aux.get_xdata()
                y = aux.get_ydata()
                ax_1d_joints[0].fill_betweenx(y, 0, x, alpha=0.15,color=mycolors[i_colors], label='_nolegend_')
                ax_1d_joints[0].set_yticks([0.25, 0.5, 0.75, 1.0])
                ax_1d_joints[0].set_yticklabels(['', '$0.5$', '', '$1$'], fontsize=fs)
                ax_1d_joints[0].set_xticks(theta)
                ax_1d_joints[0].set_xticklabels(['$\\tau_'+str(q+1)+'$' for q in [0, 1, 2, 3, 4, 5, 6, 0]])

                aux = ax_1d_joints[1].plot(theta,one_rms,marker='o', linestyle='-',color=mycolors[i_colors])[0]
                x = aux.get_xdata()
                y = aux.get_ydata()
                ax_1d_joints[1].fill_betweenx(y, 0, x, alpha=0.15,color=mycolors[i_colors], label='_nolegend_')
                ax_1d_joints[1].set_yticks([0.15, 0.3, 0.4])
                ax_1d_joints[1].set_yticklabels(['$0.15$', '$0.3$',''], fontsize=fs)
                ax_1d_joints[1].set_xticks(theta)
                ax_1d_joints[1].set_xticklabels(['$\\tau_'+str(q+1)+'$' for q in [0, 1, 2, 3, 4, 5, 6, 0]])

            else: # tau rms
                aux = ax_1d_joints[0].plot(theta,one_tau_max,marker='o', linestyle='-',color=mycolors[i_colors])[0]
                x = aux.get_xdata()
                y = aux.get_ydata()
                ax_1d_joints[0].fill_betweenx(y, 0, x, alpha=0.15,color=mycolors[i_colors], label='_nolegend_')
                ax_1d_joints[0].set_yticks([0.25, 0.5, 0.75, 1.0])
                ax_1d_joints[0].set_yticklabels(['', '$0.5$', '', '$1$'], fontsize=fs)
                ax_1d_joints[0].set_xticks(theta)
                ax_1d_joints[0].set_xticklabels(['$\\tau_'+str(q+1)+'$' for q in [0, 1, 2, 3, 4, 5, 6, 0]])

                aux = ax_1d_joints[1].plot(theta,one_rms,marker='o', linestyle='-',color=mycolors[i_colors])[0]
                x = aux.get_xdata()
                y = aux.get_ydata()
                ax_1d_joints[1].fill_betweenx(y, 0, x, alpha=0.15,color=mycolors[i_colors], label='_nolegend_')
                ax_1d_joints[1].set_yticks([0.15, 0.3, 0.4])
                ax_1d_joints[1].set_yticklabels(['$0.15$', '$0.3$',''], fontsize=fs)
                ax_1d_joints[1].set_xticks(theta)
                ax_1d_joints[1].set_xticklabels(['$\\tau_'+str(q+1)+'$' for q in [0, 1, 2, 3, 4, 5, 6, 0]])
            i_colors += 1

        legends = ['$VM-VIC$', '$VM-VIC-DIM$']

        ax_1d_joints[0].legend(legends, bbox_to_anchor=(1.45, -.15), fontsize=10)

        ax_1d_joints[0].set_title('$\\hat{\\boldsymbol{\\tau}}_{max}$')
        ax_1d_joints[0].yaxis.set_label_coords(-0.125, 0.5)
        ax_1d_joints[1].set_title('$\\hat{\\boldsymbol{\\tau}}_{RMS}$')
        ax_1d_joints[1].yaxis.set_label_coords(-0.125, 0.5)

        ax_1d_joints[0].set_ylim(0,1.15)

        # plt.show()
        fig_1d_joints.savefig('images/1d-dim-joint-polar-plots.png', pad_inches=0, dpi=400, bbox_inches='tight')


    ###############################DIM ADDITION##################################
    standard_cycler = cycler("color", ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"])
    # my_cyc = 
    plt.rc("axes", prop_cycle=standard_cycler)
    fig_dim, ax_dim  = plt.subplots(6, 1, figsize=(6, 6),# layout="constrained",
                             gridspec_kw={'wspace': 0.0, 'hspace': 0.25})
    
    labels = ['$VM-VIC$', '$VM-VIC-DIM$']
    i = 0
    idx_tau = idx_Kp+1
    colors = ['k', myc.mpl_colors[3]]

    for j, key_exp in enumerate(['vm-vic', 'vm-vic-dim']):
        print(key_exp)

        ax_dim[idx_ft].plot(data_1d[key_exp].time, data_1d[key_exp].ft_, color=colors[i], linewidth=lw, label=legend_ft[i])
        ax_dim[idx_ft].set_xlim(xlimits)
        ax_dim[idx_ft].set_ylim(ylimits_ft)

        ax_dim[idx_Kp].plot(data_1d[key_exp].time, data_1d[key_exp].Kp, color=colors[i], linewidth=lw, label=labels[i])
        ax_dim[idx_Kp].set_xlim(xlimits)
        ax_dim[idx_Kp].set_ylim(ylimits_Kp)

        off_pos = 0 if j == 1 else 0
        ax_dim[idx_pos].plot(data_1d[key_exp].time[off_pos:], data_1d[key_exp].pos[off_pos:], color=colors[i], linewidth=lw, label=legend_pos[i])
        # if i == 0:
        ax_dim[idx_pos].plot(data_1d[key_exp].time[off_pos:], data_1d[key_exp].pos_d[off_pos:], color=colors[i], linestyle='--', linewidth=lw-0.5, label=legend_pos[i][:-1]+'_d$', alpha=0.6)
        ax_dim[idx_pos].set_xlim(xlimits)
        ax_dim[idx_pos].set_ylim(ylimits_pos)

        ax_dim[idx_vel].plot(data_1d[key_exp].time[off_pos:], data_1d[key_exp].vel[off_pos:], color=colors[i], linewidth=lw, label=legend_vel[i])
        ax_dim[idx_vel].plot(data_1d[key_exp].time[off_pos:], data_1d[key_exp].vel_d[off_pos:], color=colors[i], linestyle='--', linewidth=lw-0.5, label=legend_vel[i][:-1]+'_d$', alpha=0.6)
        ax_dim[idx_vel].set_xlim(xlimits)
        ax_dim[idx_vel].set_ylim(ylimits_vel)

        # apply KF
        # ax[idx_pos].plot(time_f, z_actual_hat-(95/2/1000), color='b', linestyle='dashdot', linewidth=lw-0.5, label='$z_b$')
        if i == 1:
            ax_dim[idx_pos].plot(data_1d[key_exp].time_f[off_pos:-2], data_1d[key_exp].z_actual_hat[off_pos:-2], color='b', linestyle='dashdot', linewidth=lw-0.5, label='$z_b$')
            ax_dim[idx_vel].plot(data_1d[key_exp].time_f[off_pos:-2], data_1d[key_exp].z_dot_actual_hat[off_pos:-2], color='b', linestyle='dashdot', linewidth=lw-0.5, label='$\dot{z}_b$')

        # tau_m_norm = np.divide(data_1d[key_exp].tau_m, tau_limits)
        if j == 1:
            idx_tau += 1
        if j == 1:
            ax_dim[idx_pos].add_patch(Ellipse((data_1d[key_exp].time_f[-2], data_1d[key_exp].z_actual_hat[-2]), 2*ball_radius/5+0.01, 2*ball_radius, color='b', fill=False))

        ax_dim[idx_tau].plot(data_1d[key_exp].time, data_1d[key_exp].tau_norm, label=['$\\tau_'+str(i+1)+'$' for i in range(7)])
        ax_dim[idx_tau].set_xlim(xlimits)
        ax_dim[idx_tau].set_ylim([-1, 1])

        i += 1

    # fig.set_constrained_layout(constrained=True)
    ax_dim[idx_pos].set_ylabel('$z~[m]$')
    ax_dim[idx_vel].set_ylabel('$\dot{z}~[m]$')
    ax_dim[idx_Kp].set_ylabel('$K_p$')
    ax_dim[idx_ft].set_ylabel('$F_z~[N]$')
    ax_dim[idx_tau-1].set_ylabel(r'$\hat{\boldsymbol{\tau}}$')
    ax_dim[idx_tau].set_ylabel(r'$\hat{\boldsymbol{\tau}}_{DIM}$')
    ax_dim[idx_tau].set_xlabel('$Time~[s]$', size=15)
    fig_dim.align_ylabels()

    for ax_ in ax_dim[:-1]:
        ax_.set_xticks([])

    ax_dim[-1].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_dim[-1].set_xticklabels(['$0$', '$0.25$', '$0.5$', '$0.75$', '$1$'], size=13)

    # grids creation
    x_grids = list(np.arange(0,2,0.25))
    alpha_grids = 0.12
    y_grids_ft = [-15, -10, -5, 0, 5]
    y_grids_Kp = [0, 10, 20, 30, 40, 50]
    y_grids_pos = [i for i in list(np.arange(0, 0.81, 0.1))]
    y_grids_vel = [-2, -1, 0, 0.5]
    y_grids_tau = [-1, -0.5, 0, 0.5, 1]
    size_labels = 12
    for j, e in enumerate(ax_dim):
        [e.axvline(xg, color='k', alpha=alpha_grids) for xg in x_grids]
        if idx_ft == j:
            e.set_yticks([5, 0, -5, -10, -15, -20])
            e.set_yticklabels(['$5$', '$0$', "$-5$", '$-10$', '$-15$', '$-20$'], size=size_labels)
            [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_ft]
            # e.axhline(0, color='k')
        if idx_Kp == j:
            e.set_yticks([10, 20, 30, 40, 50])
            e.set_yticklabels(['$10$', '$20$', '$30$', '$40$', '$50$'], size=size_labels)
            [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_Kp]
        if idx_pos == j:
            aux = [0, 0.2, 0.4, 0.6, 0.8]
            e.set_yticks(aux)
            e.set_yticklabels(['$'+str(a)+'$' for a in aux], size=size_labels)
            [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_pos]
        if idx_vel == j:
            e.set_yticks([-2, -1, 0])
            e.set_yticklabels(['$-2$', '$-1$', '$0$'], size=size_labels)
            [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_vel]
        if idx_tau == j or idx_tau-1 == j:
            e.set_yticks(y_grids_tau)
            e.set_yticklabels(['$'+str(a)+'$' for a in y_grids_tau], size=size_labels)
            [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_tau]

    # for j in range(4):
    #     if j == 1:
    #         ax[j].legend(prop={'size': 10}, loc='lower right')
    #     if j == 0:
    leg = ax_dim[-1].legend(prop={'size': 8}, loc='lower right', bbox_to_anchor=(1.01, 0.75), ncol=4)
    leg.get_frame().set_alpha(None)
    leg.get_frame().set_facecolor((1.0, 1.0, 1, 1))
    # ax[-2].legend(prop={'size': 10}, loc='lower right')
    ax_dim[-3].legend(prop={'size': 10}, loc='lower right')
    ax_dim[-1].axvline(x = 0.34, ymin=0, ymax=7.25, linestyle='--', linewidth=1.1, color = 'r', label = '_nolegend_', clip_on=False)

    plt.show()
    fig_dim.savefig('images/1d_plots_dim.png', dpi=400, bbox_inches='tight')

    return data_1d

def generate_videos_1d(data_1d):
    
    def animate_video(j):
        idx = j*STEP
        plots_force.set_data(data_1d[key_exp].time[:idx], data_1d[key_exp].ft_[:idx])
        plots_k.set_data(data_1d[key_exp].time[:idx], data_1d[key_exp].Kp[:idx])
        plots_twist.set_data(data_1d[key_exp].time[:idx], data_1d[key_exp].vel[:idx])
        plots_pos.set_data(data_1d[key_exp].time[:idx], data_1d[key_exp].pos[:idx])

        plots_twist_d.set_data(data_1d[key_exp].time[:idx], data_1d[key_exp].vel_d[:idx])
        plots_pos_d.set_data(data_1d[key_exp].time[:idx], data_1d[key_exp].pos_d[:idx])

        # if data_1d[key_exp].time_f[0] < data_1d[key_exp].time[idx] and data_1d[key_exp].time_f[-1] > data_1d[key_exp].time[idx]:
        #     plots_pos_ball.set_data(data_1d[key_exp].time_f[:(j-j_init_ball)*STEP], data_1d[key_exp].z_actual_hat[:(j-j_init_ball)*STEP])
        #     plots_twist_ball.set_data(data_1d[key_exp].time_f[:(j-j_init_ball)*STEP], data_1d[key_exp].z_dot_actual_hat[:(j-j_init_ball)*STEP])
    
    ylimits_ft = [-20, 5]
    ylimits_Kp = [10, 50]
    ylimits_pos = [0, 0.8]
    ylimits_vel = [-2.75, 1]

    idx_pos = 0
    idx_vel = 1
    idx_ft = 2
    idx_Kp = 3
    add_lw = 1

    legend_ft = [r'$F_z$' for _ in range(1)]
    legend_Kp = [r'$K_H$' for _ in range(1)]
    legend_pos = [r'$z$' for _ in range(1)]
    legend_vel = [r'$\dot{z}$' for _ in range(1)]
    
    for key_exp in chosen_exps.keys():
        print(key_exp)
        fig, ax = plt.subplots(4, figsize=(6, 8), layout="constrained")

        ax[idx_ft].set_xlim(xlimits)
        ax[idx_ft].set_ylim(ylimits_ft)

        ax[idx_Kp].set_xlim(xlimits)
        ax[idx_Kp].set_ylim(ylimits_Kp)

        ax[idx_pos].set_xlim(xlimits)
        ax[idx_pos].set_ylim(ylimits_pos)

        ax[idx_vel].set_xlim(xlimits)
        ax[idx_vel].set_ylim(ylimits_vel)

        ax[idx_pos].set_ylabel('$z~[m]$')
        ax[idx_vel].set_ylabel('$\dot{z}~[m/s]$')
        ax[idx_Kp].set_ylabel('$K_p$')
        ax[idx_ft].set_ylabel('$F_z~[N]$')
        fig.align_ylabels()

        ax[0].set_xticks([])
        ax[1].set_xticks([])
        ax[2].set_xticks([])
        # ax[3].set_xticks([])
        ax[3].set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax[3].set_xticklabels(['$0$', '$0.25$', '$0.5$', '$0.75$', '$1$'], size=13)
        ax[3].set_xlabel('$Time~[s]$', size=15)


        x_grids = list(np.arange(0,2,0.25))
        alpha_grids = 0.12
        y_grids_ft = [-15, -10, -5, 0, 5]
        y_grids_Kp = [0, 10, 20, 30, 40, 50]
        y_grids_pos = [i for i in list(np.arange(0, 0.81, 0.1))]
        y_grids_vel = [-2, -1, 0, 0.5]
        for j, e in enumerate(ax):
            [e.axvline(xg, color='k', alpha=alpha_grids) for xg in x_grids]
            if idx_ft == j:
                [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_ft]
                # e.axhline(0, color='k')
                # e.set_yticks([0, -5, -15])
                # e.set_yticklabels(['$0$', '$-5$', '$-15$'], size=13)
                e.set_yticks([5, 0, -5, -10, -15, -20])
                e.set_yticklabels(['$5$', '$0$', "$-5$", '$-10$', '$-15$', '$-20$'], size=13)
            if idx_Kp == j:
                [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_Kp]
                e.set_yticks([10, 20, 30, 40, 50])
                e.set_yticklabels(['$10$', '$20$', '$30$', '$40$', '$50$'], size=13)
            if idx_pos == j:
                [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_pos]
                aux = [0, 0.2, 0.4, 0.6, 0.8]
                e.set_yticks(aux)
                e.set_yticklabels(['$'+str(a)+'$' for a in aux], size=13)
            if idx_vel == j:
                e.set_yticks([-2, -1, 0])
                e.set_yticklabels(['$-2$', '$-1$', '$0$'], size=13)
                [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_vel]

        plots_force, = ax[idx_ft].plot(data_1d[key_exp].time[0], data_1d[key_exp].ft_[0], linewidth=dh.lw, color='k')
        plots_k, = ax[idx_Kp].plot(data_1d[key_exp].time[0], data_1d[key_exp].Kp[0], linewidth=dh.lw, color='k')
        plots_twist, = ax[idx_vel].plot(data_1d[key_exp].time[0], data_1d[key_exp].vel[0], linewidth=dh.lw, color='k')
        plots_pos, = ax[idx_pos].plot(data_1d[key_exp].time[0], data_1d[key_exp].pos[0], linewidth=dh.lw, color='k')
        plots_twist_d, = ax[idx_vel].plot(data_1d[key_exp].time[0], data_1d[key_exp].vel_d[0], linewidth=dh.lw-1, color='k', linestyle='--', alpha=0.6)
        plots_pos_d, = ax[idx_pos].plot(data_1d[key_exp].time[0], data_1d[key_exp].pos_d[0], linewidth=dh.lw-1, color='k', linestyle='--', alpha=0.6)

        plots_pos_ball, = ax[idx_pos].plot(data_1d[key_exp].time_f[0], data_1d[key_exp].z_actual_hat[0], color='b', linestyle='dashdot', linewidth=lw-0.5, label='$z_b$')
        plots_twist_ball, = ax[idx_vel].plot(data_1d[key_exp].time_f[0], data_1d[key_exp].z_dot_actual_hat[0], color='b', linestyle='dashdot', linewidth=lw-0.5, label='$\dot{z}_b$')

        # labels=['$\overline{\\boldsymbol{F}}_{VM-IC}$']
        # ax[0].legend(labels=labels, borderaxespad=0.1, handlelength=0.8, fontsize=LEGEND_SIZE)

        # labels=['$\\boldsymbol{K}_{VM-IC}$']
        # ax[1].legend(labels=labels, borderaxespad=0.1, handlelength=0.8, fontsize=LEGEND_SIZE, loc='lower right')

        # labels=['$\dot{\\boldsymbol{x}}_{d}$', '$\overline{\dot{\\boldsymbol{x}}}_{VM-IC}$']
        # ax[2].legend(labels=labels, borderaxespad=0.1, handlelength=0.8, fontsize=LEGEND_SIZE, loc='lower right')

        # labels=['$\\boldsymbol{x}_{d}$', '$\overline{\\boldsymbol{x}}_{VM-IC}$']
        # ax[3].legend(labels=labels, borderaxespad=0.1, handlelength=0.8, fontsize=LEGEND_SIZE)


        STEP = 5
        N_POINTS = 1000
        # if plot_now == 'const':
        #     STEP = 1

        n_frames = int((N_POINTS)/STEP)
        print("n_frames = ", n_frames)
        # print("video duration = +-", n_frames/FPS)

        animation_1 = animation.FuncAnimation(plt.gcf(), animate_video, interval=1, repeat=False, frames=n_frames)

        ### visualization
        # plt.show()

        video_name = 'video_'+key_exp+'.mp4'
        ### creating and saving the video
        writervideo = animation.FFMpegWriter(fps=30)
        # animation_1.save('images/' + video_name, writer=writervideo)
        plt.show()

def generate_videos_2d(data_2d):
    
    def animate_video(j):
        idx = j*STEP
        if idx != 0:
            plots_force.set_data(data_2d[key_exp].time[:idx], data_2d[key_exp].ft_[:idx])
            plots_k.set_data(data_2d[key_exp].time[offset_kp:offset_kp+idx]-data_2d[key_exp].time[offset_kp], data_2d[key_exp].Kp[offset_kp:offset_kp+idx].T[2]+10)
            plots_twist_y.set_data(data_2d[key_exp].time[:idx], data_2d[key_exp].vel[:idx].T[1])
            plots_pos_y.set_data(data_2d[key_exp].time[:idx], data_2d[key_exp].pos[:idx].T[1])
            plots_twist_z.set_data(data_2d[key_exp].time[:idx], data_2d[key_exp].vel[:idx].T[2])
            plots_pos_z.set_data(data_2d[key_exp].time[:idx], data_2d[key_exp].pos[:idx].T[2])

            plots_twist_y_d.set_data(data_2d[key_exp].time[:idx], data_2d[key_exp].vel_d[:idx].T[1])
            plots_pos_y_d.set_data(data_2d[key_exp].time[:idx], data_2d[key_exp].pos_d[:idx].T[1])
            plots_twist_z_d.set_data(data_2d[key_exp].time[:idx], data_2d[key_exp].vel_d[:idx].T[2])
            plots_pos_z_d.set_data(data_2d[key_exp].time[:idx], data_2d[key_exp].pos_d[:idx].T[2])

        # if data_2d[key_exp].time_f[0] < data_2d[key_exp].time[idx] and data_2d[key_exp].time_f[-1] > data_2d[key_exp].time[idx]:
        #     plots_pos_ball.set_data(data_2d[key_exp].time_f[:(j-j_init_ball)*STEP], data_2d[key_exp].z_actual_hat[:(j-j_init_ball)*STEP])
        #     plots_twist_ball.set_data(data_2d[key_exp].time_f[:(j-j_init_ball)*STEP], data_2d[key_exp].z_dot_actual_hat[:(j-j_init_ball)*STEP])
    
    ylimits_ft = [-20, 5]
    ylimits_Kp = [0, 50]
    ylimits_pos_z = [0.2, 0.8]
    ylimits_vel_z = [-2, 2]
    ylimits_pos_y = [-1.75, 0]
    ylimits_vel_y = [-1, 4]

    idx_pos_z = 0
    idx_vel_z = 1
    idx_pos_y = 2
    idx_vel_y = 3
    idx_ft = 4
    idx_Kp = 5
    add_lw = 1

    legend_ft = [r'$F_z$' for _ in range(1)]
    legend_Kp = [r'$K_H$' for _ in range(1)]
    legend_pos = [r'$z$' for _ in range(1)]
    legend_vel = [r'$\dot{z}$' for _ in range(1)]

    chosen_exps = {'2d-1': 19, '2d-2': 21, '2d-rmm-1': 22, '2d-rmm-2': 23}
    xlimits = [0, 1.5]
    
    for j, key_exp in enumerate(chosen_exps.keys()):
        print(key_exp)
        fig, ax = plt.subplots(6, figsize=(6, 10), layout="constrained")

        ax[idx_ft].set_xlim(xlimits)
        ax[idx_ft].set_ylim(ylimits_ft)

        ax[idx_Kp].set_xlim(xlimits)
        ax[idx_Kp].set_ylim(ylimits_Kp)

        ax[idx_pos_z].set_xlim(xlimits)
        ax[idx_pos_z].set_ylim(ylimits_pos_z)

        ax[idx_pos_y].set_xlim(xlimits)
        ax[idx_pos_y].set_ylim(ylimits_pos_y)

        ax[idx_vel_z].set_xlim(xlimits)
        ax[idx_vel_z].set_ylim(ylimits_vel_z)

        ax[idx_vel_y].set_xlim(xlimits)
        ax[idx_vel_y].set_ylim(ylimits_vel_y)

        ax[idx_pos_z].set_ylabel('$z~[m]$')
        ax[idx_vel_z].set_ylabel('$\dot{z}~[m/s]$')
        ax[idx_pos_y].set_ylabel('$y~[m]$')
        ax[idx_vel_y].set_ylabel('$\dot{y}~[m/s]$')
        ax[idx_Kp].set_ylabel('$K_p$')
        ax[idx_ft].set_ylabel('$F_z~[N]$')
        fig.align_ylabels()

        ax[0].set_xticks([])
        ax[1].set_xticks([])
        ax[2].set_xticks([])
        ax[3].set_xticks([])
        ax[4].set_xticks([])

        aux = [0] + list(np.arange(0.25, 1.55, 0.25))
        ax[5].set_xticks(aux)
        ax[5].set_xticklabels(['$'+str(a)+'$' for a in aux], size=13)
        ax[5].set_xlabel('$Time~[s]$', size=16)


        x_grids = list(np.arange(0,2,0.25))
        alpha_grids = 0.12
        y_grids_ft = [-15, -10, -5, 0, 5]
        y_grids_Kp = [0, 10, 20, 30, 40, 50]
        y_grids_pos_z = [i for i in list(np.arange(0, 0.81, 0.1))]
        y_grids_pos_y = [i for i in list(np.arange(-2, .01, 0.5))]
        y_grids_vel_z = [i for i in list(np.arange(-3, 3, 1))]
        y_grids_vel_y = [i for i in list(np.arange(0, 4.01, 1))]

        if j == 0:
            offset_kp = 90
        if j == 1:
            offset_kp = 40
        if j == 2:
            offset_kp = 70
        if j == 3:
            offset_kp = 40

        for j, e in enumerate(ax):
            [e.axvline(xg, color='k', alpha=alpha_grids) for xg in x_grids]
            if idx_ft == j:
                [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_ft]
                # e.axhline(0, color='k')
                # e.set_yticks([0, -5, -15])
                # e.set_yticklabels(['$0$', '$-5$', '$-15$'], size=13)
                e.set_yticks([5, 0, -5, -10, -15, -20])
                e.set_yticklabels(['$5$', '$0$', "$-5$", '$-10$', '$-15$', '$-20$'], size=13)
            if idx_Kp == j:
                [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_Kp]
                e.set_yticks([10, 20, 30, 40, 50])
                e.set_yticklabels(['$10$', '$20$', '$30$', '$40$', '$50$'], size=13)
            if idx_pos_z == j:
                [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_pos_z]
                aux = [0.2, 0.4, 0.6, 0.8]
                e.set_yticks(aux)
                e.set_yticklabels(['$'+str(a)+'$' for a in aux], size=13)
            if idx_vel_z == j:
                e.set_yticks([-2, -1, 0, 1, 2])
                e.set_yticklabels(['$-2$', '$-1$', '$0$', '$1$', '$2$'], size=13)
                [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_vel_z]
            if idx_pos_y == j:
                [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_pos_y]
                aux = [-1.5, -1, -0.5, 0]
                e.set_yticks(aux)
                e.set_yticklabels(['$'+str(a)+'$' for a in aux], size=13)
            if idx_vel_y == j:
                aux = [0, 1, 2, 3]
                e.set_yticks(aux)
                e.set_yticklabels(['$'+str(a)+'$' for a in aux], size=13)
                [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids_vel_y]

        plots_force, = ax[idx_ft].plot(data_2d[key_exp].time[0], data_2d[key_exp].ft_[0], linewidth=dh.lw, color='k')
        plots_k, = ax[idx_Kp].plot(data_2d[key_exp].time[0], data_2d[key_exp].Kp[0][2], linewidth=dh.lw, color='k')
        plots_twist_y, = ax[idx_vel_y].plot(data_2d[key_exp].time[0], data_2d[key_exp].vel[0][1], linewidth=dh.lw, color='k')
        plots_pos_y, = ax[idx_pos_y].plot(data_2d[key_exp].time[0], data_2d[key_exp].pos[0][1], linewidth=dh.lw, color='k')
        plots_twist_z, = ax[idx_vel_z].plot(data_2d[key_exp].time[0], data_2d[key_exp].vel[0][2], linewidth=dh.lw, color='k')
        plots_pos_z, = ax[idx_pos_z].plot(data_2d[key_exp].time[0], data_2d[key_exp].pos[0][2], linewidth=dh.lw, color='k')

        plots_twist_y_d, = ax[idx_vel_y].plot(data_2d[key_exp].time[0], data_2d[key_exp].vel_d[0][1], linewidth=dh.lw-1, color='k')
        plots_pos_y_d, = ax[idx_pos_y].plot(data_2d[key_exp].time[0], data_2d[key_exp].pos_d[0][1], linewidth=dh.lw-1, color='k')
        plots_twist_z_d, = ax[idx_vel_z].plot(data_2d[key_exp].time[0], data_2d[key_exp].vel_d[0][2], linewidth=dh.lw-1, color='k')
        plots_pos_z_d, = ax[idx_pos_z].plot(data_2d[key_exp].time[0], data_2d[key_exp].pos_d[0][2], linewidth=dh.lw-1, color='k')

        # plots_pos_ball, = ax[idx_pos_z].plot(data_2d[key_exp].time_f[0], data_2d[key_exp].z_actual_hat[0], color='b', linestyle='dashdot', linewidth=lw-0.5, label='$z_b$')
        # plots_twist_ball, = ax[idx_vel_z].plot(data_2d[key_exp].time_f[0], data_2d[key_exp].z_dot_actual_hat[0], color='b', linestyle='dashdot', linewidth=lw-0.5, label='$\dot{z}_b$')

        # labels=['$\overline{\\boldsymbol{F}}_{VM-IC}$']
        # ax[0].legend(labels=labels, borderaxespad=0.1, handlelength=0.8, fontsize=LEGEND_SIZE)

        # labels=['$\\boldsymbol{K}_{VM-IC}$']
        # ax[1].legend(labels=labels, borderaxespad=0.1, handlelength=0.8, fontsize=LEGEND_SIZE, loc='lower right')

        # labels=['$\dot{\\boldsymbol{x}}_{d}$', '$\overline{\dot{\\boldsymbol{x}}}_{VM-IC}$']
        # ax[2].legend(labels=labels, borderaxespad=0.1, handlelength=0.8, fontsize=LEGEND_SIZE, loc='lower right')

        # labels=['$\\boldsymbol{x}_{d}$', '$\overline{\\boldsymbol{x}}_{VM-IC}$']
        # ax[3].legend(labels=labels, borderaxespad=0.1, handlelength=0.8, fontsize=LEGEND_SIZE)


        STEP = 5
        N_POINTS = 1500
        # if plot_now == 'const':
        #     STEP = 1

        n_frames = int((N_POINTS)/STEP)
        print("n_frames = ", n_frames)
        # print("video duration = +-", n_frames/FPS)

        animation_1 = animation.FuncAnimation(plt.gcf(), animate_video, interval=1, repeat=False, frames=n_frames)

        ### visualization
        # plt.show()

        video_name = 'video_'+key_exp+'.mp4'
        ### creating and saving the video
        writervideo = animation.FFMpegWriter(fps=30)
        animation_1.save('images/' + video_name, writer=writervideo)
        # plt.show()

if __name__=='__main__':
    # plot_all_vanilla()
    # plot_vanilla_best_all_in_one()
    # plot_vanilla_best_4subplots_4figs()
    # plot_vanilla_best_1subplot_4figs() # this is shit
    # plot_best_4subplots_1fig()
    # plot_best_4subplots_1fig_transpose()
    # plot_zoom_forces()
    # plot_zoom_forces_single_plot()  # T-RO
    # plot_zoom_position_single_plot()  # T-RO
    # data = plot_best_2d()  # T-RO
    # generate_videos_2d(data)
    # plot_rmm_1d_all_in_one_4_plots()
    plot_rmm_1d_1_plot()
    # plot_rmm_1d_1_plot_with_without_dim() # T-RO
    #   # T-RO
    # plot_1d_increased_height()  # T-RO
    # plot_rmm_2d_all_in_one()
    # plots_joints_max_torque() 
    # plots_joints_max_torque_without_KH() # T-RO
    # plot_joint_torques_vanilla()
    # polar_plots_metrics()
    # get_metrics_tables()
    # robots_ratio()
    # data = cartesian_plots_and_metrics_all()  # T-RO
    # generate_videos_1d(data)
    pass