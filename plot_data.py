import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
from data_plot_class import DataPlotHelper

path_folder = 'data/icra2022/2-exp-ImpLoop3-NoAdaptation-PARTIAL-RESULTS-DONT-EDIT/'
data_info = pd.read_csv(path_folder+'data_info.csv')

dh = DataPlotHelper(path_folder)

Z_AXIS = 2
COLORS = ['Green', 'Blue']
TRIALS_IDXS = [1, 2, 3]
HEIGHTS = [27]

params = {  
    'vic': True,
    'color': 'Blue',
    'trial_idx': 3,
    'height': 27,
    'impedance_loop': 30,
    'online_adaptation': False,
    'i_initial': 0,
    'i_final': -1,
    'data_to_plot': 'EE_twist_d',
    }

fig, ax = dh.plot_single(params, data_info, axis=Z_AXIS)
plt.show()


# individual plots
# fig, ax = plt.subplots(n_plots,1) 
# for i, file_name, i_init, i_fin in zip(range(n_plots), files_names, i_initial, i_final):
#     i_init *= 0
#     i_fin = -1
#     f = h5py.File(folder_name+file_name, 'r')

#     print("keys: ", f.keys())

#     # K = get_data(f, 'K', i_init, i_fin)
#     FT = get_data(f, 'FT_ati', i_init, i_fin)
#     EE_twist = get_data(f, 'EE_twist', i_init, i_fin)
#     time = get_data(f, 'time', i_init, i_fin)


#     # ax[i].plot(time-time[0], FT[:, 2], linewidth=3)
#     ax[i].plot(FT[:, 2], linewidth=3)
#     ax[i].plot(np.sqrt(np.mean(FT[:, 2]**2))*np.ones(n_points), 'k--')
#     # ax[i].plot(time-time[0], EE_twist[:, 2])
#     # ax[i].set_ylim(y_lim_ft)
#     # ax[i].set_xlim(x_lim)
#     ax[i].set_title(file_name)
#     ax[i].grid()

plt.show()

# fig, ax = plt.subplots(n_plots,1) 
# for i, file_name, i_init, i_fin in zip(range(n_plots), files_names, i_initial, i_final):
#     f = h5py.File(file_name, 'r')

#     print("keys: ", f.keys())

#     # K = get_data(f, 'K', i_init, i_fin)
#     EE_twist = get_data(f, 'EE_twist', i_init, i_fin)
#     EE_twist_d = get_data(f, 'EE_twist_d', i_init, i_fin)
#     time = get_data(f, 'time', i_init, i_fin)


#     ax[i].plot(time-time[0], EE_twist[:, 2], linewidth=3)
#     ax[i].plot(time-time[0], EE_twist_d[:, 2], 'r--')
#     # ax[i].plot(time-time[0], EE_twist[:, 2])
#     ax[i].set_ylim(y_lim_twist)
#     ax[i].set_xlim(x_lim)
#     ax[i].set_title(file_name)
#     ax[i].grid()

# plt.show()



# # comparison plots
# fig, ax = plt.subplots(1,1) 
# for i, file_name, i_init, i_fin in zip(range(n_plots), files_names, i_initial, i_final):
#     f = h5py.File(file_name, 'r')

#     print("keys: ", f.keys())

#     K = get_data(f, 'K', i_init, i_fin)
#     FT = get_data(f, 'FT_ati', i_init, i_fin)
#     time = get_data(f, 'time', i_init, i_fin)
#     EE_twist = get_data(f, 'EE_twist', i_init, i_fin)
#     EE_twist_d = get_data(f, 'EE_twist_d', i_init, i_fin)
#     # EE_position = get_data(f, 'EE_position', i_init, i_fin)
#     # EE_position_d = get_data(f, 'EE_position_d', i_init, i_fin)

#     ax.plot(time-time[0], FT[:, 2]/max(FT[:,2]), linewidth=3)
#     ax.plot(time-time[0], EE_twist[:, 2])
#     # ax.plot(time-time[0], EE_position[:, 2]/max(EE_position[:,2]))
#     # ax.plot(time-time[0], np.sqrt(np.mean(FT[:, 2]**2))*np.ones(n_points), 'k--')
#     # ax[i].plot(time-time[0], EE_twist[:, 2])
# ax.plot(time-time[0], K[:,2]/max(K[:,2]), 'k--')
# ax.plot(time-time[0], EE_twist_d[:,2], 'k--')
# # ax.plot(time-time[0], EE_position_d[:,2]/max(EE_position[:,2]), 'k--')
# ax.set_title("comparison")
# # ax.set_ylim(y_lim_ft)
# ax.set_xlim(x_lim)
# ax.set_ylabel('Fz [N]')
# ax.set_xlabel('time [s]')
# ax.grid()
# # ax.legend([aux[:-4] for aux in files_names])
# # ax.legend(['KMP-Fz', 'KMP-TwistZ', 'KMP-PosZ', 'VIC-Fz', 'VIC-TwistZ', 'VIC-PosZ'])
# ax.legend(['KMP-Fz', 'KMP-TwistZ', 'VIC-Fz', 'VIC-TwistZ'])
# plt.show()


# # comparison plots
# fig, ax = plt.subplots(1,1) 
# for i, file_name, i_init, i_fin in zip(range(n_plots), files_names, i_initial, i_final):
#     f = h5py.File(file_name, 'r')

#     print("keys: ", f.keys())

#     EE_twist = get_data(f, 'EE_twist', i_init, i_fin)
#     EE_twist_d = get_data(f, 'EE_twist_d', i_init, i_fin)
#     time = get_data(f, 'time', i_init, i_fin)


#     ax.plot(time-time[0], EE_twist[:, 2], linewidth=3)
#     # ax.plot(time-time[0], np.sqrt(np.mean(FT[:, 2]**2))*np.ones(n_points), 'k--')
#     # ax[i].plot(time-time[0], EE_twist[:, 2])
# ax.plot(time-time[0], EE_twist_d[:, 2], 'k--')
# ax.set_title("comparison")
# ax.set_ylim(y_lim_twist)
# ax.set_xlim(x_lim)
# ax.set_ylabel('Vel. Z [m/s]')
# ax.set_xlabel('time [s]')
# ax.grid()
# ax.legend([aux[:-4] for aux in files_names])
# plt.show()