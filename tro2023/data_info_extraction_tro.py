import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data_plot_class_tro import DataPlotHelper
from os import listdir

FZ_IDX = 2

path_folder = 'data/'
data_info = pd.DataFrame(columns=['id', 'idx_start', 'idx_end', 'data_size'])

dh = DataPlotHelper(path_folder)

TRIALS_IDXS = [1, 2, 3]
HEIGHTS = [27]
SECONDS = 2.0

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

files_names = listdir(path=path_folder)
files_names.sort()

params['type'] = 'fp'   # fp, vm, 2d
params['gain'] = 'low' # low, high, sic, poc
params['rmm'] = False
params['idx'] = 2
params['data_to_plot'] = 'ft_'

ft_ = dh.get_data(params=params, file_name=files_names[0])

for idx, file_name in enumerate(files_names):
    data = dh.get_data(params=params, file_name=file_name, data_to_plot='ft_')
    idx_start = dh.get_idx_movement_starts(params=params, file_name=file_name)
    data_size = data[:,2].size
    idx_end = dh.get_idx_movement_ends(params, data_size=data_size, file_name=file_name, seconds=SECONDS)
    df_aux = pd.DataFrame([[str(idx).zfill(2), idx_start, idx_end, data_size]], columns=['id', 'idx_start', 'idx_end', 'data_size'])
    data_info = pd.concat([data_info, df_aux], ignore_index=True)

data_info.to_csv('data_info.csv', index=False)

#                 data_info.loc[(data_info['experiment_name'] == experiment_name) &
#                               (data_info['color'] == color) &
#                               (data_info['height'] == height), 'idx_start'] = idx_start
                
#                 data_info.loc[(data_info['experiment_name'] == experiment_name) &
#                               (data_info['color'] == color) &
#                               (data_info['height'] == height), 'idx_end'] = idx_end


# file_name = path_folder + 'empty.mat'
# params['i_initial'] = 0
# params['i_final'] = -1
# data = dh.get_data(params)
# idx_start = dh.get_idx_movement_starts(params)
# idx_end = dh.get_idx_movement_ends(params)
# data_info.loc[(data_info['experiment_name'] == 'empty'), 'idx_start'] = idx_start
# data_info.loc[(data_info['experiment_name'] == 'empty'), 'idx_end'] = idx_end


# file_name = path_folder + 'const-imp.mat'
# params['i_initial'] = 0
# params['i_final'] = -1
# data = dh.get_data(params)
# idx_start = dh.get_idx_movement_starts(params)
# idx_end = dh.get_idx_movement_ends(params)
# data_info.loc[(data_info['experiment_name'] == 'const_imp'), 'idx_start'] = idx_start
# data_info.loc[(data_info['experiment_name'] == 'const_imp'), 'idx_end'] = -1

# print(data_info)

# data_info.to_csv(path_folder+'data_info.csv', index=False)


# params['data_to_plot'] = 'EE_twist_d'
# data_EE_twist_d = dh.get_data(params)

# params['data_to_plot'] = 'EE_twist'
# data_EE_twist = dh.get_data(params)

# plt.plot(data_EE_twist[idx_start:idx_ends, 2])
# plt.plot(data_EE_twist_d[idx_start:idx_ends, 2], '--')
# plt.show()

# params['data_to_plot'] = 'FT_ati'
# data_ft = dh.get_data(params)
# plt.plot(data_ft[idx_start:idx_ends, 2])
# plt.show()

# for i in [1, 2]:
#     params['trial_idx'] = i
#     data = dh.get_data(params)
#     # print(data.shape)
#     plt.plot(data[:,FZ_IDX])
#     # dh.plot_data(data)

# plt.show()