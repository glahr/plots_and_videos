import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data_plot_class import DataPlotHelper

FZ_IDX = 2

path_folder = 'data/icra2022/2-exp-ImpLoop3-NoAdaptation-PARTIAL-RESULTS-DONT-EDIT/'
data_info = pd.read_csv(path_folder+'data_info.csv')

dh = DataPlotHelper(path_folder)

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

for vic in [False, True]:
    for trial_idx in TRIALS_IDXS:
        for color in COLORS:
            for height in HEIGHTS:
                params['color'] = color
                params['height'] = height
                params['trial_idx'] = trial_idx

                data = dh.get_data(params)
                
                idx_start = dh.get_idx_movement_starts(params)
                idx_end = dh.get_idx_movement_ends(params)

                experiment_name = 'opt_kmp'
                experiment_name += '_vic' if vic else ''
                experiment_name += '_' + str(trial_idx)

                data_info.loc[(data_info['experiment_name'] == experiment_name) &
                              (data_info['color'] == color) &
                              (data_info['height'] == height), 'idx_start'] = idx_start
                
                data_info.loc[(data_info['experiment_name'] == experiment_name) &
                              (data_info['color'] == color) &
                              (data_info['height'] == height), 'idx_end'] = idx_end
                

print(data_info)
                
data_info.to_csv(path_folder+'data_info.csv')


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