import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data_plot_class import DataPlotHelper

FZ_IDX = 2

path_folder = 'data/hrii_icra2022/'
data_info = pd.read_csv(path_folder+'data_info.csv')
sub_folder = '2-exp-ImpLoop3-NoAdaptation-PARTIAL-RESULTS-DONT-EDIT/'
path_folder += sub_folder

dh = DataPlotHelper()

params = {
    'vic': True,
    'color': 'Blue',
    'trial_idx': 1,
    'height': 27,
    'impedance_loop': 30,
    'online_adaptation': False,
    'i_initial': 0,
    'i_final': -1,
    'data_to_plot': 'EE_twist_d',
    }

data = dh.get_data(params, path_folder)
plt.plot(data)
plt.show()

# idx = dh.get_idx_movement_start(params)
# print(idx)

# for i in [1, 2]:
#     params['trial_idx'] = i
#     data = dh.get_data(params)
#     # print(data.shape)
#     plt.plot(data[:,FZ_IDX])
#     # dh.plot_data(data)

# plt.show()