from tkinter import N
import matplotlib.pyplot as plt
import numpy as np
import h5py

class DataPlotHelper:
    def __init__(self) -> None:
        pass

    def get_idx_movement_start(self, params):
        params['data_to_plot'] = 'EE_twist_d'
        return np.nonzero(self.get_data(params))[0][0] 
        

    def get_idx_contact_start(self):
        pass

    def get_data(self, params={}, path_folder=''):
        if not params:
            print('EMTPY PARAMS')
            return 0
        
        # folder_name = path_folder + '0-exp-' if params['height'] == 27 else '1-exp-'
        folder_name = path_folder + '0-exp-' + params['color'] + '-Height' + str(params['height']) + '/'
        # folder_name += 'ImpLoop' + str(params['impedance_loop']) + '-'
        # folder_name += 'Height' + str(params['height']) + '/'
        
        file_name = 'opt-'
        file_name += 'kmp-vic-' if params['vic'] else 'kmp-'
        file_name += str(params['trial_idx']) + '.mat'

        import os
        print(os.getcwd()+'/'+folder_name+file_name)

        f = h5py.File(os.getcwd()+'/'+folder_name+file_name, 'r') 

        return np.array(f.get(params['data_to_plot']))[params['i_initial']:params['i_final']]

    def plot_data(self, data, show=True):
        plt.plot(data)
        if show:
            plt.show()