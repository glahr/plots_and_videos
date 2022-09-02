import matplotlib.pyplot as plt
import numpy as np
import h5py

FORCE_THRESHOLD = 3

class DataPlotHelper:
    def __init__(self, path_folder='') -> None:
        self.path_folder = path_folder
        self.idx_movement_starts = 0
        self.f_size = 20

        plt.rcParams.update({'font.size': self.f_size})
        plt.rcParams['text.usetex'] = True ## enable TeX style labels
        plt.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
    
    def copy_dict(self, my_dict):
        return {key: value for key, value in my_dict.items()}

    def get_idx_movement_starts(self, params):
        aux_params = self.copy_dict(params)
        aux_params['data_to_plot'] = 'EE_twist_d'
        data_aux = self.get_data(aux_params, axis=2)
        idx = np.nonzero(data_aux)[0][0]
        for i in range(idx, len(data_aux)):
            if abs(data_aux[i]) >= 1e-5:
                break
            idx += 1
        idx -= 1
        self.idx_movement_starts = idx
        return idx
        
    def get_idx_movement_ends(self, params, seconds=1.5):
        return self.idx_movement_starts + int(1000*seconds)

    def get_idx_contact_starts(self, params):
        aux_params = self.copy_dict(params)
        aux_params['data_to_plot'] = 'FT_ati'
        data_aux = self.get_data(aux_params)
        idx_forces_above_threshold = np.where(data_aux[self.idx_movement_starts:, 2]> FORCE_THRESHOLD)[0]  # this [0] gets only the indexes from np.where return
        idx_forces_above_threshold -= 5
        return idx_forces_above_threshold[0]


    def get_data(self, params={}, axis=0):
        if not params:
            print('EMTPY PARAMS')
            return 0
        
        # folder_name = path_folder + '0-exp-' if params['height'] == 27 else '1-exp-'
        folder_name = self.path_folder + params['color'] + '-Height' + str(params['height']) + '/'
        # folder_name += 'ImpLoop' + str(params['impedance_loop']) + '-'
        # folder_name += 'Height' + str(params['height']) + '/'
        
        file_name = 'opt-'
        file_name += 'kmp-vic-' if params['vic'] else 'kmp-'
        file_name += str(params['trial_idx']) + '.mat'

        import os

        f = h5py.File(os.getcwd()+'/'+folder_name+file_name, 'r') 

        return np.array(f.get(params['data_to_plot']))[params['i_initial']:params['i_final'], axis]

    def plot_data(self, data, show=True):
        plt.plot(data)
        if show:
            plt.show()
    
    def get_idx_from_file(self, params, data_info, idx_name=''):
        if idx_name == '':
            print('IDX EMPTY')
            return 0
        experiment_name = 'opt_kmp'
        experiment_name += '_vic' if params['vic'] else ''
        experiment_name += '_' + str(params['trial_idx'])

        idx = data_info.loc[(data_info['experiment_name'] == experiment_name) &
                                    (data_info['color'] == params['color']) &
                                    (data_info['height'] == params['height']), idx_name].values
        return idx[0]
    
    def set_axis(self, ax=None, fig=None, xlim_plot=None, xlabel=None, xticks=None, xtickslabels=None,
                                ylim_plot=None, ylabel=None, yticks=None, ytickslabels=None,
                                fig_size=[10,10], n_subplots=1):
        
        if ax is None or fig is None:
            fig, ax = plt.subplots(n_subplots,figsize=fig_size)
        
        if xlabel is not None:
            ax.set_xlabel(xlabel, size=self.f_size)
        
        if xlim_plot is not None:
            ax.set_xlim(xlim_plot)
        
        if xticks is not None:
            ax.set_xticks(xticks)
        
        if xtickslabels is not None:
            ax.set_xticklabels(xtickslabels)

        if ylabel is not None:
            ax.set_ylabel(ylabel, size=self.f_size)

        if ylim_plot is not None:
            ax.set_ylim(ylim_plot)

        if yticks is not None:
            ax.set_yticks(yticks)

        if ytickslabels is not None:
            ax.set_yticklabels(ytickslabels)

        ax.grid(which='major', alpha=0.2, linestyle='--')

        fig.tight_layout()
        return fig, ax
    
    def plot_single(self, time, data, fig, ax, color_shape='', lw=2):
        if '--' in color_shape:
            lw -= 0.5
        ax.plot(time, data, color_shape, linewidth=lw)
        return fig, ax

