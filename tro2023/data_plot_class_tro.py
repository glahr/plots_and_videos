import matplotlib.pyplot as plt
import numpy as np
import h5py
import os
from sys import exit

FORCE_THRESHOLD = 3

class DataPlotHelper:
    def __init__(self, path_folder='', files_names='') -> None:
        self.path_folder = path_folder
        self.idx_movement_starts = 0
        self.f_size = 20
        self.lw = 3
        self.files_names = files_names

        plt.rcParams.update({'font.size': self.f_size})
        plt.rcParams['text.usetex'] = True ## enable TeX style labels
        plt.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
    
    def copy_dict(self, my_dict):
        return {key: value for key, value in my_dict.items()}

    def get_idx_movement_starts(self, params, file_name):
        # # sync by impact
        aux_params = self.copy_dict(params)
        idx = self.get_idx_contact_starts(params, file_name)
        if 'fp' in file_name:
            idx -= 500
        else:
            idx -= 500

        # # sync by drop
        # aux_params = self.copy_dict(params)
        # aux_params['data_to_plot'] = 'ball_pose_'
        # data_aux = self.get_data(params=aux_params, file_name=file_name)[:,2]
        # idx = np.where(data_aux != 0)
        # if idx[0].size == 0:
        #     idx = self.get_idx_contact_starts(params, file_name)
        # else:
        #     idx = idx[0][0]
        #     # for i in range(idx, len(data_aux)):
        #     #     if abs(data_aux[i]) >= 1e-5:
        #     #         break
        #     #     idx += 1
        #     # avg = np.mean(data_aux[idx:idx+150])
        #     # std = np.std(data_aux[idx:idx+150])
        #     # print('avg  = ', avg)
        #     # print('std  = ', std)
        #     for i in range(idx, len(data_aux)):
        #         if data_aux[i] <= 0.665:
        #             break
        #         idx += 1
        # idx -= 200
        self.idx_movement_starts = idx
        return idx
        
    def get_idx_movement_ends(self, params, data_size, file_name='', seconds=1.5):
        idx_end = self.idx_movement_starts + int(1000*seconds)
        if idx_end > data_size:
            idx_end = data_size
        return idx_end

    def get_idx_contact_starts(self, params, file_name=''):
        aux_params = self.copy_dict(params)
        aux_params['data_to_plot'] = 'ft_'
        data_aux = self.get_data(aux_params, file_name=file_name, data_to_plot='ft_')
        idx_forces_above_threshold = np.where(data_aux[:, 2] < -FORCE_THRESHOLD)[0]  # this [0] gets only the indexes from np.where return
        return idx_forces_above_threshold[0]

    def _get_name(self, file_name):
        file_name = [s for s in self.files_names if str(file_name) in s][0]
        return file_name

    def get_data(self, params={}, file_name=None, data_to_plot=None):
        file_name = os.getcwd()+'/'+ self.path_folder + file_name
        # print(file_name)
        # try:
        f = h5py.File(file_name, 'r')
        if data_to_plot is None:
            data_to_plot = params['data_to_plot']
        data = np.array(f.get(data_to_plot))
        # print(f.keys())
        # print('data', data.size)
        # print('data', data[params['idx_initial']:params['idx_end']])
        # print(params)
        # print(params['idx_initial'], params['idx_end'])
        return data[params['idx_initial']:params['idx_end']]
        # except FileNotFoundError:
        #     print("WRONG FILE. CHECK INPUTS:")
        #     print([k + "=" + str(v) for k, v in zip(params.keys(), params.values())])
        #     exit()
    
    def plot_data(self, data, show=True):
        plt.plot(data)
        if show:
            plt.show()
    
    def get_idx_from_file(self, idx_exp, data_info, idx_name=''):
        # print(data_info)
        # print(data_info.loc[(data_info['id'] == idx_exp), idx_name])
        idx_exp -= 1
        idx = data_info.loc[(data_info['id'] == idx_exp), idx_name].values
        return idx[0]
    
    def set_axis(self, ax=None, fig=None, xlim_plot=None, xlabel=None, xticks=None, xtickslabels=None,
                                ylim_plot=None, ylabel=None, yticks=None, ytickslabels=None,
                                fig_size=[10,10], n_subplots=1):
        
        if ax is None or fig is None:
            fig, ax = plt.subplots(n_subplots,figsize=fig_size)
        
        if xlabel is not None:
            ax.set_xlabel(xlabel, size=self.f_size)
        elif xlabel == '':
            ax.set_xlabel(xlabel, size=self.f_size/10)
        
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
    
    def plot_single(self, time, data, fig=None, ax=None, shape='', color=None, lw=3):
        if '--' in shape:
            lw = self.lw - 0.5
        else:
            lw = self.lw
        ax.plot(time, data, shape, color=color, linewidth=lw)
        return fig, ax

