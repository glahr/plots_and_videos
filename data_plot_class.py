import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from scipy.io import loadmat
import numpy as np
import h5py
import os

FORCE_THRESHOLD = 3

class DataPlotHelper:
    def __init__(self, path_folder='', files_names='', window_earlier_than_500=50, end_poc=1000, f_size=20) -> None:
        self.path_folder = path_folder
        self.idx_movement_starts = 0
        self.f_size = f_size
        self.lw = 3
        self.files_names = files_names
        self.window_earlier_than_500 = window_earlier_than_500
        self.start_prc = window_earlier_than_500
        self.end_prc = window_earlier_than_500+500
        self.start_poc = window_earlier_than_500+500
        self.end_poc = window_earlier_than_500+end_poc
        self.adjust_start = False

        plt.rcParams.update({'font.size': self.f_size})
        plt.rcParams['text.usetex'] = True ## enable TeX style labels
        plt.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
    
    def copy_dict(self, my_dict):
        return {key: value for key, value in my_dict.items()}

    def get_idx_fz_max(self, file_name=''):
        data_aux = self.get_data(file_name=file_name, data_to_plot='ft_sensor')
        return np.argmin(data_aux[:, 2])

    def _get_name(self, params):
        file_name = ''
        for f in self.files_names:
            if params['subject'] in f[:19] and 'e'+str(params['exp_idx']) in f[:19] and params['exp_type'] in f[:19]:
                file_name = f
                break
        return file_name

    def get_data(self, file_name=None, data_to_plot=None):
        file_name = os.getcwd()+'/'+ self.path_folder + file_name
        f = h5py.File(file_name, 'r')
        data = np.array(f.get(data_to_plot))
        assert np.size(data[0]) > 0
        return data
    
    def get_nata_tlx_data(self, subject):
        file_name = os.getcwd()+'/'+ subject + '/Nasa_tlx.mat'
        data = loadmat(file_name)
        WL_HL = 0
        WL_RL = 0
        tlx_scale = np.array([[int(j) for j in i] for i in data['nasa_tlx_scale']])
        tlx_importance = np.array([i for i in data['nasa_tlx_importance']])
        WL_HL =  tlx_importance.T.dot(tlx_scale[:,0])/15
        WL_RL = tlx_importance.T.dot(tlx_scale[:,1])/15
        return WL_HL[0], WL_RL[0]
    
    def plot_data(self, data, show=True):
        plt.plot(data)
        if show:
            plt.show()
    
    def get_idx_start(self, file_name):
        return self.get_idx_fz_max(file_name=file_name) - 500 - self.window_earlier_than_500
        # return self.get_idx_obj_drop_starts()

    # def get_idx_obj_drop_starts(self, params, file_name):
    #     aux_params = self.copy_dict(params)
    #     aux_params['data_to_plot'] = 'OBJ_position'
    #     data_aux = self.get_data(aux_params, file_name=file_name, data_to_plot='ft_sensor')
    #     (np.abs(divs_rob - pi_rob[k])).argmin()
    #     return
    
    def get_idx_end(self, file_name, idx_initial):
        return idx_initial + 2000 + self.window_earlier_than_500

    def remove_bias(self, data):
        data -= np.mean(data[:50])
        return data

    def check_and_adjust_length(self, data, idx_initial, idx_end):
        if idx_end - idx_initial > len(data):
            n = (idx_end - idx_initial) - len(data)
            for _ in range(n):
                data = np.concatenate([data, [data[-1]]])
        return data
    
    def check_and_adjust_length_time(self, time, idx_initial, idx_end, adjust_start=False):
        if idx_end - idx_initial > len(time):
            n = (idx_end - idx_initial) - len(time)
            for _ in range(n):
                time = np.concatenate([time, [time[-1]+(time[-1]-time[-2])]])
        return time
    
    def smooth_plot(self, data):
        return savgol_filter(data, 50, 5)
        
    
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

        plt.rcParams.update({'font.size': self.f_size})
        plt.rcParams['text.usetex'] = True ## enable TeX style labels
        plt.rcParams['text.latex.preamble'] = r"\usepackage{amsmath}"
        return fig, ax
    
    def plot_single(self, time, data, fig=None, ax=None, shape='', color=None, lw=3):
        if '--' in shape:
            lw = self.lw - 0.5
        else:
            lw = self.lw
        ax.plot(time, data, shape, color=color, linewidth=lw)
        return fig, ax

