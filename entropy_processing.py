import os
import dit
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from data_plot_class import DataPlotHelper
from entropy_utils_jpype import calc_ais, calc_te_and_local_te, \
                                calc_cte, get_pmf, get_init_drop_idx, \
                                optim_te_destination_only, optim_te_source, optim_delay_u
from collections import Counter

def load_blindfolded_data():
    subjects_blindfolded = ['s02s07',
                            's10s03',
                            's06s09',
                            's08s04',
                            's07s02',
                            's03s10',
                            's09s06',
                            's04s08',
                            's05s01',
                            's01s05',]
    
    idx_split = {subs: {'start': [], 'end': []} for subs in subjects_blindfolded}

    idx_split['s02s07']['start'] = [4850+600+50, 25600+400+50, 33900+500+50, 43250+50, 51500+50, 64900+200+50]
    idx_split['s10s03']['start'] = [18110+50, 30750, 41310+200, 50770+200, 59550+600, 76060+250]
    idx_split['s06s09']['start'] = [17500, 28950, 40060, 50450, 59090, 69750]
    idx_split['s08s04']['start'] = [7630, 16430, 25030, 37300, 48080, 58420]
    idx_split['s05s01']['start'] = [9150, 19200, 30470, 39460, 48260, 58950]

    idx_split['s07s02']['start'] = [9600, 17000, 24500, 31150, 38880, 46170]
    idx_split['s03s10']['start'] = [3700, 14000, 23000, 31100, 40600, 50000]
    idx_split['s09s06']['start'] = [6100, 16500, 25600, 37300, 48200, 58300]
    idx_split['s04s08']['start'] = [7480, 19560, 29880, 46850, 59300, 70820]
    idx_split['s01s05']['start'] = [7070, 15790, 23750, 47280, 57430, 67470]


    # idx_split['s02s07']['end'] = [0, 0, 0, 0, 0, 0]
    # idx_split['s10s03']['end'] = [0, 0, 0, 0, 0, 0]
    # idx_split['s06s09']['end'] = [0, 0, 0, 0, 0, 0]
    # idx_split['s08s04']['end'] = [0, 0, 0, 0, 0, 0]

    for subs in subjects_blindfolded:
        idx_split[subs]['end'] = [a + 1600 for a in idx_split[subs]['start']]
    
    pos_obj_dict = []
    pos_hum_dict = []
    pos_rob_dict = []
    vel_obj_dict = []
    vel_hum_dict = []
    vel_rob_dict = []
    fz_dict = []
            
    for subs in subjects_blindfolded:
        
        path_folder = 'blindfolded/data_filtered/'
        files_names = os.listdir(path=path_folder)

        dh = DataPlotHelper(path_folder=path_folder, files_names=files_names)

        files_with_exp_type = [fn for fn in files_names if subs in fn]
        files_with_exp_type.sort()

        for file_name in files_with_exp_type:
            with open(path_folder+file_name[:-4]+'.pkl', 'rb') as f:
                data = pickle.load(f)

            print(file_name)
    
            for i_hat_start, i_hat_end in zip(idx_split[subs]['start'], idx_split[subs]['end']):
                idx_start_offset = get_init_drop_idx(data['pos_obj'][i_hat_start:i_hat_end])

                idx_start = i_hat_start + idx_start_offset - 600
                idx_end = i_hat_end + idx_start_offset + 600

                pos_obj_dict.append(data['pos_obj'][idx_start:idx_end])
                pos_hum_dict.append(data['pos_hum'][idx_start:idx_end])
                pos_rob_dict.append(data['pos_rob'][idx_start:idx_end])

                vel_obj_dict.append(data['vel_obj'][idx_start:idx_end])
                vel_hum_dict.append(data['vel_hum'][idx_start:idx_end])
                vel_rob_dict.append(data['vel_rob'][idx_start:idx_end])

                fz_dict.append(data['fz'][idx_start:idx_end])
    
    return pos_obj_dict, pos_hum_dict, pos_rob_dict, vel_obj_dict, vel_hum_dict, vel_rob_dict, fz_dict


def load_data_by_exp_type(exp_type, ti_max=1000):
    if exp_type == 'HB':
        return load_blindfolded_data()

    pos_obj_dict = []
    pos_hum_dict = []
    pos_rob_dict = []
    vel_obj_dict = []
    vel_hum_dict = []
    vel_rob_dict = []
    fz_dict = []

    path_folder = 'data_filtered/'
    files_names = os.listdir(path=path_folder)

    dh = DataPlotHelper(path_folder=path_folder, files_names=files_names)
    files_with_exp_type = [a for a in files_names if exp_type in a]
    files_with_exp_type.sort()

    for file_name in files_with_exp_type:
        with open(path_folder+file_name, 'rb') as f:
            data = pickle.load(f)
        
        print(file_name)
        idx = get_init_drop_idx(data['pos_obj'])
        
        phase_steps = slice(idx-600, idx+ti_max+600)

        if 'franka_a_s07_HH_e03__2023_06_13__16_32_19' in file_name:
            data['pos_obj'][:550] = np.ones_like(data['pos_obj'][:550])*data['pos_obj'][550]
            data['pos_hum'][:550] = np.ones_like(data['pos_hum'][:550])*data['pos_hum'][550]
            data['pos_rob'][:550] = np.ones_like(data['pos_rob'][:550])*data['pos_rob'][550]

            data['vel_obj'][:550] = np.ones_like(data['vel_obj'][:550])*data['vel_obj'][550]
            data['vel_hum'][:550] = np.ones_like(data['vel_hum'][:550])*data['vel_hum'][550]
            data['vel_rob'][:550] = np.ones_like(data['vel_rob'][:550])*data['vel_rob'][550]

            idx = get_init_drop_idx(data['pos_obj']) 
            phase_steps = slice(idx-600, idx+ti_max+600)

        pos_obj_dict.append(data['pos_obj'][phase_steps])
        pos_hum_dict.append(data['pos_hum'][phase_steps])
        pos_rob_dict.append(data['pos_rob'][phase_steps])

        vel_obj_dict.append(data['vel_obj'][phase_steps])
        vel_hum_dict.append(data['vel_hum'][phase_steps])
        vel_rob_dict.append(data['vel_rob'][phase_steps])

        fz_dict.append(data['fz'][phase_steps])

    return pos_obj_dict, pos_hum_dict, pos_rob_dict, vel_obj_dict, vel_hum_dict, vel_rob_dict, fz_dict


def calculate_info_dynamics(w_r, w_k, w_l, delay_u, exps_types, ti_max=1000, n_bins=10):
    entropies_dict = {'hum': 0,
                      'rob': 0,
                      'obj': 0,
                      'joint_obj_hum': 0,
                      'joint_rob_hum': 0,
                      'joint_obj_rob': 0,
                      'cond_hum_given_obj': 0,
                      'cond_hum_given_rob': 0,
                      'cond_rob_given_obj': 0,
                      'cond_obj_given_hum': 0,
                      'cond_rob_given_hum': 0,
                      'cond_obj_given_rob': 0,

                      'ais_hum': 0,
                      'ais_rob': 0,
                      'ais_obj': 0,
                      
                      'te_obj_to_hum': 0,
                      'te_rob_to_hum': 0,
                      'te_obj_to_rob': 0,
                      'te_hum_to_obj': 0,
                      'te_hum_to_rob': 0,
                      'te_rob_to_obj': 0,

                      'te_obj_to_hum_multi': 0,

                      'te_obj_to_hum_avg': 0,
                      'te_rob_to_hum_avg': 0,
                      'te_obj_to_rob_avg': 0,
                      'te_hum_to_obj_avg': 0,
                      'te_hum_to_rob_avg': 0,
                      'te_rob_to_obj_avg': 0,

                      'te_obj_to_hum_subs': 0,
                      'te_rob_to_hum_subs': 0,
                      'te_obj_to_rob_subs': 0,
                      'te_hum_to_obj_subs': 0,
                      'te_hum_to_rob_subs': 0,
                      'te_rob_to_obj_subs': 0,

                      'local_te_obj_to_hum': 0,
                      'local_te_rob_to_hum': 0,
                      'local_te_obj_to_rob': 0,
                      'local_te_hum_to_obj': 0,
                      'local_te_hum_to_rob': 0,
                      'local_te_rob_to_obj': 0,

                      'local_te_obj_to_hum_avg': 0,
                      'local_te_rob_to_hum_avg': 0,
                      'local_te_obj_to_rob_avg': 0,
                      'local_te_hum_to_obj_avg': 0,
                      'local_te_hum_to_rob_avg': 0,
                      'local_te_rob_to_obj_avg': 0,

                      'te_obj_to_hum_std': 0,
                      'te_rob_to_hum_std': 0,
                      'te_obj_to_rob_std': 0,
                      'te_hum_to_obj_std': 0,
                      'te_hum_to_rob_std': 0,
                      'te_rob_to_obj_std': 0,

                      'cte_obj_to_hum_cond_rob': 0,
                      'cte_rob_to_hum_cond_obj': 0,
                      'cte_obj_to_rob_cond_hum': 0,
                      'cte_hum_to_obj_cond_rob': 0,
                      'cte_hum_to_rob_cond_obj': 0,
                      'cte_rob_to_obj_cond_hum': 0,

                      'cte_obj_to_hum_cond_rob_avg': 0,
                      'cte_rob_to_hum_cond_obj_avg': 0,
                      'cte_obj_to_rob_cond_hum_avg': 0,
                      'cte_hum_to_obj_cond_rob_avg': 0,
                      'cte_hum_to_rob_cond_obj_avg': 0,
                      'cte_rob_to_obj_cond_hum_avg': 0,

                      'cte_obj_to_hum_cond_rob_subs': 0,
                      'cte_rob_to_hum_cond_obj_subs': 0,
                      'cte_obj_to_rob_cond_hum_subs': 0,
                      'cte_hum_to_obj_cond_rob_subs': 0,
                      'cte_hum_to_rob_cond_obj_subs': 0,
                      'cte_rob_to_obj_cond_hum_subs': 0,

                      'local_cte_obj_to_hum_cond_rob': 0,
                      'local_cte_rob_to_hum_cond_obj': 0,
                      'local_cte_obj_to_rob_cond_hum': 0,
                      'local_cte_hum_to_obj_cond_rob': 0,
                      'local_cte_hum_to_rob_cond_obj': 0,
                      'local_cte_rob_to_obj_cond_hum': 0,

                      'mutual_obj_hum': 0,
                      'mutual_rob_hum': 0,
                      'mutual_obj_rob': 0,
                      'pos_obj': 0,
                      'pos_hum': 0,
                      'pos_rob': 0,
                      'pos_obj_dict': 0,
                      'pos_hum_dict': 0,
                      'pos_rob_dict': 0,
                      'fz_conv': 0,
                      'fz': 0,
                      'd_obj_hum': 0,
                      'd_rob_hum': 0,
                      'd_obj_rob': 0,
                      'd_hum_given_obj': 0,
                      'd_hum_given_rob': 0,
                      'd_rob_given_obj': 0,
                      'd_obj_given_hum': 0,
                      'd_rob_given_hum': 0,
                      'd_obj_given_rob': 0,
                      }    

    pos_obj_dict = {exp_type: [] for exp_type in exps_types}
    pos_hum_dict = {exp_type: [] for exp_type in exps_types}
    pos_rob_dict = {exp_type: [] for exp_type in exps_types}
    vel_obj_dict = {exp_type: [] for exp_type in exps_types}
    vel_hum_dict = {exp_type: [] for exp_type in exps_types}
    vel_rob_dict = {exp_type: [] for exp_type in exps_types}
    fz_dict = {exp_type: [] for exp_type in exps_types}
    dh = DataPlotHelper()

    # k_hist = 200
    # k_tau  = 0
    # l_hist = 200
    # l_tau  = 0
    # delay  = 100

    k_hist = 2
    k_tau  = 3
    l_hist = 2
    l_tau  = 3
    u  = 0
            
    for i, exp_type in enumerate(exps_types):

        pos_obj_dict[exp_type], pos_hum_dict[exp_type], pos_rob_dict[exp_type],  vel_obj_dict[exp_type], vel_hum_dict[exp_type], vel_rob_dict[exp_type], fz_dict[exp_type] = load_data_by_exp_type(exp_type=exp_type)

        entropies_all = {exp_type: {e: [] for e in entropies_dict} for exp_type in exps_types}

        for ti in range(600, ti_max+600):
            if (ti%100) == 0:
                print('ws', w_r, '\tdelay_u', delay_u, '\tsteps', str(ti-600)+'/'+str(ti_max))
                print('n_bins = ', n_bins)

            slice_current = slice(ti-w_k, ti)
            slice_past = slice(ti-w_l-delay_u, ti-delay_u)

            pos_obj = np.array(pos_obj_dict[exp_type], dtype='float64')[:, slice_past]
            pos_hum = np.array(pos_hum_dict[exp_type], dtype='float64')[:, slice_current]
            n_bins = int(np.sqrt(pos_obj.size))
            # vel_obj = np.array(vel_obj_dict[exp_type], dtype='float64')[:, slice_past]
            # vel_hum = np.array(vel_hum_dict[exp_type], dtype='float64')[:, slice_current]
            d_hum_given_obj = get_pmf(data_x=pos_obj, data_y=pos_hum, n_bins=n_bins)
            entropies_all[exp_type]['cond_hum_given_obj'].append(dit.multivariate.entropy(d_hum_given_obj, rvs=[1], crvs=[0]))
            
            pos_rob = np.array(pos_rob_dict[exp_type], dtype='float64')[:, slice_past]
            pos_hum = np.array(pos_hum_dict[exp_type], dtype='float64')[:, slice_current]
            d_hum_given_rob = get_pmf(data_x=pos_rob, data_y=pos_hum, n_bins=n_bins)
            entropies_all[exp_type]['cond_hum_given_rob'].append(dit.multivariate.entropy(d_hum_given_rob, rvs=[1], crvs=[0]))
            
            pos_obj = np.array(pos_obj_dict[exp_type], dtype='float64')[:, slice_past]
            pos_rob = np.array(pos_rob_dict[exp_type], dtype='float64')[:, slice_current]
            d_rob_given_obj = get_pmf(data_x=pos_obj, data_y=pos_rob, n_bins=n_bins)
            entropies_all[exp_type]['cond_rob_given_obj'].append(dit.multivariate.entropy(d_rob_given_obj, rvs=[1], crvs=[0]))
            
            ###############
            pos_hum = np.array(pos_hum_dict[exp_type], dtype='float64')[:, slice_past]
            pos_obj = np.array(pos_obj_dict[exp_type], dtype='float64')[:, slice_current]
            d_obj_given_hum = get_pmf(data_x=pos_hum, data_y=pos_obj, n_bins=n_bins)
            entropies_all[exp_type]['cond_obj_given_hum'].append(dit.multivariate.entropy(d_obj_given_hum, rvs=[1], crvs=[0]))
            
            pos_hum = np.array(pos_hum_dict[exp_type], dtype='float64')[:, slice_past]
            pos_rob = np.array(pos_rob_dict[exp_type], dtype='float64')[:, slice_current]
            d_rob_given_hum = get_pmf(data_x=pos_hum, data_y=pos_rob, n_bins=n_bins)
            entropies_all[exp_type]['cond_rob_given_hum'].append(dit.multivariate.entropy(d_rob_given_hum, rvs=[1], crvs=[0]))
            
            pos_rob = np.array(pos_rob_dict[exp_type], dtype='float64')[:, slice_past]
            pos_obj = np.array(pos_obj_dict[exp_type], dtype='float64')[:, slice_current]
            d_obj_given_rob = get_pmf(data_x=pos_rob, data_y=pos_obj, n_bins=n_bins)
            entropies_all[exp_type]['cond_obj_given_rob'].append(dit.multivariate.entropy(d_obj_given_rob, rvs=[1], crvs=[0]))
            
            ###############
            ## INFORMATION DYNAMICS
            slice_current = slice(ti-w_r, ti)
            pos_obj = np.array(pos_obj_dict[exp_type], dtype='float64')[:, slice_current]
            pos_hum = np.array(pos_hum_dict[exp_type], dtype='float64')[:, slice_current]
            pos_rob = np.array(pos_rob_dict[exp_type], dtype='float64')[:, slice_current]
            fz = np.array(fz_dict[exp_type], dtype='float64')[:, slice_current]

            te_obj_to_hum, local_te_obj_to_hum  = calc_te_and_local_te(sourceArray=pos_obj.tolist(), destArray=pos_hum.tolist(), k_hist=k_hist, k_tau=k_tau, l_hist=l_hist, l_tau=l_tau, delay=u)
            te_rob_to_hum, local_te_rob_to_hum  = calc_te_and_local_te(sourceArray=pos_rob.tolist(), destArray=pos_hum.tolist(), k_hist=k_hist, k_tau=k_tau, l_hist=l_hist, l_tau=l_tau, delay=u)
            te_obj_to_rob, local_te_obj_to_rob  = calc_te_and_local_te(sourceArray=pos_obj.tolist(), destArray=pos_rob.tolist(), k_hist=k_hist, k_tau=k_tau, l_hist=l_hist, l_tau=l_tau, delay=u)
            te_hum_to_obj, local_te_hum_to_obj  = calc_te_and_local_te(sourceArray=pos_hum.tolist(), destArray=pos_obj.tolist(), k_hist=k_hist, k_tau=k_tau, l_hist=l_hist, l_tau=l_tau, delay=u)
            te_hum_to_rob, local_te_hum_to_rob  = calc_te_and_local_te(sourceArray=pos_hum.tolist(), destArray=pos_rob.tolist(), k_hist=k_hist, k_tau=k_tau, l_hist=l_hist, l_tau=l_tau, delay=u)
            te_rob_to_obj, local_te_rob_to_obj  = calc_te_and_local_te(sourceArray=pos_rob.tolist(), destArray=pos_obj.tolist(), k_hist=k_hist, k_tau=k_tau, l_hist=l_hist, l_tau=l_tau, delay=u)

            entropies_all[exp_type]['te_obj_to_hum'].append(te_obj_to_hum)
            entropies_all[exp_type]['te_rob_to_hum'].append(te_rob_to_hum)
            entropies_all[exp_type]['te_obj_to_rob'].append(te_obj_to_rob)
            entropies_all[exp_type]['te_hum_to_obj'].append(te_hum_to_obj)
            entropies_all[exp_type]['te_hum_to_rob'].append(te_hum_to_rob)
            entropies_all[exp_type]['te_rob_to_obj'].append(te_rob_to_obj)

            entropies_all[exp_type]['local_te_obj_to_hum_avg'].append(np.mean(np.array(local_te_obj_to_hum)))
            entropies_all[exp_type]['local_te_rob_to_hum_avg'].append(np.mean(np.array(local_te_rob_to_hum)))
            entropies_all[exp_type]['local_te_obj_to_rob_avg'].append(np.mean(np.array(local_te_obj_to_rob)))
            entropies_all[exp_type]['local_te_hum_to_obj_avg'].append(np.mean(np.array(local_te_hum_to_obj)))
            entropies_all[exp_type]['local_te_hum_to_rob_avg'].append(np.mean(np.array(local_te_hum_to_rob)))
            entropies_all[exp_type]['local_te_rob_to_obj_avg'].append(np.mean(np.array(local_te_rob_to_obj)))

            # entropies_all[exp_type]['cte_obj_to_hum_cond_rob'].append(calc_cte(sourceArray=pos_obj.tolist(), destArray=pos_hum.tolist(), condArray=pos_rob.tolist(), k_hist=1, k_tau=1, l_hist=6, l_tau=1, delay=0))
            entropies_all[exp_type]['cte_rob_to_hum_cond_obj'].append(calc_cte(sourceArray=pos_rob.tolist(), destArray=pos_hum.tolist(), condArray=pos_obj.tolist(), k_hist=1, k_tau=1, l_hist=6, l_tau=1, delay=0))
            # entropies_all[exp_type]['cte_obj_to_rob_cond_hum'].append(calc_cte(sourceArray=pos_obj.tolist(), destArray=pos_rob.tolist(), condArray=pos_hum.tolist(), k_hist=1, k_tau=1, l_hist=6, l_tau=1, delay=0))
            # entropies_all[exp_type]['cte_hum_to_obj_cond_rob'].append(calc_cte(sourceArray=pos_hum.tolist(), destArray=pos_obj.tolist(), condArray=pos_rob.tolist(), k_hist=1, k_tau=1, l_hist=6, l_tau=1, delay=0))
            entropies_all[exp_type]['cte_hum_to_rob_cond_obj'].append(calc_cte(sourceArray=pos_hum.tolist(), destArray=pos_rob.tolist(), condArray=pos_obj.tolist(), k_hist=1, k_tau=1, l_hist=6, l_tau=1, delay=0))
            # entropies_all[exp_type]['cte_rob_to_obj_cond_hum'].append(calc_cte(sourceArray=pos_rob.tolist(), destArray=pos_obj.tolist(), condArray=pos_hum.tolist(), k_hist=1, k_tau=1, l_hist=6, l_tau=1, delay=0))

            ###############
            ## INFORMATION THEORY METRICS
            d_obj_hum = get_pmf(data_x=pos_obj, data_y=pos_hum, n_bins=n_bins)
            d_rob_hum = get_pmf(data_x=pos_rob, data_y=pos_hum, n_bins=n_bins)
            d_obj_rob = get_pmf(data_x=pos_obj, data_y=pos_rob, n_bins=n_bins)

            entropies_all[exp_type]['mutual_obj_hum'].append(dit.shannon.mutual_information(d_obj_hum, [0], [1], rv_mode='indexes'))
            entropies_all[exp_type]['mutual_rob_hum'].append(dit.shannon.mutual_information(d_rob_hum, [0], [1], rv_mode='indexes'))
            entropies_all[exp_type]['mutual_obj_rob'].append(dit.shannon.mutual_information(d_obj_rob, [0], [1], rv_mode='indexes'))

            entropies_all[exp_type]['joint_obj_hum'].append(dit.multivariate.entropy(d_obj_hum, [0,1]))
            entropies_all[exp_type]['joint_rob_hum'].append(dit.multivariate.entropy(d_rob_hum, [0,1]))
            entropies_all[exp_type]['joint_obj_rob'].append(dit.multivariate.entropy(d_obj_rob, [0,1]))

        ###############
        ## LOGGING INFO
        pos_obj = np.array(pos_obj_dict[exp_type], dtype='float64')[:, 600:1600]
        pos_hum = np.array(pos_hum_dict[exp_type], dtype='float64')[:, 600:1600]
        pos_rob = np.array(pos_rob_dict[exp_type], dtype='float64')[:, 600:1600]
        fz = np.array([dh.remove_bias(data=a) for a in fz_dict[exp_type]], dtype='float64')[:, 600:1600]

        entropies_all[exp_type]['pos_obj'] = pos_obj
        entropies_all[exp_type]['pos_hum'] = pos_hum
        entropies_all[exp_type]['pos_rob'] = pos_rob
        entropies_all[exp_type]['fz'] = fz
        
        with open('data/entropy_'+exp_type+'_'+'_ws'+str(w_r)+'_delay'+str(delay_u)+'.pkl', 'wb') as f:
            print('saving entropy_'+exp_type+'_ws'+str(w_r)+'_delay'+str(delay_u)+'.pkl')
            pickle.dump(entropies_all[exp_type], f)

    return


def calculate_info_dynamics_increasing_window(w_r, w_k, w_l, shift, exps_types, ti_max=1000, n_bins=10):
    entropies_dict = {'hum': 0,
                      'rob': 0,
                      'obj': 0,
                      'joint_obj_hum': 0,
                      'joint_rob_hum': 0,
                      'joint_obj_rob': 0,
                      'cond_hum_given_obj': 0,
                      'cond_hum_given_rob': 0,
                      'cond_rob_given_obj': 0,
                      'cond_obj_given_hum': 0,
                      'cond_rob_given_hum': 0,
                      'cond_obj_given_rob': 0,

                      'ais_hum': 0,
                      'ais_rob': 0,
                      'ais_obj': 0,
                      
                      'te_obj_to_hum': 0,
                      'te_rob_to_hum': 0,
                      'te_obj_to_rob': 0,
                      'te_hum_to_obj': 0,
                      'te_hum_to_rob': 0,
                      'te_rob_to_obj': 0,

                      'te_obj_to_hum_multi': 0,

                      'te_obj_to_hum_avg': 0,
                      'te_rob_to_hum_avg': 0,
                      'te_obj_to_rob_avg': 0,
                      'te_hum_to_obj_avg': 0,
                      'te_hum_to_rob_avg': 0,
                      'te_rob_to_obj_avg': 0,

                      'te_obj_to_hum_subs': 0,
                      'te_rob_to_hum_subs': 0,
                      'te_obj_to_rob_subs': 0,
                      'te_hum_to_obj_subs': 0,
                      'te_hum_to_rob_subs': 0,
                      'te_rob_to_obj_subs': 0,

                      'local_te_obj_to_hum': 0,
                      'local_te_rob_to_hum': 0,
                      'local_te_obj_to_rob': 0,
                      'local_te_hum_to_obj': 0,
                      'local_te_hum_to_rob': 0,
                      'local_te_rob_to_obj': 0,

                      'local_te_obj_to_hum_avg': 0,
                      'local_te_rob_to_hum_avg': 0,
                      'local_te_obj_to_rob_avg': 0,
                      'local_te_hum_to_obj_avg': 0,
                      'local_te_hum_to_rob_avg': 0,
                      'local_te_rob_to_obj_avg': 0,

                      'te_obj_to_hum_std': 0,
                      'te_rob_to_hum_std': 0,
                      'te_obj_to_rob_std': 0,
                      'te_hum_to_obj_std': 0,
                      'te_hum_to_rob_std': 0,
                      'te_rob_to_obj_std': 0,

                      'cte_obj_to_hum_cond_rob': 0,
                      'cte_rob_to_hum_cond_obj': 0,
                      'cte_obj_to_rob_cond_hum': 0,
                      'cte_hum_to_obj_cond_rob': 0,
                      'cte_hum_to_rob_cond_obj': 0,
                      'cte_rob_to_obj_cond_hum': 0,

                      'cte_obj_to_hum_cond_rob_avg': 0,
                      'cte_rob_to_hum_cond_obj_avg': 0,
                      'cte_obj_to_rob_cond_hum_avg': 0,
                      'cte_hum_to_obj_cond_rob_avg': 0,
                      'cte_hum_to_rob_cond_obj_avg': 0,
                      'cte_rob_to_obj_cond_hum_avg': 0,

                      'cte_obj_to_hum_cond_rob_subs': 0,
                      'cte_rob_to_hum_cond_obj_subs': 0,
                      'cte_obj_to_rob_cond_hum_subs': 0,
                      'cte_hum_to_obj_cond_rob_subs': 0,
                      'cte_hum_to_rob_cond_obj_subs': 0,
                      'cte_rob_to_obj_cond_hum_subs': 0,

                      'local_cte_obj_to_hum_cond_rob': 0,
                      'local_cte_rob_to_hum_cond_obj': 0,
                      'local_cte_obj_to_rob_cond_hum': 0,
                      'local_cte_hum_to_obj_cond_rob': 0,
                      'local_cte_hum_to_rob_cond_obj': 0,
                      'local_cte_rob_to_obj_cond_hum': 0,

                      'mutual_obj_hum': 0,
                      'mutual_rob_hum': 0,
                      'mutual_obj_rob': 0,
                      'pos_obj': 0,
                      'pos_hum': 0,
                      'pos_rob': 0,
                      'pos_obj_dict': 0,
                      'pos_hum_dict': 0,
                      'pos_rob_dict': 0,
                      'fz_conv': 0,
                      'fz': 0,
                      'd_obj_hum': 0,
                      'd_rob_hum': 0,
                      'd_obj_rob': 0,
                      'd_hum_given_obj': 0,
                      'd_hum_given_rob': 0,
                      'd_rob_given_obj': 0,
                      'd_obj_given_hum': 0,
                      'd_rob_given_hum': 0,
                      'd_obj_given_rob': 0,
                      }    

    pos_obj_dict = {exp_type: [] for exp_type in exps_types}
    pos_hum_dict = {exp_type: [] for exp_type in exps_types}
    pos_rob_dict = {exp_type: [] for exp_type in exps_types}
    vel_obj_dict = {exp_type: [] for exp_type in exps_types}
    vel_hum_dict = {exp_type: [] for exp_type in exps_types}
    vel_rob_dict = {exp_type: [] for exp_type in exps_types}
    fz_dict = {exp_type: [] for exp_type in exps_types}
    dh = DataPlotHelper()

    k_hist = 2
    k_tau  = 3
    l_hist = 2
    l_tau  = 3
    delay  = 0

    k_hist_cte = 2
    k_tau_cte  = 3
    l_hist_cte = 10
    l_tau_cte  = 20
    m_hist_cte = 2
    m_tau_cte  = 1
    delay_u_cte  = 0
    delay_v_cte  = 130
            
    for i, exp_type in enumerate(exps_types):

        pos_obj_dict[exp_type], pos_hum_dict[exp_type], pos_rob_dict[exp_type],  vel_obj_dict[exp_type], vel_hum_dict[exp_type], vel_rob_dict[exp_type], fz_dict[exp_type] = load_data_by_exp_type(exp_type=exp_type)

        entropies_all = {exp_type: {e: [] for e in entropies_dict} for exp_type in exps_types}

        for i, ti in enumerate(range(600, ti_max+600)):
            if (ti%100) == 0:
                print('ws', w_r, '\tdelay_u', shift, '\tsteps', str(ti-600)+'/'+str(ti_max))
                print('n_bins = ', n_bins)

            idx_k = min(i, w_k)
            idx_l = min(i, w_l)

            # slice_current = slice(ti-w_k, ti)
            # slice_past = slice(ti-w_l-delay_u, ti-delay_u)

            slice_current = slice(ti-idx_k, ti+1)
            slice_past = slice(ti-idx_l-shift, ti+1-shift)

            pos_obj = np.array(pos_obj_dict[exp_type], dtype='float64')[:, slice_past]
            pos_hum = np.array(pos_hum_dict[exp_type], dtype='float64')[:, slice_current]
            n_bins = int(np.sqrt(pos_obj.size))
            # vel_obj = np.array(vel_obj_dict[exp_type], dtype='float64')[:, slice_past]
            # vel_hum = np.array(vel_hum_dict[exp_type], dtype='float64')[:, slice_current]
            d_hum_given_obj = get_pmf(data_x=pos_obj, data_y=pos_hum, n_bins=n_bins)
            entropies_all[exp_type]['cond_hum_given_obj'].append(dit.multivariate.entropy(d_hum_given_obj, rvs=[1], crvs=[0]))
            
            pos_rob = np.array(pos_rob_dict[exp_type], dtype='float64')[:, slice_past]
            pos_hum = np.array(pos_hum_dict[exp_type], dtype='float64')[:, slice_current]
            d_hum_given_rob = get_pmf(data_x=pos_rob, data_y=pos_hum, n_bins=n_bins)
            entropies_all[exp_type]['cond_hum_given_rob'].append(dit.multivariate.entropy(d_hum_given_rob, rvs=[1], crvs=[0]))
            
            pos_obj = np.array(pos_obj_dict[exp_type], dtype='float64')[:, slice_past]
            pos_rob = np.array(pos_rob_dict[exp_type], dtype='float64')[:, slice_current]
            d_rob_given_obj = get_pmf(data_x=pos_obj, data_y=pos_rob, n_bins=n_bins)
            entropies_all[exp_type]['cond_rob_given_obj'].append(dit.multivariate.entropy(d_rob_given_obj, rvs=[1], crvs=[0]))
            
            ###############
            pos_hum = np.array(pos_hum_dict[exp_type], dtype='float64')[:, slice_past]
            pos_obj = np.array(pos_obj_dict[exp_type], dtype='float64')[:, slice_current]
            d_obj_given_hum = get_pmf(data_x=pos_hum, data_y=pos_obj, n_bins=n_bins)
            entropies_all[exp_type]['cond_obj_given_hum'].append(dit.multivariate.entropy(d_obj_given_hum, rvs=[1], crvs=[0]))
            
            pos_hum = np.array(pos_hum_dict[exp_type], dtype='float64')[:, slice_past]
            pos_rob = np.array(pos_rob_dict[exp_type], dtype='float64')[:, slice_current]
            d_rob_given_hum = get_pmf(data_x=pos_hum, data_y=pos_rob, n_bins=n_bins)
            entropies_all[exp_type]['cond_rob_given_hum'].append(dit.multivariate.entropy(d_rob_given_hum, rvs=[1], crvs=[0]))
            
            pos_rob = np.array(pos_rob_dict[exp_type], dtype='float64')[:, slice_past]
            pos_obj = np.array(pos_obj_dict[exp_type], dtype='float64')[:, slice_current]
            d_obj_given_rob = get_pmf(data_x=pos_rob, data_y=pos_obj, n_bins=n_bins)
            entropies_all[exp_type]['cond_obj_given_rob'].append(dit.multivariate.entropy(d_obj_given_rob, rvs=[1], crvs=[0]))
            
            ###############
            ## INFORMATION DYNAMICS
            idx_te = min(i, w_r)
            slice_current = slice(ti-idx_te, ti+1)
            pos_obj = np.array(pos_obj_dict[exp_type], dtype='float64')[:, slice_current]
            pos_hum = np.array(pos_hum_dict[exp_type], dtype='float64')[:, slice_current]
            pos_rob = np.array(pos_rob_dict[exp_type], dtype='float64')[:, slice_current]
            fz = np.array(fz_dict[exp_type], dtype='float64')[:, slice_current]

            ###############
            ## INFORMATION THEORY METRICS
            d_obj_hum = get_pmf(data_x=pos_obj, data_y=pos_hum, n_bins=n_bins)
            d_rob_hum = get_pmf(data_x=pos_rob, data_y=pos_hum, n_bins=n_bins)
            d_obj_rob = get_pmf(data_x=pos_obj, data_y=pos_rob, n_bins=n_bins)

            entropies_all[exp_type]['mutual_obj_hum'].append(dit.shannon.mutual_information(d_obj_hum, [0], [1], rv_mode='indexes'))
            entropies_all[exp_type]['mutual_rob_hum'].append(dit.shannon.mutual_information(d_rob_hum, [0], [1], rv_mode='indexes'))
            entropies_all[exp_type]['mutual_obj_rob'].append(dit.shannon.mutual_information(d_obj_rob, [0], [1], rv_mode='indexes'))

            entropies_all[exp_type]['joint_obj_hum'].append(dit.multivariate.entropy(d_obj_hum, [0,1]))
            entropies_all[exp_type]['joint_rob_hum'].append(dit.multivariate.entropy(d_rob_hum, [0,1]))
            entropies_all[exp_type]['joint_obj_rob'].append(dit.multivariate.entropy(d_obj_rob, [0,1]))
            
            ###############
            ## INFORMATION DYNAMICS
            te_obj_to_hum, local_te_obj_to_hum  = [0, 0] if pos_obj[0].size < (k_hist+1)*k_tau else calc_te_and_local_te(sourceArray=pos_obj.tolist(), destArray=pos_hum.tolist(), k_hist=k_hist, k_tau=k_tau, l_hist=l_hist, l_tau=l_tau, delay=delay)
            te_rob_to_hum, local_te_rob_to_hum  = [0, 0] if pos_obj[0].size < (k_hist+1)*k_tau else calc_te_and_local_te(sourceArray=pos_rob.tolist(), destArray=pos_hum.tolist(), k_hist=k_hist, k_tau=k_tau, l_hist=l_hist, l_tau=l_tau, delay=delay)
            te_obj_to_rob, local_te_obj_to_rob  = [0, 0] if pos_obj[0].size < (k_hist+1)*k_tau else calc_te_and_local_te(sourceArray=pos_obj.tolist(), destArray=pos_rob.tolist(), k_hist=k_hist, k_tau=k_tau, l_hist=l_hist, l_tau=l_tau, delay=delay)
            te_hum_to_obj, local_te_hum_to_obj  = [0, 0] if pos_obj[0].size < (k_hist+1)*k_tau else calc_te_and_local_te(sourceArray=pos_hum.tolist(), destArray=pos_obj.tolist(), k_hist=k_hist, k_tau=k_tau, l_hist=l_hist, l_tau=l_tau, delay=delay)
            te_hum_to_rob, local_te_hum_to_rob  = [0, 0] if pos_obj[0].size < (k_hist+1)*k_tau else calc_te_and_local_te(sourceArray=pos_hum.tolist(), destArray=pos_rob.tolist(), k_hist=k_hist, k_tau=k_tau, l_hist=l_hist, l_tau=l_tau, delay=delay)
            te_rob_to_obj, local_te_rob_to_obj  = [0, 0] if pos_obj[0].size < (k_hist+1)*k_tau else calc_te_and_local_te(sourceArray=pos_rob.tolist(), destArray=pos_obj.tolist(), k_hist=k_hist, k_tau=k_tau, l_hist=l_hist, l_tau=l_tau, delay=delay)

            entropies_all[exp_type]['te_obj_to_hum'].append(te_obj_to_hum)
            entropies_all[exp_type]['te_rob_to_hum'].append(te_rob_to_hum)
            entropies_all[exp_type]['te_obj_to_rob'].append(te_obj_to_rob)
            entropies_all[exp_type]['te_hum_to_obj'].append(te_hum_to_obj)
            entropies_all[exp_type]['te_hum_to_rob'].append(te_hum_to_rob)
            entropies_all[exp_type]['te_rob_to_obj'].append(te_rob_to_obj)

            entropies_all[exp_type]['local_te_obj_to_hum_avg'].append(np.mean(np.array(local_te_obj_to_hum)))
            entropies_all[exp_type]['local_te_rob_to_hum_avg'].append(np.mean(np.array(local_te_rob_to_hum)))
            entropies_all[exp_type]['local_te_obj_to_rob_avg'].append(np.mean(np.array(local_te_obj_to_rob)))
            entropies_all[exp_type]['local_te_hum_to_obj_avg'].append(np.mean(np.array(local_te_hum_to_obj)))
            entropies_all[exp_type]['local_te_hum_to_rob_avg'].append(np.mean(np.array(local_te_hum_to_rob)))
            entropies_all[exp_type]['local_te_rob_to_obj_avg'].append(np.mean(np.array(local_te_rob_to_obj)))

            cte_rob_to_hum_cond_obj = 0 if pos_obj[0].size < 200 else calc_cte(sourceArray=pos_rob.tolist(), destArray=pos_hum.tolist(), condArray=pos_obj.tolist(), k_hist=k_hist_cte, k_tau=k_tau_cte, l_hist=l_hist_cte, l_tau=l_tau_cte, delay=delay_u_cte, c_hist=m_hist_cte, c_tau=m_tau_cte, c_delay=delay_v_cte)
            cte_hum_to_rob_cond_obj = 0 if pos_obj[0].size < 200 else calc_cte(sourceArray=pos_hum.tolist(), destArray=pos_rob.tolist(), condArray=pos_obj.tolist(), k_hist=k_hist_cte, k_tau=k_tau_cte, l_hist=l_hist_cte, l_tau=l_tau_cte, delay=delay_u_cte, c_hist=m_hist_cte, c_tau=m_tau_cte, c_delay=delay_v_cte)

            entropies_all[exp_type]['cte_rob_to_hum_cond_obj'].append(cte_rob_to_hum_cond_obj)
            entropies_all[exp_type]['cte_hum_to_rob_cond_obj'].append(cte_hum_to_rob_cond_obj)

        ###############
        ## LOGGING INFO
        pos_obj = np.array(pos_obj_dict[exp_type], dtype='float64')[:, 600:1600]
        pos_hum = np.array(pos_hum_dict[exp_type], dtype='float64')[:, 600:1600]
        pos_rob = np.array(pos_rob_dict[exp_type], dtype='float64')[:, 600:1600]
        fz = np.array([dh.remove_bias(data=a) for a in fz_dict[exp_type]], dtype='float64')[:, 600:1600]

        entropies_all[exp_type]['pos_obj'] = pos_obj
        entropies_all[exp_type]['pos_hum'] = pos_hum
        entropies_all[exp_type]['pos_rob'] = pos_rob
        entropies_all[exp_type]['fz'] = fz
        
        with open('data/entropy_increasing_window_'+exp_type+'_'+'_ws'+str(w_r)+'_shift'+str(shift)+'_u'+str(delay)+'.pkl', 'wb') as f:
            print('saving entropy_increasing_window_'+exp_type+'_ws'+str(w_r)+'_shift'+str(shift)+'_u'+str(delay)+'.pkl')
            pickle.dump(entropies_all[exp_type], f)

    return


def entropies_global_table(exps_types, window_size, shift):
    entropies_dict = {'hum': 0,
                      'rob': 0,
                      'obj': 0,
                      'joint_obj_hum': 0,
                      'joint_rob_hum': 0,
                      'joint_obj_rob': 0,
                      'cond_hum_given_obj': 0,
                      'cond_hum_given_rob': 0,
                      'cond_rob_given_obj': 0,
                      'cond_obj_given_hum': 0,
                      'cond_rob_given_hum': 0,
                      'cond_obj_given_rob': 0,
                      'mutual_obj_hum': 0,
                      'mutual_rob_hum': 0,
                      'mutual_obj_rob': 0,
                      
                      'te_obj_to_hum': 0,
                      'te_rob_to_hum': 0,
                      'te_obj_to_rob': 0,
                      'te_hum_to_obj': 0,
                      'te_hum_to_rob': 0,
                      'te_rob_to_obj': 0,

                      'cte_obj_to_hum_cond_rob': 0,
                      'cte_rob_to_hum_cond_obj': 0,
                      'cte_obj_to_rob_cond_hum': 0,
                      'cte_hum_to_obj_cond_rob': 0,
                      'cte_hum_to_rob_cond_obj': 0,
                      'cte_rob_to_obj_cond_hum': 0,

                      'ais_hum': 0,
                      'ais_rob': 0,
                      'ais_obj': 0,
                      }

    entropies_all = {exp_type: {e: 0 for e in entropies_dict} for exp_type in exps_types}

    k_hist = 2
    k_tau  = 3
    l_hist = 2
    l_tau  = 3
    delay  = 0

    all_dfe = []

    # for phase, phase_steps in zip(['prc', 'poc'], [slice(0+50, 500+50), slice(500+50, 1000+50)]):
    for phase, phase_steps in zip(['all', 'prc', 'poc'], [slice(0,1000), slice(0,467), slice(467,1000)]):
    # for phase, phase_steps in zip(['whole'], [slice(600,1100)]):
    # for phase, phase_steps in zip(['whole'], [slice(1100,1600)]):
        for i, exp_type in enumerate(exps_types):
            for win_size in [window_size]:

                f_name = 'data/entropy_'+exp_type+'__ws'+str(win_size)+'_delay'+str(shift)+'.pkl'

                with open(f_name, 'rb') as f:
                    entropies_all[exp_type] = pickle.load(f)

                print(f_name)

                pos_obj = np.array(entropies_all[exp_type]['pos_obj'], dtype='float64')[:, phase_steps]
                pos_hum = np.array(entropies_all[exp_type]['pos_hum'], dtype='float64')[:, phase_steps]
                pos_rob = np.array(entropies_all[exp_type]['pos_rob'], dtype='float64')[:, phase_steps]

                lims_pos_obj = np.array([np.min(pos_obj), np.max(pos_obj)])
                lims_pos_hum = np.array([np.min(pos_hum), np.max(pos_hum)])
                lims_pos_rob = np.array([np.min(pos_rob), np.max(pos_rob)])

                # n_bins = N_BINS
                n_bins = int(np.sqrt(pos_obj.size))
                pxy_obj_hum, _, _ = np.histogram2d(pos_obj[0], pos_hum[0], bins=n_bins, range=[lims_pos_obj,lims_pos_hum])
                pxy_rob_hum, _, _ = np.histogram2d(pos_rob[0], pos_hum[0], bins=n_bins, range=[lims_pos_rob,lims_pos_hum])
                pxy_obj_rob, _, _ = np.histogram2d(pos_obj[0], pos_rob[0], bins=n_bins, range=[lims_pos_obj,lims_pos_rob])

                px_hum, _ = np.histogram(pos_hum, bins=n_bins, range=lims_pos_hum)
                px_obj, _ = np.histogram(pos_obj, bins=n_bins, range=lims_pos_obj)
                px_rob, _ = np.histogram(pos_rob, bins=n_bins, range=lims_pos_rob)

                for p_o, p_h, p_r in zip(pos_obj[1:], pos_hum[1:], pos_rob[1:]):
                    pxy_obj_hum += np.histogram2d(p_o, p_h, bins=n_bins, range=[lims_pos_obj,lims_pos_hum])[0]
                    pxy_rob_hum += np.histogram2d(p_r, p_h, bins=n_bins, range=[lims_pos_rob,lims_pos_hum])[0]
                    pxy_obj_rob += np.histogram2d(p_o, p_r, bins=n_bins, range=[lims_pos_obj,lims_pos_rob])[0]
                
                pxy_obj_hum /= np.sum(pxy_obj_hum)
                pxy_rob_hum /= np.sum(pxy_rob_hum)
                pxy_obj_rob /= np.sum(pxy_obj_rob)

                px_hum = px_hum/np.sum(px_hum, dtype='float64')
                px_obj = px_obj/np.sum(px_obj, dtype='float64')
                px_rob = px_rob/np.sum(px_rob, dtype='float64')

                d_hum = dit.Distribution.from_ndarray(px_hum)
                d_obj = dit.Distribution.from_ndarray(px_obj)
                d_rob = dit.Distribution.from_ndarray(px_rob)

                d_obj_hum = dit.Distribution.from_ndarray(pxy_obj_hum)
                d_rob_hum = dit.Distribution.from_ndarray(pxy_rob_hum)
                d_obj_rob = dit.Distribution.from_ndarray(pxy_obj_rob)

                # entropies
                entropies_all[exp_type]['hum'] = dit.shannon.entropy(d_hum)
                entropies_all[exp_type]['rob'] = dit.shannon.entropy(d_obj)
                entropies_all[exp_type]['obj'] = dit.shannon.entropy(d_rob)

                # joint entropy
                entropies_all[exp_type]['joint_obj_hum'] = dit.multivariate.entropy(d_obj_hum, [0,1])
                entropies_all[exp_type]['joint_rob_hum'] = dit.multivariate.entropy(d_rob_hum, [0,1])
                entropies_all[exp_type]['joint_obj_rob'] = dit.multivariate.entropy(d_obj_rob, [0,1])

                # conditional entropy                
                entropies_all[exp_type]['cond_hum_given_obj'] = dit.multivariate.entropy(d_obj_hum, rvs=[1], crvs=[0])
                entropies_all[exp_type]['cond_hum_given_rob'] = dit.multivariate.entropy(d_rob_hum, rvs=[1], crvs=[0])
                entropies_all[exp_type]['cond_rob_given_obj'] = dit.multivariate.entropy(d_obj_rob, rvs=[1], crvs=[0])
                entropies_all[exp_type]['cond_obj_given_hum'] = dit.multivariate.entropy(d_obj_hum, rvs=[0], crvs=[1])
                entropies_all[exp_type]['cond_rob_given_hum'] = dit.multivariate.entropy(d_rob_hum, rvs=[0], crvs=[1])
                entropies_all[exp_type]['cond_obj_given_rob'] = dit.multivariate.entropy(d_obj_rob, rvs=[0], crvs=[1])

                # mutual info:  I(X; Y) = I(Y; X) (2.46)
                entropies_all[exp_type]['mutual_obj_hum'] = dit.shannon.mutual_information(d_obj_hum, [0], [1], rv_mode='indexes')
                entropies_all[exp_type]['mutual_rob_hum'] = dit.shannon.mutual_information(d_rob_hum, [0], [1], rv_mode='indexes')
                entropies_all[exp_type]['mutual_obj_rob'] = dit.shannon.mutual_information(d_obj_rob, [0], [1], rv_mode='indexes')

                # ## obj to agents
                # s = pos_obj[exp_type].tolist()
                # d = pos_rob[exp_type].tolist()
                # k_hist, k_tau = optim_te_destination_only(destArray=d, sourceArray=s)
                # l_hist, l_tau = optim_te_source(destArray=d, sourceArray=s, k_hist=k_hist, k_tau=k_tau)
                # u = [optim_delay_u(destArray=d,
                #                 sourceArray=s,
                #                 k_hist='k_hist',
                #                 k_tau='k_tau',
                #                 l_hist='l_hist',
                #                 l_tau='l_tau',
                #                 u=ui) for ui in range(200)]
                # delay = np.argmax(u)
                te_obj_to_hum, local_te_obj_to_hum  = calc_te_and_local_te(sourceArray=pos_obj.tolist(), destArray=pos_hum.tolist(), k_hist=k_hist, k_tau=k_tau, l_hist=l_hist, l_tau=l_tau, delay=delay)
                te_obj_to_rob, local_te_obj_to_rob  = calc_te_and_local_te(sourceArray=pos_obj.tolist(), destArray=pos_rob.tolist(), k_hist=k_hist, k_tau=k_tau, l_hist=l_hist, l_tau=l_tau, delay=delay)
                te_rob_to_hum, local_te_rob_to_hum  = calc_te_and_local_te(sourceArray=pos_rob.tolist(), destArray=pos_hum.tolist(), k_hist=k_hist, k_tau=k_tau, l_hist=l_hist, l_tau=l_tau, delay=delay)
                te_hum_to_obj, local_te_hum_to_obj  = calc_te_and_local_te(sourceArray=pos_hum.tolist(), destArray=pos_obj.tolist(), k_hist=k_hist, k_tau=k_tau, l_hist=l_hist, l_tau=l_tau, delay=delay)
                te_hum_to_rob, local_te_hum_to_rob  = calc_te_and_local_te(sourceArray=pos_hum.tolist(), destArray=pos_rob.tolist(), k_hist=k_hist, k_tau=k_tau, l_hist=l_hist, l_tau=l_tau, delay=delay)
                te_rob_to_obj, local_te_rob_to_obj  = calc_te_and_local_te(sourceArray=pos_rob.tolist(), destArray=pos_obj.tolist(), k_hist=k_hist, k_tau=k_tau, l_hist=l_hist, l_tau=l_tau, delay=delay)

                entropies_all[exp_type]['te_obj_to_hum'] = te_obj_to_hum
                entropies_all[exp_type]['te_rob_to_hum'] = te_rob_to_hum
                entropies_all[exp_type]['te_obj_to_rob'] = te_obj_to_rob
                entropies_all[exp_type]['te_hum_to_obj'] = te_hum_to_obj
                entropies_all[exp_type]['te_hum_to_rob'] = te_hum_to_rob
                entropies_all[exp_type]['te_rob_to_obj'] = te_rob_to_obj

                entropies_all[exp_type]['local_te_obj_to_hum_avg'].append(np.mean(np.array(local_te_obj_to_hum)))
                entropies_all[exp_type]['local_te_rob_to_hum_avg'].append(np.mean(np.array(local_te_rob_to_hum)))
                entropies_all[exp_type]['local_te_obj_to_rob_avg'].append(np.mean(np.array(local_te_obj_to_rob)))
                entropies_all[exp_type]['local_te_hum_to_obj_avg'].append(np.mean(np.array(local_te_hum_to_obj)))
                entropies_all[exp_type]['local_te_hum_to_rob_avg'].append(np.mean(np.array(local_te_hum_to_rob)))
                entropies_all[exp_type]['local_te_rob_to_obj_avg'].append(np.mean(np.array(local_te_rob_to_obj)))

                entropies_all[exp_type]['cte_rob_to_hum_cond_obj'] = calc_cte(sourceArray=pos_rob.tolist(), destArray=pos_hum.tolist(), condArray=pos_obj.tolist(), k_hist=k_hist, k_tau=k_tau, l_hist=l_hist, l_tau=l_tau, delay=delay)
                entropies_all[exp_type]['cte_hum_to_rob_cond_obj'] = calc_cte(sourceArray=pos_hum.tolist(), destArray=pos_rob.tolist(), condArray=pos_obj.tolist(), k_hist=k_hist, k_tau=k_tau, l_hist=l_hist, l_tau=l_tau, delay=delay)

        keys = ['hum',
                'rob',
                'obj',
                'cond_hum_given_obj',
                'cond_rob_given_obj',
                'cond_rob_given_hum',
                'cond_hum_given_rob',
                'cond_obj_given_hum',
                'cond_obj_given_rob',
                'mutual_obj_hum',
                'mutual_rob_hum',
                'mutual_obj_rob',
                'te_obj_to_hum',
                'te_obj_to_rob',
                'te_hum_to_rob',
                'te_rob_to_hum',
                'te_hum_to_obj',
                'te_rob_to_obj',
                'cte_rob_to_hum_cond_obj',
                'cte_hum_to_rob_cond_obj',
                ]
        
        for exp_type, exp_label in zip(['HH', 'HB', 'HL', 'RL'], ['HH+', 'HH-', 'HR-', 'HR+']):
            # dfe = pd.DataFrame.from_dict(entropies_all[exp_type], orient='index').T
            dfe = pd.DataFrame.from_dict({key: entropies_all[exp_type][key] for key in keys}, orient='index', columns=[exp_label+'/'+phase]).T
            all_dfe.append(dfe)
            # dfe.to_excel(writer, sheet_name=exp_label)
    
    df_all = pd.concat(all_dfe)
    with pd.ExcelWriter('entropy_global.xlsx') as writer:
        df_all.to_excel(writer)

    keys_plots = ['cond_hum_given_obj',
                    'cond_rob_given_obj',
                    'cond_rob_given_hum',
                    'cond_hum_given_rob',
                    'cond_obj_given_hum',
                    'cond_obj_given_rob',
                    'te_obj_to_hum',
                    'te_obj_to_rob',
                    'te_hum_to_rob',
                    'te_rob_to_hum',
                    'te_hum_to_obj',
                    'te_rob_to_obj',]

    df_all[keys_plots]
    return


def calculate_learning_effect(w_r, w_k, w_l, exps_types, shift):
    entropies_dict = {'hum': 0,
                      'rob': 0,
                      'obj': 0,
                      'joint_obj_hum': 0,
                      'joint_rob_hum': 0,
                      'joint_obj_rob': 0,
                      'cond_hum_given_obj': 0,
                      'cond_hum_given_rob': 0,
                      'cond_rob_given_obj': 0,
                      'cond_obj_given_hum': 0,
                      'cond_rob_given_hum': 0,
                      'cond_obj_given_rob': 0,
                      'mutual_obj_hum': 0,
                      'mutual_rob_hum': 0,
                      'mutual_obj_rob': 0,
                      'pos_obj': 0,
                      'pos_hum': 0,
                      'pos_rob': 0,
                      'pos_obj_dict': 0,
                      'pos_hum_dict': 0,
                      'pos_rob_dict': 0,
                      'fz_conv': 0,
                      'fz': 0,
                      'd_obj_hum': 0,
                      'd_rob_hum': 0,
                      'd_obj_rob': 0,
                      'd_hum_given_obj': 0,
                      'd_hum_given_rob': 0,
                      'd_rob_given_obj': 0,
                      'd_obj_given_hum': 0,
                      'd_rob_given_hum': 0,
                      'd_obj_given_rob': 0,
                      }
    
    dh = DataPlotHelper()

    ti_max = 1000
            
    for i, exp_type in enumerate(exps_types):
        pos_obj_dict = {exp_type: [] for exp_type in exps_types}
        pos_hum_dict = {exp_type: [] for exp_type in exps_types}
        pos_rob_dict = {exp_type: [] for exp_type in exps_types}
        vel_obj_dict = {exp_type: [] for exp_type in exps_types}
        vel_hum_dict = {exp_type: [] for exp_type in exps_types}
        vel_rob_dict = {exp_type: [] for exp_type in exps_types}
        fz_dict = {exp_type: [] for exp_type in exps_types}

        pos_obj_dict[exp_type], pos_hum_dict[exp_type], pos_rob_dict[exp_type], fz_dict[exp_type], vel_obj_dict[exp_type], vel_hum_dict[exp_type], vel_rob_dict[exp_type] = load_data_by_exp_type(exp_type=exp_type)
        
        idx_list = [0]
        for _ in range(11):
            idx_list.append(idx_list[-1]+5)
        
        idx_list = np.array(idx_list)
        
        entropies_all = {str(i_): {exp_type: {e: [] for e in entropies_dict} for exp_type in exps_types} for i_ in range(5)}
        for idx_exp in [str(ii) for ii in range(5)]:

            exps_selected_by_sub = idx_list+int(idx_exp)
            for ti in range(600, ti_max+600):
                if (ti%w_r) == 0:
                    print('ws', w_r, '\tshift', shift, '\tsteps', str(ti-600)+'/'+str(ti_max))

                slice_future = slice(ti-w_k, ti)
                slice_current = slice(ti-w_l-shift, ti-shift)

                pos_obj = np.array(pos_obj_dict[exp_type], dtype='float64')[exps_selected_by_sub, slice_current]
                pos_hum = np.array(pos_hum_dict[exp_type], dtype='float64')[exps_selected_by_sub, slice_future]
                n_bins = int(np.sqrt(pos_obj.size))
                d_hum_given_obj = get_pmf(data_x=pos_obj, data_y=pos_hum, n_bins=n_bins)
                entropies_all[idx_exp][exp_type]['cond_hum_given_obj'].append(dit.multivariate.entropy(d_hum_given_obj, rvs=[1], crvs=[0]))
                
                pos_rob = np.array(pos_rob_dict[exp_type], dtype='float64')[exps_selected_by_sub, slice_current]
                pos_hum = np.array(pos_hum_dict[exp_type], dtype='float64')[exps_selected_by_sub, slice_future]
                d_hum_given_rob = get_pmf(data_x=pos_rob, data_y=pos_hum, n_bins=n_bins)
                entropies_all[idx_exp][exp_type]['cond_hum_given_rob'].append(dit.multivariate.entropy(d_hum_given_rob, rvs=[1], crvs=[0]))

                pos_obj = np.array(pos_obj_dict[exp_type], dtype='float64')[exps_selected_by_sub, slice_current]
                pos_rob = np.array(pos_rob_dict[exp_type], dtype='float64')[exps_selected_by_sub, slice_future]
                d_rob_given_obj = get_pmf(data_x=pos_obj, data_y=pos_rob, n_bins=n_bins)
                entropies_all[idx_exp][exp_type]['cond_rob_given_obj'].append(dit.multivariate.entropy(d_rob_given_obj, rvs=[1], crvs=[0]))

                ###############
                pos_hum = np.array(pos_hum_dict[exp_type], dtype='float64')[exps_selected_by_sub, slice_current]
                pos_obj = np.array(pos_obj_dict[exp_type], dtype='float64')[exps_selected_by_sub, slice_future]
                d_obj_given_hum = get_pmf(data_x=pos_hum, data_y=pos_obj, n_bins=n_bins)
                entropies_all[idx_exp][exp_type]['cond_obj_given_hum'].append(dit.multivariate.entropy(d_obj_given_hum, rvs=[1], crvs=[0]))

                pos_hum = np.array(pos_hum_dict[exp_type], dtype='float64')[exps_selected_by_sub, slice_current]
                pos_rob = np.array(pos_rob_dict[exp_type], dtype='float64')[exps_selected_by_sub, slice_future]
                d_rob_given_hum = get_pmf(data_x=pos_hum, data_y=pos_rob, n_bins=n_bins)
                entropies_all[idx_exp][exp_type]['cond_rob_given_hum'].append(dit.multivariate.entropy(d_rob_given_hum, rvs=[1], crvs=[0]))

                pos_rob = np.array(pos_rob_dict[exp_type], dtype='float64')[exps_selected_by_sub, slice_current]
                pos_obj = np.array(pos_obj_dict[exp_type], dtype='float64')[exps_selected_by_sub, slice_future]
                d_obj_given_rob = get_pmf(data_x=pos_rob, data_y=pos_obj, n_bins=n_bins)
                entropies_all[idx_exp][exp_type]['cond_obj_given_rob'].append(dit.multivariate.entropy(d_obj_given_rob, rvs=[1], crvs=[0]))


                ###############
                # SYNCED INFO: mutual, joint
                slice_current = slice(ti-w_k, ti)
                pos_obj = np.array(pos_obj_dict[exp_type], dtype='float64')[exps_selected_by_sub, slice_current]
                pos_hum = np.array(pos_hum_dict[exp_type], dtype='float64')[exps_selected_by_sub, slice_current]
                pos_rob = np.array(pos_rob_dict[exp_type], dtype='float64')[exps_selected_by_sub, slice_current]
                fz = np.array(fz_dict[exp_type], dtype='float64')[exps_selected_by_sub, ti:ti+w_r]

                d_obj_hum = get_pmf(data_x=pos_obj, data_y=pos_hum, n_bins=n_bins)
                d_rob_hum = get_pmf(data_x=pos_rob, data_y=pos_hum, n_bins=n_bins)
                d_obj_rob = get_pmf(data_x=pos_obj, data_y=pos_rob, n_bins=n_bins)

                entropies_all[idx_exp][exp_type]['mutual_obj_hum'].append(dit.shannon.mutual_information(d_obj_hum, [0], [1], rv_mode='indexes'))
                entropies_all[idx_exp][exp_type]['mutual_rob_hum'].append(dit.shannon.mutual_information(d_rob_hum, [0], [1], rv_mode='indexes'))
                entropies_all[idx_exp][exp_type]['mutual_obj_rob'].append(dit.shannon.mutual_information(d_obj_rob, [0], [1], rv_mode='indexes'))

                entropies_all[idx_exp][exp_type]['joint_obj_hum'].append(dit.multivariate.entropy(d_obj_hum, [0,1]))
                entropies_all[idx_exp][exp_type]['joint_rob_hum'].append(dit.multivariate.entropy(d_rob_hum, [0,1]))
                entropies_all[idx_exp][exp_type]['joint_obj_rob'].append(dit.multivariate.entropy(d_obj_rob, [0,1]))

                entropies_all[idx_exp][exp_type]['fz_conv'].append(np.sum(fz)/fz.size)

                entropies_all[idx_exp][exp_type]['d_obj_hum'].append(d_obj_hum)
                entropies_all[idx_exp][exp_type]['d_rob_hum'].append(d_rob_hum)
                entropies_all[idx_exp][exp_type]['d_obj_rob'].append(d_obj_rob)

                entropies_all[idx_exp][exp_type]['d_hum_given_obj'].append(d_hum_given_obj)
                entropies_all[idx_exp][exp_type]['d_hum_given_rob'].append(d_hum_given_rob)
                entropies_all[idx_exp][exp_type]['d_rob_given_obj'].append(d_rob_given_obj)
                entropies_all[idx_exp][exp_type]['d_obj_given_hum'].append(d_obj_given_hum)
                entropies_all[idx_exp][exp_type]['d_rob_given_hum'].append(d_rob_given_hum)
                entropies_all[idx_exp][exp_type]['d_obj_given_rob'].append(d_obj_given_rob)

            pos_obj = np.array(pos_obj_dict[exp_type], dtype='float64')[exps_selected_by_sub, 600:1600]
            pos_hum = np.array(pos_hum_dict[exp_type], dtype='float64')[exps_selected_by_sub, 600:1600]
            pos_rob = np.array(pos_rob_dict[exp_type], dtype='float64')[exps_selected_by_sub, 600:1600]
            # fz = dh.remove_bias(data=np.array(fz_dict[exp_type], dtype='float64')[exps_selected_by_sub, 600:1600])
            fz = np.array([dh.remove_bias(data=a) for a in fz_dict[exp_type]], dtype='float64')[exps_selected_by_sub, 600:1600]

            entropies_all[idx_exp][exp_type]['pos_obj'] = pos_obj
            entropies_all[idx_exp][exp_type]['pos_hum'] = pos_hum
            entropies_all[idx_exp][exp_type]['pos_rob'] = pos_rob
            entropies_all[idx_exp][exp_type]['fz'] = fz
                    
        with open('data/entropy_learning_effect_'+exp_type+'_'+'_ws'+str(w_r)+'_shift'+str(shift)+'.pkl', 'wb') as f:
            print('saving enotrpy_learning_effect_'+exp_type+'_ws'+str(w_r)+'_shift'+str(shift)+'.pkl')
            pickle.dump(entropies_all, f)

    return


def optimize_hyperparam_TE_all_steps(exps_types):
    print('\n\n\n\n')
    agents_list = ['obj_to_hum', 'obj_to_rob',
                   'hum_to_rob', 'rob_to_hum',
                   'hum_to_obj', 'rob_to_obj']
    metrics_te = {exp_type: {key: {'k': 0, 'l': 0} for key in agents_list} for exp_type in exps_types}

    for exp_type in exps_types:
        print('exp_type = ', exp_type)
        pos_obj_dict = {exp_type: [] for exp_type in exps_types}
        pos_hum_dict = {exp_type: [] for exp_type in exps_types}
        pos_rob_dict = {exp_type: [] for exp_type in exps_types}
        vel_obj_dict = {exp_type: [] for exp_type in exps_types}
        vel_hum_dict = {exp_type: [] for exp_type in exps_types}
        vel_rob_dict = {exp_type: [] for exp_type in exps_types}
        fz_dict = {exp_type: [] for exp_type in exps_types}


        pos_obj_dict[exp_type], pos_hum_dict[exp_type], pos_rob_dict[exp_type], fz_dict[exp_type], vel_obj_dict[exp_type], vel_hum_dict[exp_type], vel_rob_dict[exp_type] = load_data_by_exp_type(exp_type=exp_type)
        pos_obj_dict[exp_type] = np.array(pos_obj_dict[exp_type], dtype='float64')
        pos_hum_dict[exp_type] = np.array(pos_hum_dict[exp_type], dtype='float64')
        pos_rob_dict[exp_type] = np.array(pos_rob_dict[exp_type], dtype='float64')
        fz_dict[exp_type] = np.array(fz_dict[exp_type], dtype='float64')
        vel_obj_dict[exp_type] = np.array(vel_obj_dict[exp_type], dtype='float64')
        vel_hum_dict[exp_type] = np.array(vel_hum_dict[exp_type], dtype='float64')
        vel_rob_dict[exp_type] = np.array(vel_rob_dict[exp_type], dtype='float64')

        for ti in range(600, 1600, 50):

            my_slice = slice(ti, ti+300)

            print('steps = ', my_slice)

            for key in agents_list:
                print(key)
                if key[:3] == 'obj':
                    s = pos_obj_dict[exp_type][:, my_slice].tolist()
                elif key[:3] == 'hum':
                    s = pos_hum_dict[exp_type][:, my_slice].tolist()
                elif key[:3] == 'rob':
                    s = pos_rob_dict[exp_type][:, my_slice].tolist()

                if key[-3:] == 'obj':
                    d = pos_obj_dict[exp_type][:, my_slice].tolist()
                elif key[-3:] == 'hum':
                    d = pos_hum_dict[exp_type][:, my_slice].tolist()
                elif key[-3:] == 'rob':
                    d = pos_rob_dict[exp_type][:, my_slice].tolist()

                metrics_te[exp_type][key]['k_hist'], metrics_te[exp_type][key]['k_tau'] = optim_te_destination_only(destArray=d, sourceArray=s)
                metrics_te[exp_type][key]['l_hist'], metrics_te[exp_type][key]['l_tau'] = optim_te_source(destArray=d, sourceArray=s, k_hist=metrics_te[exp_type][key]['k_hist'], k_tau=metrics_te[exp_type][key]['k_tau'])

                metrics_te[exp_type][key]['u'] = [optim_delay_u(destArray=d,
                                                                sourceArray=s,
                                                                k_hist=metrics_te[exp_type][key]['k_hist'],
                                                                k_tau=metrics_te[exp_type][key]['k_tau'],
                                                                l_hist=metrics_te[exp_type][key]['l_hist'],
                                                                l_tau=metrics_te[exp_type][key]['l_tau'],
                                                                u=ui) for ui in range(200)]
                
            # ## obj to agents
            # s = pos_obj_dict[exp_type][:, my_slice].tolist()
            # d = pos_rob_dict[exp_type][:, my_slice].tolist()
            # k_hist_obj_to_rob, k_tau_obj_to_rob = optim_te_destination_only(destArray=d, sourceArray=s)
            # l_hist_obj_to_rob, l_tau_obj_to_rob = optim_te_source(destArray=d, sourceArray=s, k_hist=k_hist_obj_to_rob, k_tau=k_tau_obj_to_rob)

            # s = pos_obj_dict[exp_type][:, my_slice].tolist()
            # d = pos_hum_dict[exp_type][:, my_slice].tolist()
            # k_hist_obj_to_hum, k_tau_obj_to_hum = optim_te_destination_only(destArray=d, sourceArray=s)
            # l_hist_obj_to_hum, l_tau_obj_to_hum = optim_te_source(destArray=d, sourceArray=s, k_hist=k_hist_obj_to_hum, k_tau=k_tau_obj_to_hum)

            # ## agents to obj
            # s = pos_rob_dict[exp_type][:, my_slice].tolist()
            # d = pos_obj_dict[exp_type][:, my_slice].tolist()
            # k_hist_rob_to_obj, k_tau_rob_to_obj = optim_te_destination_only(destArray=d, sourceArray=s)
            # l_hist_rob_to_obj, l_tau_rob_to_obj = optim_te_source(destArray=d, sourceArray=s, k_hist=k_hist_rob_to_obj, k_tau=k_tau_rob_to_obj)

            # s = pos_hum_dict[exp_type][:, my_slice].tolist()
            # d = pos_obj_dict[exp_type][:, my_slice].tolist()
            # k_hist_hum_to_obj, k_tau_hum_to_obj = optim_te_destination_only(destArray=d, sourceArray=s)
            # l_hist_hum_to_obj, l_tau_hum_to_obj = optim_te_source(destArray=d, sourceArray=s, k_hist=k_hist_hum_to_obj, k_tau=k_tau_hum_to_obj)

            # ## agents to agents
            # s = pos_rob_dict[exp_type][:, my_slice].tolist()
            # d = pos_hum_dict[exp_type][:, my_slice].tolist()
            # k_hist_rob_to_hum, k_tau_rob_to_hum = optim_te_destination_only(destArray=d, sourceArray=s)
            # l_hist_rob_to_hum, l_tau_rob_to_hum = optim_te_source(destArray=d, sourceArray=s, k_hist=k_hist_rob_to_hum, k_tau=k_tau_rob_to_hum)

            # s = pos_hum_dict[exp_type][:, my_slice].tolist()
            # d = pos_rob_dict[exp_type][:, my_slice].tolist()
            # k_hist_hum_to_rob, k_tau_hum_to_rob = optim_te_destination_only(destArray=d, sourceArray=s)
            # l_hist_hum_to_rob, l_tau_hum_to_rob = optim_te_source(destArray=d, sourceArray=s, k_hist=k_hist_hum_to_rob, k_tau=k_tau_hum_to_rob)

            
            # u = list(range(200))

            # u_hum_to_rob = [optim_delay_u(destArray=pos_rob_dict[exp_type][:, my_slice].tolist(),
            #                     sourceArray=pos_hum_dict[exp_type][:, my_slice].tolist(),
            #                     k_hist=k_hist_hum_to_rob,
            #                     k_tau=k_tau_hum_to_rob,
            #                     l_hist=l_hist_hum_to_rob,
            #                     l_tau=l_tau_hum_to_rob,
            #                     u=ui) for ui in u]

            # u_rob_to_hum = [optim_delay_u(destArray=pos_hum_dict[exp_type][:, my_slice].tolist(),
            #                     sourceArray=pos_rob_dict[exp_type][:, my_slice].tolist(),
            #                     k_hist=k_hist_rob_to_hum,
            #                     k_tau=k_tau_rob_to_hum,
            #                     l_hist=l_hist_rob_to_hum,
            #                     l_tau=l_tau_rob_to_hum,
            #                     u=ui) for ui in u]
            
            # u_hum_to_obj = [optim_delay_u(destArray=pos_obj_dict[exp_type][:, my_slice].tolist(),
            #                     sourceArray=pos_hum_dict[exp_type][:, my_slice].tolist(),
            #                     k_hist=k_hist_hum_to_obj,
            #                     k_tau=k_tau_hum_to_obj,
            #                     l_hist=l_hist_hum_to_obj,
            #                     l_tau=l_tau_hum_to_obj,
            #                     u=ui) for ui in u]
            
            # u_rob_to_obj = [optim_delay_u(destArray=pos_obj_dict[exp_type][:, my_slice].tolist(),
            #                     sourceArray=pos_rob_dict[exp_type][:, my_slice].tolist(),
            #                     k_hist=k_hist_rob_to_obj,
            #                     k_tau=k_tau_rob_to_obj,
            #                     l_hist=l_hist_rob_to_obj,
            #                     l_tau=l_tau_rob_to_obj,
            #                     u=ui) for ui in u]
            
            # u_obj_to_hum = [optim_delay_u(destArray=pos_hum_dict[exp_type][:, my_slice].tolist(),
            #                     sourceArray=pos_obj_dict[exp_type][:, my_slice].tolist(),
            #                     k_hist=k_hist_obj_to_hum,
            #                     k_tau=k_tau_obj_to_hum,
            #                     l_hist=l_hist_obj_to_hum,
            #                     l_tau=l_tau_obj_to_hum,
            #                     u=ui) for ui in u]
            
            # u_obj_to_rob = [optim_delay_u(destArray=pos_hum_dict[exp_type][:, my_slice].tolist(),
            #                     sourceArray=pos_obj_dict[exp_type][:, my_slice].tolist(),
            #                     k_hist=k_hist_obj_to_rob,
            #                     k_tau=k_tau_obj_to_rob,
            #                     l_hist=l_hist_obj_to_rob,
            #                     l_tau=l_tau_obj_to_rob,
            #                     u=ui) for ui in u]
            
            # metrics_te[exp_type].append([k_hist_obj_to_rob, k_tau_obj_to_rob, l_hist_obj_to_rob, l_tau_obj_to_rob,
            #                              k_hist_obj_to_hum, k_tau_obj_to_hum, l_hist_obj_to_hum, l_tau_obj_to_hum,
            #                              k_hist_rob_to_obj, k_tau_rob_to_obj, l_hist_rob_to_obj, l_tau_rob_to_obj,
            #                              k_hist_hum_to_obj, k_tau_hum_to_obj, l_hist_hum_to_obj, l_tau_hum_to_obj,
            #                              k_hist_rob_to_hum, k_tau_rob_to_hum, l_hist_rob_to_hum, l_tau_rob_to_hum,
            #                              k_hist_hum_to_rob, k_tau_hum_to_rob, l_hist_hum_to_rob, l_tau_hum_to_rob,
            #                              u_hum_to_rob, u_rob_to_hum, u_hum_to_obj, u_obj_to_hum, u_rob_to_obj, u_obj_to_rob])
            
            # print('hum > rob = ', np.argmax(res1))
            # print('rob > hum = ', np.argmax(res2))
            print('ti = ', ti)

        print('change')
        print('\n\n\n\n')
    print('done all')

    with open('data/optim_metrics.pkl', 'wb') as f:
        print('saving optim_metrics.pkl')
        pickle.dump(metrics_te, f)

    return


def optimize_check_params(exps_types):
    with open('data/optim_metrics.pkl', 'rb') as f:
        print('saving optim_metrics.pkl')
        metrics_te = pickle.load(f)
    for exp_type in exps_types:
        print('\n\n\n Counting for ', exp_type)

        print('k_hist = ', Counter([k_hist for k_hist, k_tau, l_hist, l_tau, res1, res2 in metrics_te[exp_type]]).most_common(3))
        print('k_tau = ', Counter([k_tau for k_hist, k_tau, l_hist, l_tau, res1, res2 in metrics_te[exp_type]]).most_common(3))
        print('l_hist = ', Counter([l_hist for k_hist, k_tau, l_hist, l_tau, res1, res2 in metrics_te[exp_type]]).most_common(3))
        print('l_tau = ', Counter([l_tau for k_hist, k_tau, l_hist, l_tau, res1, res2 in metrics_te[exp_type]]).most_common(3))
        print('u_delay = ', Counter([np.argmax(res1) for k_hist, k_tau, l_hist, l_tau, res1, res2 in metrics_te[exp_type]]).most_common(3))
        print('v_delay = ', Counter([np.argmax(res2) for k_hist, k_tau, l_hist, l_tau, res1, res2 in metrics_te[exp_type]]).most_common(3))

    return


if __name__ == '__main__':
    # calculate_info_dynamics(w_r=300, w_k=200, w_l=200, delay_u=100, exps_types=['HH', 'HB', 'RL', 'HL'])
    # calculate_info_dynamics(w_r=300, w_k=200, w_l=200, delay_u=100, exps_types=['HH'])
    # calculate_info_dynamics(w_r=300, w_k=200, w_l=200, delay_u=100, exps_types=['HB'])
    # calculate_info_dynamics(w_r=300, w_k=200, w_l=200, delay_u=100, exps_types=['RL'])
    # calculate_info_dynamics(w_r=300, w_k=200, w_l=200, delay_u=100, exps_types=['HL'])

    # calculate_info_dynamics_increasing_window(w_r=300, w_k=200, w_l=200, shift=100, exps_types=['HH'])
    # calculate_info_dynamics_increasing_window(w_r=300, w_k=200, w_l=200, shift=100, exps_types=['HB'])
    # calculate_info_dynamics_increasing_window(w_r=300, w_k=200, w_l=200, shift=100, exps_types=['RL'])
    # calculate_info_dynamics_increasing_window(w_r=300, w_k=200, w_l=200, shift=100, exps_types=['HL'])

    # entropies_global_table(exps_types=['HH', 'HB', 'RL', 'HL'], window_size=300, shift=100)

    # calculate_learning_effect(w_r=300, w_k=200, w_l=200, exps_types=['HH'], shift=100)
    # calculate_learning_effect(w_r=300, w_k=200, w_l=200, exps_types=['HB'], shift=100)
    # calculate_learning_effect(w_r=300, w_k=200, w_l=200, exps_types=['RL'], shift=100)
    # calculate_learning_effect(w_r=300, w_k=200, w_l=200, exps_types=['HL'], shift=100)

    optimize_hyperparam_TE_all_steps(exps_types=['HH', 'HB', 'RL', 'HL'])
    # optimize_check_params(exps_types=['HH', 'HB', 'RL', 'HL'])
    

    pass
