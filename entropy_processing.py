import os
import dit
import pickle
import numpy as np
import matplotlib.pyplot as plt
from data_plot_class import DataPlotHelper
from entropy_utils_jpype import calc_ais, calc_te_and_local_te, calc_cte, get_pmf, get_init_drop_idx


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


def calculate_info_dynamics(win_size, delay_u, exps_types, ti_max=1000, n_bins=10):
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

    k_hist = 1
    k_tau  = 3
    l_hist = 6
    l_tau  = 1
    delay  = 0
            
    for i, exp_type in enumerate(exps_types):

        pos_obj_dict[exp_type], pos_hum_dict[exp_type], pos_rob_dict[exp_type],  vel_obj_dict[exp_type], vel_hum_dict[exp_type], vel_rob_dict[exp_type], fz_dict[exp_type] = load_data_by_exp_type(exp_type=exp_type)

        entropies_all = {exp_type: {e: [] for e in entropies_dict} for exp_type in exps_types}

        for ti in range(600, ti_max+600):
            if (ti%100) == 0:
                print('ws', win_size, '\tdelay_u', delay_u, '\tsteps', str(ti-600)+'/'+str(ti_max))

            slice_current = slice(ti-win_size, ti)
            slice_past = slice(ti-win_size-delay_u, ti-delay_u)

            pos_obj = np.array(pos_obj_dict[exp_type], dtype='float64')[:, slice_past]
            pos_hum = np.array(pos_hum_dict[exp_type], dtype='float64')[:, slice_current]
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
            slice_current = slice(ti-win_size, ti)
            pos_obj = np.array(pos_obj_dict[exp_type], dtype='float64')[:, slice_current]
            pos_hum = np.array(pos_hum_dict[exp_type], dtype='float64')[:, slice_current]
            pos_rob = np.array(pos_rob_dict[exp_type], dtype='float64')[:, slice_current]
            fz = np.array(fz_dict[exp_type], dtype='float64')[:, slice_current]

            te_obj_to_hum, local_te_obj_to_hum  = calc_te_and_local_te(sourceArray=pos_obj.tolist(), destArray=pos_hum.tolist(), k_hist=k_hist, k_tau=k_tau, l_hist=l_hist, l_tau=l_tau, delay=delay)
            te_rob_to_hum, local_te_rob_to_hum  = calc_te_and_local_te(sourceArray=pos_rob.tolist(), destArray=pos_hum.tolist(), k_hist=k_hist, k_tau=k_tau, l_hist=l_hist, l_tau=l_tau, delay=delay)
            te_obj_to_rob, local_te_obj_to_rob  = calc_te_and_local_te(sourceArray=pos_obj.tolist(), destArray=pos_rob.tolist(), k_hist=k_hist, k_tau=k_tau, l_hist=l_hist, l_tau=l_tau, delay=delay)
            te_hum_to_obj, local_te_hum_to_obj  = calc_te_and_local_te(sourceArray=pos_hum.tolist(), destArray=pos_obj.tolist(), k_hist=k_hist, k_tau=k_tau, l_hist=l_hist, l_tau=l_tau, delay=delay)
            te_hum_to_rob, local_te_hum_to_rob  = calc_te_and_local_te(sourceArray=pos_hum.tolist(), destArray=pos_rob.tolist(), k_hist=k_hist, k_tau=k_tau, l_hist=l_hist, l_tau=l_tau, delay=delay)
            te_rob_to_obj, local_te_rob_to_obj  = calc_te_and_local_te(sourceArray=pos_rob.tolist(), destArray=pos_obj.tolist(), k_hist=k_hist, k_tau=k_tau, l_hist=l_hist, l_tau=l_tau, delay=delay)

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
        
        with open('data/entropy_'+exp_type+'_'+'_ws'+str(win_size)+'_delay'+str(delay_u)+'.pkl', 'wb') as f:
            print('saving entropy_'+exp_type+'_ws'+str(win_size)+'_delay'+str(delay_u)+'.pkl')
            pickle.dump(entropies_all[exp_type], f)

    return


if __name__ == '__main__':
    # calculate_info_dynamics(win_size=300, delay_u=100, exps_types=['HH', 'HB', 'RL', 'HL'])
    # calculate_info_dynamics(win_size=300, delay_u=100, exps_types=['HH'])
    # calculate_info_dynamics(win_size=300, delay_u=100, exps_types=['HB'])
    # calculate_info_dynamics(win_size=300, delay_u=100, exps_types=['RL'])
    calculate_info_dynamics(win_size=300, delay_u=100, exps_types=['HH'])
