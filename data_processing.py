import os
import pickle
import numpy as np
import pandas as pd
from bagpy import bagreader 
import matplotlib.pyplot as plt
from pynumdiff import kalman_smooth as KS
from data_plot_class import DataPlotHelper

def save_filtered_data(exps_types):

    subjects_blindfolded = [
                            'franka_a_s02s07_HB_e01__2023_07_20__16_40_13',
                            'franka_a_s10s03_HB_e01__2023_07_20__16_55_14',
                            'franka_a_s06s09_HB_e01__2023_07_20__17_01_53',
                            'franka_a_s08s04_HB_e01__2023_07_20__17_12_43',
                            'franka_a_s05s01_HB_e01__2023_07_27__11_48_20',
                            
                            'franka_a_s03s10_HB_e01__2023_07_20__16_57_19',
                            'franka_a_s04s08_HB_e01__2023_07_20__17_14_39',
                            'franka_a_s07s02_HB_e01__2023_07_20__16_44_09',
                            'franka_a_s09s06_HB_e01__2023_07_20__17_03_58',
                            'franka_a_s01s05_HB_e01__2023_07_27__11_50_40',
                            ]

    subjects = ['s'+str(i+1).zfill(2) for i in range(12)]

    for i, exp_type in enumerate(exps_types):

        subs = subjects if exp_type != 'HB' else subjects_blindfolded

        k = 0

        for j, subject in enumerate(subs):

            path_folder = 'data_raw/'+subject+'/' if exp_type != 'HB' else 'blindfolded/raw/'
            files_names = os.listdir(path=path_folder) 

            dh = DataPlotHelper(path_folder=path_folder, files_names=files_names, window_earlier_than_500=800)
            files_with_exp_type = [a for a in files_names if exp_type in a] if exp_type != 'HB' else [a for a in files_names if subject in a]
                        
            files_with_exp_type.sort()
            files_to_filter = files_with_exp_type[1:] if exp_type != 'HB' else files_with_exp_type

            assert len(files_with_exp_type) > 0, 'Empty exp_type'

            for file_name in files_to_filter:

                # file_name = dh._get_name(params=params)
                print(file_name)
                data_dict = {'pos_obj': None, 'pos_hum': None, 'pos_rob': None, 'fz': None}
                idx_initial = dh.get_idx_start(file_name=file_name) if exp_type != 'HB' else 0
                idx_end = dh.get_idx_end(file_name=file_name, idx_initial=idx_initial) if exp_type != 'HB' else -1

                dh.adjust_start = True if idx_initial < 0 else False
                aux = 0
                if dh.adjust_start:
                    aux = np.abs(idx_initial)
                    idx_initial = 0

                time = dh.get_data(file_name=file_name, data_to_plot='time')[idx_initial:idx_end]
                # time = dh.check_and_adjust_length_time(time=time, params=params, adjust_start=dh.adjust_start)
                time = dh.check_and_adjust_length_time(time=time, idx_initial=idx_initial, idx_end=idx_end)
                time -= time[0]

                axis = 2 if exp_type != 'HB' else 1

                z_obj = dh.get_data(file_name=file_name, data_to_plot='OBJ_position')[idx_initial:idx_end,axis]
                z_hum = dh.get_data(file_name=file_name, data_to_plot='HUM_position')[idx_initial:idx_end,axis]
                z_rob = dh.get_data(file_name=file_name, data_to_plot='ROB_position')[idx_initial:idx_end,axis]
                fz = dh.get_data(file_name=file_name, data_to_plot='ft_sensor')[idx_initial:idx_end,2]

                if dh.adjust_start:
                    z_obj = np.concatenate([np.ones(aux)*z_obj[0], z_obj])
                    z_hum = np.concatenate([np.ones(aux)*z_hum[0], z_hum])
                    z_rob = np.concatenate([np.ones(aux)*z_rob[0], z_rob])
                    fz = np.concatenate([np.ones(aux)*fz[0], fz])
                    idx_end = 2000 + dh.window_earlier_than_500
                
                if 'franka_a_s02s07_HB_e01__2023_07_20__16_40_13' in file_name:
                    file_name = 'franka_a_s02s07_HB_e01__2023_07_20__16_42_39.mat'
                    z_obj_pt2 = dh.get_data(file_name=file_name, data_to_plot='OBJ_position')[idx_initial:idx_end,axis]
                    z_hum_pt2 = dh.get_data(file_name=file_name, data_to_plot='HUM_position')[idx_initial:idx_end,axis]
                    z_rob_pt2 = dh.get_data(file_name=file_name, data_to_plot='ROB_position')[idx_initial:idx_end,axis]
                    fz_pt2 = dh.get_data(file_name=file_name, data_to_plot='ft_sensor')[idx_initial:idx_end,2]

                    z_obj = np.concatenate([z_obj, z_obj_pt2])
                    z_hum = np.concatenate([z_hum, z_hum_pt2])
                    z_rob = np.concatenate([z_rob, z_rob_pt2])
                    fz = np.concatenate([fz, fz_pt2])

                p_obj_filt = np.zeros_like(z_obj)
                p_hum_filt = np.zeros_like(z_hum)
                p_rob_filt = np.zeros_like(z_rob)

                p_obj_filt, v_obj_filt = KS.constant_acceleration(z_obj, 0.001, [0.01**2, 1])
                p_hum_filt, v_hum_filt = KS.constant_acceleration(z_hum, 0.001, [0.01**2, 1])
                p_rob_filt, v_rob_filt = KS.constant_acceleration(z_rob, 0.001, [0.01**2, 1])

                p_obj_filt = dh.check_and_adjust_length(data=p_obj_filt, idx_initial=idx_initial, idx_end=idx_end)
                p_hum_filt = dh.check_and_adjust_length(data=p_hum_filt, idx_initial=idx_initial, idx_end=idx_end)
                p_rob_filt = dh.check_and_adjust_length(data=p_rob_filt, idx_initial=idx_initial, idx_end=idx_end)
                
                v_obj_filt = dh.check_and_adjust_length(data=v_obj_filt, idx_initial=idx_initial, idx_end=idx_end)
                v_hum_filt = dh.check_and_adjust_length(data=v_hum_filt, idx_initial=idx_initial, idx_end=idx_end)
                v_rob_filt = dh.check_and_adjust_length(data=v_rob_filt, idx_initial=idx_initial, idx_end=idx_end)
                
                fz_filt = dh.check_and_adjust_length(data=fz, idx_initial=idx_initial, idx_end=idx_end)

                data_dict['pos_obj'] = p_obj_filt
                data_dict['pos_hum'] = p_hum_filt
                data_dict['pos_rob'] = p_rob_filt
                data_dict['vel_obj'] = v_obj_filt
                data_dict['vel_hum'] = v_hum_filt
                data_dict['vel_rob'] = v_rob_filt
                data_dict['fz'] = fz_filt

                with open('data_filtered/'+file_name[:-4]+'.pkl', 'wb') as f:
                    pickle.dump(data_dict, f)
                
                k += 1
                
    return


def bag_to_pose_optitrack(bag, frame):
    pose = {'position.x': 0,
            'position.y': 0,
            'position.z': 0,
            'orientation.x': 0,
            'orientation.y': 0,
            'orientation.z': 0,
            'orientation.w': 0,
            }
    pose['position.x'] = pd.read_csv(bag.message_by_topic('optitrack/'+frame))['pose.position.x'].values
    pose['position.y'] = pd.read_csv(bag.message_by_topic('optitrack/'+frame))['pose.position.y'].values
    pose['position.z'] = pd.read_csv(bag.message_by_topic('optitrack/'+frame))['pose.position.z'].values
    
    pose['orientation.x'] = pd.read_csv(bag.message_by_topic('optitrack/'+frame))['pose.orientation.x'].values
    pose['orientation.y'] = pd.read_csv(bag.message_by_topic('optitrack/'+frame))['pose.orientation.y'].values
    pose['orientation.z'] = pd.read_csv(bag.message_by_topic('optitrack/'+frame))['pose.orientation.z'].values
    pose['orientation.z'] = pd.read_csv(bag.message_by_topic('optitrack/'+frame))['pose.orientation.w'].values
    
    return pose


def save_filtered_data_throwing():

    path_folder = 'throwing/raw/'
    files_names = os.listdir(path=path_folder)

    file_name = [fn for fn in files_names if 'bag' in fn]

    bag = bagreader(path_folder+file_name[0])

    hum1 = {'left': None, 'right': None}
    hum2 = {'left': None, 'right': None}

    idx_init = 0
    obj_ball = bag_to_pose_optitrack(bag, 'frame_1')
    obj_box = bag_to_pose_optitrack(bag, 'frame_6')
    
    hum1['right'] = bag_to_pose_optitrack(bag, 'frame_2')
    hum1['left'] = bag_to_pose_optitrack(bag, 'frame_3')
    
    hum2['left'] = bag_to_pose_optitrack(bag, 'frame_4')
    hum2['right'] = bag_to_pose_optitrack(bag, 'frame_5')

    print(file_name)
    data_dict = {'pos_obj': None, 'pos_hum1': None, 'pos_hum2': None}

    z_hum1 = (hum1['right']['position.z'][1:] + hum1['left']['position.z'])/2
    z_hum2 = (hum2['right']['position.z'] + hum2['left']['position.z'])/2

    z_hum1 = z_hum1[idx_init:]
    z_hum2 = z_hum2[idx_init:]
    z_box = obj_box['position.z'][idx_init:]

    p_obj_filt = np.zeros_like(z_box)
    p_hum1_filt = np.zeros_like(z_hum1)
    p_hum2_filt = np.zeros_like(z_hum2)
    p_obj_filt, v_obj_filt = KS.constant_acceleration(z_box, 0.0075, [0.01**2, 1])
    p_hum1_filt, v_hum1_filt = KS.constant_acceleration(z_hum1, 0.0075, [0.01**2, 1])
    p_hum2_filt, v_hum2_filt = KS.constant_acceleration(z_hum2, 0.0075, [0.01**2, 1])

    data_dict['pos_obj'] = p_obj_filt
    data_dict['pos_hum1'] = p_hum1_filt
    data_dict['pos_hum2'] = p_hum2_filt

    data_dict['vel_obj'] = v_obj_filt
    data_dict['vel_hum1'] = v_hum1_filt
    data_dict['vel_hum2'] = v_hum2_filt

    with open('throwing/data_filtered/'+file_name[0][:-4]+'.pkl', 'wb') as f:
        pickle.dump(data_dict, f)

    return

if __name__ == "__main__":
    '''DATA PROCESSING'''
    # save_filtered_data(exps_types=['HH', 'HB', 'RL', 'HL'])
    save_filtered_data(exps_types=['HB'])
    # save_filtered_data_throwing()
