import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
from data_plot_class import DataPlotHelper

def LOI(ft, time):
    idx = 0
    loi = 0
    fw = ft[-1]
    int_term_prev = 0
    for i, val in enumerate(ft):
        if val > 3:
            idx = i-1
            break
    for i in range(idx, len(ft)):
        int_term = np.abs(ft[i] - fw)
        loi += np.abs((int_term + int_term_prev)*(time[i]-time[i-1])/2)
        int_term_prev = int_term
    return loi

def PEI(pos, pos_d):
    error = np.abs(pos_d - pos)
    return np.max(error)

def DRI(ft, exp=''):
    if exp == 'opt_kmp':
        i1 = 100
        i2 = 154
    if exp == 'opt_kmp_vic':
        i1 = 100
        i2 = 169
    if exp == 'empty':
        i1 = 0
        i2 = 1
    if exp == 'const-imp':
        i1 = 104
        i2 = 183
    delta = np.log10(ft[i1]/ft[i2])
    dri = 1/(np.sqrt(1+(2*np.pi/delta)**2))
    return dri

def BTI(ft, time, exp=''):
    # if force <= 0 we start counting. But i hard coded based on the means
    if exp == 'opt_kmp':
        i1 = 225
        i2 = 275
    if exp == 'opt_kmp_vic':
        i1 = 1
        i2 = 1
    if exp == 'empty':
        i1 = 1
        i2 = 1
    if exp == 'const-imp':
        i1 = 1
        i2 = 1
    return time[i2] - time[i1]

def WORK(ft, ee_pos, exp=''):
    idx = 0
    work = 0
    for i, val in enumerate(ft):
        if val > 3:
            idx = i-1
            break
    for i in range(idx, len(ft)):
        work += (ft[i]+ft[i-1])*(ee_pos[i]-ee_pos[i-1])/2
    return work

def ENERGY(ft, power, time, exp=''):
    idx = 0
    work = 0
    for i, val in enumerate(ft):
        if val > 3:
            idx = i-1
            break
    for i in range(idx, len(ft)):
        work += (power[i]+power[i-1])*(time[i]-time[i-1])/2
    return work

path_folder = 'data/icra2023/2-exp-ImpLoop3-NoAdaptation-PARTIAL-RESULTS-DONT-EDIT/'
data_info = pd.read_csv(path_folder+'data_info.csv')

dh = DataPlotHelper(path_folder)

# LOAD DATA
Z_AXIS = 2
STEP_XAXIS=0.25
COLORS = ['Green', 'Blue']
TRIALS_IDXS = [1, 2, 3]
HEIGHTS = [27]
LEGEND_SIZE = 14

params = {
    'empty': False,
    'vic': False,
    'color': 'Blue',
    'trial_idx': 1,
    'height': 27,
    'impedance_loop': 30,
    'online_adaptation': False,
    'i_initial': 0,
    'i_final': -1,
    'data_to_plot': 'EE_twist_d',
    }

#------------------ MEAN FORCES

params['color'] = 'Blue'
params['height'] = 27
params['trial_idx'] = 1
params['vic'] = False

fts = np.zeros((1000, 3))

for i in [1, 2, 3]:
    params['trial_idx'] = i

    if i == 1:
        offset = -14
    if i == 2:
        offset = 0
    if i==3:
        offset = 20

    idx_start = dh.get_idx_from_file(params, data_info, idx_name='idx_start')
    idx_end = dh.get_idx_from_file(params, data_info, idx_name='idx_end')
    params['i_initial'] = idx_start + offset
    params['i_final'] = idx_end + offset

    params['data_to_plot'] = 'time'
    time = dh.get_data(params, axis=0)
    time = time - time[0]

    params['data_to_plot'] = 'FT_ati'
    # FT_ati = dh.get_data(params, axis=Z_AXIS
    # fts.append(dh.get_data(params, axis=Z_AXIS))
    fts[:,i-1] = dh.get_data(params, axis=Z_AXIS)


params['color'] = 'Blue'
params['height'] = 27
params['trial_idx'] = 1
params['vic'] = True

fts_vic = np.zeros((1000, 3))

vic_offset = 24

for i in [1, 2, 3]:
    params['trial_idx'] = i

    offset = 6 if i==2 else 0

    idx_start = dh.get_idx_from_file(params, data_info, idx_name='idx_start')
    idx_end = dh.get_idx_from_file(params, data_info, idx_name='idx_end')
    params['i_initial'] = idx_start + offset + vic_offset
    params['i_final'] = idx_end + offset + vic_offset

    params['data_to_plot'] = 'time'
    time = dh.get_data(params, axis=0)
    time = time - time[0]

    params['data_to_plot'] = 'FT_ati'
    # FT_ati = dh.get_data(params, axis=Z_AXIS
    # fts_vic.append(dh.get_data(params, axis=Z_AXIS))
    fts_vic[:,i-1] = dh.get_data(params, axis=Z_AXIS)

FT = np.mean(fts, axis=1)
FT_vic = np.mean(fts_vic, axis=1)

# plt.plot(FT)
# plt.plot(fts)
# plt.legend(['mean','1', '2', '3'])
# plt.show()

# plt.plot(FT_vic)
# plt.plot(fts_vic)

params = {
    'empty': False,
    'vic': False,
    'color': 'Blue',
    'trial_idx': 1,
    'height': 27,
    'impedance_loop': 30,
    'online_adaptation': False,
    'i_initial': 0,
    'i_final': -1,
    'data_to_plot': 'EE_twist_d',
    }

# EMPTY
file_name = path_folder + 'empty.mat'
idx_start = dh.get_idx_from_file(params, data_info, idx_name='idx_start', file_name=file_name)
idx_end = dh.get_idx_from_file(params, data_info, idx_name='idx_end', file_name=file_name)
params['i_initial'] = idx_start
params['i_final'] = idx_end
params['data_to_plot'] = 'time'
time_em = dh.get_data(params, axis=0)
time_em = time_em - time_em[0]
params['data_to_plot'] = 'FT_ati'
FT_empty = dh.get_data(params, axis=Z_AXIS, file_name=file_name)
# params['data_to_plot'] = 'EE_twist_d'
# ee_d_emp = dh.get_data(params, axis=Z_AXIS)

# fpIC
file_name = path_folder + 'const-imp.mat'
idx_start = dh.get_idx_from_file(params, data_info, idx_name='idx_start', file_name=file_name)
idx_end = dh.get_idx_from_file(params, data_info, idx_name='idx_end', file_name=file_name)
params['i_initial'] = idx_start
params['i_final'] = idx_end
params['data_to_plot'] = 'time'
time_const_imp = dh.get_data(params, axis=0, file_name=file_name)
time_const_imp = time_const_imp - time_const_imp[0]
params['data_to_plot'] = 'FT_ati'
ft_const_imp = dh.get_data(params, axis=Z_AXIS, file_name=file_name)

# ------------------------------ MEAN POSITION
params['color'] = 'Blue'
params['height'] = 27
params['trial_idx'] = 1
params['vic'] = False

ee_pose = np.zeros((1000, 3))

for i in [1, 2, 3]:
    params['trial_idx'] = i

    if i == 1:
        offset = -14
    if i == 2:
        offset = 0
    if i==3:
        offset = 20

    idx_start = dh.get_idx_from_file(params, data_info, idx_name='idx_start')
    idx_end = dh.get_idx_from_file(params, data_info, idx_name='idx_end')
    params['i_initial'] = idx_start + offset
    params['i_final'] = idx_end + offset

    params['data_to_plot'] = 'time'
    time = dh.get_data(params, axis=0)
    time = time - time[0]

    params['data_to_plot'] = 'EE_position'
    ee_pose[:,i-1] = dh.get_data(params, axis=Z_AXIS)

    params['data_to_plot'] = 'EE_position_d'
    EE_position_d = dh.get_data(params, axis=Z_AXIS)

params['color'] = 'Blue'
params['height'] = 27
params['trial_idx'] = 1
params['vic'] = True

ee_pose_vic = np.zeros((1000, 3))

vic_offset = 24

for i in [1, 2, 3]:
    params['trial_idx'] = i

    offset = 6 if i==2 else 0

    idx_start = dh.get_idx_from_file(params, data_info, idx_name='idx_start')
    idx_end = dh.get_idx_from_file(params, data_info, idx_name='idx_end')
    params['i_initial'] = idx_start + offset + vic_offset
    params['i_final'] = idx_end + offset + vic_offset

    params['data_to_plot'] = 'time'
    time = dh.get_data(params, axis=0)
    time = time - time[0]

    params['data_to_plot'] = 'EE_position'
    ee_pose_vic[:,i-1] = dh.get_data(params, axis=Z_AXIS)

EE_pos = np.mean(ee_pose, axis=1)
EE_pos_vic = np.mean(ee_pose_vic, axis=1)


file_name = path_folder + 'empty.mat'
idx_start = dh.get_idx_from_file(params, data_info, idx_name='idx_start', file_name=file_name)
idx_end = dh.get_idx_from_file(params, data_info, idx_name='idx_end', file_name=file_name)
params['i_initial'] = idx_start
params['i_final'] = idx_end
params['data_to_plot'] = 'time'
time_empty = dh.get_data(params, axis=0)
time_empty = time_empty - time_empty[0]
params['data_to_plot'] = 'EE_position'
EE_pos_empty = dh.get_data(params, Z_AXIS, file_name)


file_name = path_folder + 'const-imp.mat'
idx_start = dh.get_idx_from_file(params, data_info, idx_name='idx_start', file_name=file_name)
idx_end = dh.get_idx_from_file(params, data_info, idx_name='idx_end', file_name=file_name)
params['i_initial'] = idx_start
params['i_final'] = idx_end
params['data_to_plot'] = 'time'
time_const_imp = dh.get_data(params, axis=0, file_name=file_name)
time_const_imp = time_const_imp - time_const_imp[0]
params['data_to_plot'] = 'EE_position'
EE_const_imp = dh.get_data(params, Z_AXIS, file_name)


# ------------------------------ MEAN TWIST
params['color'] = 'Blue'
params['height'] = 27
params['trial_idx'] = 1
params['vic'] = False

ees = np.zeros((1000, 3))

for i in [1, 2, 3]:
    params['trial_idx'] = i

    if i == 1:
        offset = -14
    if i == 2:
        offset = 0
    if i==3:
        offset = 20

    idx_start = dh.get_idx_from_file(params, data_info, idx_name='idx_start')
    idx_end = dh.get_idx_from_file(params, data_info, idx_name='idx_end')
    params['i_initial'] = idx_start + offset
    params['i_final'] = idx_end + offset

    params['data_to_plot'] = 'time'
    time = dh.get_data(params, axis=0)
    time = time - time[0]

    params['data_to_plot'] = 'EE_twist'
    ees[:,i-1] = dh.get_data(params, axis=Z_AXIS)


params['color'] = 'Blue'
params['height'] = 27
params['trial_idx'] = 1
params['vic'] = True

ees_vic = np.zeros((1000, 3))

vic_offset = 24

for i in [1, 2, 3]:
    params['trial_idx'] = i

    offset = 6 if i==2 else 0

    idx_start = dh.get_idx_from_file(params, data_info, idx_name='idx_start')
    idx_end = dh.get_idx_from_file(params, data_info, idx_name='idx_end')
    params['i_initial'] = idx_start + offset + vic_offset
    params['i_final'] = idx_end + offset + vic_offset

    params['data_to_plot'] = 'time'
    time = dh.get_data(params, axis=0)
    time = time - time[0]

    params['data_to_plot'] = 'EE_twist'
    ees_vic[:,i-1] = dh.get_data(params, axis=Z_AXIS)

EE_twist = np.mean(ees, axis=1)
EE_twist_vic = np.mean(ees_vic, axis=1)


file_name = path_folder + 'empty.mat'
idx_start = dh.get_idx_from_file(params, data_info, idx_name='idx_start', file_name=file_name)
idx_end = dh.get_idx_from_file(params, data_info, idx_name='idx_end', file_name=file_name)
params['i_initial'] = idx_start
params['i_final'] = idx_end
params['data_to_plot'] = 'time'
time_empty = dh.get_data(params, axis=0)
time_empty = time_empty - time_empty[0]
params['data_to_plot'] = 'EE_twist'
EE_twist_empty = dh.get_data(params, Z_AXIS, file_name)


file_name = path_folder + 'const-imp.mat'
idx_start = dh.get_idx_from_file(params, data_info, idx_name='idx_start', file_name=file_name)
idx_end = dh.get_idx_from_file(params, data_info, idx_name='idx_end', file_name=file_name)
params['i_initial'] = idx_start
params['i_final'] = idx_end
params['data_to_plot'] = 'time'
time_const_imp = dh.get_data(params, axis=0, file_name=file_name)
time_const_imp = time_const_imp - time_const_imp[0]
params['data_to_plot'] = 'EE_twist'
EE_twist_const_imp = dh.get_data(params, Z_AXIS, file_name)

# ---------------------- POWER

power = np.multiply(EE_twist, FT)
power_vic = np.multiply(EE_twist_vic, FT_vic)
power_empty = np.multiply(EE_twist_empty, FT_empty)
power_const_imp = np.multiply(EE_twist_const_imp, ft_const_imp)

# ---------------------- METRICS CALCULATION

metrics = { 'loi': 0,
            'pei': 0,
            'dri': 0,
            'work': 0,
            'energy': 0,
            'f_max': 0}

metrics_opt_kmp = {key: value for key, value in metrics.items()}
metrics_opt_kmp_vic = {key: value for key, value in metrics.items()}
metrics_empty = {key: value for key, value in metrics.items()}
metrics_fpic = {key: value for key, value in metrics.items()}

metrics_opt_kmp['loi'] = LOI(FT, time)
metrics_opt_kmp_vic['loi'] = LOI(FT_vic, time)
metrics_empty['loi'] = LOI(FT_empty, time)
metrics_fpic['loi'] = LOI(ft_const_imp, time_const_imp)

metrics_opt_kmp['pei'] = PEI(EE_pos, EE_position_d)
metrics_opt_kmp_vic['pei'] = PEI(EE_pos_vic, EE_position_d)
metrics_empty['pei'] = PEI(EE_pos_empty, EE_position_d)
metrics_fpic['pei'] = PEI(EE_const_imp, np.ones(len(EE_const_imp))*EE_const_imp[0])

metrics_opt_kmp['dri'] = DRI(FT, 'opt_kmp')
metrics_opt_kmp_vic['dri'] = DRI(FT_vic, 'opt_kmp_vic')
metrics_empty['dri'] = DRI(FT_empty, 'empty')
metrics_fpic['dri'] = DRI(ft_const_imp, 'const-imp')

metrics_opt_kmp['bti'] = BTI(FT, time, 'opt_kmp')
metrics_opt_kmp_vic['bti'] = BTI(FT_vic, time, 'opt_kmp_vic')
metrics_empty['bti'] = BTI(FT_empty, time, 'empty')
metrics_fpic['bti'] = BTI(ft_const_imp, time_const_imp, 'const-imp')

metrics_opt_kmp['work'] = WORK(FT, EE_pos, 'opt_kmp')
metrics_opt_kmp_vic['work'] = WORK(FT_vic, EE_pos_vic, 'opt_kmp_vic')
metrics_empty['work'] = WORK(FT_empty, EE_pos_empty, 'empty')
metrics_fpic['work'] = WORK(ft_const_imp, EE_const_imp, 'const-imp')

metrics_opt_kmp['f_max'] = np.max(FT)
metrics_opt_kmp_vic['f_max'] = np.max(FT_vic)
metrics_empty['f_max'] = np.max(FT_empty)
metrics_fpic['f_max'] = np.max(ft_const_imp)

metrics_opt_kmp['energy'] = ENERGY(FT, power, time, 'opt_kmp')
metrics_opt_kmp_vic['energy'] = ENERGY(FT_vic, power_vic, time, 'opt_kmp_vic')
metrics_empty['energy'] = ENERGY(FT_empty, power_empty, time, 'empty')
metrics_fpic['energy'] = ENERGY(ft_const_imp, power_const_imp, time, 'empty')

print('fpic = ', metrics_fpic)
print('vm = ', metrics_opt_kmp)
print('vic = ', metrics_opt_kmp_vic)
# print('empty = ', metrics_empty)
