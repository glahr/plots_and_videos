def load_forces_mean():
    #------------------ MEAN
    params['color'] = 'Blue'
    params['height'] = 27
    params['trial_idx'] = 1
    params['vic'] = False

    fts = np.zeros((N_POINTS, 3))

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

    fts_vic = np.zeros((N_POINTS, 3))

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
    ft_emp = dh.get_data(params, axis=Z_AXIS, file_name=file_name)
    # params['data_to_plot'] = 'EE_twist_d'
    # ee_d_emp = dh.get_data(params, axis=Z_AXIS)

    # CONST IMP
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
    ft_const_imp_tail = np.ones(N_POINTS-len(ft_const_imp))*ft_const_imp[-1]
    # params['data_to_plot'] = 'EE_twist_d'
    # ee_d_emp = dh.get_data(params, axis=Z_AXIS)