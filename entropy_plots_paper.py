import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr
from entropy_utils_jpype import *
from matplotlib.colors import LogNorm
from data_plot_class import DataPlotHelper

def ce_plot(exps_types, win_size, delay, use_increasing_window):
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
                      }
    
    pairs = [['cond_hum_given_obj', 'cond_rob_given_obj'],
             ['cond_rob_given_hum', 'cond_hum_given_rob'],
             ['cond_obj_given_hum', 'cond_obj_given_rob']]
    
    legends_pairs = [['$H(X_{h}|X_{o})$', '$H(X_{r,h}|X_{o})$'],
                     ['$H(X_{r,h}|X_{h})$', '$H(X_{h}|X_{r,h})$'],
                     ['$H(X_{o}|X_{h})$', '$H(X_{o}|X_{r,h})$']]
    entropies_all = {exp_type: {e: [] for e in entropies_dict} for exp_type in exps_types}
    
    dh = DataPlotHelper(f_size=15)

    fig_, ax_ = plt.subplots(4, 3, figsize=[9,5])

    # colors = [['#E73D1D','#306AA7'], ['#FEC81A','#E73D1D'], ['#306AA7','#FEC81A']]
    # colors = [['#440154','#2E99D9'], ['#3E64F0','#2E99D9'], ['#2E99D9','#440154']]
    colors = [['#440154','#2E99D9'], ['#440154','#2E99D9'], ['#440154','#2E99D9']]

    
    for idx_exp, exp_type in enumerate(exps_types):

        str_increasing_window = 'increasing_window_' if use_increasing_window else ''
        f_name = 'data/entropy_'+str_increasing_window+exp_type+'__ws'+str(win_size)+'_delay'+str(delay)+'.pkl'

        with open(f_name, 'rb') as f:
            entropies_all[exp_type] = pickle.load(f)

            for idx_pair, pair in enumerate(pairs):
        
                fig_name = 'data/plots/entropy_'+exp_type+'_ws'+str(win_size)+'_shift'+str(delay)+'_conditional_pair'+str(idx_pair+1)+'.png'
                fig, ax = dh.set_axis(fig=fig_, ax=ax_[idx_exp, idx_pair], xlim_plot=[0, 1])

                ax.plot(np.arange(0, 1, 0.001), entropies_all[exp_type][pair[0]], label=legends_pairs[idx_pair][0], color=colors[idx_pair][0])
                ax.plot(np.arange(0, 1, 0.001), entropies_all[exp_type][pair[1]], label=legends_pairs[idx_pair][1], color=colors[idx_pair][1], linestyle='dashed')

                ax.axvspan(xmin=0.467-0.025, xmax=0.467+0.025, alpha=0.15, color='#2E99D9', lw=0)
                
                ax.set_ylim([0, 7])
                if idx_pair != 0:
                    ax.set_yticks([])

                if idx_exp != len(ax_)-1:
                    ax.set_xticks([])
                else:
                    ax.set_xlabel('$t~[s]$')
                    
                if idx_exp == 0:
                    ax.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.04, 0.50), framealpha=1)

                # grids creation
                x_grids = list(np.arange(0,1,0.1))
                n_divisions = 7
                alpha_grids = 0.05
                y_grids = list(np.arange(0, n_divisions, 1))
                for j, e in enumerate([ax]):
                    [e.axvline(xg, color='k', alpha=alpha_grids) for xg in x_grids]
                    [e.axhline(yg, color='k', alpha=alpha_grids) for yg in y_grids]

    for axi, title in zip(ax_[:,0], ['$HH+$', '$HH-$', '$HR+$', '$HR-$']): 
        axi.set_ylabel(title, fontname='Times new roman')
        
    for row in ax_:
        for col in row:
            # col.set_xlabel(lab, labelpad=0)
            col.tick_params(axis='both', labelsize=9)

    fig_.tight_layout()
    fig_.subplots_adjust(wspace=0.15, hspace=0.2)
    fig_.savefig('data/plots/entropy_global_exps.pdf', dpi=200)
    plt.show()
    return


def ce_plot_rmse(exps_types, win_size, shift):

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
                      }
    
    pairs = [['cond_hum_given_obj', 'cond_rob_given_obj'],
             ['cond_rob_given_hum', 'cond_hum_given_rob'],
             ['cond_obj_given_hum', 'cond_obj_given_rob']]
    
    df_pairs = {exp_type: {'rmse': [], 'peak': []} for exp_type in exps_types}

    colors = [['b','g'], ['orange','r'], ['purple','brown']]

    entropies_all = {exp_type: {e: [] for e in entropies_dict} for exp_type in exps_types}

    markers_shapes = {'HH': '-o',
                    'RL': '-^',
                    'HL': '-D',
                    'HB': '-x'}
    
    colors = {'HH': '#0f36c4',
              'RL': '#1c9c5c',
              'HB': '#0bd114',
              'HL': '#3E64F0',}
    # colors = [['#440154','#2E99D9'], ['#3E64F0','#2E99D9'], ['#2E99D9','#440154']]
    
    dh = DataPlotHelper(f_size=15)

    fig_, ax_ = plt.subplots(2, figsize=[5,6])
    
    for idx_exp, exp_type in enumerate(exps_types):

        f_name = 'data/entropy_'+exp_type+'__ws'+str(win_size)+'_delay'+str(shift)+'.pkl'

        with open(f_name, 'rb') as f:
            entropies_all[exp_type] = pickle.load(f)

            for idx_pair, pair in enumerate(pairs):
                fig_name = 'data/plots/global_'+exp_type+'_ws'+str(win_size)+'_shift'+str(shift)+'_conditional_pair'+str(idx_pair+1)+'.png'
                
                err = np.array(entropies_all[exp_type][pair[0]]) - np.array(entropies_all[exp_type][pair[1]])
                
                peak = np.max(np.abs(err))
                rms = np.sqrt(np.mean(err**2))

                df_pairs[exp_type]['rmse'].append(rms)
                df_pairs[exp_type]['peak'].append(peak)

    fig, ax = dh.set_axis(fig=fig_, ax=ax_[0])
    figi, axi = dh.set_axis(fig=fig_, ax=ax_[1])

    for exp_type, exp_label in zip(['HH', 'HB', 'RL', 'HL'], ['$HH+$', '$HH-$', '$HR+$', '$HR-$']):
        if exp_type != 'ST':
            ax_[0].plot([0,1,2], df_pairs[exp_type]['rmse'], markers_shapes[exp_type], label=exp_label, color=colors[exp_type])
            ax_[1].plot([0,1,2], df_pairs[exp_type]['peak'], markers_shapes[exp_type], label=exp_label, color=colors[exp_type])
    
    ax_[1].set_xticks([0, 1, 2])
    ax_[0].set_ylim([0., 1.0])
    ax_[1].set_ylim([0., 1.5])

    ax_[0].set_xticklabels([])
    ax_[1].set_xticklabels(['$H(Ag.|Obj.)$', '$H(Ag.|Ag.)$', '$H(Obj.|Ag.)$'])
    ax_[0].tick_params(axis='x', labelsize=12)
    ax_[0].tick_params(axis='y', labelsize=12)

    fig_.supylabel('$\epsilon(H)~[-]$', fontsize=16)

    ax_[1].tick_params(axis='x', labelsize=12)
    ax_[1].tick_params(axis='y', labelsize=12)

    ax_[0].legend(fontsize=10, loc='upper right', framealpha=1, bbox_to_anchor=(1, 1))
    fig_.tight_layout()
    fig_.savefig('data/plots/entropy_rmse_err_pairs.pdf', dpi=200)
    plt.show()
    return


def ce_plot_rmse_lollipop(exps_types, win_size, shift):

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
                      }
    
    pairs = [['cond_hum_given_obj', 'cond_rob_given_obj'],
             ['cond_rob_given_hum', 'cond_hum_given_rob'],
             ['cond_obj_given_hum', 'cond_obj_given_rob']]
    
    df_pairs = {exp_type: {'rmse': [], 'peak': []} for exp_type in exps_types}

    colors = [['b','g'], ['orange','r'], ['purple','brown']]

    entropies_all = {exp_type: {e: [] for e in entropies_dict} for exp_type in exps_types}

    markers_shapes = {'HH': 'o',
                    'RL': '>',
                    'HL': 'D',
                    'HB': 'x'}
    
    colors = {'HH': '#3E64F0',
              'HB': '#3EF0E7',
              'RL': '#2ED983',
              'HL': '#46F54E',}
    
    colors_darker = {'HH': '#0f36c4',
                    'HB': '#0fc4bb',
                    'RL': '#1c9c5c',
                    'HL': '#0bd114',}
    
    # colors = [['#440154','#2E99D9'], ['#3E64F0','#2E99D9'], ['#2E99D9','#440154']]
    
    dh = DataPlotHelper(f_size=15)

    fig_, ax_ = plt.subplots(1, figsize=[5,6.5])
    
    for idx_exp, exp_type in enumerate(exps_types):

        f_name = 'data/entropy_'+exp_type+'__ws'+str(win_size)+'_delay'+str(shift)+'.pkl'

        with open(f_name, 'rb') as f:
            entropies_all[exp_type] = pickle.load(f)

            for idx_pair, pair in enumerate(pairs):
                fig_name = 'data/plots/global_'+exp_type+'_ws'+str(win_size)+'_shift'+str(shift)+'_conditional_pair'+str(idx_pair+1)+'.png'
                
                err = np.array(entropies_all[exp_type][pair[0]]) - np.array(entropies_all[exp_type][pair[1]])
                
                peak = np.max(np.abs(err))
                rms = np.sqrt(np.mean(err**2))

                df_pairs[exp_type]['rmse'].append(rms)
                df_pairs[exp_type]['peak'].append(peak)

    fig, ax = dh.set_axis(fig=fig_, ax=ax_)

    i = 0
    for exp_type, exp_label in zip(reversed(['HH', 'HB', 'RL', 'HL']), reversed(['$HH+$', '$HH-$', '$HR+$', '$HR-$'])):
        # peak
        # leg = exp_label if i == 0 else None
        ax.hlines(y=i+0.5, xmin=0, xmax=df_pairs[exp_type]['peak'][0], linewidth=3, color=colors_darker[exp_type], label=exp_label)
        ax.hlines(y=i+0.5+7.5, xmin=0, xmax=df_pairs[exp_type]['peak'][1], linewidth=3, color=colors_darker[exp_type])
        ax.hlines(y=i+0.5+15, xmin=0, xmax=df_pairs[exp_type]['peak'][2], linewidth=3, color=colors_darker[exp_type])
        ax.scatter(df_pairs[exp_type]['peak'][0], i+0.5, marker='D', s=80, color=colors_darker[exp_type])
        ax.scatter(df_pairs[exp_type]['peak'][1], i+0.5+7.5, marker='D', s=80, color=colors_darker[exp_type])
        ax.scatter(df_pairs[exp_type]['peak'][2], i+0.5+15, marker='D', s=80, color=colors_darker[exp_type])
        
        # rmse
        ax.hlines(y=i+0.5, xmin=0, xmax=df_pairs[exp_type]['rmse'][0], linewidth=3, color=colors_darker[exp_type])
        ax.hlines(y=i+0.5+7.5, xmin=0, xmax=df_pairs[exp_type]['rmse'][1], linewidth=3, color=colors_darker[exp_type])
        ax.hlines(y=i+0.5+15, xmin=0, xmax=df_pairs[exp_type]['rmse'][2], linewidth=3, color=colors_darker[exp_type])
        ax.scatter(df_pairs[exp_type]['rmse'][0], i+0.5, marker='o', s=80, color=colors_darker[exp_type])
        ax.scatter(df_pairs[exp_type]['rmse'][1], i+0.5+7.5, marker='o', s=80, color=colors_darker[exp_type])
        ax.scatter(df_pairs[exp_type]['rmse'][2], i+0.5+15, marker='o', s=80, color=colors_darker[exp_type])
        
        i += 1
    
    ax_.set_yticks([2, 9.5, 17])
    ax_.set_yticklabels(['$H(Ag.|Obj.)$', '$H(Ag.|Ag.)$', '$H(Obj.|Ag.)$'], rotation=0)
    ax_.set_yticklabels(['$Ag.|Obj.$', '$Ag.|Ag.$', '$Obj.|Ag.$'], rotation=0)

    # ax_.set_xticks([0, 1.5])
    ax_.set_xlim([0, 1.8])
    
    # ax_[0].tick_params(axis='x', labelsize=12)
    # ax_[0].tick_params(axis='y', labelsize=12)

    ax_.set_xlabel('$\epsilon(H)$', fontsize=16)

    # ax_[1].tick_params(axis='x', labelsize=12)
    # ax_[1].tick_params(axis='y', labelsize=12)

    handles, labels = ax_.get_legend_handles_labels()
    ax_.legend(reversed(handles), reversed(labels), fontsize=11, loc='lower right', framealpha=1)#, bbox_to_anchor=(1, 1))
    ax.spines[['right', 'top']].set_visible(False)
    fig_.tight_layout()
    fig_.savefig('data/plots/entropy_rmse_err_pairs.pdf', dpi=200)
    plt.show()
    return


def forces_and_time_pdf(exps_types, win_size, delay):
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
                      }


    entropies_all = {exp_type: {e: 0 for e in entropies_dict} for exp_type in exps_types}
       
    fig_, ax_ = plt.subplots(2, 1, figsize=[4.2,5])
    
    titles = {'HH': '$HH+$',
              'HB': '$HH-$',
              'RL': '$HR+$',
              'HL': '$HR-$',
              }
    
    # colors = {'HH': '#E73D1D',
    #           'RL': '#FEC81A',
    #           'HL': '#306AA7',
    #           'HB': '#000000',}
    colors_ = {'HH': '#440154',
              'HB': '#1c9c5c',
              'RL': '#3E64F0',
              'HL': '#46F54E',}
    linestyle_ = {'HH': '-',
                'HB': '--',
                'RL': '-',
                'HL': '--',}
    # colors = [['#440154','#2E99D9'], ['#3E64F0','#2E99D9'], ['#2E99D9','#440154']]
    
    # colors = {'h': '#0f36c4',
    #           'r,h': '#46F54E',
    #           'o': '#1c9c5c',}
    
    fz_max = []
    t_fz_max = []

    dh = DataPlotHelper()

    fig_, ax_[0] = dh.set_axis(fig=fig_, ax=ax_[0])
    fig_, ax_[1] = dh.set_axis(fig=fig_, ax=ax_[1])

    # for phase, phase_steps in zip(['prc', 'poc'], [slice(0+50, 500+50), slice(500+50, 1000+50)]):
    for phase, phase_steps in zip(['whole'], [slice(600,1600)]):
        for i, exp_type in enumerate(exps_types):
            j = 0
            f_name = 'data/entropy_'+exp_type+'__ws'+str(win_size)+'_delay'+str(delay)+'.pkl'

            with open(f_name, 'rb') as f:
                entropies_all[exp_type] = pickle.load(f)

            # for idx_count, fz in enumerate(entropies_all[exp_type]['fz']):
            #     leg = titles[exp_type] if idx_count == 0 else None
            #     ax_[0].plot(np.arange(0, 1, 0.001), fz, color=colors[exp_type], alpha=1, label=leg, linestyle='--')

            leg = titles[exp_type]
            fz_mean = np.mean([fz for fz in entropies_all[exp_type]['fz'] if np.abs(np.mean(fz)) > 0.1], axis=0)
            fz_mean[600:] = 0.5*fz_mean[600:]
            fz_std = np.std([fz for fz in entropies_all[exp_type]['fz'] if np.abs(np.mean(fz)) > 0.1], axis=0)
            ax_[0].plot(np.arange(0, 1, 0.001), fz_mean, color=colors_[exp_type], alpha=1, label=leg, linewidth=2.5, linestyle=linestyle_[exp_type])

            [fz_max.append(np.min(fz)) for fz in entropies_all[exp_type]['fz'] if np.min(fz) < -5]
            [t_fz_max.append(np.argmin(fz)/1000) for fz in entropies_all[exp_type]['fz'] if np.min(fz) < -5]

    ax_[0].set_yticks([-15, -10, -5, 0])
    # ax_[0].set_yticks([-12., -5, 0])
    ax_[0].set_xticks([0.3, 0.4, 0.5, 0.6])
    ax_[0].set_xlabel('$t~[s]$', fontsize=12)
    ax_[0].set_xlim([0.3, .6])
    ax_[0].set_ylabel(r'$\bar{F}_z~[N]$', fontsize=12)
    ax_[0].grid(alpha=0.35)

    my_sns = sns.histplot(t_fz_max, kde=False, ax=ax_[1], color='#2E99D9')#, stat='probability')
    # idx_max_prob = np.argmax(my_sns.get_lines()[0].get_ydata())
    # print('t_imp = ', my_sns.get_lines()[0].get_xdata()[idx_max_prob])
    t_fz_max.sort()
    print('\nt_imp_mean = ', np.mean(t_fz_max))
    print('\nt_imp_std = ', np.std(t_fz_max))
    # ax_[1].set_xticks([0.4, 0.425, 0.450, 0.475, 0.5, 0.525, 0.550])
    ax_[1].set_xticks([0.4, 0.450, 0.5, 0.550])#, 0.600])
    ax_[1].set_xlabel('$t_{imp}~[s]$', fontsize=12)
    ax_[1].set_ylabel('$N_{t_{imp}}$', fontsize=12)
    # ax_[1].set_ylim([0, 0.2])
    ax_[1].set_xlim([0.4, 0.55])
    ax_[1].set_yticks([0, 10, 20, 30, 40])

    ax_[0].tick_params(axis='both', labelsize=9)
    ax_[1].tick_params(axis='both', labelsize=9)

    fig_.legend(framealpha=1, loc='lower right', bbox_to_anchor=(.435, .6325), fontsize=9)
    
    fig_.tight_layout()

    fig_name = 'data/plots/forces_and_time_impact.pdf'
    fig_.savefig(fig_name, dpi=200)

    plt.show()
    return


def agents_positions(exps_types, win_size, shift):
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
                      }

    colors = [['b','g'], ['orange','r'], ['purple','brown']]

    entropies_all = {exp_type: {e: [] for e in entropies_dict} for exp_type in exps_types}

    titles = {'HH': '$HH+$',
              'RL': '$HR+$',
              'HL': '$HR-$',
              'HB': '$HH-$'}
   
    dh = DataPlotHelper(f_size=15)

    fig_, ax_ = plt.subplots(2,2)
    seqs = [((0),(0)), ((0),(1)), ((1),(0)), ((1),(1))]

    # colors = {'h': '#E73D1D',
    #           'r,h': '#FEC81A',
    #           'o': '#306AA7',}

    colors = {'h': '#0f36c4',
              'r,h': '#46F54E',
              'o': '#1c9c5c',}

    for idx_exp, exp_type in enumerate(exps_types):

        f_name = 'data/entropy_'+exp_type+'__ws'+str(win_size)+'_delay'+str(shift)+'.pkl'

        with open(f_name, 'rb') as f:
            entropies_all[exp_type] = pickle.load(f)

            fig, ax = dh.set_axis(fig=fig_, ax=ax_[seqs[idx_exp]], fig_size=[5, 5], xlim_plot=[0, 1], ylim_plot=[0.4, 2.35], ylabel=titles[exp_type])

            for i_, obj in enumerate(entropies_all[exp_type]['pos_obj']):
                offset = 0 if obj[0] < 2.25 else 0.05
                if obj[200] < 1.9 or obj[574] > 1.4:
                    continue
                else:
                    if obj[500] > 1.4:
                        continue
                    leg = None if i_ > 0 or seqs[idx_exp] != ((0), (1)) else '$p_{h}$'
                    ax.plot(np.arange(0, 1, 0.001), entropies_all[exp_type]['pos_hum'][i_], color=colors['h'], label=leg)
                    leg = None if i_ > 0 or seqs[idx_exp] != ((0), (1)) else '$p_{r,h}$'
                    ax.plot(np.arange(0, 1, 0.001), entropies_all[exp_type]['pos_rob'][i_], color=colors['r,h'], label=leg)
                    leg = None if i_ > 0 or seqs[idx_exp] != ((0), (1)) else '$p_{o}$'
                    ax.plot(np.arange(0, 1, 0.001), obj-offset, color=colors['o'], label=leg, linestyle='--')
    
    fig_.legend(framealpha=1, fontsize=10)
    ax_[1, 0].set_xlabel('$t~[s]$')
    ax_[1, 1].set_xlabel('$t~[s]$')
    
    fig_name = 'data/plots/data_charac_positions.pdf'
    fig_.tight_layout()
    fig_.savefig(fig_name)
    plt.show()
    return


def pdfs_heatmap_whole_task(exps_types, win_size, shift, n_bins, u):
    import matplotlib.colors as colors
    import matplotlib.cm as cm
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
                      }


    entropies_all = {exp_type: {e: 0 for e in entropies_dict} for exp_type in exps_types}
    
    fig_, ax_ = plt.subplots(3,4, figsize=[7,5])
    
    max_of_max = 0

    titles = {'HH': '$HH+$',
              'RL': '$HR+$',
              'HL': '$HR-$',
              'HB': '$HH-$'}
    

    lims_pos_obj = np.array([10., 0.])
    lims_pos_hum = np.array([10., 0.])
    lims_pos_rob = np.array([10., 0.])

    for phase, phase_steps in zip(['whole'], [slice(600,1600)]):
        for i, exp_type in enumerate(exps_types):
                
            f_name = 'data/entropy_'+exp_type+'__ws'+str(win_size)+'_delay'+str(shift)+'.pkl'

            with open(f_name, 'rb') as f:
                entropies_all[exp_type] = pickle.load(f)

            pos_obj = np.array(entropies_all[exp_type]['pos_obj'], dtype='float64')
            pos_hum = np.array(entropies_all[exp_type]['pos_hum'], dtype='float64')
            pos_rob = np.array(entropies_all[exp_type]['pos_rob'], dtype='float64')

            lims_pos_obj[0] = min(lims_pos_obj[0], np.min(pos_obj))
            lims_pos_hum[0] = min(lims_pos_hum[0], np.min(pos_hum))
            lims_pos_rob[0] = min(lims_pos_rob[0], np.min(pos_rob))
            lims_pos_obj[1] = max(lims_pos_obj[1], np.max(pos_obj))
            lims_pos_hum[1] = max(lims_pos_hum[1], np.max(pos_hum))
            lims_pos_rob[1] = max(lims_pos_rob[1], np.max(pos_rob))

            # lims_pos_obj = np.array([np.min(pos_obj), np.max(pos_obj)])
            # lims_pos_hum = np.array([np.min(pos_hum), np.max(pos_hum)])
            # lims_pos_rob = np.array([np.min(pos_rob), np.max(pos_rob)])
    
    for phase, phase_steps in zip(['whole'], [slice(600,1600)]):
        for i, exp_type in enumerate(exps_types):
                
            f_name = 'data/entropy_'+exp_type+'__ws'+str(win_size)+'_delay'+str(shift)+'.pkl'

            with open(f_name, 'rb') as f:
                entropies_all[exp_type] = pickle.load(f)

            pos_obj = np.array(entropies_all[exp_type]['pos_obj'], dtype='float64')
            pos_hum = np.array(entropies_all[exp_type]['pos_hum'], dtype='float64')
            pos_rob = np.array(entropies_all[exp_type]['pos_rob'], dtype='float64')

            # lims_pos_obj = np.array([np.min(pos_obj), np.max(pos_obj)])
            # lims_pos_hum = np.array([np.min(pos_hum), np.max(pos_hum)])
            # lims_pos_rob = np.array([np.min(pos_rob), np.max(pos_rob)])

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

            max_of_max = np.max([np.max(pxy_obj_hum), np.max(pxy_rob_hum), np.max(pxy_obj_rob)])

            px_hum = px_hum/np.sum(px_hum, dtype='float64')
            px_obj = px_obj/np.sum(px_obj, dtype='float64')
            px_rob = px_rob/np.sum(px_rob, dtype='float64')

            n_ticks = 11

            my_ytickslabels_0 = np.linspace(lims_pos_obj[0], lims_pos_obj[1], n_ticks)
            my_ytickslabels_1 = np.linspace(lims_pos_rob[0], lims_pos_rob[1], n_ticks)
            my_ytickslabels_2 = np.linspace(lims_pos_obj[0], lims_pos_obj[1], n_ticks)
            my_ytickslabels_0 = ["{:.1f}".format(a) if i in [1, 5, 9] else None for i, a in enumerate(reversed(my_ytickslabels_0))]
            my_ytickslabels_1 = ["{:.1f}".format(a) if i in [1, 5, 9] else None for i, a in enumerate(reversed(my_ytickslabels_1))]
            my_ytickslabels_2 = ["{:.1f}".format(a) if i in [1, 5, 9] else None for i, a in enumerate(reversed(my_ytickslabels_2))]

            my_xtickslabels_0 = np.linspace(lims_pos_hum[0], lims_pos_hum[1], n_ticks)
            my_xtickslabels_1 = np.linspace(lims_pos_hum[0], lims_pos_hum[1], n_ticks)
            my_xtickslabels_2 = np.linspace(lims_pos_rob[0], lims_pos_rob[1], n_ticks)
            my_xtickslabels_0 = ["{:.1f}".format(a) if i in [1, 5, 9] else None for i, a in enumerate(reversed(my_xtickslabels_0))]
            my_xtickslabels_1 = ["{:.1f}".format(a) if i in [1, 5, 9] else None for i, a in enumerate(reversed(my_xtickslabels_1))]
            my_xtickslabels_2 = ["{:.1f}".format(a) if i in [1, 5, 9] else None for i, a in enumerate(reversed(my_xtickslabels_2))]

            ax0 = sns.heatmap(np.flip(pxy_obj_hum), ax=ax_[0,i], cbar=False, cmap=cm.winter, norm=LogNorm())
            ax1 = sns.heatmap(np.flip(pxy_rob_hum), ax=ax_[1,i], cbar=False, cmap=cm.winter, norm=LogNorm())
            ax2 = sns.heatmap(np.flip(pxy_obj_rob), ax=ax_[2,i], cbar=False, cmap=cm.winter, norm=LogNorm())
            # sns.heatmap(pxy_obj_hum, ax=ax_[0,i], norm=colors.PowerNorm(gamma=1), vmin=0 vmax=.5, square=True, cbar=False, xticklabels=my_xtickslabels_0, yticklabels=my_ytickslabels_0, cmap=cm.coolwarm)#norm=colors.LogNorm(vmin=0, vmax=1))
            # sns.heatmap(pxy_rob_hum, ax=ax_[1,i], norm=colors.PowerNorm(gamma=1), vmin=0, vmax=.5, square=True, cbar=False, xticklabels=my_xtickslabels_1, yticklabels=my_ytickslabels_1, cmap=cm.coolwarm)#norm=colors.LogNorm(vmin=0, vmax=1))
            # sns.heatmap(pxy_obj_rob, ax=ax_[2,i], norm=colors.PowerNorm(gamma=1), vmin=0, vmax=.5, square=True, cbar=False, xticklabels=my_xtickslabels_2, yticklabels=my_ytickslabels_2, cmap=cm.coolwarm)#norm=colors.LogNorm(vmin=0, vmax=1))

            ax0.set_yticks(np.linspace(0, n_bins, n_ticks))
            ax1.set_yticks(np.linspace(0, n_bins, n_ticks))
            ax2.set_yticks(np.linspace(0, n_bins, n_ticks))

            ax0.set_xticks(np.linspace(0, n_bins, n_ticks))
            ax1.set_xticks(np.linspace(0, n_bins, n_ticks))
            ax2.set_xticks(np.linspace(0, n_bins, n_ticks))

            ax0.imshow(np.flip(pxy_obj_hum), interpolation='bilinear')
            ax1.imshow(np.flip(pxy_rob_hum), interpolation='bilinear')
            ax2.imshow(np.flip(pxy_obj_rob), interpolation='bilinear')
            
            if i == 0:
                ax0.set_yticklabels(my_ytickslabels_0)
                ax1.set_yticklabels(my_ytickslabels_1)
                ax2.set_yticklabels(my_ytickslabels_2)

            ax0.set_xticklabels(my_xtickslabels_0)
            ax1.set_xticklabels(my_xtickslabels_1)
            ax2.set_xticklabels(my_xtickslabels_2)

    print(max_of_max)
    for axi, title in zip(ax_[0], exps_types):
        axi.set_title(titles[title])
    
    ax_[0,0].set_ylabel('$p_{o}~[m]$')
    ax_[1,0].set_ylabel('$p_{r,h}~[m]$')
    ax_[2,0].set_ylabel('$p_{o}~[m]$')

    for row, lab in zip(ax_, ['$p_{h}~[m]$', '$p_{h}~[m]$', '$p_{r,h}~[m]$']):
        for col in row:
            col.set_xlabel(lab, labelpad=0)
            col.tick_params(axis='y', labelsize=8, rotation=90)
            col.tick_params(axis='x', labelsize=8, rotation=0)
    
    cbar_ax = fig_.add_axes([.92, .1135, .02, .765])
    # fig_.colorbar(a, cax=cbar_ax)
    formatter = tkr.ScalarFormatter(useMathText=True)
    formatter.set_scientific(True)
    formatter.set_powerlimits((-2, 2))

    cb = fig_.colorbar(ax_[-1, -1].collections[0], cax=cbar_ax, format=formatter)
    cb.ax.tick_params(labelsize=8)
    cb.ax.yaxis.set_offset_position('left')
    cb.ax.yaxis.get_offset_text().set_fontsize(8.5)
    cb.update_ticks()
    fig_.tight_layout(rect=[0, 0, .9, 1])
    fig_.subplots_adjust(wspace=0.1, hspace=0.45)
    fig_name = 'data/plots/data_charac_heatmaps.pdf'
    fig_.savefig(fig_name, dpi=200)
    plt.show()
    return


def plot_transfer_entropy(exps_types, win_size, use_increasing_window, shift, delay):
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
                      'ais_hum': 0,
                      'ais_rob': 0,
                      'ais_obj': 0,
                      }
    
    pairs = [['te_obj_to_hum', 'te_obj_to_rob'],
             ['te_hum_to_rob', 'te_rob_to_hum'],
             ['te_hum_to_obj', 'te_rob_to_obj']]
    
    legends_pairs = [[r'$T_{{o}\rightarrow{h}}$',   r'$T_{{o}\rightarrow{r,h}}$'],
                     [r'$T_{{h}\rightarrow{r,h}}$', r'$T_{{r,h}\rightarrow{h}}$'],
                     [r'$T_{{h}\rightarrow{o}}$',   r'$T_{{r,h}\rightarrow{o}}$']]
    
    entropies_all = {exp_type: {e: [] for e in entropies_dict} for exp_type in exps_types}
    
    dh = DataPlotHelper(f_size=15)

    fig_te, ax_te = plt.subplots(4, 3, figsize=[15,9])
    fig_cte, ax_cte = plt.subplots(4, 1, figsize=[5,9])

    colors = ['#440154','#2E99D9']

    colors_ = {'HH': '#440154',
              'HB': '#1c9c5c',
              'RL': '#3E64F0',
              'HL': '#46F54E',}

    for idx_exp, exp_type in enumerate(exps_types):
        print("PLOTTING WITH WINDOW SIZE = ", win_size)

        str_increasing_window = 'increasing_window_' if use_increasing_window else ''
        f_name = 'data/entropy_'+str_increasing_window+exp_type+'__ws'+str(win_size)+'_shift'+str(shift)+'_u'+str(delay)+'.pkl'
        # f_name = 'data/entropy_'+str_increasing_window+exp_type+'__ws'+str(win_size)+'_delay'+str(shift)+'.pkl'

        # f_name = 'data/entropy_'+exp_type+'__ws'+str(win_size)+'_delay'+str(shift)+'.pkl'                    

        with open(f_name, 'rb') as f:
            entropies_all[exp_type] = pickle.load(f)

            fig_cte_, ax_cte_ = dh.set_axis(fig=fig_cte, ax=ax_cte[idx_exp], xlim_plot=[0, 1])#, ylabel='$'+titles[exp_type]+'$')
            leg = r'$T_{h \rightarrow h,r | o}$' if idx_exp == 0 else ''
            ax_cte_.plot(np.arange(0, 1, 0.001), entropies_all[exp_type]['cte_hum_to_rob_cond_obj'], colors[0], label=leg)
            leg = r'$T_{h,r \rightarrow h | o}$' if idx_exp == 0 else ''
            ax_cte_.plot(np.arange(0, 1, 0.001), entropies_all[exp_type]['cte_rob_to_hum_cond_obj'], color=colors[1], linestyle='--', label=leg)

            for idx_pair, pair in enumerate(pairs):

                fig_te_, ax_te_ = dh.set_axis(fig=fig_te, ax=ax_te[idx_exp, idx_pair], xlim_plot=[0, 1])

                ax_te_.plot(np.arange(0, 1, 0.001), entropies_all[exp_type][pair[0]], label=legends_pairs[idx_pair][0], color=colors[0])
                ax_te_.plot(np.arange(0, 1, 0.001), entropies_all[exp_type][pair[1]], label=legends_pairs[idx_pair][1], color=colors[1], linestyle='dashed')

                ax_te_.axvspan(xmin=0.467-0.025, xmax=0.467+0.025, alpha=0.15, color='#2E99D9', lw=0)

                ax_te_.set_ylim([-0.1, 0.3])
                ax_cte_.set_ylim([-0.005, 0.1])
                
                if idx_pair != 0:
                    ax_te_.set_yticks([])
                ax_te_.hlines(0,xmin=0,xmax=1,color='k')

                if idx_exp != len(ax_te)-1:
                    ax_te_.set_xticks([])
                else:
                    ax_te_.set_xlabel('$t~[s]$')
                    
                print(idx_exp)
                if idx_exp == 0:
                    ax_te_.legend(fontsize=13, loc='upper right', bbox_to_anchor=(1, 1.02), framealpha=1)
                    ax_cte_.legend(fontsize=13, loc='upper right', bbox_to_anchor=(.37, 1.02), framealpha=1)

    
    # for axi in ax_cte:
    plt.ticklabel_format(axis='y', style='sci')

    for axi, title in zip(ax_te[:,0], ['$HH+$', '$HH-$', '$HR+$', '$HR-$']): 
        axi.set_ylabel(title, fontname='Times new roman')
    
    for axi, title in zip(ax_cte, ['$HH+$', '$HH-$', '$HR+$', '$HR-$']): 
        axi.set_ylabel(title, fontname='Times new roman')
        
    for row in ax_te:
        for col in row:
            col.tick_params(axis='both', labelsize=11)

    ax_cte[3].set_xlabel('$t~[s]$')

    fig_te.tight_layout()
    fig_cte.tight_layout()
    fig_te.subplots_adjust(wspace=0.15, hspace=0.2)
    fig_te.savefig('data/plots/te_global_exps_ws'+str(win_size)+'_'+str_increasing_window[:-1]+'_delay_'+str(shift)+'.pdf', dpi=200)
    fig_cte.savefig('data/plots/cte_global_exps_ws'+str(win_size)+'_'+str_increasing_window[:-1]+'_delay_'+str(shift)+'.pdf', dpi=200)
    ax_te[0,1].set_title('TE')
    plt.show()
    return


if __name__ == '__main__':
    # ce_plot(exps_types=['HH', 'HB', 'RL', 'HL'], win_size=300, delay=100, use_increasing_window=True)
    # ce_plot_rmse_lollipop(exps_types=['HH', 'HB', 'RL', 'HL'], win_size=300, shift=100)
    # forces_and_time_pdf(exps_types=['HH', 'HB', 'RL', 'HL'], win_size=300, delay=100)
    # agents_positions(exps_types=['HH', 'HB', 'RL', 'HL'], win_size=300, shift=100)
    # pdfs_heatmap_whole_task(exps_types=['HH', 'HB', 'RL', 'HL'], win_size=300, shift=100, n_bins=80)
    plot_transfer_entropy(exps_types=['HH', 'HB', 'RL', 'HL'], win_size=300, use_increasing_window=True, shift=100, delay=0)



    ### DEPRECATED
    # ce_plot_rmse(exps_types=['HH', 'HB', 'RL', 'HL'], win_size=300, shift=100)

    pass
