import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from entropy_utils_jpype import *
from data_plot_class import DataPlotHelper

def ce_plot(exps_types, win_size, delay):
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

    colors = [['#E73D1D','#306AA7'], ['#FEC81A','#E73D1D'], ['#306AA7','#FEC81A']]
    
    for idx_exp, exp_type in enumerate(exps_types):

        f_name = 'data/entropy_'+exp_type+'__ws'+str(win_size)+'_delay'+str(delay)+'.pkl'

        with open(f_name, 'rb') as f:
            entropies_all[exp_type] = pickle.load(f)

            for idx_pair, pair in enumerate(pairs):
        
                fig_name = 'data/plots/entropy_'+exp_type+'_ws'+str(win_size)+'_shift'+str(delay)+'_conditional_pair'+str(idx_pair+1)+'.png'
                fig, ax = dh.set_axis(fig=fig_, ax=ax_[idx_exp, idx_pair], xlim_plot=[0, 1])

                ax.plot(np.arange(0, 1, 0.001), entropies_all[exp_type][pair[0]], label=legends_pairs[idx_pair][0], color=colors[idx_pair][0])
                ax.plot(np.arange(0, 1, 0.001), entropies_all[exp_type][pair[1]], label=legends_pairs[idx_pair][1], color=colors[idx_pair][1], linestyle='dashed')
                
                ax.set_ylim([0, 4])
                if idx_pair != 0:
                    ax.set_yticks([])

                if idx_exp != len(ax_)-1:
                    ax.set_xticks([])
                else:
                    ax.set_xlabel('$t~[s]$')
                    
                if idx_exp == 0:
                    ax.legend(fontsize=9, loc='upper right', bbox_to_anchor=(1.06, 1.225), framealpha=1)

                # grids creation
                x_grids = list(np.arange(0,1,0.1))
                n_divisions = 4
                alpha_grids = 0.05
                y_grids = list(np.arange(0, 4, 4/n_divisions))
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
    # fig_.savefig('data/plots/entropy_global_exps.pdf', dpi=200)
    plt.show()
    return


def forces_and_time_pdf(exps_types, win_size, delay):
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
       
    fig_, ax_ = plt.subplots(2, 1, figsize=[4,5])
    
    titles = {'HH': '$HH+$',
              'HB': '$HH-$',
              'RL': '$HR+$',
              'HL': '$HR-$',
              }
    
    colors = {'HH': '#E73D1D',
              'RL': '#FEC81A',
              'HL': '#306AA7',
              'HB': '#000000',}
    # colors = {'HH': '#f1eef6',
    #           'RL': '#bdc9e1',
    #           'HL': '#74a9cf',
    #           'HH-': '#0570b0',}
    
    fz_max = []
    t_fz_max = []

    # for phase, phase_steps in zip(['prc', 'poc'], [slice(0+50, 500+50), slice(500+50, 1000+50)]):
    for phase, phase_steps in zip(['whole'], [slice(600,1600)]):
        for i, exp_type in enumerate(exps_types):
            j = 0
            f_name = 'data/entropy_'+exp_type+'__ws'+str(win_size)+'_delay'+str(delay)+'.pkl'

            with open(f_name, 'rb') as f:
                entropies_all[exp_type] = pickle.load(f)

            for idx_count, fz in enumerate(entropies_all[exp_type]['fz']):
                leg = titles[exp_type] if idx_count == 0 else None
                ax_[0].plot(np.arange(0, 1, 0.001), fz, color=colors[exp_type], alpha=0.5, label=leg)

            [fz_max.append(np.min(fz)) for fz in entropies_all[exp_type]['fz'] if np.min(fz) < -5]
            [t_fz_max.append(np.argmin(fz)/1000) for fz in entropies_all[exp_type]['fz'] if np.min(fz) < -5]

    ax_[0].set_yticks([-25, -15, -5, 0])
    ax_[0].set_xlabel('$t~[s]$')
    ax_[0].set_xlim([0, 1])
    ax_[0].set_ylabel('$F_z~[N]$')

    my_sns = sns.histplot(t_fz_max, kde=True, ax=ax_[1], stat='probability')
    idx_max_prob = np.argmax(my_sns.get_lines()[0].get_ydata())
    print('t_occur = ', my_sns.get_lines()[0].get_xdata()[idx_max_prob])
    ax_[1].set_xticks([0.4, 0.425, 0.450, 0.475, 0.5, 0.525, 0.550])
    ax_[1].set_xticks([0.4, 0.450, 0.5, 0.550, 0.600])
    ax_[1].set_xlabel('$t_{occur}~[s]$')
    ax_[1].set_ylabel('$p(t_{occur})$')
    ax_[1].set_ylim([0, 0.2])

    ax_[0].tick_params(axis='both', labelsize=11)
    ax_[1].tick_params(axis='both', labelsize=11)

    fig_.legend(framealpha=1, loc='lower right', bbox_to_anchor=(.922, .625), fontsize=9)
    
    fig_.tight_layout()

    fig_name = 'data/plots/forces_and_time_impact.pdf'
    # fig_.savefig(fig_name, dpi=200)

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

    colors = {'h': '#E73D1D',
              'r,h': '#FEC81A',
              'o': '#306AA7',}

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
                    ax.plot(np.arange(0, 1, 0.001), obj-offset, color=colors['o'], label=leg)
    
    fig_.legend(framealpha=1)
    ax_[1, 0].set_xlabel('$t~[s]$')
    ax_[1, 1].set_xlabel('$t~[s]$')
    
    fig_name = 'data/plots/data_charac_positions.pdf'
    fig_.tight_layout()
    # fig_.savefig(fig_name)
    plt.show()
    return


def pdfs_heatmap_whole_task(exps_types, win_size, shift):
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
    
    for phase, phase_steps in zip(['whole'], [slice(600,1600)]):
        for i, exp_type in enumerate(['HH', 'HB', 'RL', 'HL']):
                
            f_name = 'data/entropy_'+exp_type+'__ws'+str(win_size)+'_delay'+str(shift)+'.pkl'

            with open(f_name, 'rb') as f:
                entropies_all[exp_type] = pickle.load(f)

            pos_obj = np.array(entropies_all[exp_type]['pos_obj'], dtype='float64')
            pos_hum = np.array(entropies_all[exp_type]['pos_hum'], dtype='float64')
            pos_rob = np.array(entropies_all[exp_type]['pos_rob'], dtype='float64')

            lims_pos_obj = np.array([np.min(pos_obj), np.max(pos_obj)])
            lims_pos_hum = np.array([np.min(pos_hum), np.max(pos_hum)])
            lims_pos_rob = np.array([np.min(pos_rob), np.max(pos_rob)])

            n_bins = 10
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

            my_xtickslabels_0 = np.linspace(lims_pos_obj[0], lims_pos_obj[1], 10)
            my_xtickslabels_1 = np.linspace(lims_pos_rob[0], lims_pos_rob[1], 10)
            my_xtickslabels_2 = np.linspace(lims_pos_obj[0], lims_pos_obj[1], 10)
            my_xtickslabels_0 = ["{:.1f}".format(a) if (i%3)==0 else None for i, a in enumerate(reversed(my_xtickslabels_0))]
            my_xtickslabels_1 = ["{:.1f}".format(a) if (i%3)==0 else None for i, a in enumerate(reversed(my_xtickslabels_1))]
            my_xtickslabels_2 = ["{:.1f}".format(a) if (i%3)==0 else None for i, a in enumerate(reversed(my_xtickslabels_2))]

            my_ytickslabels_0 = np.linspace(lims_pos_hum[0], lims_pos_hum[1], 10)
            my_ytickslabels_1 = np.linspace(lims_pos_hum[0], lims_pos_hum[1], 10)
            my_ytickslabels_2 = np.linspace(lims_pos_rob[0], lims_pos_rob[1], 10)
            my_ytickslabels_0 = ["{:.1f}".format(a) if (i%4)==0 else None for i, a in enumerate(reversed(my_ytickslabels_0))]
            my_ytickslabels_1 = ["{:.1f}".format(a) if (i%4)==0 else None for i, a in enumerate(reversed(my_ytickslabels_1))]
            my_ytickslabels_2 = ["{:.1f}".format(a) if (i%4)==0 else None for i, a in enumerate(reversed(my_ytickslabels_2))]

            sns.heatmap(np.flip(pxy_obj_hum), ax=ax_[0,i], vmin=0, vmax=.51, center=0.15, cbar=False, xticklabels=my_xtickslabels_0, yticklabels=my_ytickslabels_0, cmap=cm.viridis)
            sns.heatmap(np.flip(pxy_rob_hum), ax=ax_[1,i], vmin=0, vmax=.51, center=0.15, cbar=False, xticklabels=my_xtickslabels_1, yticklabels=my_ytickslabels_1, cmap=cm.viridis)
            sns.heatmap(np.flip(pxy_obj_rob), ax=ax_[2,i], vmin=0, vmax=.51, center=0.15, cbar=False, xticklabels=my_xtickslabels_2, yticklabels=my_ytickslabels_2, cmap=cm.viridis)
            # sns.heatmap(pxy_obj_hum, ax=ax_[0,i], norm=colors.PowerNorm(gamma=1), vmin=0, vmax=.5, square=True, cbar=False, xticklabels=my_xtickslabels_0, yticklabels=my_ytickslabels_0, cmap=cm.coolwarm)#norm=colors.LogNorm(vmin=0, vmax=1))
            # sns.heatmap(pxy_rob_hum, ax=ax_[1,i], norm=colors.PowerNorm(gamma=1), vmin=0, vmax=.5, square=True, cbar=False, xticklabels=my_xtickslabels_1, yticklabels=my_ytickslabels_1, cmap=cm.coolwarm)#norm=colors.LogNorm(vmin=0, vmax=1))
            # sns.heatmap(pxy_obj_rob, ax=ax_[2,i], norm=colors.PowerNorm(gamma=1), vmin=0, vmax=.5, square=True, cbar=False, xticklabels=my_xtickslabels_2, yticklabels=my_ytickslabels_2, cmap=cm.coolwarm)#norm=colors.LogNorm(vmin=0, vmax=1))
    
    print(max_of_max)
    for axi, title in zip(ax_[0], ['HH', 'HB', 'RL', 'HL']):
        axi.set_title(titles[title], fontname='Times new roman')

    
    ax_[0,0].set_ylabel('$p_{h}~[m]$')
    ax_[1,0].set_ylabel('$p_{h}~[m]$')
    ax_[2,0].set_ylabel('$p_{r,h}~[m]$')

    for row, lab in zip(ax_, ['$p_{o}~[m]$', '$p_{r,h}~[m]$', '$p_{o}~[m]$']):
        for col in row:
            col.set_xlabel(lab, labelpad=0)
            col.tick_params(axis='both', labelsize=9)
    
    cbar_ax = fig_.add_axes([.92, .1135, .02, .765])
    # fig_.colorbar(a, cax=cbar_ax)
    cb = fig_.colorbar(ax_[-1, -1].collections[0], cax=cbar_ax)
    cb.ax.tick_params(labelsize=9)
    fig_.tight_layout(rect=[0, 0, .9, 1])
    fig_.subplots_adjust(wspace=0.25, hspace=0.5)
    fig_name = 'data/plots/data_charac_heatmaps.pdf'
    # fig_.savefig(fig_name)
    plt.show()
    return


if __name__ == '__main__':
    # ce_plot(exps_types=['HH', 'HB', 'RL', 'HL'], win_size=300, delay=100)
    # forces_and_time_pdf(exps_types=['HH', 'HB', 'RL', 'HL'], win_size=300, delay=100)
    # agents_positions(exps_types=['HH', 'HB', 'RL', 'HL'], win_size=300, shift=100)
    # pdfs_heatmap_whole_task(exps_types=['HH', 'HB', 'RL', 'HL'], win_size=300, shift=100)

    pass
