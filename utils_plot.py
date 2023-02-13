import pandas as pd
import os
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt


def plot_from_files(file_paths, x, y=[], save_path=None, filter={}, save_name='', weight=False, curr_a_t2plot='0.05',
                    reuse=False, only_e=False, order=None):
    if len(y) == 0:
        if reuse:
            y_col_options = {'FDR', 'TDR', 'r-variance'}
        else:
            y_col_options = {'FDR', 'TDR'}
    else:
        y_col_options = y
    results = pd.DataFrame({})

    for i, file_path in enumerate(file_paths):
        result = pd.read_pickle(file_path)
        result['Index'] = i
        for key in filter:
            if key == 'random_params':
                result = result[result[key].astype(int) == int(filter[key])]
            else:
                result = result[result[key].astype(str) == str(filter[key])]
        if result.empty:
            continue
        if result['alpha_t'].values[0] != curr_a_t2plot and 'E_value' in result['algorithm'].values[0] and x != 'alpha_t':
            continue
        if 'E_value' in result['algorithm'].values[0] and (not hasattr(result, 'weight_metric') or result['weight_metric'].values[0] != 't-test') and not weight:
            continue
        if not 'E_value' in result['algorithm'].values[0] and (weight or only_e):
            continue
        if not 'E_value' in result['algorithm'].values[0]:
            result['n_e_value'] = 1

        results = pd.concat([results, result])
    if results.empty:
        print('No results for this set of filters.')
        return
    if x == 'mu_o':
        results['mu_o'] = results['mu_o'].astype(float)
    elif x == 'alpha_t':
        results['alpha_t'] = results['alpha_t'].astype(float)

    for y_col in y_col_options:
        plot_figure(x, y_col, results, save_path, save_name, weight, order)



wanted_legend_order = ['AdaDetect', 'E-AdaDetect']
wanted_legend_order_OC = ['OC-Conformal', 'E-OC-Conformal']
wanted_legend_order_weight = ['E-AdaDetect (uniform)', 'E-AdaDetect (avg. score)', 'E-AdaDetect (t-test)']
wanted_legend_order_OC_weight = ['E-OC-Conformal (uniform)', 'E-OC-Conformal (avg. score)', 'E-OC-Conformal (t-test)']


def get_label(algorithm, a_t, weight_metric, weight, x):
    label = ''
    if 'E_value' in algorithm:
        label += 'E-value ' + algorithm.lstrip('E_value_')
        if x != 'alpha_t':
            label += ' ' + a_t
        if weight:
            label += ' ' + weight_metric
        else:
            if weight_metric != 'uniform':
                label += ' ' + weight_metric
    else:
        label += algorithm
    return legend4paper(label, weight)


def legend4paper(x, weight):
    legend4paper_dict = {'AdaDetectERM': 'AdaDetect',
                         'ConformalOCC': 'OC-Conformal',
                         'E-value AdaDetectERM': 'E-AdaDetect',
                         'E-value AdaDetectERM 0.05': 'E-AdaDetect',
                         'E-value AdaDetectERM 0.05_0.07': 'E-AdaDetect',
                         'E-value ConformalOCC': 'E-value OC-Conformal',
                         'E-value ConformalOCC 0.05': 'E-value OC-Conformal',
                         'E-value ConformalOCC 0.05_0.07': 'E-OC-Conformal',

                         'E-value AdaDetectERM t-test': 'E-AdaDetect',
                         'E-value AdaDetectERM 0.05 t-test': 'E-AdaDetect',
                         'E-value AdaDetectERM 0.05_0.07 t-test': 'E-AdaDetect',
                         'E-value ConformalOCC t-test': 'E-OC-Conformal',
                         'E-value ConformalOCC 0.05 t-test': 'E-OC-Conformal',
                         'E-value ConformalOCC 0.05_0.07 t-test': 'E-OC-Conformal',
                         }
    # for weight
    if weight:
        legend4paper_dict = {
                               'AdaDetectERMcv': 'AdaDetect',
                               'E-value AdaDetectERM 0.05 t-test': 'E-AdaDetect (t-test)',
                               'E-value AdaDetectERM 0.05 uniform': 'E-AdaDetect (uniform)',
                               'E-value AdaDetectERM 0.05 avg_score': 'E-AdaDetect (avg. score)',
                               'E-value AdaDetectERM 0.05_0.07 t-test': 'E-AdaDetect (t-test)',
                               'E-value AdaDetectERM 0.05_0.07 uniform': 'E-AdaDetect (uniform)',
                               'E-value AdaDetectERM 0.05_0.07 avg_score': 'E-AdaDetect (avg. score)',
                            }
    if x in legend4paper_dict.keys():
        return legend4paper_dict[x]
    return x


def desc4paper(x, reuse, box):
    desc4paper_dict = {'mu_o': 'Signal amplitude',
                       'TDR': 'Power',
                       'n_cal': 'Size of calibration set',
                       'actual_n_train': 'Number of training samples',
                       'alpha_t': r'$\alpha_{bh}$',
                       'n_e_value': 'Number of iterations',
                       'dataset': 'Dataset',
                       'r-variance': 'Variance',
                       }
    if x in desc4paper_dict.keys():
        return desc4paper_dict[x]
    else:
        return x


def plot_figure(x, y_col, results, save_path, save_name, weight, order):
    palette = {
               'AdaDetect': 'red',
               'OC-Conformal': 'red',
               'E-AdaDetect': 'blue',
               'E-OC-Conformal': 'blue',
               'E-AdaDetect (t-test)': 'blue',
               'E-AdaDetect (uniform)': 'limegreen',
               'E-AdaDetect (avg. score)': 'purple',
               }

    markers = {
                'AdaDetect': '>',
                'E-AdaDetect': 'o',
                'OC-Conformal': '>',
                'E-OC-Conformal': 'o',
                'E-AdaDetect (t-test)': 'o',
                'E-AdaDetect (uniform)': 'X',
                'E-AdaDetect (avg. score)': '*',
                }
    if not os.path.exists(save_path + '/for_paper/legend/'):
        os.makedirs(save_path + '/for_paper/legend/')
    if not os.path.exists(save_path + '/for_paper/no_legend/'):
        os.makedirs(save_path + '/for_paper/no_legend/')
    flds = ['algorithm', 'alpha_t', 'n_e_value', 'weight_metric']
    hue = results[flds].apply(
        lambda row: f"{get_label(algorithm=row.algorithm, a_t=row.alpha_t, weight_metric=row.weight_metric, weight=weight, x=x)}" if hasattr(row, 'weight_metric') and not pd.isnull(row.weight_metric) else
        f"{get_label(algorithm=row.algorithm, a_t=row.alpha_t, weight_metric='uniform', weight=weight, x=x)}", axis=1)

    # arrange markers list
    _, idx = np.unique(hue.values, return_index=True)
    hue_order = hue.values[np.sort(idx)]
    markers_list = [markers[h] for h in hue_order]

    font_size = 14
    font_size_labels = 15
    font_size_legend = 15
    if type(y_col) is set:
        return
    for box in [True, False]:
        desc = '_point' if not box else ''
        for legend in [True, False]:
            fig = plt.figure(figsize=([5, 3]))
            ax = fig.add_subplot(111)
            if box:
                ax1 = sns.boxplot(x=x, y=y_col, hue=hue, data=results, ax=ax, palette=palette, order=order)
            else:
                if x == 'dataset':
                    ax1 = sns.barplot(x=x, y=y_col, hue=hue, data=results, ax=ax, palette=palette, order=order)
                    desc = '_bar'
                else:
                    ax1 = sns.pointplot(x=x, y=y_col, hue=hue, data=results, ax=ax, palette=palette, markers=markers_list, order=order)

            ax1.set_xlabel(desc4paper(x, reuse=bool('reuse' in save_path and 'no_reuse' not in save_path), box=box),
                           fontsize=font_size_labels)
            ax1.set_ylabel(desc4paper(y_col, reuse=bool('reuse' in save_path and 'no_reuse' not in save_path), box=box),
                           fontsize=font_size_labels)
            plt.rc('legend', fontsize = font_size_legend)
            if x == 'mu_o' or x == 'dataset':
                plt.xticks(rotation=55)
            if x == 'mu_o':
                locs, labels = plt.xticks()
                new_locs = []
                new_labels = []
                for i in range(len(locs)):
                    if i%2 == 0:
                        new_locs.append(locs[i])
                        new_labels.append(labels[i])
                plt.xticks(new_locs, new_labels)
            if not box:
                plt.setp(ax1.lines, alpha=.3)
            if y_col == 'FDR':
                ax.axhline(y = 0.1, color = 'black', linestyle = 'dashed')
                ax1.set_ylim(0,1)
            plt.tick_params(axis='both', which='major', labelsize=font_size)
            if not legend:
                ax1.get_legend().remove()
                plt.savefig(
                    save_path + '/for_paper/no_legend/' + y_col.replace(' ', '-').replace('/', '-') + save_name + desc + '.pdf', bbox_inches="tight")
            else:
                plt.legend()
                #sns.move_legend(ax, "center left", bbox_to_anchor=(1, 0.5))
                plt.savefig(
                    save_path + '/for_paper/legend/' + y_col.replace(' ', '-').replace('/', '-') + save_name + desc + '.pdf', bbox_inches="tight")

