import pandas as pd
import os
import errno
import seaborn as sns
import numpy as np
from numpy import load
import matplotlib.pyplot as plt
import yaml


def plot_from_files(file_paths, x, y=[], save_path=None, filter={}, save_name='', weight=False, curr_a_t2plot='0.05',
                    reuse=False, only_e=False, order=None, wo_e=False, no_rank_desc=False):
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
                if key == 'mu_o':
                    result = result[result[key].astype(float) == float(filter[key])]
                else:
                    result = result[result[key].astype(str) == str(filter[key])]
        if result.empty:
            continue
        if wo_e and 'E_value' in result['algorithm'].values[0]:
            continue
        if curr_a_t2plot is not None and result['alpha_t'].values[0] != curr_a_t2plot and 'E_value' in result['algorithm'].values[0] and x != 'alpha_t':
            continue
        if 'E_value' in result['algorithm'].values[0] and (not hasattr(result, 'weight_metric') or result['weight_metric'].values[0] != 't-test') and not weight:
            continue
        if (not 'E_value' in result['algorithm'].values[0] and only_e) or (not 'E_value' in result['algorithm'].values[0] and not 'Calibrator' in result['algorithm'].values[0] and weight):
            continue
        if not 'E_value' in result['algorithm'].values[0] and not 'Calibrator' in result['algorithm'].values[0]:
            result['n_e_value'] = 1
        if weight and (float(result['mu_o'].values[0]) > 3.6 or float(result['mu_o'].values[0]) < 2):
            continue
        if x == 'mu_o':
            if 'test_p_0.1' in save_path:
                if 'LR' in save_name:
                    if float(result['mu_o'].values[0]) > 4.6 or float(result['mu_o'].values[0]) < 1.8:
                        continue
                elif 'OCSVM' in save_name:
                    if float(result['mu_o'].values[0]) > 4.6 or float(result['mu_o'].values[0]) < 1.8:
                        continue
            elif 'test_p_0.5' in save_path:
                if 'LR' in save_name:
                    if float(result['mu_o'].values[0]) > 2.5 or float(result['mu_o'].values[0]) < 0.5:
                        continue
                elif 'OCSVM' in save_name:
                    if float(result['mu_o'].values[0]) > 4.6 or float(result['mu_o'].values[0]) < 1.8:
                        continue
            elif 'test_p_0.05' in save_path:
                if 'LR' in save_name:
                    if float(result['mu_o'].values[0]) > 6.4 or float(result['mu_o'].values[0]) < 3.6:
                        continue
                elif 'OCSVM' in save_name:
                    if float(result['mu_o'].values[0]) > 4.6 or float(result['mu_o'].values[0]) < 1.8:
                        continue
            elif 'test_p_0.2' in save_path:
                if 'LR' in save_name:
                    if float(result['mu_o'].values[0]) > 3.5 or float(result['mu_o'].values[0]) < 1.1:
                        continue
                elif 'OCSVM' in save_name:
                    if float(result['mu_o'].values[0]) > 4.6 or float(result['mu_o'].values[0]) < 1.8:
                        continue
            elif 'test_p_0.3' in save_path:
                if 'LR' in save_name:
                    if float(result['mu_o'].values[0]) > 3 or float(result['mu_o'].values[0]) < 0.9:
                        continue
                elif 'OCSVM' in save_name:
                    if float(result['mu_o'].values[0]) > 4.6 or float(result['mu_o'].values[0]) < 1.8:
                        continue
            elif 'test_p_0.4' in save_path:
                if 'LR' in save_name:
                    if float(result['mu_o'].values[0]) > 2.5 or float(result['mu_o'].values[0]) < 0.7:
                        continue
                elif 'OCSVM' in save_name:
                    if float(result['mu_o'].values[0]) > 4.6 or float(result['mu_o'].values[0]) < 1.8:
                        continue
        if x == 'test_purity' and result['dataset'].values[0] == 'artificial_gaussian':
            if result['model'].values[0] == 'OC-SVM':
                if float(result['test_purity'].values[0]) == 0.05:
                    if float(result['mu_o'].values[0]) != 3.6:
                        continue
                elif float(result['test_purity'].values[0]) == 0.1:
                    if float(result['mu_o'].values[0]) != 3.4:
                        continue
                elif float(result['test_purity'].values[0]) == 0.2:
                    if float(result['mu_o'].values[0]) != 3.2:
                        continue
                elif float(result['test_purity'].values[0]) == 0.3:
                    if float(result['mu_o'].values[0]) != 3.1:
                        continue
                elif float(result['test_purity'].values[0]) == 0.4:
                    if float(result['mu_o'].values[0]) != 3:
                        continue
                elif float(result['test_purity'].values[0]) == 0.5:
                    if float(result['mu_o'].values[0]) != 3:
                        continue
            elif result['model'].values[0] == 'LogisticRegression':
                if float(result['test_purity'].values[0]) == 0.05:
                    if float(result['mu_o'].values[0]) != 6.5:
                        continue
                elif float(result['test_purity'].values[0]) == 0.1:
                    if float(result['mu_o'].values[0]) != 4.3:
                        continue
                elif float(result['test_purity'].values[0]) == 0.2:
                    if float(result['mu_o'].values[0]) != 2.9:
                        continue
                elif float(result['test_purity'].values[0]) == 0.3:
                    if float(result['mu_o'].values[0]) != 2.3:
                        continue
                elif float(result['test_purity'].values[0]) == 0.4:
                    if float(result['mu_o'].values[0]) != 2:
                        continue
                elif float(result['test_purity'].values[0]) == 0.5:
                    if float(result['mu_o'].values[0]) != 1.8:
                        continue
        results = pd.concat([results, result])
    if results.empty:
        print('No results for this set of filters.')
        return
    if x == 'mu_o':
        results['mu_o'] = results['mu_o'].astype(float)
    elif x == 'alpha_t':
        results['alpha_t'] = results['alpha_t'].astype(float)
    elif x == 'alpha':
        results['alpha'] = results['alpha'].astype(float)

    for y_col in y_col_options:
        plot_figure(x, y_col, results, save_path, save_name, weight, order, at_legend=(curr_a_t2plot == None), no_rank_desc=no_rank_desc)


def get_label(algorithm, a_t, weight_metric, weight, x, calibrator_type, r, at_legend, no_rank_desc):
    label = ''
    if 'E_value' in algorithm:
        label += 'E-value ' + algorithm.lstrip('E_value_')
        if at_legend and x != 'alpha_t' and x != 'alpha':
            return legend4paper(label, weight) + ' ' + a_t
        if weight:
            label += ' ' + weight_metric
    elif 'Calibrator' in algorithm:
        label += algorithm + ' ' + calibrator_type
        if calibrator_type == 'soft-rank' and not no_rank_desc:
            label += ' r=' + str(int(r))
    else:
        label += algorithm
    return legend4paper(label, weight, no_rank_desc)


def legend4paper(x, weight, no_rank_desc=False):
    legend4paper_dict = {'AdaDetectERM': 'AdaDetect',
                         'ConformalOCC': 'OC-Conformal',
                         'E-value AdaDetectERM': 'E-AdaDetect',
                         'E-value ConformalOCC': 'E-OC-Conformal',

                         'CalibratorAdaDetectERM Shafer': 'Shafer AdaDetect',
                         'CalibratorConformalOCC Shafer': 'Shafer OC-Conformal',
                         'CalibratorAdaDetectERM VS': 'VS AdaDetect',
                         'CalibratorConformalOCC VS': 'VS OC-Conformal',
                         'CalibratorAdaDetectERM soft-rank': 'soft-rank AdaDetect',
                         'CalibratorConformalOCC soft-rank': 'soft-rank OC-Conformal',
                         'CalibratorAdaDetectERM integral': 'integral AdaDetect',
                         'CalibratorConformalOCC integral': 'integral OC-Conformal',
                         }
    # for weight
    if weight:
        legend4paper_dict = {
                               'AdaDetectERMcv': 'AdaDetect',
                               'E-value AdaDetectERM t-test': 'E-AdaDetect (t-test)',
                               'E-value AdaDetectERM uniform': 'E-AdaDetect (uniform)',
                               'E-value AdaDetectERM avg_score': 'E-AdaDetect (avg. score)',
                            }
    if 'soft-rank' in x and not no_rank_desc:
        r_idx = x.find(' r=')
        return legend4paper_dict[x[:r_idx]] + x[r_idx:]
    if x in legend4paper_dict.keys():
        return legend4paper_dict[x]
    return x


def desc4paper(x):
    desc4paper_dict = {'mu_o': 'Signal amplitude',
                       'TDR': 'Power',
                       'n_cal': 'Size of calibration set',
                       'actual_n_train': 'Number of training samples',
                       'alpha_t': r'$\alpha_{bh}$',
                       'n_e_value': 'Number of iterations',
                       'dataset': 'Dataset',
                       'r-variance': 'Variance',
                       'test_purity': 'Outliers\' proportion',
                       'alpha': 'Target FDR level',
                       'soft_rank_r': '$r$',
                       }
    if x in desc4paper_dict.keys():
        return desc4paper_dict[x]
    else:
        return x


def plot_figure(x, y_col, results, save_path, save_name, weight, order, at_legend=False, no_rank_desc=False):
    palette = {
               'AdaDetect': 'red',
               'OC-Conformal': 'red',
               'E-AdaDetect': 'blue',
               'E-OC-Conformal': 'blue',
               'E-AdaDetect (t-test)': 'blue',
               'E-AdaDetect (uniform)': 'limegreen',
               'E-AdaDetect (avg. score)': 'deeppink',
               'Shafer AdaDetect': 'deeppink',
               'Shafer OC-Conformal': 'deeppink',
               'VS AdaDetect': 'limegreen',
               'VS OC-Conformal': 'limegreen',
               'soft-rank AdaDetect': 'purple',
               'soft-rank OC-Conformal': 'purple',
               'integral AdaDetect': 'darkorange',
               'integral OC-Conformal': 'darkorange',
               }

    markers = {
                'AdaDetect': '>',
                'E-AdaDetect': 'o',
                'OC-Conformal': '>',
                'E-OC-Conformal': 'o',
                'E-AdaDetect (t-test)': 'o',
                'E-AdaDetect (uniform)': 'X',
                'E-AdaDetect (avg. score)': '*',
                'Shafer AdaDetect': 'D',
                'Shafer OC-Conformal': 'D',
                'VS AdaDetect': 'P',
                'VS OC-Conformal': 'P',
                'soft-rank AdaDetect': 'X',
                'soft-rank OC-Conformal': 'X',
                'integral AdaDetect': 'p',
                'integral OC-Conformal': 'p',
                }

    if not os.path.exists(save_path + '/for_paper/legend/'):
        try:
            os.makedirs(save_path + '/for_paper/legend/')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    if not os.path.exists(save_path + '/for_paper/no_legend/'):
        try:
            os.makedirs(save_path + '/for_paper/no_legend/')
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    flds = ['algorithm', 'alpha_t', 'n_e_value', 'weight_metric', 'calibrator_type', 'soft_rank_r']
    hue = results[flds].apply(
        lambda row: f"{get_label(algorithm=row.algorithm, a_t=row.alpha_t, weight_metric=row.weight_metric, weight=weight, x=x, calibrator_type=row.calibrator_type, r=row.soft_rank_r, at_legend=at_legend, no_rank_desc=no_rank_desc)}" if hasattr(row, 'soft_rank_r') and not pd.isnull(row.soft_rank_r) else
        f"{get_label(algorithm=row.algorithm, a_t=row.alpha_t, weight_metric=row.weight_metric, weight=weight, x=x, calibrator_type=row.calibrator_type, r=0, at_legend=at_legend, no_rank_desc=no_rank_desc)}", axis=1)

    # check if all hue exist in palette and markers
    use_default_palette_markers = False
    for h in hue:
        if h not in palette or h not in markers:
            use_default_palette_markers = True
    if use_default_palette_markers:
        print('Not all hue descriptions available in the predefined colors and markers.\n'
              'Using bright palette and o markers.')
        palette = 'bright'
        markers_list = 'o'
    else:
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

            ax1.set_xlabel(desc4paper(x), fontsize=font_size_labels)
            ax1.set_ylabel(desc4paper(y_col), fontsize=font_size_labels)
            plt.rc('legend', fontsize = font_size_legend)
            if x == 'mu_o' or x == 'dataset' or (x == 'n_cal' and 'extreme' in save_path) \
                    or x == 'alpha_t' or x == 'soft_rank_r':
                plt.xticks(rotation=55)
            if x == 'mu_o' or x == 'alpha_t':
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
            if y_col == 'FDR' and x != 'alpha':
                ax.axhline(y = 0.1, color = 'black', linestyle = 'dashed')
                ax1.set_ylim(0,1)
            elif y_col == 'FDR' and x == 'alpha':
                alphas = sorted(list(set(results[x])))
                ax1.axline((0, alphas[0]), (len(alphas) - 1, alphas[-1]), color = 'black', linestyle = 'dashed')
            plt.tick_params(axis='both', which='major', labelsize=font_size)
            if not legend:
                ax1.get_legend().remove()
                plt.savefig(
                    save_path + '/for_paper/no_legend/' + y_col.replace(' ', '-').replace('/', '-') + save_name + desc + '.pdf', bbox_inches="tight")
            else:
                plt.legend()
                plt.savefig(
                    save_path + '/for_paper/legend/' + y_col.replace(' ', '-').replace('/', '-') + save_name + desc + '.pdf', bbox_inches="tight")


def plot_visualization(x_null, x_test, y_test, all_r, algorithm, save_path=None, seed_list=[]):
    x1 = x_test[:, 0]
    x2 = x_test[:, 1]
    if not os.path.exists(save_path + '/illustrations/legend/'):
        os.makedirs(save_path + '/illustrations/legend/')
    if not os.path.exists(save_path + '/illustrations/legend_side/'):
        os.makedirs(save_path + '/illustrations/legend_side/')
    if not os.path.exists(save_path + '/illustrations/no_legend/'):
        os.makedirs(save_path + '/illustrations/no_legend/')
    if not os.path.exists(save_path + '/illustrations/no_legend_no_cb/'):
        os.makedirs(save_path + '/illustrations/no_legend_no_cb/')
    # plot xnull
    fig = plt.figure(figsize=([5, 3]))
    ax = plt.scatter(x_null[:, 0], x_null[:, 1], marker='o', color='gray', alpha=0.3, edgecolor='black', linewidths=0.5, label='Inlier')  #inliers
    plt.legend()
    if save_path is not None:
        plt.savefig(
            save_path + '/illustrations/legend/' + algorithm + '_train_calibration.pdf', bbox_inches="tight")
        plt.gca().get_legend().remove()
        plt.savefig(
            save_path + '/illustrations/no_legend/' + algorithm + '_train_calibration.pdf', bbox_inches="tight")
    else:
        plt.show()
    plt.close()
    # cmap
    fig = plt.figure(figsize=([5, 3]))
    rejected_value =  np.mean(all_r, axis=0)
    ax1  = plt.scatter(x1[y_test == 0], x2[y_test == 0], marker='o', c=rejected_value[y_test == 0], cmap="gist_heat_r", alpha=0.5, linewidths=0, vmin=0, vmax=1.0)
    ax1  = plt.scatter(x1[y_test == 1], x2[y_test == 1], marker='s', c=rejected_value[y_test == 1], cmap="gist_heat_r", alpha=0.5, linewidths=0, vmin=0, vmax=1.0)
    plt.scatter(x1[y_test == 0], x2[y_test == 0], marker='o', color='none', edgecolor='black', linewidths=0.8, label='Inlier')  #inliers
    plt.scatter(x1[y_test == 1], x2[y_test == 1], marker='s', color='none', edgecolor='black', linewidths=0.8, label='Outlier')  #outliers
    plt.legend()
    cb = fig.colorbar(ax1)
    if save_path is not None:
        plt.savefig(
            save_path + '/illustrations/legend/' + algorithm + '_cmap.pdf', bbox_inches="tight")
        plt.legend(loc='center left', bbox_to_anchor=(1.25, 0.5))
        plt.savefig(
            save_path + '/illustrations/legend_side/' + algorithm + '_cmap.pdf', bbox_inches="tight")
        plt.gca().get_legend().remove()
        plt.savefig(
            save_path + '/illustrations/no_legend/' + algorithm + '_cmap.pdf', bbox_inches="tight")
        cb.remove()
        plt.savefig(
            save_path + '/illustrations/no_legend_no_cb/' + algorithm + '_cmap.pdf', bbox_inches="tight")
    else:
        plt.show()
    plt.close()
