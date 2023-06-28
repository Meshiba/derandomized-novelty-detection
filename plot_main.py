import os
import errno

import click
from utils_plot import plot_from_files

################################################################################
# Settings
################################################################################
@click.command()
@click.argument('save_path', type=str, default=None)
@click.option('--files', type=click.Path(exists=True), multiple=True, default=[],
              help='Path to stats file.')
@click.option('--x', type=str, default="n_e_value",
              help='')
@click.option('--y', type=str, multiple=True, default=[],
              help='')
@click.option('--filter_keys', type=str, multiple=True, default=[],
              help='Please add the filter keys in the following order - int, float, str')
@click.option('--filter_values', type=int, multiple=True, default=[],
              help='Corresponding int values to the int filter keys.')
@click.option('--filter_values_float', type=float, multiple=True, default=[],
              help='Corresponding float values to the float filter keys.')
@click.option('--filter_values_str', type=str, multiple=True, default=[],
              help='Corresponding string values to the string filter keys.')
@click.option('--save_name', default='', type=str,
              help='Suffix of the save name')
@click.option('--weight', is_flag=True,
              help='Plot comparison between different weight metrics.')
@click.option('--plot_variance', is_flag=True,
              help='Plot variance (only available for fixed settings experiment).')
@click.option('--only_e', is_flag=True,
              help='Plot only e-value methods.')
@click.option('--a_t2plot', default=None, help='Specify which alpha_bh value to plot.')
@click.option('--order', multiple=True, default=None,
              help='Order to plot the categorical levels in; otherwise the levels are inferred from the data objects.')
@click.option('--wo_e', is_flag=True,
              help='Do not plot martingale-based e-value methods.')
@click.option('--no_rank_desc', is_flag=True,
              help='Do not plot soft-rank hyper-parameter in legend.')
def main(save_path, files, x, y, filter_keys, filter_values, filter_values_float,
         filter_values_str, save_name, weight, a_t2plot, plot_variance, only_e, order,
         wo_e, no_rank_desc):
    if len(filter_keys) != (len(filter_values) + len(filter_values_float) + len(filter_values_str)):
        raise ValueError(f'filter keys and values must be in the same length\n\tfilter_keys = {filter_keys}'
                         f'\n\tfilter_values = {filter_values}\tfilter_values_float = {filter_values_float}\tfilter_values_str = {filter_values_str}')
    if not os.path.exists(save_path):
        try:
            os.makedirs(save_path)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise
    if len(order) == 0:
        order = None
    filter_ = {}
    filter_values = list(filter_values)
    filter_values.extend(list(filter_values_float))
    filter_values.extend(list(filter_values_str))
    if len(filter_values):
        for i in range(len(filter_values)):
            if filter_values[i] == 'None':
                v = None
            else:
                v = filter_values[i]
            filter_[filter_keys[i]] = v

    all_files = []
    for f in files:
        if os.path.isdir(f):
            dir_name = f
            files_ = [os.path.join(dir_name, f) for f in os.listdir(dir_name)
                      if os.path.isfile(os.path.join(dir_name, f)) and f.endswith('.pkl')]
            all_files.extend(files_)
        else:
            all_files.append(f)

    plot_from_files(all_files, save_path=save_path, x=x, y=y, filter=filter_,
                    save_name=save_name, weight=weight, curr_a_t2plot=a_t2plot, reuse=plot_variance, only_e=only_e,
                    order=order, wo_e=wo_e, no_rank_desc=no_rank_desc)


if __name__ == '__main__':
    main()
