import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from cdr.config import Config
from cdr.data import compare_elementwise_perf
from cdr.util import filter_models, get_partition_list, nested, stderr, extract_cdr_prediction_files


def scale(*a):
    df = np.stack(a, axis=1)
    scaling_factor = df.std()
    return [x / scaling_factor for x in a]

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
        Plot elementwise differences in model performance.
    ''')
    argparser.add_argument('config_paths', nargs='*', help='Path(s) to configuration (*.ini) file')
    argparser.add_argument('-m', '--models', nargs='*', default=[], help='List of models (or model basenames if using -a) to compare. Regex permitted. If unspecified, uses all models.')
    argparser.add_argument('-q', '--quantile_range', type=float, default=0.9, help='Range of quantiles to plot. If `1.0`, plots entire dataset.')
    argparser.add_argument('-P', '--pool', action='store_true', help='Pool test statistic across models by basename using all ablation configurations common to all basenames, forces -a. Evaluation data must already exist for all ablation configurations common to all basenames.')
    argparser.add_argument('-a', '--ablation', action='store_true', help='Only compare models within an ablation set (those defined using the "ablate" param in the config file)')
    argparser.add_argument('-A', '--ablation_components', type=str, nargs='*', help='Names of variables to consider in ablative tests. Useful for excluding some ablated models from consideration')
    argparser.add_argument('-p', '--partition', type=str, default='dev', help='Name of partition to use (one of "train", "dev", "test")')
    argparser.add_argument('-M', '--metric', type=str, default='err', help='Metric to use for comparison ("err", "loglik", or "corr")')
    argparser.add_argument('-r', '--response', nargs='*', default=None, help='Name(s) of response(s) to test. If left unspecified, tests all responses.')
    argparser.add_argument('-o', '--outdir', default=None, help='Output directory. If ``None``, placed in same directory as the config.')
    args, unknown = argparser.parse_known_args()

    metric = args.metric
    if metric == 'err':
        metric = 'mse'
    assert metric in ['err', 'mse', 'loglik', 'corr'], 'Metric must be one of ["err", "loglik"].'

    if args.pool:
        args.ablation = True
        ablations = None
        basenames_to_pool = None
        exps_outdirs = []

    ablation_components = args.ablation_components

    partitions = get_partition_list(args.partition)
    partition_str = '-'.join(partitions)

    for path in args.config_paths:
        p = Config(path)

        models = filter_models(p.model_list, args.models)
        cdr_models = [x for x in models if (x.startswith('CDR') or x.startswith('DTSR'))]

        if args.ablation:
            comparison_sets = {}
            for model_name in cdr_models:
                model_basename = model_name.split('!')[0]
                if model_basename not in comparison_sets:
                    comparison_sets[model_basename] = []
                comparison_sets[model_basename].append(model_name)
            for model_name in p.model_list:
                model_basename = model_name.split('!')[0]
                if model_basename in comparison_sets and model_name not in comparison_sets[model_basename]:
                    if ablation_components is None or len(ablation_components) > 0:
                        components = model_name.split('!')[1:]
                        hit = True
                        for c in components:
                            if ablation_components is not None and c not in ablation_components:
                                hit = False
                        if hit:
                            comparison_sets[model_basename].append(model_name)
                    else:
                        comparison_sets[model_basename].append(model_name)
            if args.pool:
                ablations_cur = None
                basenames_to_pool_cur = []
                for s in comparison_sets:
                    s_ablated = set([tuple(x.split('!')[1:]) for x in comparison_sets[s]])
                    if len(s_ablated) > 1:
                        basenames_to_pool_cur.append(s)
                        if ablations_cur is None:
                            ablations_cur = s_ablated
                        else:
                            ablations_cur = ablations_cur.intersection(s_ablated)
                # ablations_cur = sorted(list(ablations_cur))
                if ablations is None:
                    ablations = sorted(list(ablations_cur), key=lambda x: len(x))
                else:
                    ablations = sorted(set(ablations).intersection(ablations_cur), key=lambda x: len(x))
                if basenames_to_pool is None:
                    basenames_to_pool = sorted(basenames_to_pool_cur)
                else:
                    basenames_to_pool = sorted(list(set(basenames_to_pool).intersection(set(basenames_to_pool_cur))))
                exps_outdirs.append(p.outdir)
        else:
            comparison_sets = {
                None: cdr_models
            }

        if not args.pool:
            for s in comparison_sets:
                model_set = comparison_sets[s]
                if len(model_set) > 1:
                    if s is not None:
                        stderr('Comparing models within ablation set "%s"...\n' % s)
                    for i in range(len(model_set)):
                        m1 = model_set[i]
                        p.set_model(m1)

                        for j in range(i+1, len(model_set)):
                            m2 = model_set[j]
                            is_nested = nested(m1, m2)
                            if is_nested or not args.ablation:
                                if is_nested:
                                    if m1.count('!') > m2.count('!'):
                                        a_model = m1
                                        b_model = m2
                                    else:
                                        a_model = m2
                                        b_model = m1
                                else:
                                    a_model = m1
                                    b_model = m2
                                a_model_path = a_model.replace(':', '+')
                                b_model_path = b_model.replace(':', '+')
                                name = '%s_v_%s' % (a_model_path, b_model_path)
                                a_files = extract_cdr_prediction_files(p.outdir + '/' + a_model_path)
                                b_files = extract_cdr_prediction_files(p.outdir + '/' + b_model_path)

                                for response in a_files:
                                    for filenum in a_files[response]:
                                        if partition_str in a_files[response][filenum] and \
                                                partition_str in b_files[response][filenum] and \
                                                (args.response is None or response in args.response):
                                            if 'table' in a_files[response][filenum][partition_str]:
                                                a = pd.read_csv(
                                                    a_files[response][filenum][partition_str]['table']['direct'],
                                                    sep=' ',
                                                    skipinitialspace=True
                                                )
                                                if metric in ['mse', 'err']:
                                                    a = (a['CDRobs'] - a['CDRpreds'])**2
                                                elif metric == 'loglik':
                                                    a = a['CDRloglik']
                                                elif metric == 'corr':
                                                    y = a['CDRobs']
                                                    a = a['CDRpreds']
                                                else:
                                                    raise ValueError('Unrecognized metric %s' % metric)
                                            else:
                                                assert not metric == 'corr', 'Metric "corr" not supported for input tables with a single column like %s.' % a_files[response][filenum][partition_str][metric]['direct']
                                                a = pd.read_csv(
                                                    a_files[response][filenum][partition_str][metric]['direct'],
                                                    sep=' ',
                                                    header=None,
                                                    skipinitialspace=True
                                                )
                                                y = None
                                            if 'table' in b_files[response][filenum][partition_str]:
                                                b = pd.read_csv(
                                                    b_files[response][filenum][partition_str]['table']['direct'],
                                                    sep=' ',
                                                    skipinitialspace=True
                                                )
                                                if metric in ['mse', 'err']:
                                                    b = (b['CDRobs'] - b['CDRpreds'])**2
                                                elif metric == 'loglik':
                                                    b = b['CDRloglik']
                                                elif metric == 'corr':
                                                    b = b['CDRpreds']
                                                else:
                                                    raise ValueError('Unrecognized metric %s' % metric)
                                            else:
                                                b = pd.read_csv(
                                                    b_files[response][filenum][partition_str][metric]['direct'],
                                                    sep=' ',
                                                    header=None,
                                                    skipinitialspace=True
                                                )

                                            select = np.logical_and(np.isfinite(np.array(a)), np.isfinite(np.array(b)))
                                            if metric == 'corr':
                                                select = np.logical_and(select, np.isfinite(np.array(y)))
                                            diff = float(len(a) - select.sum())
                                            performance_diffs = compare_elementwise_perf(
                                                a[select],
                                                b[select],
                                                y=None if y is None else y[select],
                                                mode=metric
                                            )
                                            if args.quantile_range < 1.:
                                                alpha = 1 - args.quantile_range
                                                lq = np.quantile(performance_diffs, alpha / 2)
                                                uq = np.quantile(performance_diffs, 1 - alpha / 2)
                                                sel = (performance_diffs > lq) & (performance_diffs < uq)
                                                performance_diffs = performance_diffs[sel]
                                            plt.hist(performance_diffs, bins=1000)
                                            plt.xlabel('Difference in %s' % metric)
                                            plt.ylabel('Count')

                                            name_base = '%s_diffplot_%s_f%s_%s.png' % (name, response, filenum, partition_str)
                                            outdir = args.outdir
                                            if outdir is None:
                                                outdir = p.outdir
                                            if not os.path.exists(outdir):
                                                os.makedirs(outdir)
                                            plt.savefig(outdir + '/' + name_base)

                                            plt.close('all')

    if args.pool:
        pooled_data = {}
        if metric == 'corr':
            pooled_obs = {}
        for a in ablations:
            pooled_data[a] = {}
            for exp_outdir in exps_outdirs:
                pooled_data[a][exp_outdir] = {}
                for m in basenames_to_pool:
                    m_name = '!'.join([m] + list(a)).replace(':', '+')
                    m_files = extract_cdr_prediction_files(exp_outdir + '/' + m_name)
                    for response in m_files:
                        for filenum in m_files[response]:
                            if partition_str in m_files[response][filenum] and \
                                    (args.response is None or response in args.response):
                                if 'table' in m_files[response][filenum][partition_str]:
                                    v = pd.read_csv(
                                        m_files[response][filenum][partition_str]['table']['direct'],
                                        sep=' ',
                                        skipinitialspace=True
                                    )
                                    if metric in ['err', 'mse']:
                                        v = (v['CDRobs'] - v['CDRpreds']) ** 2
                                    elif metric == 'loglik':
                                        v = v['CDRloglik']
                                    elif metric == 'corr':
                                        y = v['CDRobs']
                                        v = v['CDRpreds']
                                    else:
                                        raise ValueError('Unrecognized metric %s' % metric)
                                else:
                                    v = pd.read_csv(
                                        m_files[response][filenum][partition_str][metric]['direct'],
                                        sep=' ',
                                        header=None,
                                        skipinitialspace=True
                                    )
                                if a not in pooled_data:
                                    pooled_data[a] = {}
                                if exp_outdir not in pooled_data[a]:
                                    pooled_data[a][exp_outdir] = {}
                                if m not in pooled_data[a][exp_outdir]:
                                    pooled_data[a][exp_outdir][m] = {}
                                if response not in pooled_data[a][exp_outdir][m]:
                                    pooled_data[a][exp_outdir][m][response] = {}
                                pooled_data[a][exp_outdir][m][response][filenum] = v
                                
                                if metric == 'corr':
                                    if a not in pooled_obs:
                                        pooled_obs[a] = {}
                                    if exp_outdir not in pooled_obs[a]:
                                        pooled_obs[a][exp_outdir] = {}
                                    if m not in pooled_obs[a][exp_outdir]:
                                        pooled_obs[a][exp_outdir][m] = {}
                                    if response not in pooled_obs[a][exp_outdir][m]:
                                        pooled_obs[a][exp_outdir][m][response] = {}
                                    pooled_obs[a][exp_outdir][m][response][filenum] = y

        for i in range(len(ablations)):
            a1 = ablations[i]
            m1 = '!'.join(a1)
            m1_dummy = 'DUMMY' + (('!' + m1) if m1 != '' else m1)
            for j in range(i + 1, len(ablations)):
                a2 = ablations[j]
                m2 = '!'.join(a2)
                m2_dummy = 'DUMMY' + (('!' + m2) if m2 != '' else m2)
                is_nested = nested(m1_dummy, m2_dummy)
                if is_nested:
                    if m1.count('!') > m2.count('!'):
                        a_model = a1
                        b_model = a2
                        a_name = 'FULL' if m2 == '' else '!' + m2
                        b_name = 'FULL' if m1 == '' else '!' + m1
                    else:
                        a_model = a2
                        b_model = a1
                        a_name = 'FULL' if m1 == '' else '!' + m1
                        b_name = 'FULL' if m2 == '' else '!' + m2
                    df1 = {}
                    df2 = {}
                    if metric == 'corr':
                        y = {}
                    for exp in exps_outdirs:
                        for m in basenames_to_pool:
                            for response in pooled_data[a_model][exp][m]:
                                for filenum in pooled_data[a_model][exp][m][response]:
                                    if response not in df1:
                                        df1[response] = {}
                                    if filenum not in df1[response]:
                                        df1[response][filenum] = []
                                    if response not in df2:
                                        df2[response] = {}
                                    if filenum not in df2[response]:
                                        df2[response][filenum] = []
                                    if metric == 'corr':
                                        if response not in y:
                                            y[response] = {}
                                        if filenum not in df2[response]:
                                            y[response][filenum] = []
                                    scale_in = [
                                        pooled_data[a_model][exp][m][response][filenum],
                                        pooled_data[b_model][exp][m][response][filenum]
                                    ]
                                    if metric == 'corr':
                                        scale_in = pooled_obs[a_model][exp][m][response][filenum]
                                    scale_out = scale(*scale_in)
                                    df1[response][filenum].append(scale_out[0])
                                    df2[response][filenum].append(scale_out[1])
                                    if metric == 'corr':
                                        y[response][filenum].append(scale_out[2])
                    for response in df1:
                        for filenum in df1[response]:
                            _df1 = np.concatenate(df1[response][filenum], axis=0)
                            _df2 = np.concatenate(df2[response][filenum], axis=0)
                            if metric == 'corr':
                                _y = np.concatenate(y[response][filenum], axis=0)
                            else:
                                _y = None
                            assert len(_df1) == len(_df2), 'Shape mismatch between datasets %s and %s: %s vs. %s' % (
                                a_name,
                                b_name,
                                _df1.shape,
                                _df2.shape
                            )

                            performance_diffs = compare_elementwise_perf(
                                _df1,
                                _df2,
                                y=_y,
                                mode=metric
                            )
                            if args.quantile_range < 1.:
                                alpha = 1 - args.quantile_range
                                lq = np.quantile(performance_diffs, alpha / 2)
                                uq = np.quantile(performance_diffs, 1 - alpha / 2)
                                sel = (performance_diffs > lq) & (performance_diffs < uq)
                                performance_diffs = performance_diffs[sel]
                            plt.hist(performance_diffs, bins=1000)
                            plt.xlabel('Difference in %s' % metric)
                            plt.ylabel('Count')

                            name = '%s_v_%s' % (a_name.replace(':', '+'), b_name.replace(':', '+'))
                            name_base = '%s_diffplot_pooled_%s_f%s_%s.png' % (name, response, filenum, partition_str)
                            outdir = args.outdir
                            if outdir is None:
                                outdir = p.outdir
                            if not os.path.exists(outdir):
                                os.makedirs(outdir)
                            plt.savefig(outdir + '/' + name_base)

                            plt.close('all')
