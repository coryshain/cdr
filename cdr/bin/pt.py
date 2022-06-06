import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from cdr.config import Config
from cdr.signif import permutation_test
from cdr.util import filter_models, get_partition_list, nested, stderr, extract_cdr_prediction_data


def scale(a, b):
    df = np.stack([np.array(a), np.array(b)], axis=1)
    scaling_factor = df.std()
    return a/scaling_factor, b/scaling_factor


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
        Performs pairwise permutation test for significance of differences in prediction quality between models.
        Can be used for in-sample and out-of-sample evaluation.
        Used (1) to perform hypothesis testing between CDR models within one or more ablation sets, or (2) (using the "-P" flag) to perform a pooled test comparing CDR models fitted to multiple responses. 
    ''')
    argparser.add_argument('config_paths', nargs='*', help='Path(s) to configuration (*.ini) file')
    argparser.add_argument('-m', '--models', nargs='*', default=[], help='List of models (or model basenames if using -a) to compare. Regex permitted. If unspecified, uses all models.')
    argparser.add_argument('-n', '--n_iter', type=int, default=10000, help='Number of resampling iterations.')
    argparser.add_argument('-P', '--pool', action='store_true', help='Pool test statistic across models by basename using all ablation configurations common to all configs in ``config_paths``, forces -a. Evaluation data must already exist for all ablation configurations common to all basenames.')
    argparser.add_argument('-a', '--ablation', action='store_true', help='Only compare models within an ablation set (those defined using the "ablate" param in the config file)')
    argparser.add_argument('-A', '--ablation_components', type=str, nargs='*', help='Names of variables to consider in ablative tests. Useful for excluding some ablated models from consideration')
    argparser.add_argument('-p', '--partition', type=str, default='dev', help='Name of partition to use (one of "train", "dev", "test")')
    argparser.add_argument('-M', '--metric', type=str, default='loglik', help='Metric to use for comparison (either "mse" or "loglik")')
    argparser.add_argument('-t', '--twostep', action='store_true', help='For DTSR models, compare predictions from fitted LME model from two-step hypothesis test.')
    argparser.add_argument('-T', '--tails', type=int, default=2, help='Number of tails (1 or 2)')
    argparser.add_argument('-r', '--response', nargs='*', default=None, help='Name(s) of response(s) to test. If left unspecified, tests all responses.')
    argparser.add_argument('-o', '--outdir', default=None, help='Output directory. If ``None``, placed in same directory as the config.')
    args = argparser.parse_args()

    metric = args.metric
    if metric == 'err':
        metric = 'mse'
    assert metric in ['mse', 'loglik'], 'Metric must be one of ["mse", "loglik"].'

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

        model_list = sorted(set(p.model_list) | set(p.ensemble_list))
        models = filter_models(model_list, args.models)
        cdr_models = [x for x in filter_models(models, cdr_only=True)]

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
                                a_data = extract_cdr_prediction_data(p.outdir + '/' + a_model_path, metric=metric)
                                b_data = extract_cdr_prediction_data(p.outdir + '/' + b_model_path, metric=metric)
                                for response in a_data:
                                    for filenum in a_data[response]:
                                        if partition_str in a_data[response][filenum] and \
                                                (args.response is None or response in args.response):
                                            a = a_data[response][filenum][partition_str]['direct']
                                            try:
                                                b = b_data[response][filenum][partition_str]['direct']
                                            except KeyError:
                                                continue

                                            a = np.stack([a[x] for x in a], axis=1)
                                            b = np.stack([b[x] for x in b], axis=1)

                                            p_value, base_diff, diffs = permutation_test(
                                                a,
                                                b,
                                                n_iter=args.n_iter,
                                                n_tails=args.tails,
                                                mode=metric,
                                                nested=is_nested
                                            )
                                            stderr('\n')

                                            name_base = '%s_PT_%s_f%s_%s' % (name, response, filenum, partition_str)
                                            outdir = args.outdir
                                            if outdir is None:
                                                outdir = p.outdir
                                            if not os.path.exists(outdir):
                                                os.makedirs(outdir)
                                            out_path = outdir + '/' + name_base + '.txt'
                                            with open(out_path, 'w') as f:
                                                stderr('Saving output to %s...\n' % out_path)

                                                summary = '='*50 + '\n'
                                                summary += 'Model comparison: %s vs %s\n' % (a_model, b_model)
                                                summary += 'Partition: %s\n' % partition_str
                                                summary += 'Metric: %s\n' % metric
                                                summary += 'Difference: %.4f\n' % base_diff
                                                summary += 'p: %.4e%s\n' % (p_value, '' if p_value > 0.05 \
                                                    else '*' if p_value > 0.01 else '**' if p_value > 0.001 else '***')
                                                summary += '='*50 + '\n'

                                                f.write(summary)
                                                sys.stdout.write(summary)

                                            plt.hist(diffs, bins=1000)
                                            plt.savefig(outdir + '/' + name_base + '.png')
                                            plt.close('all')

    if args.pool:
        pooled_data = {}
        for a in ablations:
            pooled_data[a] = {}
            for exp_outdir in exps_outdirs:
                pooled_data[a][exp_outdir] = {}
                for m in basenames_to_pool:
                    m_name = '!'.join([m] + list(a)).replace(':', '+')
                    m_files = extract_cdr_prediction_data(exp_outdir + '/' + m_name)
                    for response in m_files:
                        for filenum in m_files[response]:
                            if partition_str in m_files[response][filenum] and \
                                    (args.response is None or response in args.response):
                                v = m_files[response][filenum][partition_str]['direct']
                                v = np.stack([v[x] for x in v], axis=1)

                                if a not in pooled_data:
                                    pooled_data[a] = {}
                                if exp_outdir not in pooled_data[a]:
                                    pooled_data[a][exp_outdir] = {}
                                if m not in pooled_data[a][exp_outdir]:
                                    pooled_data[a][exp_outdir][m] = {}
                                if response not in pooled_data[a][exp_outdir][m]:
                                    pooled_data[a][exp_outdir][m][response] = {}
                                pooled_data[a][exp_outdir][m][response][filenum] = v

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
                                    a, b = scale(
                                        pooled_data[a_model][exp][m][response][filenum],
                                        pooled_data[b_model][exp][m][response][filenum]
                                    )
                                    df1[response][filenum].append(a)
                                    df2[response][filenum].append(b)
                    for response in df1:
                        for filenum in df1[response]:
                            _df1 = np.concatenate(df1[response][filenum], axis=0)
                            _df2 = np.concatenate(df2[response][filenum], axis=0)
                            assert len(_df1) == len(_df2), 'Shape mismatch between datasets %s and %s: %s vs. %s' % (
                                a_name,
                                b_name,
                                _df1.shape,
                                _df2.shape
                            )
                            p_value, diff, diffs = permutation_test(
                                _df1,
                                _df2,
                                n_iter=args.n_iter,
                                n_tails=args.tails,
                                mode=metric,
                                nested=is_nested
                            )
                            stderr('\n')
                            name = '%s_v_%s' % (a_name.replace(':', '+'), b_name.replace(':', '+'))
                            name_base = '%s_PT_pooled_%s_f%s_%s' % (name, response, filenum, partition_str)
                            outdir = args.outdir
                            if outdir is None:
                                outdir = p.outdir
                            if not os.path.exists(outdir):
                                os.makedirs(outdir)
                            out_path = outdir + '/' + name_base + '.txt'
                            with open(out_path, 'w') as f:
                                stderr('Saving output to %s...\n' % out_path)

                                summary = '=' * 50 + '\n'
                                summary += 'Model comparison: %s vs %s\n' % (a_name, b_name)
                                summary += 'Partition: %s\n' % partition_str
                                summary += 'Metric: %s\n' % metric
                                summary += 'Experiments pooled:\n'
                                for exp in exps_outdirs:
                                    summary += '  %s\n' % exp
                                summary += 'Ablation sets pooled:\n'
                                for basename in basenames_to_pool:
                                    summary += '  %s\n' % basename
                                summary += 'n: %s\n' % _df1.shape[0]
                                summary += 'Difference: %.4f\n' % diff
                                summary += 'p: %.4e%s\n' % (
                                    p_value,
                                    '' if p_value > 0.05 else '*' if p_value > 0.01
                                    else '**' if p_value > 0.001 else '***'
                                )
                                summary += '=' * 50 + '\n'

                                f.write(summary)
                                sys.stdout.write(summary)

                            plt.hist(diffs, bins=1000)
                            plt.savefig(outdir + '/' + name_base + '.png')
                            plt.close('all')
