import sys
import argparse
import numpy as np
import pandas as pd
from cdr.config import Config
from cdr.signif import permutation_test
from cdr.util import filter_models, get_partition_list, nested
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from cdr.util import stderr

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
    argparser.add_argument('-P', '--pool', action='store_true', help='Pool test statistic across models by basename using all ablation configurations common to all basenames, forces -a. Evaluation data must already exist for all ablation configurations common to all basenames.')
    argparser.add_argument('-a', '--ablation', action='store_true', help='Only compare models within an ablation set (those defined using the "ablate" param in the config file)')
    argparser.add_argument('-A', '--ablation_components', type=str, nargs='*', help='Names of variables to consider in ablative tests. Useful for excluding some ablated models from consideration')
    argparser.add_argument('-p', '--partition', type=str, default='dev', help='Name of partition to use (one of "train", "dev", "test")')
    argparser.add_argument('-M', '--metric', type=str, default='loss', help='Metric to use for comparison (either "loss" or "loglik")')
    argparser.add_argument('-t', '--twostep', action='store_true', help='For DTSR models, compare predictions from fitted LME model from two-step hypothesis test.')
    argparser.add_argument('-T', '--tails', type=int, default=2, help='Number of tails (1 or 2)')
    args, unknown = argparser.parse_known_args()

    assert args.metric in ['loss', 'loglik'], 'Metric must be one of ["loss", "loglik"].'

    if args.pool:
        args.ablation = True
        ablations = None
        basenames_to_pool = None
        exps_outdirs = []

    for path in args.config_paths:
        p = Config(path)

        models = filter_models(p.model_list, args.models)
        cdr_models = [x for x in models if (x.startswith('CDR') or x.startswith('DTSR'))]

        partitions = get_partition_list(args.partition)
        partition_str = '-'.join(partitions)

        if args.metric == 'loss':
            file_name = 'losses_mse_%s.txt' % partition_str
        else:
            file_name = 'loglik_%s.txt' % partition_str
        if args.twostep:
            file_name = 'LM_2STEP_' + file_name 

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
                    if len(args.ablation_components) > 0:
                        components = model_name.split('!')[1:]
                        hit = True
                        for c in components:
                            if c not in args.ablation_components:
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
                None: models
            }

        if not args.pool:
            for s in comparison_sets:
                model_set = comparison_sets[s]
                if len(model_set) > 1:
                    if s is not None:
                        stderr('Comparing models within ablation set "%s"...\n' %s)
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
                                name = '%s_v_%s' %(a_model, b_model)
                                a = pd.read_csv(p.outdir + '/' + a_model + '/' + file_name, sep=' ', header=None, skipinitialspace=True)
                                b = pd.read_csv(p.outdir + '/' + b_model + '/' + file_name, sep=' ', header=None, skipinitialspace=True)
                                select = np.logical_and(np.isfinite(np.array(a)), np.isfinite(np.array(b)))
                                diff = float(len(a) - select.sum())
                                p_value, base_diff, diffs = permutation_test(a[select], b[select], n_iter=10000, n_tails=args.tails, mode=args.metric, nested=is_nested)
                                stderr('\n')
                                out_path = p.outdir + '/' + name + '_PT_' + partition_str + '.txt'
                                with open(out_path, 'w') as f:
                                    stderr('Saving output to %s...\n' %out_path)

                                    summary = '='*50 + '\n'
                                    summary += 'Model comparison: %s vs %s\n' %(a_model, b_model)
                                    if diff > 0:
                                        summary += '%d NaN rows filtered out (out of %d)\n' % (diff, len(a))
                                    summary += 'Partition: %s\n' % partition_str
                                    summary += 'Metric: %s\n' % args.metric
                                    summary += 'Difference: %.4f\n' % base_diff
                                    summary += 'p: %.4e%s\n' % (p_value, '' if p_value > 0.05 else '*' if p_value > 0.01 else '**' if p_value > 0.001 else '***')
                                    summary += '='*50 + '\n'

                                    f.write(summary)
                                    sys.stdout.write(summary)

                                plt.hist(diffs, bins=1000)
                                plt.savefig(p.outdir + '/' + name + '_PT_' + partition_str + '.png')
                                plt.close('all')

    if args.pool:
        pooled_data = {}
        for a in ablations:
            pooled_data[a] = {}
            for exp_outdir in exps_outdirs:
                pooled_data[a][exp_outdir] = {}
                for m in basenames_to_pool:
                    m_name = '!'.join([m] + list(a))
                    pooled_data[a][exp_outdir][m] = pd.read_csv(exp_outdir + '/' + m_name + '/' + file_name, sep=' ', header=None, skipinitialspace=True)

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
                    df1 = []
                    df2 = []
                    for exp in exps_outdirs:
                        for m in basenames_to_pool:
                            a, b = scale(pooled_data[a_model][exp][m], pooled_data[b_model][exp][m])
                            df1.append(a)
                            df2.append(b)
                    df1 = np.concatenate(df1, axis=0)
                    df2 = np.concatenate(df2, axis=0)
                    assert len(df1) == len(df2), 'Shape mismatch between datasets %s and %s: %s vs. %s' %(
                        a_name,
                        b_name,
                        df1.shape,
                        df2.shape
                    )
                    p_value, diff, diffs = permutation_test(df1, df2, n_iter=10000, n_tails=args.tails, mode=args.metric)
                    stderr('\n')
                    name = '%s_v_%s' % (a_name, b_name)
                    out_path = p.outdir + '/' + name + '_PT_pooled_' + partition_str + '.txt'
                    with open(out_path, 'w') as f:
                        stderr('Saving output to %s...\n' % out_path)

                        summary = '=' * 50 + '\n'
                        summary += 'Model comparison: %s vs %s\n' % (a_name, b_name)
                        summary += 'Partition: %s\n' % partition_str
                        summary += 'Metric: %s\n' % args.metric
                        summary += 'Experiments pooled:\n'
                        for exp in exps_outdirs:
                            summary += '  %s\n' %exp
                        summary += 'Ablation sets pooled:\n'
                        for basename in basenames_to_pool:
                            summary += '  %s\n' %basename
                        summary += 'n: %s\n' %df1.shape[0]
                        summary += 'Difference: %.4f\n' % diff
                        summary += 'p: %.4e%s\n' % (
                            p_value,
                            '' if p_value > 0.05 else '*' if p_value > 0.01 else '**' if p_value > 0.001 else '***')
                        summary += '=' * 50 + '\n'

                        f.write(summary)
                        sys.stdout.write(summary)

                    plt.hist(diffs, bins=1000)
                    plt.savefig(p.outdir + '/' + name + '_PT_pooled_' + partition_str + '.png')
                    plt.close('all')
