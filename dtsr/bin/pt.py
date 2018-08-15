import sys
import argparse
import numpy as np
import pandas as pd
from dtsr.config import Config
from dtsr.signif import bootstrap
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from dtsr.util import nested

def scale(a, b):
    df = np.stack([np.array(a), np.array(b)], axis=1)
    scaling_factor = df.std()
    return a/scaling_factor, b/scaling_factor

if __name__ == '__main__':

    argparser = argparse.ArgumentParser('''
        Performs pairwise permutation test for significance of differences in prediction quality between models.
        Can be used for in-sample and out-of-sample evaluation.
        Can be used (1) to compare models of different structure (e.g. DTSR vs. LME), (2) (using the "-a" flag) to perform hypothesis testing between DTSR models within one or more ablation sets, or (3) (using the "-P" flag) to perform a pooled test comparing DTSR models fitted to multiple responses. 
    ''')
    argparser.add_argument('config_paths', nargs='*', help='Path(s) to configuration (*.ini) file')
    argparser.add_argument('-m', '--models', nargs='*', default=[], help='Models (or model basenames if using -a) to compare. Defaults to all models in the config file.')
    argparser.add_argument('-P', '--pool', action='store_true', help='Pool test statistic across models by basename using all ablation configurations common to all basenames, forces -a. Evaluation data must already exist for all ablation configurations common to all basenames.')
    argparser.add_argument('-a', '--ablation', action='store_true', help='Only compare models within an ablation set (those defined using the "ablate" param in the config file)')
    argparser.add_argument('-p', '--partition', type=str, default='dev', help='Name of partition to use (one of "train", "dev", "test")')
    argparser.add_argument('-M', '--metric', type=str, default='loss', help='Metric to use for comparison (either "loss" or "loglik")')
    argparser.add_argument('-t', '--tails', type=int, default=2, help='Number of tails (1 or 2)')
    args, unknown = argparser.parse_known_args()

    if args.pool:
        args.ablation = True
        ablations = None
        basenames_to_pool = None
        exps_outdirs = []

    for path in args.config_paths:
        p = Config(path)
        if len(args.models) > 0:
            models = args.models
        else:
            models = p.model_list[:]

        if args.metric == 'loss':
            file_name = '%s_losses_%s.txt' % (p['loss_name'], args.partition)
        else:
            file_name = 'loglik_%s.txt' % args.partition

        dtsr_models = [x for x in models if x.startswith('DTSR')]

        if args.ablation:
            comparison_sets = {}
            for model_name in dtsr_models:
                model_basename = model_name.split('!')[0]
                if model_basename not in comparison_sets:
                    comparison_sets[model_basename] = []
                comparison_sets[model_basename].append(model_name)
            for model_name in p.model_list:
                model_basename = model_name.split('!')[0]
                if model_basename in comparison_sets and model_name not in comparison_sets[model_basename]:
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
                None: dtsr_models
            }

        if not args.pool:
            for s in comparison_sets:
                model_set = comparison_sets[s]
                if len(model_set) > 1:
                    if s is not None:
                        sys.stderr.write('Comparing models within ablation set "%s"...\n' %s)
                    for i in range(len(model_set)):
                        m1 = model_set[i]
                        p.set_model(m1)

                        for j in range(i+1, len(model_set)):
                            m2 = model_set[j]
                            if nested(m1, m2) or not args.ablation:
                                name = '%s_v_%s' %(m1, m2)
                                a = pd.read_csv(p.outdir + '/' + m1 + '/' + file_name, sep=' ', header=None, skipinitialspace=True)
                                b = pd.read_csv(p.outdir + '/' + m2 + '/' + file_name, sep=' ', header=None, skipinitialspace=True)
                                select = np.logical_and(np.isfinite(np.array(a)), np.isfinite(np.array(b)))
                                diff = float(len(a) - select.sum())
                                p_value, base_diff, diffs = bootstrap(a[select], b[select], n_iter=10000, n_tails=args.tails, mode=args.metric)
                                sys.stderr.write('\n')
                                out_path = p.outdir + '/' + name + '_PT_' + args.partition + '.txt'
                                with open(out_path, 'w') as f:
                                    sys.stderr.write('Saving output to %s...\n' %out_path)

                                    summary = '='*50 + '\n'
                                    summary += 'Model comparison: %s vs %s\n' %(m1, m2)
                                    if diff > 0:
                                        summary += '%d NaN rows filtered out (out of %d)\n' %(diff, len(a))
                                    summary += 'Partition: %s\n' %args.partition
                                    summary += 'Loss difference: %.4f\n' %base_diff
                                    summary += 'p: %.4e%s\n' %(p_value, '' if p_value > 0.05 else '*' if p_value > 0.01 else '**' if p_value > 0.001 else '***')
                                    summary += '='*50 + '\n'

                                    f.write(summary)
                                    sys.stdout.write(summary)

                                plt.hist(diffs, bins=1000)
                                plt.savefig(p.outdir + '/' + name + '_PT_' + args.partition + '.png')
                                plt.close('all')

    if args.pool:
        pooled_data = {}
        for a in ablations:
            pooled_data[a] = {}
            for exp_outdir in exps_outdirs:
                pooled_data[a][exp_outdir] = {}
                for m in basenames_to_pool:
                    m_name = '!'.join([m] + list(a))
                    pooled_data[a][exp_outdir][m] = pd.read_csv(p.outdir + '/' + m_name + '/' + file_name, sep=' ', header=None, skipinitialspace=True)

        for i in range(len(ablations)):
            a1 = ablations[i]
            m1 = '!'.join(a1)
            for j in range(i + 1, len(ablations)):
                a2 = ablations[j]
                m2 = '!'.join(a2)
                if nested('DUMMY' + (('!' + m1) if m1 != '' else m1), 'DUMMY' + (('!' + m2) if m2 != '' else m2)):
                    df1 = []
                    df2 = []
                    for exp in exps_outdirs:
                        for m in basenames_to_pool:
                            a, b = scale(pooled_data[a1][exp][m], pooled_data[a2][exp][m])
                            df1.append(a)
                            df2.append(b)
                    df1 = np.concatenate(df1, axis=0)
                    df2 = np.concatenate(df2, axis=0)
                    assert len(df1) == len(df2), 'Shape mismatch between datasets %s and %s: %s vs. %s' %(
                        'FULL' if m1 == '' else '!' + m1,
                        'FULL' if m2 == '' else '!' + m2,
                        df1.shape,
                        df2.shape
                    )
                    p_value, diff, diffs = bootstrap(df1, df2, n_iter=10000, n_tails=args.tails, mode=args.metric)
                    sys.stderr.write('\n')
                    name = '%s_v_%s' % ('FULL' if m1 == '' else '!' + m1, 'FULL' if m2 == '' else '!' + m2)
                    out_path = p.outdir + '/' + name + '_PT_pooled_' + args.partition + '.txt'
                    with open(out_path, 'w') as f:
                        sys.stderr.write('Saving output to %s...\n' % out_path)

                        summary = '=' * 50 + '\n'
                        summary += 'Model comparison: %s vs %s\n' % (
                        'FULL' if m1 == '' else '!' + m1, 'FULL' if m2 == '' else '!' + m2)
                        summary += 'Partition: %s\n' % args.partition
                        summary += 'Experiments pooled:\n'
                        for exp in exps_outdirs:
                            summary += '  %s\n' %exp
                        summary += 'Ablation sets pooled:\n'
                        for basename in basenames_to_pool:
                            summary += '  %s\n' %basename
                        summary += 'n: %s\n' %df1.shape[0]
                        summary += 'Loss difference: %.4f\n' % diff
                        summary += 'p: %.4e%s\n' % (
                            p_value,
                            '' if p_value > 0.05 else '*' if p_value > 0.01 else '**' if p_value > 0.001 else '***')
                        summary += '=' * 50 + '\n'

                        f.write(summary)
                        sys.stdout.write(summary)

                    plt.hist(diffs, bins=1000)
                    plt.savefig(p.outdir + '/' + name + '_PT_pooled_' + args.partition + '.png')
                    plt.close('all')
