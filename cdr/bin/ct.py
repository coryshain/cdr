import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from cdr.config import Config
from cdr.signif import permutation_test, correlation_test
from cdr.util import filter_models, get_partition_list, nested, stderr


def scale(a):
    return (a - a.mean()) / a.std()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
        Performs parametric correlation test for significance of differences in prediction quality between models.
        Can be used for in-sample and out-of-sample evaluation.
        Used (1) to perform hypothesis testing between CDR models within one or more ablation sets, or (2) (using the "-P" flag) to perform a pooled test comparing CDR models fitted to multiple responses. 
    ''')
    argparser.add_argument('config_paths', nargs='*', help='Path(s) to configuration (*.ini) file')
    argparser.add_argument('-m', '--models', nargs='*', default=[], help='List of models (or model basenames if using -a) to compare. Regex permitted. If unspecified, uses all models.')
    argparser.add_argument('-P', '--pool', action='store_true', help='Pool test statistic across models by basename using all ablation configurations common to all basenames, forces -a. Evaluation data must already exist for all ablation configurations common to all basenames.')
    argparser.add_argument('-a', '--ablation', action='store_true', help='Only compare models within an ablation set (those defined using the "ablate" param in the config file)')
    argparser.add_argument('-A', '--ablation_components', type=str, nargs='*', help='Names of variables to consider in ablative tests. Useful for excluding some ablated models from consideration')
    argparser.add_argument('-p', '--partition', type=str, default='dev', help='Name of partition to use (one of "train", "dev", "test")')
    argparser.add_argument('-t', '--twostep', action='store_true', help='For DTSR models, compare predictions from fitted LME model from two-step hypothesis test.')
    argparser.add_argument('-T', '--permutation_test', action='store_true', help='Use a permutation test of correlation difference. If ``False``, use a parametric test (Steiger, 1980).')
    args, unknown = argparser.parse_known_args()

    if args.pool:
        args.ablation = True
        ablations = None
        basenames_to_pool = None
        exps_outdirs = []

    ablation_components = args.ablation_components

    for path in args.config_paths:
        p = Config(path)

        models = filter_models(p.model_list, args.models)
        cdr_models = [x for x in models if (x.startswith('CDR') or x.startswith('DTSR'))]

        partitions = get_partition_list(args.partition)
        partition_str = '-'.join(partitions)

        obs_file_name = 'obs_%s.txt' % partition_str
        pred_file_name = 'preds_%s.txt' % partition_str
        pred_table_name = 'preds_table_%s.csv' % partition_str
        if args.twostep:
            pred_file_name = 'LM_2STEP_' + pred_file_name

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
                                a_model_path = a_model.replace(':', '+')
                                b_model_path = b_model.replace(':', '+')
                                name = '%s_v_%s' %(a_model_path, b_model_path)
                                try:
                                    if os.path.exists(p.outdir + '/' + a_model_path + '/' + pred_table_name):
                                        df = pd.read_csv(p.outdir + '/' + a_model_path + '/' + pred_table_name, sep=' ', skipinitialspace=True)
                                        a = scale(df.CDRpreds)
                                        if 'y' in df.columns:
                                            y = scale(df.y)
                                        elif 'yStandardized' in df.columns:
                                            y = scale(df.yStandardized)
                                        else:
                                            y = scale(df.CDRobs)
                                    else:
                                        y = scale(pd.read_csv(p.outdir + '/' + a_model_path + '/' + obs_file_name, sep=' ', header=None, skipinitialspace=True)[0])
                                        a = scale(pd.read_csv(p.outdir + '/' + a_model_path + '/' + pred_file_name, sep=' ', header=None, skipinitialspace=True)[0])
                                    if os.path.exists(p.outdir + '/' + b_model_path + '/' + pred_table_name):
                                        b = scale(pd.read_csv(p.outdir + '/' + b_model_path + '/' + pred_table_name, sep=' ', skipinitialspace=True).CDRpreds)
                                    else:
                                        b = scale(pd.read_csv(p.outdir + '/' + b_model_path + '/' + pred_file_name, sep=' ', header=None, skipinitialspace=True)[0])
                                except FileNotFoundError as e:
                                    sys.stderr.write(str(e) + '\n')
                                    continue
                                select = np.logical_and(np.isfinite(y), np.logical_and(np.isfinite(np.array(a)), np.isfinite(np.array(b))))
                                diff = float(len(y) - select.sum())
                                stderr('\n')
                                if args.permutation_test:
                                    out_path = p.outdir + '/' + name + '_CPT_' + partition_str + '.txt'
                                else:
                                    out_path = p.outdir + '/' + name + '_CT_' + partition_str + '.txt'
                                with open(out_path, 'w') as f:
                                    stderr('Saving output to %s...\n' %out_path)

                                    summary = '='*50 + '\n'
                                    summary += 'Model comparison: %s vs %s\n' %(a_model, b_model)
                                    if diff > 0:
                                        summary += '                  %d NaN rows filtered out (out of %d)\n' % (diff, len(a))
                                    summary += 'Partition:        %s\n' % partition_str
                                    summary += 'Metric:     corr\n'
                                    summary += 'n:          %s\n' %y.shape[0]
                                    if args.permutation_test:
                                        summary += 'Test type:  Permutation\n'
                                        denom = select.sum() - 1
                                        v1 = a[select] * y[select]
                                        v2 = b[select] * y[select]
                                        r1 = v1.sum() / denom
                                        r2 = v2.sum() / denom
                                        rx = (a[select] * b[select]).sum() / (len(a[select]) - 1)
                                        p_value, rdiff, _ = permutation_test(
                                            v1,
                                            v2,
                                            mode='corr',
                                            nested=is_nested,
                                            verbose=True
                                        )
                                    else:
                                        summary += 'Test type:  Seiger (1980)\n'
                                        p_value, Z, r1, r2, rx, rdiff = correlation_test(
                                            y[select],
                                            a[select],
                                            b[select],
                                            nested=is_nested
                                        )
                                        summary += 'Z:          %.4f\n' % Z
                                    summary += 'r(a,y):     %s\n' %r1
                                    summary += 'r(b,y):     %s\n' %r2
                                    summary += 'r(a,b):     %s\n' %rx
                                    summary += 'Difference: %.4f\n' % rdiff
                                    summary += 'p:          %.4e%s\n' % (p_value, '' if p_value > 0.05 else '*' if p_value > 0.01 else '**' if p_value > 0.001 else '***')
                                    summary += '=' * 50 + '\n'

                                    f.write(summary)
                                    sys.stdout.write(summary)

    if args.pool:
        pooled_data = {}
        targets = {}
        for a in ablations:
            pooled_data[a] = {}
            for exp_outdir in exps_outdirs:
                pooled_data[a][exp_outdir] = {}
                for m in basenames_to_pool:
                    m_name = '!'.join([m] + list(a))
                    m_name_path = m_name.replace(':', '+')
                    if os.path.exists(exp_outdir + '/' + m_name_path + '/' + pred_table_name):
                        df = pd.read_csv(exp_outdir + '/' + m_name_path + '/' + pred_table_name, sep=' ', skipinitialspace=True)
                        pooled_data[a][exp_outdir][m] = scale(df.CDRpreds)
                        if exp_outdir not in targets:
                            if 'y' in df.columns:
                                targets[exp_outdir] = scale(df.y)
                            elif 'yStandardized' in df.columns:
                                targets[exp_outdir] = scale(df.yStandardized)
                            else:
                                targets[exp_outdir] = scale(df.CDRobs)
                    else:
                        pooled_data[a][exp_outdir][m] = scale(pd.read_csv(exp_outdir + '/' + m_name_path + '/' + pred_file_name, sep=' ', header=None, skipinitialspace=True)[0])
                        if exp_outdir not in targets:
                            targets[exp_outdir] = scale(pd.read_csv(exp_outdir + '/' + m_name_path + '/' + obs_file_name, sep=' ', header=None, skipinitialspace=True)[0])

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
                    a_name_path = a_name.replace(':', '+')
                    b_name_path = b_name.replace(':', '+')
                    df1 = []
                    df2 = []
                    df_targ = []
                    for exp in exps_outdirs:
                        for m in basenames_to_pool:
                            a = pooled_data[a_model][exp][m]
                            b = pooled_data[b_model][exp][m]
                            y = targets[exp]
                            df1.append(a)
                            df2.append(b)
                            df_targ.append(y)
                    df1 = np.concatenate(df1, axis=0)
                    df2 = np.concatenate(df2, axis=0)
                    df_targ = np.concatenate(df_targ, axis=0)
                    assert len(df1) == len(df2), 'Shape mismatch between datasets %s and %s: %s vs. %s' %(
                        a_name,
                        b_name,
                        df1.shape,
                        df2.shape
                    )
                    stderr('\n')
                    name = '%s_v_%s' % (a_name_path, b_name_path)
                    out_path = p.outdir + '/' + name + '_CT_pooled_' + partition_str + '.txt'
                    with open(out_path, 'w') as f:
                        stderr('Saving output to %s...\n' % out_path)

                        summary = '=' * 50 + '\n'
                        summary += 'Model comparison:   %s vs %s\n' % (a_name, b_name)
                        summary += 'Partition:          %s\n' % partition_str
                        summary += 'Experiments pooled:\n'
                        for exp in exps_outdirs:
                            summary += '  %s\n' %exp
                        summary += 'Ablation sets pooled:\n'
                        for basename in basenames_to_pool:
                            summary += '  %s\n' %basename
                        summary += 'Metric:     corr\n'
                        summary += 'n:          %s\n' %df1.shape[0]
                        if args.permutation_test:
                            summary += 'Test type:  Permutation'
                            denom = len(df1) - 1
                            v1 = df1 * df_targ
                            v2 = df2 * df_targ
                            r1 = v1.sum() / denom
                            r2 = v2.sum() / denom
                            rx = (df1 * df2).sum() / denom
                            p_value, rdiff, _ = permutation_test(
                                df1,
                                df2,
                                mode='corr',
                                nested=is_nested,
                                verbose=True
                            )
                        else:
                            summary += 'Test type:  Seiger (1980)\n'
                            p_value, Z, r1, r2, rx, rdiff = correlation_test(
                                df_targ,
                                df1,
                                df2,
                                nested=is_nested
                            )
                            summary += 'Z:          %.4f\n' % Z
                        summary += 'r(a,y):     %s\n' %r1
                        summary += 'r(b,y):     %s\n' %r2
                        summary += 'r(a,b):     %s\n' %rx
                        summary += 'Difference: %s\n' %rdiff
                        summary += 'p:          %.4e%s\n' % (p_value, '' if p_value > 0.05 else '*' if p_value > 0.01 else '**' if p_value > 0.001 else '***')
                        summary += '=' * 50 + '\n'

                        f.write(summary)
                        sys.stdout.write(summary)
