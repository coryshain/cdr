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


def scale(a):
    return (a - a.mean()) / a.std()


def joint_scale(a, b):
    df = np.stack([np.array(a), np.array(b)], axis=1)
    scaling_factor = df.std()
    return a/scaling_factor, b/scaling_factor


def get_summary(
        a_model,
        b_model,
        a_perf,
        b_perf,
        base_diff,
        p_value,
        metric,
        partition_str,
        n,
        a_ensemble_size=None,
        b_ensemble_size=None,
        agg_fn=None,
        r1=None,
        r2=None,
        rx=None
    ):
    summary = '='*50 + '\n'
    summary += 'Model comparison:      %s vs %s\n' % (a_model, b_model)
    summary += 'Partition:             %s\n' % partition_str
    summary += 'Metric:                %s\n' % metric
    if a_ensemble_size is not None or b_ensemble_size is not None:
        if agg_fn is not None:
            summary += 'Ensemble agg fn:       %s\n' % agg_fn
        if a_ensemble_size is not None:
            summary += 'Model A ensemble size: %s\n' % a_ensemble_size
        if b_ensemble_size is not None:
            summary += 'Model B ensemble size: %s\n' % b_ensemble_size
    summary += 'n: %s\n' % n
    if r1 is not None:
        summary += 'r(a,y):                %s\n' % r1
    if r2 is not None:
        summary += 'r(b,y):                %s\n' % r2
    if rx is not None:
        summary += 'r(a,b):            %s\n' % rx
    summary += 'Model A:               %s\n' % a_model
    summary += 'Model B:               %s\n' % b_model
    summary += 'Model A score:         %.4f\n' % a_perf
    summary += 'Model B score:         %.4f\n' % b_perf
    summary += 'Difference:            %.4f\n' % base_diff
    summary += 'p:                     %.4e%s\n' % (p_value, '' if p_value > 0.05 \
        else '*' if p_value > 0.01 else '**' if p_value > 0.001 else '***')
    summary += '='*50 + '\n'

    return summary




if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
        Performs pairwise permutation test for significance of differences in prediction quality between models.
        Can be used for in-sample and out-of-sample evaluation.
        Used (1) to perform hypothesis testing between CDR models within one or more ablation sets, or (2) (using the "-P" flag) to perform a pooled test comparing CDR models fitted to multiple responses. 
    ''')
    argparser.add_argument('config_paths', nargs='*', help='Path(s) to configuration (*.ini) file')
    argparser.add_argument('-m', '--models', nargs='*', default=[], help='List of models (or model basenames if using -a) to compare. Regex permitted. If unspecified, uses all models.')
    argparser.add_argument('-n', '--n_iter', type=int, default=10000, help='Number of resampling iterations.')
    argparser.add_argument('-P', '--pool', action='store_true', help='Pool test statistic by model name across config files (experiments). Only applied to models that are (i) named identically across config files an (ii) have evaluation data for the relevant partition for every config file.')
    argparser.add_argument('-a', '--ablation', action='store_true', help='Only compare models within an ablation set (those defined using the "ablate" param in the config file)')
    argparser.add_argument('-p', '--partition', type=str, default='dev', help='Name of partition to use (one of "train", "dev", "test")')
    argparser.add_argument('-g', '--agg', type=str, default='median', help='Aggregation function to use over ensembles. E.g., ``"mean"``, ``"median"``, ``"min"``, ``"max"``.')
    argparser.add_argument('-M', '--metric', type=str, default='loglik', help='Metric to use for comparison ("mse", "loglik", or "corr")')
    argparser.add_argument('-T', '--tails', type=int, default=2, help='Number of tails (1 or 2)')
    argparser.add_argument('-r', '--response', nargs='*', default=None, help='Name(s) of response(s) to test. If left unspecified, tests all responses.')
    argparser.add_argument('-o', '--outdir', default=None, help='Output directory. If ``None``, placed in same directory as the config.')
    args = argparser.parse_args()

    metric = args.metric
    if metric == 'err':
        metric = 'mse'
    if metric.lower() == 'll':
        metric = 'loglik'
    assert metric in ['mse', 'loglik', 'corr'], 'Metric must be one of ["mse", "loglik", "corr"].'

    exps_outdirs = []
    partitions = get_partition_list(args.partition)
    partition_str = '-'.join(partitions)

    pooled_comparison_sets = None
    for path in args.config_paths:
        p = Config(path)
        exps_outdirs.append(p.outdir)

        model_list = sorted(set(p.model_names) | set(p.ensemble_names) | set(p.crossval_family_names))
        models = filter_models(model_list, args.models)
        cdr_models = [x for x in filter_models(models, cdr_only=True)]

        if args.ablation:
            comparison_sets = {}
            for model_name in cdr_models:
                model_basename = model_name.split('!')[0]
                if model_basename not in comparison_sets:
                    comparison_sets[model_basename] = []
                comparison_sets[model_basename].append(model_name)
            for model_name in p.model_names:
                model_basename = model_name.split('!')[0]
                if model_basename in comparison_sets and model_name not in comparison_sets[model_basename]:
                    comparison_sets[model_basename].append(model_name)
            if pooled_comparison_sets is None:
                pooled_comparison_sets = comparison_sets.copy()
            else:
                for comparison_set in comparison_sets:
                    if comparison_set not in pooled_comparison_sets:
                        del pooled_comparison_sets[comparison_set]
                    else:
                        pooled_comparison_sets[comparison_set] = sorted(list(set(pooled_comparison_sets[comparison_set]) & set(comparison_sets[comparison_set])))
        else:
            comparison_sets = {
                None: cdr_models
            }
            if pooled_comparison_sets is None:
                pooled_comparison_sets = comparison_sets.copy()
            else:
                pooled_comparison_sets[None] = sorted(list(set(pooled_comparison_sets[None]) & set(comparison_sets[None])))

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
                                            if not (len(a) and len(b)):
                                                # Need at least one array per model
                                                continue

                                            if metric == 'corr':
                                                y = None
                                                for x in a:
                                                    if y is None:
                                                        y = a[x]['CDRobs']
                                                        break
                                                a = np.stack([a[x]['CDRpreds'] for x in a], axis=1)
                                                b = np.stack([b[x]['CDRpreds'] for x in b], axis=1)

                                                y = scale(y)
                                                denom = len(y) - 1
                                                r1 = []
                                                r2 = []
                                                rx = []
                                                for k in range(a.shape[-1]):
                                                    _a = a[..., k]
                                                    _b = b[..., k]

                                                    _a = scale(_a)
                                                    _b = scale(_b)

                                                    _a = _a * y
                                                    _b = _b * y

                                                    a[..., k] = _a
                                                    b[..., k] = _b

                                                    r1.append(_a.sum() / denom)
                                                    r2.append(_b.sum() / denom)
                                                    rx.append((_a * _b).sum() / denom)
                                                r1 = np.mean(r1)
                                                r2 = np.mean(r2)
                                                rx = np.mean(rx)
                                            else:
                                                a = np.stack([a[x] for x in a], axis=1)
                                                b = np.stack([b[x] for x in b], axis=1)
                                                r1 = r2 = rx = None

                                            p_value, a_perf, b_perf, base_diff, diffs = permutation_test(
                                                a,
                                                b,
                                                n_iter=args.n_iter,
                                                n_tails=args.tails,
                                                mode=metric,
                                                agg=args.agg,
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
                                                summary = get_summary(
                                                        a_model,
                                                        b_model,
                                                        a_perf,
                                                        b_perf,
                                                        base_diff,
                                                        p_value,
                                                        metric,
                                                        partition_str,
                                                        a.shape[0],
                                                        a_ensemble_size=a.shape[1],
                                                        b_ensemble_size=b.shape[1],
                                                        agg_fn=args.agg,
                                                        r1=r1,
                                                        r2=r2,
                                                        rx=rx
                                                    )

                                                f.write(summary)
                                                sys.stdout.write(summary)

                                            plt.hist(diffs, bins=1000)
                                            plt.savefig(outdir + '/' + name_base + '.png')
                                            plt.close('all')

    if args.pool:
        pooled_data = {}
        for s in pooled_comparison_sets:
            pooled_data[s] = {}
            for exp_outdir in exps_outdirs:
                pooled_data[s][exp_outdir] = {}
                for m in pooled_comparison_sets[s]:
                    m_name = m.replace(':', '+')
                    m_files = extract_cdr_prediction_data(exp_outdir + '/' + m_name, metric=metric)
                    for response in m_files:
                        for filenum in m_files[response]:
                            if partition_str in m_files[response][filenum] and \
                                    (args.response is None or response in args.response):
                                v = m_files[response][filenum][partition_str]['direct']
                                if metric == 'corr':
                                    y = None
                                    for x in v:
                                        if y is None:
                                            y = v[x]['CDRobs']
                                            break
                                    v = np.stack([v[x]['CDRpreds'] for x in v], axis=1)
                                    for k in range(v.shape[-1]):
                                        _v = v[..., k]
                                        _v = scale(_v)
                                        _v = _v * y
                                        v[..., k] = _v
                                    v = (y, v)
                                else:
                                    v = np.stack([v[x] for x in v], axis=1)

                                if s not in pooled_data:
                                    pooled_data[s] = {}
                                if exp_outdir not in pooled_data[s]:
                                    pooled_data[s][exp_outdir] = {}
                                if m not in pooled_data[s][exp_outdir]:
                                    pooled_data[s][exp_outdir][m] = {}
                                if response not in pooled_data[s][exp_outdir][m]:
                                    pooled_data[s][exp_outdir][m][response] = {}
                                pooled_data[s][exp_outdir][m][response][filenum] = v

        for s in pooled_comparison_sets:
            model_set = pooled_comparison_sets[s]
            if len(model_set) > 1:
                if s is not None:
                    stderr('Comparing models within ablation set "%s"...\n' % s)
                for i in range(len(model_set)):
                    m1 = model_set[i]
                    for j in range(i + 1, len(model_set)):
                        m2 = model_set[j]
                        is_nested = nested(m1, m2)
                        if is_nested or not args.ablation:
                            if m1.count('!') > m2.count('!'):
                                a_model = m1
                                b_model = m2
                                a_name = m2 if m2 else 'FULL'
                                b_name = m1 if m1 else 'FULL'
                            else:
                                a_model = m2
                                b_model = m1
                                a_name = m1 if m1 else 'FULL'
                                b_name = m2 if m2 else 'FULL'
                            df1 = []
                            df2 = []
                            for exp in exps_outdirs:
                                for response in pooled_data[s][exp][a_model]:
                                    for filenum in pooled_data[s][exp][b_model][response]:
                                        if metric == 'mse':
                                            a, b = joint_scale(
                                                pooled_data[s][exp][a_model][response][filenum],
                                                pooled_data[s][exp][b_model][response][filenum]
                                            )
                                        else:
                                            a = pooled_data[s][exp][a_model][response][filenum]
                                            b = pooled_data[s][exp][b_model][response][filenum]
                                        df1.append(a)
                                        df2.append(b)
                            a = df1
                            b = df2
                            if not (len(a) and len(b)):
                                # Need at least one array per model
                                continue
        
                            if metric == 'corr':
                                y = np.concatenate([x[0] for x in a], axis=0)
                                a = np.concatenate([x[1] for x in a], axis=0)
                                b = np.concatenate([x[1] for x in b], axis=0)
        
                                y = scale(y)
                                a = scale(a)
                                b = scale(b)
        
                                a = a * y
                                b = b * y
        
                                denom = len(y) - 1
                                r1 = (a.sum(axis=0) / denom).mean()
                                r2 = (b.sum(axis=0) / denom).mean()
                                rx = ((a * b).sum(axis=0) / denom).mean()
        
                            else:
                                a = np.concatenate(a, axis=0)
                                print(a.shape)
                                b = np.concatenate(b, axis=0)
                                print(b.shape)
                                r1 = r2 = rx = None
        
                            assert len(a) == len(b), 'Shape mismatch between datasets %s and %s: %s vs. %s' % (
                                a_name,
                                b_name,
                                a.shape,
                                b.shape
                            )
                            p_value, a_perf, b_perf, diff, diffs = permutation_test(
                                a,
                                b,
                                n_iter=args.n_iter,
                                n_tails=args.tails,
                                mode=metric,
                                agg=args.agg,
                                nested=is_nested
                            )
                            stderr('\n')
                            name = '%s_v_%s' % (a_name.replace(':', '+'), b_name.replace(':', '+'))
                            name_base = '%s_PT_pooled_%s' % (name, partition_str)
                            outdir = args.outdir
                            if outdir is None:
                                outdir = 'signif_pooled' 
                            if not os.path.exists(outdir):
                                os.makedirs(outdir)
                            out_path = outdir + '/' + name_base + '.txt'
                            with open(out_path, 'w') as f:
                                stderr('Saving output to %s...\n' % out_path)
                                summary = get_summary(
                                        a_name,
                                        b_name,
                                        a_perf,
                                        b_perf,
                                        diff,
                                        p_value,
                                        metric,
                                        partition_str,
                                        a.shape[0],
                                        a_ensemble_size=a.shape[1],
                                        b_ensemble_size=b.shape[1],
                                        agg_fn=args.agg,
                                        r1=r1,
                                        r2=r2,
                                        rx=rx
                                )
         
                                f.write(summary)
                                sys.stdout.write(summary)
        
                            plt.hist(diffs, bins=1000)
                            plt.savefig(outdir + '/' + name_base + '.png')
                            plt.close('all')
