import sys
import argparse
import numpy as np
import pandas as pd
from dtsr.signif import permutation_test
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from dtsr.util import stderr


def scale(a, b):
    df = np.stack([np.array(a), np.array(b)], axis=1)
    df = df[np.where(np.isfinite(df))] 
    scaling_factor = df.std()
    print(scaling_factor)
    return a/scaling_factor, b/scaling_factor


if __name__ == '__main__':

    argparser = argparse.ArgumentParser('''
        Performs pairwise permutation test for significance of differences in prediction quality between arbitrary models.
        Used to compare DTSR to other kinds of statistical models trained on the same data.
        Appropriate for in-sample and out-of-sample evaluation.
    ''')
    argparser.add_argument('model_error_paths', type=str, help='";"-delimited list of paths to files containing model error/likelihood vectors.')
    argparser.add_argument('-n', '--name', type=str, default='bakeoff', help='Name for this set of bakeoff comparisons. Default: "bakeoff".')
    argparser.add_argument('-b', '--baseline_error_paths', nargs='*', default=[], help='List of ";"-delimited lists of paths to files containing baseline error/likelihood vectors.')
    argparser.add_argument('-M', '--metric', type=str, default='loss', help='Metric to use for comparison (either "loss" or "loglik")')
    argparser.add_argument('-t', '--tails', type=int, default=2, help='Number of tails (1 or 2)')
    argparser.add_argument('-o', '--outdir', type=str, default='./', help='Path to output directory')
    args, unknown = argparser.parse_known_args()

    assert args.metric in ['loss', 'loglik'], 'Metric must be one of ["loss", "loglik"].'

    model_errors = []
    for path in args.model_error_paths.split(';'):
        model_errors.append(pd.read_csv(path, sep=' ', header=None, skipinitialspace=True))

    baseline_errors = []
    for baseline in args.baseline_error_paths:
        baseline_errors.append([])
        for path in baseline.split(';'):
            baseline_errors[-1].append(pd.read_csv(path, sep=' ', header=None, skipinitialspace=True))

    for i, baseline in enumerate(baseline_errors):
        assert len(model_errors) == len(baseline), 'Model and baseline must contain the same number of datasets. Saw %d and %d, respectively.' % (len(model_errors), len(baseline))

        if len(model_errors) > 1:
            model_cur = []
            baseline_cur = []
            for m, b in zip(model_errors, baseline):
                m_cur, b_cur = scale(m, b)
                model_cur.append(m_cur)
                baseline_cur.append(b_cur)
            model_cur = np.concatenate(model_cur, axis=0)
            baseline_cur = np.concatenate(baseline_cur, axis=0)
        else:
            model_cur = np.array(model_errors[0])
            baseline_cur = np.array(baseline_errors[i][0])

        select = np.logical_and(np.isfinite(np.array(model_cur)), np.isfinite(np.array(baseline_cur)))
        diff = float(len(model_cur) - select.sum())
        p_value, base_diff, diffs = permutation_test(baseline_cur[select], model_cur[select], n_iter=10000, n_tails=args.tails, mode=args.metric, nested=True)
        stderr('\n')
        out_path = args.outdir + '/%s_%d_PT.txt' % (args.name, i)
        with open(out_path, 'w') as f:
            stderr('Saving output to %s...\n' %out_path)

            summary = '='*50 + '\n'
            summary += 'Model comparison:\n'
            summary += '  %s\n' % args.baseline_error_paths[i]
            summary += '  vs\n'
            summary += '  %s\n' % args.model_error_paths
            if diff > 0:
                summary += '%d NaN rows filtered out (out of %d)\n' % (diff, len(model_cur))
            summary += 'Metric: %s\n' % args.metric
            summary += 'Difference: %.4f\n' % base_diff
            summary += 'p: %.4e%s\n' % (p_value, '' if p_value > 0.05 else '*' if p_value > 0.01 else '**' if p_value > 0.001 else '***')
            summary += '='*50 + '\n'

            f.write(summary)
            sys.stdout.write(summary)

        plt.hist(diffs, bins=1000)
        plt.savefig(args.outdir + '/%s_%d_PT.png' % (args.name, i))
        plt.close('all')

