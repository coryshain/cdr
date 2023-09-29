import sys
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import argparse

from cdr.signif import permutation_test
from cdr.util import stderr


def get_model_path(name, experiment):
    name = name.replace('1.00', '')
    return results_path + '/lme_%s_%s_output_test.csv' % (experiment, name)


if __name__ == '__main__':
    if os.path.exists('bk21_results_path.txt'):
        with open('bk21_results_path.txt') as f:
            for line in f:
               line = line.strip()
               if line:
                   results_path = line
                   break
    else:
        results_path = 'results/bk21'

    argparser = argparse.ArgumentParser('''Run signif tests for the Brothers & Kuperberg data''')
    argparser.add_argument('experiment', help='Name of experiment (one of ``spr``, ``naming``).')
    argparser.add_argument('test_name', help='Name of test to run.')
    args = argparser.parse_args()

    experiment = args.experiment
    test_name = args.test_name

    response_name = 'rt'

    a_name, b_name = test_name.split('_v_')

    a_paths = [get_model_path(a_name, experiment)]
    b_paths = [get_model_path(b_name, experiment)]
    a = pd.read_csv(a_paths[0])['loglik'].values[:, None]
    b = pd.read_csv(b_paths[0])['loglik'].values[:, None]

    sel = np.all(np.isfinite(a), axis=1) & np.all(np.isfinite(b), axis=1)
    dropped = (~sel).sum()
    a = a[sel]
    b = b[sel]

    assert len(a) == len(b), 'Length mismatch: %s vs. %s' % (len(a), len(b))

    name = test_name
    outdir = results_path + '/signif/%s/' % experiment
    name_base = '%s_PT_%s_test' % (name, response_name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    out_path = os.path.join(outdir, name_base + '.txt')
   
    p_value, a_perf, b_perf, diff, diffs = permutation_test(
        a,
        b,
        mode='loglik',
        agg='median',
        nested=False
    )
    stderr('\n')
    with open(out_path, 'w') as f:
        stderr('Saving output to %s...\n' % out_path)

        summary = '=' * 50 + '\n'
        summary += 'Model comparison: %s vs %s\n' % (a_name, b_name)
        summary += 'Partition: test\n'
        summary += 'Metric: loglik\n'
        if a.shape[1] > 1:
            summary += 'Ens agg fn: median\n'
        summary += 'Model A paths pooled:\n'
        for path in a_paths:
            summary += '  %s\n' % path
        summary += 'Model B paths pooled:\n'
        for path in b_paths:
            summary += '  %s\n' % path
        summary += 'N:            %s\n' % a.shape[0]
        summary += 'N Ensemble A: %s\n' % a.shape[1]
        summary += 'N Ensemble B: %s\n' % b.shape[1]
        if dropped:
            summary += 'N dropped:    %d\n' % dropped
        summary += 'Model A:      %.4f\n' % a_perf
        summary += 'Model B:      %.4f\n' % b_perf
        summary += 'Difference:   %.4f\n' % diff
        summary += 'p:            %.4e%s\n' % (
            p_value,
            '' if p_value > 0.05 else '*' if p_value > 0.01
            else '**' if p_value > 0.001 else '***'
        )
        summary += '=' * 50 + '\n'

        f.write(summary)
        sys.stdout.write(summary)

    plt.hist(diffs, bins=1000)
    plt.savefig(os.path.join(outdir, name_base + '.png'))
    plt.close('all')
