import sys
import os
import numpy as np
import pandas as pd
import argparse

from cdr.signif import permutation_test
from cdr.util import stderr


def get_suffix(name):
    for s in SUFFIXES:
        if name.endswith(s):
            return s
    return ''

datasets = [
    'brown.fdur',
    'dundee.fdurSPsummed',
    'dundee.fdurFP',
    'dundee.fdurGP',
    'geco.fdurFP',
    'geco.fdurGP',
    'natstor.fdur',
    'natstormaze.rt',
    'provo.fdurSPsummed',
    'provo.fdurFP',
    'provo.fdurGP',
]

if os.path.exists('lme_results_path.txt'):
    with open('lme_results_path.txt') as f:
        for line in f:
           line = line.strip()
           if line:
               results_path = line
               break
else:
    results_path = 'results/lme'

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''Run LME-based signif tests for the CDRNN frequency-predictability study''')
    argparser.add_argument('dataset', help='Name of dataset to test on, one of <brown.fdur,dundee.{fdurSPsummed,fdurFP,fdurGP},geco.{fdurFP,fdurGP},natstor.{fdur},natstormaze.{rt},provo.{fdurSPsummed,fdurFP,fdurGP}>')
    argparser.add_argument('test_name', help='Name of test to run. Format: {lin,sub,sup,notsub,notsup}_v_{lin,sub,sup,notsub,notsup}')
    argparser.add_argument('-f', '--force', action='store_true', help='Re-run test even if results already exist. Otherwise, finished tests will be skipped.')
    args = argparser.parse_args()

    dataset, response = args.dataset.split('.')
    test_name = args.test_name

    a_name, b_name = test_name.split('_v_')
    out_dir = '{results_path}/{dataset}'.format(
        results_path=results_path,
        dataset=dataset,
    )
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    out_file = '{dataset}_{response}_test_{test_name}_outofsample.txt'.format(
        dataset=dataset,
        response=response,
        test_name=test_name
    )
    out_path = os.path.join(out_dir, out_file)

    a = []
    b = []
    a_paths = []
    b_paths = []
    a_hypoth = [a, a_name, a_paths]
    b_hypoth = [b, b_name, b_paths]
    for hypoth in (a_hypoth, b_hypoth):
        h, h_name, paths = hypoth
        path = os.path.join(results_path, dataset, '%s_%s_%s_outofsample_output.csv' % (dataset, response, h_name))
        ll = pd.read_csv(path).LMEloglik.values[..., None]
        h.append(ll)
    summary_path = os.path.join(results_path, dataset, '%s_%s_%s_outofsample_summary.txt' % (dataset, response, b_name))
    with open(summary_path, 'r') as f:
        summary = f.read()

    a = np.concatenate(a, axis=1)
    b = np.concatenate(b, axis=1)

    sel = np.all(np.isfinite(a), axis=1) & np.all(np.isfinite(b), axis=1)
    dropped = (~sel).sum()
    a = a[sel]
    b = b[sel]

    assert len(a) == len(b), 'Length mismatch: %s vs. %s' % (len(a), len(b))

    if not args.force and os.path.exists(out_path):
        stderr('Test results exist. Exiting.\n')
        exit()
 
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

        summary += '\n' + '=' * 50 + '\n'
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

