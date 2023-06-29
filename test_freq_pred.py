import sys
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import argparse

from cdr.signif import permutation_test
from cdr.util import stderr


DATASETS = [
    'brown',
    'dundee',
    'geco',
    'natstor',
    'natstormaze',
    'provo'
]

RESPONSE_MAP = {
    'brown':              ['fdur'],
    'dundee':             ['fdurSPsummed', 'fdurFP', 'fdurGP'],
    'geco':               ['fdurFP', 'fdurGP'],
    'natstor':            ['fdur'],
    'natstormaze':        ['rt'],
    'provo':              ['fdurSPsummed', 'fdurFP', 'fdurGP'],
}

# Change to your desired output path for testing results
results_path = '../results/cdrnn_freq_pred_owt/'

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''Run extra signif tests for the CDRNN surprisal study''')
    argparser.add_argument('dataset', help='Name of dataset to test on, one of <all,brown.fdur,dundee.{fdurSPsummed,fdurFP,fdurGP},geco.{fdurFP,fdurGP},natstor.{fdur},natstormaze.{rt},provo.{fdurSPsummed,fdurFP,fdurGP}>')
    argparser.add_argument('test_name', help='Name of test to run. Format: MODELA_v_MODELB')
    argparser.add_argument('-f', '--force', action='store_true', help='Re-run test even if results already exist. Otherwise, finished tests will be skipped.')
    args = argparser.parse_args()

    dataset = args.dataset
    test_name = args.test_name

    a_name, b_name = test_name.split('_v_')

    response_map = RESPONSE_MAP

    if dataset == 'all':
        datasets = DATASETS
    else:
        dataset, response = dataset.split('.')
        datasets = [dataset]
        response_map = {dataset: [response]}

    a_datasets = datasets 
    b_datasets = datasets

    a = []
    b = []
    a_paths = []
    b_paths = []
    a_hypoth = [a, a_datasets, a_name, a_paths]
    b_hypoth = [b, b_datasets, b_name, b_paths]
    responses = set()
    for hypoth in (a_hypoth, b_hypoth):
        h, h_datasets, h_name, paths = hypoth
        for _dataset in h_datasets:
            for response in response_map[_dataset]:
                responses.add(response)
                if 'log' in h_name:
                    response = 'log.%s.' % response
                _h = []
                for i in range(10):
                    _model = '%s.m%s' % (h_name, i)
                    path = os.path.join(results_path, _dataset, _model, 'output_%s_test.csv' % response)
                    ll = pd.read_csv(path, sep=' ').CDRloglik.values
                    _h.append(ll)
                    paths.append(path)
                _h = np.stack(_h, axis=1)
                h.append(_h)

    a = np.concatenate(a, axis=0)
    b = np.concatenate(b, axis=0)

    assert len(a) == len(b), 'Length mismatch: %s vs. %s' % (len(a), len(b))

    name = test_name
    outdir = '../results/cdrnn_freq_pred_owt/signif/%s' % dataset
    if len(responses) == 1:
        response_name = list(responses)[0]
    else:
        response_name = 'pooled'
    name_base = '%s_PT_%s_test' % (name, response_name)
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    out_path = os.path.join(outdir, name_base + '.txt')
   
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
