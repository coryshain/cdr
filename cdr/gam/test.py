import sys
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import argparse

from cdr.signif import permutation_test
from cdr.util import stderr


def get_lm_names(name):
    out = None
    for s in SUFFIXES:
        if name.endswith(s):
            out = name[:-len(s)]
            break
    if out is None:
        out = name
    if out == 'allLM':
        out = LMS[1:]
    else:
        out = [out]
    return out


def get_suffix(name):
    for s in SUFFIXES:
        if name.endswith(s):
            return s
    return ''

SUFFIXES = [
    'nosurp',
    'prob',
    '0.50',
    '0.75',
    '1.00',
    '1.33',
    '2.00',
]
SUFFIXES = sorted(SUFFIXES, key=lambda x: len(x), reverse=True)


DATASETS = [
    'brown',
    'dundee',
    'geco',
    'natstor',
    'natstormaze',
    'provo'
]

LMS = [
    'cloze',
    'ngram',
    'pcfg',
    'gpt',
    'gptj',
    'gpt3',
]

RESPONSE_MAP = {
    'brown':              ['fdur'],
    'dundee':             ['fdurSPsummed', 'fdurFP', 'fdurGP'],
    'geco':               ['fdurFP', 'fdurGP'],
    'natstor':            ['fdur'],
    'natstormaze':        ['rt'],
    'provo':              ['fdurSPsummed', 'fdurFP', 'fdurGP'],
}

MODELS_MAP = {
    'nosurp':          ['nosurp'],
    '':                ['%s'],
    'prob':            ['%sprob'],
    '0.50':            ['pow0_5_%s_'],
    '0.75':            ['pow0_75_%s_'],
    '1.00':            ['pow1_%s_'],
    '1.33':            ['pow1_33_%s_'],
    '2.00':            ['pow2_%s_'],
}

if os.path.exists('gam_results_path.txt'):
    with open('gam_results_path.txt') as f:
        for line in f:
           line = line.strip()
           if line:
               results_path = line
               break
else:
    results_path = 'results/gam'

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''Run extra signif tests for the CDRNN surprisal study''')
    argparser.add_argument('dataset', help='Name of dataset to test on, one of <all,brown.fdur,dundee.{fdurSPsummed,fdurFP,fdurGP},geco.{fdurFP,fdurGP},natstor.{fdur},natstormaze.{rt},provo.{fdurSPsummed,fdurFP,fdurGP}>')
    argparser.add_argument('test_name', help='Name of test to run. Format: {lin,sub,sup,notsub,notsup}_v_{lin,sub,sup,notsub,notsup}')
    argparser.add_argument('-f', '--force', action='store_true', help='Re-run test even if results already exist. Otherwise, finished tests will be skipped.')
    args = argparser.parse_args()

    dataset = args.dataset
    test_name = args.test_name

    a_name, b_name = test_name.split('_v_')
    a_lm = get_lm_names(a_name)
    b_lm = get_lm_names(b_name)

    a_models = MODELS_MAP[get_suffix(a_name)]
    b_models = MODELS_MAP[get_suffix(b_name)]

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
    a_hypoth = [a, a_datasets, a_models, a_lm, a_paths]
    b_hypoth = [b, b_datasets, b_models, b_lm, b_paths]
    responses = set()
    for hypoth in (a_hypoth, b_hypoth):
        h, h_datasets, h_models, lm_names, paths = hypoth
        for _dataset in h_datasets:
            for response in response_map[_dataset]:
                responses.add(response)
                _h = []
                for model in h_models:
                    for lm_name in lm_names:
                        lm_name = lm_name.replace('pcfg', 'totsurp')
                        if lm_name != 'cloze' or _dataset.startswith('provo'):
                            if model == 'nosurp':
                               model_name = model
                            else:
                               model_name = model % lm_name
                            path = os.path.join(results_path, _dataset, '%s_%s_output_test.csv' % (response, model_name))
                            ll = pd.read_csv(path).GAMloglik.values
                            _h.append(ll)
                            paths.append(path)
                _h = np.stack(_h, axis=1)
                h.append(_h)

    a = np.concatenate(a, axis=0)
    b = np.concatenate(b, axis=0)

    sel = np.all(np.isfinite(a), axis=1) & np.all(np.isfinite(b), axis=1)
    dropped = (~sel).sum()
    a = a[sel]
    b = b[sel]

    assert len(a) == len(b), 'Length mismatch: %s vs. %s' % (len(a), len(b))

    name = test_name
    outdir = os.path.join(results_path, 'signif/%s' % dataset)
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
