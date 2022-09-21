import os
import numpy as np
import pandas as pd

from cdr.signif import permutation_test

for dataset in [
    'natstor_log',
    'natstor_raw',
    'dundee_raw_sp',
    'dundee_log_sp',
    'dundee_raw_fp',
    'dundee_log_fp',
    'dundee_raw_gp',
    'dundee_log_gp',
    'fmri'
]:
    paths = [x for x in os.listdir('../results/cdrnn_journal/%s/CDR_main' % dataset) if x.startswith('output_') and x.endswith('_test.csv')]
    assert len(paths) == 1, 'Wrong number of paths for %s: %s' % (dataset, paths)
    path = paths[0]
    ll_cdrnn = pd.read_csv('../results/cdrnn_journal/%s/CDR_main/' % dataset + path, sep=' ').CDRloglik.values

    if dataset == 'fmri':
        baselines = [
            '../results/cognition/fMRI/fMRI_convolved/LME/conditional_ll_test.txt',
            '../results/cognition/fMRI/fMRI_interpolated/LME/conditional_ll_test.txt',
            '../results/cognition/fMRI/fMRI_averaged/LME/conditional_ll_test.txt',
            '../results/cognition/fMRI/fMRI_lanczos/LME/conditional_ll_test.txt',
            '../results/cdrnn_journal/fmriraw/GAMnoS/conditional_ll_test.txt',
            '../results/cdrnn_journal/fmriraw/GAMLSSnoS/conditional_ll_test.txt',
            '../results/cognition/fMRI/fMRI_cdr/CDR_DG5_bbvi/loglik_test.txt',
        ]
    elif dataset.endswith('sp'):
        baselines = [
            '../results/cognition/reading/%s/LMEnoS/conditional_ll_test.txt' % dataset,
            '../results/cognition/reading/%s/GAMnoS/conditional_ll_test.txt' % dataset,
            '../results/cognition/reading/%s/GAMfullS/conditional_ll_test.txt' % dataset,
            '../results/cdrnn_journal/%s/GAMLSSnoS/conditional_ll_test.txt' % dataset,
            '../results/cdrnn_journal/%s/GAMLSSfullS/conditional_ll_test.txt' % dataset,
            '../results/cognition/reading/%s/CDR_G_bbvi/loglik_test.txt' % dataset,
        ]
    else:
        baselines = [
            '../results/cognition/reading/%s/LMEnoS/conditional_ll_test.txt' % dataset,
            '../results/cognition/reading/%s/LMEfullS/conditional_ll_test.txt' % dataset,
            '../results/cognition/reading/%s/GAMnoS/conditional_ll_test.txt' % dataset,
            '../results/cognition/reading/%s/GAMfullS/conditional_ll_test.txt' % dataset,
            '../results/cdrnn_journal/%s/GAMLSSnoS/conditional_ll_test.txt' % dataset,
            '../results/cdrnn_journal/%s/GAMLSSfullS/conditional_ll_test.txt' % dataset,
            '../results/cognition/reading/%s/CDR_G_bbvi/loglik_test.txt' % dataset,
        ]

    for baseline in baselines:
        ll_base = pd.read_csv(baseline, header=None)
        p, diff, _ = permutation_test(ll_base, ll_cdrnn, mode='loglik')
        
        print('=' * 50)
        print(dataset)
        print(baseline)
        print('p = %f' % p)
        print('diff = %f' % diff)
        print()
