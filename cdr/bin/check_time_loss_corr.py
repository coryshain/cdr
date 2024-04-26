import os
import numpy as np
import numpy.ma as ma
import pandas as pd
import argparse
import textwrap
from io import StringIO


if __name__ == '__main__':
    argparser = argparse.ArgumentParser(textwrap.dedent('''Compute the correlation between training iterations and model performance.
        At convergence, this should be near 0 (the outputs of this script are only meaningful if all models have finished training).
        If converged models show a strong positive correlation, this could be a sign of under-training.
        Considering increased ``convergence_n_iterates`` and re-running.'''))
    argparser.add_argument('config_paths', nargs='+', help='Path(s) to configuration (*.ini) file(s)')
    argparser.add_argument('-p', '--partition', default='dev', help='Partition to check.')
    args = argparser.parse_args()
    config_paths = args.config_paths
    partition = args.partition

    ll = pd.read_csv(StringIO(os.popen('python -m cdr.bin.metrics %s -c' % ' '.join(config_paths)).read()), na_rep='---')
    ll = ll.set_index('model')
    n_iter = pd.read_csv(StringIO(os.popen('python -m cdr.bin.metrics %s -m iter -c' % ' '.join(config_paths)).read()), na_rep='---')
    n_iter = n_iter.set_index('model')

    datasets = [x for x in ll if x.endswith(partition)]

    for dataset in datasets:
        a = ma.masked_invalid(ll[dataset])
        b = ma.masked_invalid(n_iter[dataset])
        print(dataset, 'r:', np.round(ma.corrcoef(a, b)[0, 1], 2))

