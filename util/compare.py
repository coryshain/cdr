import sys
import argparse
import numpy as np
import pandas as pd
from dtsr import Config, bootstrap

if __name__ == '__main__':

    argparser = argparse.ArgumentParser('''
        Computes pairwise significance of error statistics between models
    ''')
    argparser.add_argument('config_path', help='Path to configuration (*.ini) file')
    argparser.add_argument('-m', '--models', nargs='*', default=[], help='Path to configuration (*.ini) file')
    argparser.add_argument('-p', '--partition', type=str, default='dev', help='Name of partition to use (one of "train", "dev", "test")')
    args, unknown = argparser.parse_known_args()

    p = Config(args.config_path)
    if len(args.models) > 0:
        models = args.models
    else:
        models = p.model_list[:]

    run_baseline = False
    run_dtsr = False
    for m in models:
        if not run_baseline and m.startswith('LM') or m.startswith('GAM'):
            run_baseline = True
        elif not run_dtsr and m.startswith('DTSR'):
            run_dtsr = True

    sys.stderr.write('\n')
    for i in range(len(models)):
        for j in range(i+1, len(models)):
            a = pd.read_csv(p.logdir + '/' + models[i] + '/%s_losses_%s.txt'%(p.loss, args.partition), sep=' ', header=None, skipinitialspace=True)
            b = pd.read_csv(p.logdir + '/' + models[j] + '/%s_losses_%s.txt'%(p.loss, args.partition), sep=' ', header=None, skipinitialspace=True)
            select = np.logical_and(np.isfinite(np.array(a)), np.isfinite(np.array(b)))
            diff = float(len(a) - select.sum())
            p_value, base_diff = bootstrap(a[select], b[select], n_iter=1000)
            sys.stderr.write('\n')
            print('='*50)
            print('Model comparison: %s vs %s' %(models[i], models[j]))
            if diff > 0:
                print('%d NaN rows filtered out (out of %d)' %(diff, len(a)))
            print('Partition: %s' %args.partition)
            print('Loss difference: %.4f' %base_diff)
            print('p: %.4e' %p_value)
            print('='*50)
            print()
            print()
