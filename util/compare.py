import sys
import argparse
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
            a = pd.read_csv(p.logdir + '/' + models[i] + '/%s_losses_%s.txt'%(p.loss, args.partition))
            b = pd.read_csv(p.logdir + '/' + models[j] + '/%s_losses_%s.txt'%(p.loss, args.partition))
            p_value, base_diff = bootstrap(a, b, n_iter=10000)
            sys.stderr.write('\n')
            print('='*50)
            print('Model comparison: %s vs %s' %(models[i], models[j]))
            print('Partition: %s' %args.partition)
            print('Loss difference: %.4f' %base_diff)
            print('p: %.4f' %p_value)
            print('='*50)
            print()
            print()
