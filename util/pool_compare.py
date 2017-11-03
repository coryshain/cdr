import sys
import argparse
import numpy as np
import pandas as pd
from dtsr import Config, bootstrap

def scale(a, b):
    df = pd.DataFrame(a, columns=['a'])
    df['b'] = b
    scaling_factor = df.std()
    return a/scaling_factor, b/scaling_factor

if __name__ == '__main__':

    argparser = argparse.ArgumentParser('''
        Computes pairwise significance of error differences between DTSR models and competitors across tasks.
        Assumes models are named using the template <MODELNAME>_<TASKNAME>, where <TASKNAME> is
        shared between models that should be compared. For example, if the config file contains
        4 models --- DTSR_TASK1, DTSR_TASK2, COMPETITOR_TASK1, and COMPETITOR_TASK2 --- the script
        will perform 2 comparisons: DTSR_TASK1 vs COMPETITOR_TASK1 and DTSR_TASK2 vs. COMPETITOR_TASK2.
    ''')
    argparser.add_argument('-t', '--tasks', nargs='*', default=[], help='List of paths to results files.')
    argparser.add_argument('-p', '--partition', type=str, default='dev', help='Name of partition to use (one of "train", "dev", "test")')
    args, unknown = argparser.parse_known_args()

    losses = {}
    select = {}

    for task in args.tasks:
        print(args.tasks)
        p = Config(task + '.ini')
        models = p.model_list[:]
        for m in models:
            l = pd.read_csv(p.logdir + '/' + m + '/%s_losses_%s.txt'%(p.loss, args.partition), sep=' ', header=None, skipinitialspace=True)
            s = np.isfinite(np.array(l))
            if m not in losses:
                losses[m] = {}
            if m not in select:
                select[m] = {}
            if task not in losses[m]:
                losses[m][task] = l
            if task not in select[m]:
                select[m][task] = s


    dtsr_models = [x for x in models if x.startswith('DTSR')]
    for m1 in dtsr_models:
        trial_name = m1.strip().split('_')[1]
        competitors = [x for x in models if not x == m1 and x.split('_')[1] == trial_name]
        for m2 in competitors:
            m1_scaled_losses = []
            m2_scaled_losses = []
            for task in losses[m2]:
                s = np.logical_and(select[m1][task], select[m2][task])
                a, b = scale(losses[m1][task][s], losses[m2][task][s])
                m1_scaled_losses.append(a)
                m2_scaled_losses.append(b)
            p_value, base_diff = bootstrap(m1_scaled_losses, m2_scaled_losses, n_iter=10000)

            print('='*50)
            print('Model comparison: %s vs %s' %(m1, m2))
            print('Partition: %s' %args.partition)
            print('Loss difference: %.4f' %base_diff)
            print('p: %.4e' %p_value)
            print('='*50)
            print()
            print()
