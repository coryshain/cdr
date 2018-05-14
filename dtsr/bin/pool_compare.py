import os
import argparse
import numpy as np
import pandas as pd
from dtsr.config import Config
from dtsr.signif import bootstrap

def scale(a, b):
    df = np.stack([np.array(a), np.array(b)], axis=1)
    scaling_factor = df.std()
    return a/scaling_factor, b/scaling_factor

def model2name(s):
    return {
        'LMnoS': 'LMEnoS',
        'LMoptS': 'LMEoptS',
        'LMfullS': 'LMEfullS'
    }.get(s, s)

def name2model(s):
    return {
        'LMEnoS': 'LMnoS',
        'LMEoptS': 'LMoptS',
        'LMEfullS': 'LMfullS'
    }.get(s, s)

if __name__ == '__main__':

    argparser = argparse.ArgumentParser('''
        Computes pairwise significance of error differences between DTSR models and competitors across tasks.
        Assumes models are named using the template <MODELNAME>_<TASKNAME>, where <TASKNAME> is
        shared between models that should be compared. For example, if the config file contains
        4 models --- DTSR_TASK1, DTSR_TASK2, COMPETITOR_TASK1, and COMPETITOR_TASK2 --- the script
        will perform 2 comparisons: DTSR_TASK1 vs COMPETITOR_TASK1 and DTSR_TASK2 vs. COMPETITOR_TASK2.
    ''')
    argparser.add_argument('-t', '--tasks', nargs='*', default=[], help='List of paths to config files.')
    argparser.add_argument('-p', '--partition', type=str, default='dev', help='Name of partition to use (one of "train", "dev", "test")')
    args, unknown = argparser.parse_known_args()

    losses = {}
    select = {}

    systems = set()
    for task in args.tasks:
        p = Config(task + '.ini')
        models = p.model_list[:]
        systems = systems.union(set([model2name(x.split('_')[0].strip()) for x in models]))
        for m in models:
            if os.path.exists(p.logdir + '/' + m + '/%s_losses_%s.txt'%(p.loss, args.partition)):
                sys_name, trial_name = m.split('_')
                sys_name = model2name(sys_name)
                if sys_name.startswith('LME') and trial_name == 'noRE':
                    l = pd.read_csv(p.logdir + '/' + m + '/%s_losses_%s.txt'%(p.loss, args.partition), sep=' ', header=None, skipinitialspace=True)
                else:
                    l = pd.read_csv(p.logdir + '/' + name2model(m) + '/%s_losses_%s.txt'%(p.loss, args.partition), sep=' ', header=None, skipinitialspace=True)
                s = np.isfinite(np.array(l))
                if sys_name not in losses:
                    losses[sys_name] = {}
                if sys_name not in select:
                    select[sys_name] = {}
                if task not in losses[sys_name]:
                    losses[sys_name][task] = {}
                if task not in select[sys_name]:
                    select[sys_name][task] = {}
                if not trial_name in losses[sys_name][task]:
                    losses[sys_name][task][trial_name] = l
                if not trial_name in select[sys_name][task]:
                    select[sys_name][task][model2name(trial_name)] = s

    systems = sorted(list(systems))
    dtsr_models = [x for x in systems if x.startswith('DTSR')]
    for m1 in dtsr_models:
        competitors = [x for x in systems if not x == m1]
        for m2 in competitors:
            m1_scaled_losses = []
            m2_scaled_losses = []
            for task in args.tasks:
                for trial in losses[m2][task]:
                    s = np.logical_and(select[m1][task][trial], select[m2][task][trial])
                    a, b = scale(losses[m1][task][trial][s], losses[m2][task][trial][s])
                    m1_scaled_losses.append(a)
                    m2_scaled_losses.append(b)
            m1_scaled_losses = np.concatenate(m1_scaled_losses, axis=0)
            m2_scaled_losses = np.concatenate(m2_scaled_losses, axis=0)
            p_value, base_diff = bootstrap(m1_scaled_losses, m2_scaled_losses, n_iter=10000)

            print('='*50)
            print('Model comparison: %s vs %s' %(m1, m2))
            print('Partition: %s' %args.partition)
            print('Number of observations: %d' %len(m1_scaled_losses))
            print('Loss difference: %.4f' %base_diff)
            print('p: %.4e' %p_value)
            print('='*50)
            print()
            print()
