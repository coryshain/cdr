import sys
import argparse
import pickle
import numpy as np
import pandas as pd
from dtsr.baselines import anova
from dtsr.config import Config
from dtsr.signif import bootstrap

if __name__ == '__main__':

    argparser = argparse.ArgumentParser('''
        Pairwise likelihood ratio test between LMER models fitted to DTSR-convolved data.
        Facilitates 2-step analysis in which DTSR learns the transform and LME/LRT performs the hypothesis test.
    ''')
    argparser.add_argument('config_path', help='Path to configuration (*.ini) file')
    argparser.add_argument('-m', '--models', nargs='*', default=[], help='Path to configuration (*.ini) file')
    argparser.add_argument('-a', '--ablation', action='store_true', help='Only compare models within an ablation set (those defined using the "ablate" param in the config file)')
    argparser.add_argument('-p', '--partition', type=str, default='dev', help='Name of partition to use (one of "train", "dev", "test")')
    args, unknown = argparser.parse_known_args()

    p = Config(args.config_path)
    if len(args.models) > 0:
        models = args.models
    else:
        models = p.model_list[:]

    sys.stderr.write('\n')
    dtsr_models = [x for x in models if x.startswith('DTSR')]

    if args.ablation:
        comparison_sets = {}
        for model_name in dtsr_models:
            model_basename = model_name.split('!')[0]
            if model_basename not in comparison_sets:
                comparison_sets[model_basename] = []
            comparison_sets[model_basename].append(model_name)
        for model_name in p.model_list:
            model_basename = model_name.split('!')[0]
            if model_basename in comparison_sets and model_name not in comparison_sets[model_basename]:
                comparison_sets[model_basename].append(model_name)
    else:
        comparison_sets = {
            None: dtsr_models
        }

    for s in comparison_sets:
        model_set = comparison_sets[s]
        if len(model_set) > 1:
            if s is not None:
                sys.stderr.write('Comparing models within ablation set "%s"...\n' %s)
            for i in range(len(model_set)):
                model_name_1 = model_set[i]
                lme_path_1 = p.outdir + '/' + model_name_1 + '/lmer_%s.obj' % args.partition
                with open(lme_path_1, 'rb') as m_file:
                    lme1 = pickle.load(m_file)

                for j in range(i+1, len(model_set)):
                    model_name_2 = model_set[j]
                    lme_path_2 = p.outdir + '/' + model_name_2 + '/lmer_%s.obj' % args.partition
                    with open(lme_path_2, 'rb') as m_file:
                        lme2 = pickle.load(m_file)
                    anova_summary = anova(lme1, lme2)
                    name = '%s_v_%s' %(model_name_1, model_name_2)
                    with open(p.outdir + '/' + name + '_2stepLRT_' + args.partition + '.txt', 'w') as f:
                        f.write('='*50 + '\n')
                        f.write('Model comparison: %s vs %s\n' % (model_name_1, model_name_2))
                        f.write('Partition: %s\n\n' %args.partition)
                        f.write(anova_summary)
