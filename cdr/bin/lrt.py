import sys
import argparse
import pickle

from cdr.baselines import anova
from cdr.config import Config
from cdr.util import nested, filter_models, get_partition_list, stderr

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
            Performs pairwise likelihood ratio test for significance of differences in prediction quality between LME models fitted to data convolved using CDR ("2-step" hypothesis test).
            Can be used for in-sample and out-of-sample evaluation.
            Can be used either to compare arbitrary sets of LME models or (using the "-a" flag) to perform hypothesis testing between CDR models within one or more ablation sets. 
        ''')
    argparser.add_argument('config_path', help='Path to configuration (*.ini) file')
    argparser.add_argument('-m', '--models', nargs='*', default=[], help='List of model names for which to compute LRT tests. Regex permitted. If unspecified, uses all CDR models.')
    argparser.add_argument('-z', '--zscore', action='store_true', help='Compare models fitted with Z-transformed predictors.')
    argparser.add_argument('-a', '--ablation', action='store_true', help='Only compare models within an ablation set (those defined using the "ablate" param in the config file)')
    argparser.add_argument('-A', '--ablation_components', type=str, nargs='*', help='Names of variables to consider in ablative tests. Useful for excluding some ablated models from consideration')
    argparser.add_argument('-p', '--partition', type=str, default='train', help='Name of partition to use (one of "train", "dev", "test")')
    args = argparser.parse_args()

    p = Config(args.config_path)

    models = filter_models(p.model_names, args.models, cdr_only=True)

    partitions = get_partition_list(args.partition)
    partition_str = '-'.join(partitions)

    stderr('\n')

    if args.ablation:
        comparison_sets = {}
        for model_name in models:
            model_basename = model_name.split('!')[0]
            if model_basename not in comparison_sets:
                comparison_sets[model_basename] = []
            comparison_sets[model_basename].append(model_name)
        for model_name in p.model_names:
            model_basename = model_name.split('!')[0]
            if model_basename in comparison_sets and model_name not in comparison_sets[model_basename]:
                if len(args.ablation_components) > 0:
                    components = model_name.split('!')[1:]
                    hit = True
                    for c in components:
                        if c not in args.ablation_components:
                            hit = False
                    if hit:
                        comparison_sets[model_basename].append(model_name)
                else:
                    comparison_sets[model_basename].append(model_name)
    else:
        comparison_sets = {
            None: models
        }

    for s in comparison_sets:
        model_set = comparison_sets[s]
        if len(model_set) > 1:
            if s is not None:
                stderr('Comparing models within ablation set "%s"...\n' %s)
            for i in range(len(model_set)):
                model_name_1 = model_set[i]
                model_name_1_path = model_name_1.replace(':', '+')
                lme_path_1 = p.outdir + '/' + model_name_1_path + '/lm_%s.obj' % ('z_%s' % partition_str if args.zscore else partition_str)
                with open(lme_path_1, 'rb') as m_file:
                    lme1 = pickle.load(m_file)

                for j in range(i+1, len(model_set)):
                    model_name_2 = model_set[j]
                    model_name_2_path = model_name_2.replace(':', '+')
                    if nested(model_name_1, model_name_2) or not args.ablation:
                        lme_path_2 = p.outdir + '/' + model_name_2_path + '/lm_%s.obj' % ('z_%s' % partition_str if args.zscore else partition_str)
                        with open(lme_path_2, 'rb') as m_file:
                            lme2 = pickle.load(m_file)
                        anova_summary = str(anova(lme1.m, lme2.m))
                        name = '%s_v_%s' %(model_name_1_path, model_name_2_path)
                        out_path = p.outdir + '/' + name + '_2stepLRT_%s.txt' % ('z_%s' % partition_str if args.zscore else partition_str)
                        with open(out_path, 'w') as f:
                            stderr('Saving output to %s...\n' %out_path)
                            summary = '='*50 + '\n'
                            summary += 'Model comparison: %s vs %s\n' % (model_name_1, model_name_2)
                            summary += 'Partition: %s\n\n' % partition_str
                            if not lme1.converged():
                                summary += 'Model %s had the following convergence warnings:\n' %model_name_1
                                summary += '%s\n\n' %lme1.convergence_warnings()
                            if not lme2.converged():
                                summary += 'Model %s had the following convergence warnings:\n' %model_name_2
                                summary += '%s\n\n' %lme2.convergence_warnings()
                            summary += anova_summary + '\n'

                            f.write(summary)
                            sys.stdout.write(summary)
