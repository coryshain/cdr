import sys
import os
import re
import numpy as np
import pandas as pd
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
from statsmodels.stats.multitest import fdrcorrection
import argparse

delimiters = [
    '_v_',
    '_vs_',
    ' vs ',
    ' vs. ',
    ' vs.\\ ',
]

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Collate significance testing results into a table (by default, LaTeX).
    ''')
    argparser.add_argument('paths', nargs='+', help='Path(s) to directories containing signif testing results (*.txt files)')
    argparser.add_argument('-s', '--settings_path', default=None, help='Path to YAML file with settings controlling the output table.')
    argparser.add_argument('-c', '--csv', action='store_true', help='Output as csv.')
    args = argparser.parse_args()

    if args.settings_path:
        with open(args.settings_path, 'r') as f:
            settings = yaml.load(f, Loader=Loader)
    else:
        settings = {}
    dataset_order = settings.get('dataset_order', [])
    response_order = settings.get('response_order', [])
    partitions = settings.get('partitions', [])
    comparison_name_map = settings.get('comparison_name_map', {})
    comparison_name_keys = sorted(list(comparison_name_map.keys()), key=lambda x: len(x), reverse=True)
    def process_name(x):
        for key in comparison_name_keys:
            x = re.sub(key, comparison_name_map[key], x)
        return x
    
    positive_only = settings.get('positive_only', [])
    if 'all' in positive_only:
        all_positive = True
    else:
        all_positive = False
    bold_signif = settings.get('bold_signif', True)
    color_by_sign = settings.get('color_by_sign', False)
    include_missing_rows = settings.get('include_missing_rows', False)
    group2comparison = settings.get('groups', {})
    comparison2group = {}
    for group in group2comparison:
        for comparison in group2comparison[group]:
            if comparison not in comparison2group:
                 comparison2group[comparison] = []
            comparison2group[comparison].append(group)
    group_order = settings.get('group_order', None)
    if group_order is None:
        group_order = sorted(list(group2comparison.keys()))    
    fdr_by_group = settings.get('fdr_by_group', False)

    results = {}
    for dir_path in args.paths:
        _results = []
        result_paths = [x for x in os.listdir(dir_path) if ('_v_' in x and x.endswith('.txt'))]
        for result_path in result_paths:
            result = {}
            a, b = result_path.split('_v_')
            b, other = b.split('_PT_')
            other = other.split('_')
            response = other[0]
            partition = other[-1][:-4]
            if not partitions or partition in partitions:
                result['comparison'] = '%s_v_%s' % (b, a)
                result['response'] = response
                result['partition'] = partition
                with open(os.path.join(dir_path, result_path), 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith('Difference:'):
                            val = line.strip().split()[1]
                            result['diff'] = int(round(-float(val)))
                        elif line.startswith('p:'):
                            val = line.strip().split()[1]
                            result['p'] = float(val.replace('*', '')) 
                _results.append(result)
            results[dir_path] = _results

    datasets = list(results.keys())
    responses = {}
    comparisons = []
    partitions = []
    all_results = {}

    for dataset in results:
        if dataset not in responses:
            responses[dataset] = []
        for result in results[dataset]:
            response = result['response']
            comparison = result['comparison']
            if comparison not in comparison2group:
                if comparison not in comparison2group:
                    comparison2group[comparison] = []
                comparison2group[comparison].append(None)
                if None not in group2comparison:
                    group2comparison[None] = []
                group2comparison[None].append(comparison)
                if None not in group_order:
                    group_order.append(None)
            partition = result['partition']
            if response not in responses[dataset]:
                responses[dataset].append(response)
            if comparison not in comparisons:
                comparisons.append(comparison)
            if partition not in partitions:
                partitions.append(partition)
            if comparison not in all_results:
                all_results[comparison] = {}
            if dataset not in all_results[comparison]:
                all_results[comparison][dataset] = {}
            if response not in all_results[comparison][dataset]:
                all_results[comparison][dataset][response] = {}
            if partition not in all_results[comparison][dataset][response]:
                all_results[comparison][dataset][response][partition] = {
                    'p': result['p'],
                    'diff': result['diff']
                }

    if dataset_order:
        dataset_basenames = [os.path.basename(os.path.normpath(x)) for x in datasets]
        _datasets = []
        for x in dataset_order:
            for i, y in enumerate(dataset_basenames):
                if x == y:
                    _datasets.append(datasets[i])
                    break
        datasets = _datasets
    if response_order:
        for dataset in responses:
            ord = responses[dataset]
            _ord = []
            for x in response_order:
                if x in ord:
                    _ord.append(x)
            responses[dataset] = _ord
    if partitions:
        _partitions = []
        for x in partitions:
            if x in partitions:
                _partitions.append(x)
        partitions = _partitions                

    if fdr_by_group:
        for group in group2comparison:
            fdr = {}
            for comparison in group2comparison[group]:
                if comparison in all_results:
                    for dataset in all_results[comparison]:
                        for response in all_results[comparison][dataset]:
                            for partition in all_results[comparison][dataset][response]:
                                key = (dataset, response, partition)
                                if key not in fdr:
                                   fdr[key] = {}
                                fdr[key][comparison] = all_results[comparison][dataset][response][partition]['p']
            for key in fdr:
                comparisons = sorted(list(fdr[key].keys()))
                p = [fdr[key][comparison] for comparison in comparisons]
                _, p = fdrcorrection(p, method='negcorr')
                
                for comparison, _p in zip(comparisons, p):
                    dataset, response, parition = key
                    all_results[comparison][dataset][response][partition]['p'] = _p

    rows = []
    if group2comparison:
        comparisons = group2comparison
    else:
        comparisons = {None: comparisons}
        group_order = [None]
    comparisons_found = {}
    for group in group_order:
        for comparison in comparisons[group]:
            if args.csv:
                comparison_name = comparison
            else:
                comparison_name = process_name(comparison)
            row = [comparison_name, group]
            res1 = all_results.get(comparison, None)
            if res1 is None:
                if not include_missing_rows:
                    continue
                for dataset in datasets:
                    for response in responses[dataset]:
                        for partition in partitions:
                            row += ['---', '---']
                if group not in comparisons_found:
                    comparisons_found[group] = []
                comparisons_found[group].append(comparison_name)
            else:
                if group not in comparisons_found:
                    comparisons_found[group] = []
                comparisons_found[group].append(comparison_name)
                _positive_only = all_positive or comparison in positive_only
                if color_by_sign and not _positive_only:
                    for delim in delimiters:
                        if delim in comparison_name:
                            split = comparison_name.split(delim)
                            if len(split) == 2:
                                if args.csv:
                                    _comparison_name = comparison_name
                                else:
                                    a, b = split
                                    a = '{\\color{cyan}%s}' % a
                                    b = '{\\color{magenta}%s}' % b
                                    _comparison_name = delim.join([a, b])
                            else:
                                _comparison_name = comparison_name
                            row[0] = _comparison_name
                            comparisons_found[group][-1] = row[0]
                for dataset in datasets:
                    res2 = res1.get(dataset, {})
                    for response in responses[dataset]:
                        res3 = res2.get(response, {})
                        for partition in partitions:
                            res4 = res3.get(partition, {})
                            diff = res4.get('diff', np.nan)
                            if np.isnan(diff):
                                diff_str = '---'
                            else:
                                diff_str = str(diff)
                            p = res4.get('p', np.nan)
                            if np.isnan(diff) or np.isnan(p) or (_positive_only and diff < 0):
                                p_str = '---'
                            else:
                                p_str = '%0.4f' % p
                                if not args.csv:
                                    if bold_signif and p <= 0.05:
                                        diff_str = '\\textbf{%s}' % diff_str
                                        p_str = '\\textbf{%s}' % p_str
                                    if color_by_sign and not _positive_only:
                                        if diff > 0:
                                            diff_str = '{\\color{cyan}%s}' % diff_str
                                            p_str = '{\\color{cyan}%s}' % p_str
                                        elif diff < 0:
                                            diff_str = '{\\color{magenta}%s}' % diff_str
                                            p_str = '{\\color{magenta}%s}' % p_str
                            row.append(diff_str)
                            row.append(p_str)
            rows.append(row)

    if args.csv:
        cols = ['comparison', 'group'] + \
               ['%s.%s.%s.%s' % (dataset, response, partition, val) for dataset in datasets for response in responses[dataset] for partition in partitions for val in ('LLDelta', 'p')]
        out = pd.DataFrame(rows, columns=cols)
        out.to_csv(sys.stdout, index=False)
    else:
        if len(comparisons) > 1:
            prefix = '    & '
        else:
            prefix = '    '
        print('\\begin{table}')
        print('  \\begin{tabular}{l|%s}' % '|'.join(['cc' * len(partitions) for dataset in datasets for response in responses[dataset]]))
        print(prefix + '& %s\\\\' % ' & '.join(['\\multicolumn{%d}{c}{%s}' % (2 * len(partitions) * len(responses[dataset]), dataset) for dataset in datasets]))
        print(prefix + '& %s\\\\' % ' & '.join(['\\multcolumns{%d}{c}{%s}' % (2 * len(partitions), response) for dataset in datasets for response in responses[dataset]]))
        if len(partitions) > 1:
            print('    & & %s\\\\' % ' & '.join(['\\multcolumns{2}{c}{%s}' % partition for dataset in datasets for response in responses[dataset] for partition in partitions]))
        print('    & Comparison & %s\\\\' % ' & '.join(['$\\Delta$LL & $p$' * (len(responses[dataset]) * len(partitions)) for dataset in datasets]))
        for row in rows:
            comparison = row[0]
            group = row[1]
            group_len = len(comparisons_found[group])
            _prefix = prefix
            if comparison == comparisons_found[group][0]:  # First in group
                print('\n    \hline')
                if len(comparisons) > 1:
                    _prefix = '    \multirow{%s}{*}{\\rotatebox[origin=c]{90}{%s}} & ' % (group_len, group)
            print(_prefix + ' & '.join(row[0:1] + row[2:]) + '\\\\')
        print('  \\end{tabular}')
        print('\\end{table}')
