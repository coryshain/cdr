import sys
import os
import pandas as pd

import argparse

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Collate significance testing results into a table (by default, LaTeX).
    ''')
    argparser.add_argument('paths', nargs='+', help='Path(s) to directories containing signif testing results (*.txt files)')
    argparser.add_argument('-D', '--dataset_order', nargs='+', help='Order in which to report datasets (left to right). Use only the directory basename.')
    argparser.add_argument('-r', '--response_order', nargs='+', help='Order in which to report responses (left to right).')
    argparser.add_argument('-p', '--partitions', nargs='+', help='Partition(s) in order over which to report signif (left to right). Defaults to all available partitions.')
    argparser.add_argument('-P', '--positive_only', action='store_true', help='Only report positive improvements.')
    argparser.add_argument('-c', '--csv', action='store_true', help='Output as csv.')
    args = argparser.parse_args()

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
            result['comparison'] = '%s vs %s' % (b, a)
            result['response'] = response
            result['partition'] = partition
            with open(os.path.join(dir_path, result_path), 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('Difference:'):
                        result['diff'] = int(round(-float(line[12:])))
                    elif line.startswith('p:'):
                        result['p'] = float(line[12:].replace('*', '')) 
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

    if args.dataset_order:
        dataset_basenames = [os.path.basename(os.path.normpath(x)) for x in datasets]
        _datasets = []
        for x in args.dataset_order:
            for i, y in enumerate(dataset_basenames):
                if x == y:
                    _datasets.append(datasets[i])
                    break
        datasets = _datasets
    if args.response_order:
        for dataset in responses:
            ord = responses[dataset]
            _ord = []
            for x in args.response_order:
                if x in ord:
                    _ord.append(x)
            responses[dataset] = _ord
    if args.partitions:
        _partitions = []
        for x in args.partitions:
            if x in partitions:
                _partitions.append(x)
        partitions = _partitions                

    rows = []
    for comparison in comparisons:
        row = [comparison]
        res1 = all_results.get(comparison)
        for dataset in datasets:
            res2 = res1.get(dataset, {})
            for response in responses[dataset]:
                res3 = res2.get(response, {})
                for partition in partitions:
                    res4 = res3.get(partition, {})
                    diff = str(res4.get('diff', '---'))
                    p = res4.get('p', '---')
                    if diff == '---' or res4['diff'] < 0:
                        p = '---'
                    if p != '---':
                        p = '%0.4f' % p
                    
                    row.append(diff)
                    row.append(p)
        rows.append(row)

    if args.csv:
        cols = ['comparison'] + ['%s.%s.%s.%s' % (dataset, response, partition, val) for dataset in datasets for response in responses[dataset] for partition in partitions for val in ('LLDelta', 'p')]
        out = pd.DataFrame(rows, columns=cols)
        out.to_csv(sys.stdout, index=False)
    else:
        print('\\begin{table}')
        print('  \\begin{tabular}{l|%s}' % '|'.join(['cc' * len(partitions) for dataset in datasets for response in responses[dataset]]))
        print('    & %s\\\\' % ' & '.join(['\\multicolumn{%d}{c}{%s}' % (2 * len(partitions) * len(responses[dataset]), dataset) for dataset in datasets]))
        print('    & %s\\\\' % ' & '.join(['\\multcolumns{%d}{c}{%s}' % (2 * len(partitions), response) for dataset in datasets for response in responses[dataset]]))
        if len(partitions) > 1:
            print('    & %s\\\\' % ' & '.join(['\\multcolumns{2}{c}{%s}' % partition for dataset in datasets for response in responses[dataset] for partition in partitions]))
        print('    Comparison & %s\\\\' % ' & '.join(['$\\Delta$LL & $p$'] * (len(datasets) * len(responses) * len(partitions))))
        for row in rows:
            print('    ' + ' & '.join(row) + '\\\\')
        print('  \\end{tabular}')
        print('\\end{table}')
