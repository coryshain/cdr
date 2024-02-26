import sys
import os
import re
import argparse
import numpy as np
import pandas as pd

from cdr.config import Config
from cdr.util import sn


ENSEMBLE = re.compile('([^ ]+)\.m\d+')
CROSSVAL = re.compile('([^ ]+)\.CV([^.~]+)~([^.~]+)(\.m(\d+))?')


def new_row(system, results, tasks, base_partitions=None):
    if base_partitions is None:
        base_partitions = ['train', 'dev', 'test']
    s = system
    out = s
    for t in tasks:
        if t in results and s in results[t]:
            row = []
            for partition in base_partitions:
                if partition in results[t][s]:
                    val = '%.4f' % results[t][s][partition]['loss']
                    if len(val.split('.')[0]) > 3:
                        val = '%d' % round(float(val))
                    val = str(val)
                    if not results[t][s][partition]['converged']:
                        val += '\\textsuperscript{\\textdagger}'
                else:
                    val = '---'
                row.append(val)
            out += ' & ' + ' & '.join(row)
        else:
            out += ' & ' + ' & '.join(['---'] * len(base_partitions))
    out += '\\\\\n'
    return out
    

def results_to_table(results, systems, baselines=None, indent=4, base_partitions=None):
    if base_partitions is None:
        base_partitions = ['train', 'dev', 'test']
    base_partition_names = [x[0].upper() + x[1:] for x in base_partitions]
    tasks = results.keys()

    out = ''
    out += '\\begin{table}\n'
    out += ' ' * indent + '\\begin{tabular}{r|%s}\n' % ('|'.join(['ccc'] * len(results)))
   
    out += ' ' * (indent * 2) + ' & '.join(['Model'] + ['\\multicolumn{3}{|c}{%s}' % t for t in tasks]) + '\\\\\n'
    out += ' ' * (indent * 2) + '& ' + ' & '.join(base_partition_names * len(tasks)) + '\\\\\n'
    out += ' ' * (indent * 2) + '\\hline\n'

    if baselines is None:
        baselines = []

    for b in baselines:
        out += ' ' * (indent * 2) + new_row(b, results, tasks, base_partitions=base_partitions)
    if len(baselines) > 0:
        out += ' ' * (indent * 2) + '\\hline\n'
    for s in systems:
        out += ' ' * (indent * 2) + new_row(s, results, tasks, base_partitions=base_partitions)
 
    out += ' ' * indent + '\\end{tabular}\n'
    out += '\\end{table}\n'

    return out

def results_to_csv(results, systems, baselines=None, indent=4, base_partitions=None):
    if base_partitions is None:
        base_partitions = ['train', 'dev', 'test']
    tasks = results.keys()
   
    cols = ['model'] + ['%s.%s' % (t.replace(' ', '.'), p) for t in tasks for p in ('train', 'dev', 'test')]
    out = []

    if baselines is None:
        baselines = []

    for b in baselines:
        out.append(tuple(new_row(b, results, tasks, base_partitions=base_partitions)[:-3].split(' & ')))
    for s in systems:
        out.append(tuple(new_row(s, results, tasks, base_partitions=base_partitions)[:-3].split(' & ')))

    out = pd.DataFrame(out, columns=cols)

    return out.to_csv(None, index=False)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Generate a table (LaTeX by default) summarizing results from CDR vs. baseline models in some output directory.
    Tasks are defined as sets of experiments within the same config file (because they are constrained to use the same data).
    ''')
    argparser.add_argument('config_paths', nargs='+', help='Path(s) to config files defining models to compare.')
    argparser.add_argument('-r', '--response', default=None, help='Name of response to evaluate.')
    argparser.add_argument('-m', '--metric', default='loglik', help='Metric to report. One of ``["err", "loglik", "pve", "iter", "time"]``.')
    argparser.add_argument('-t', '--task_names', nargs='+', default=None, help='Task names to use (should be in 1-1 alignment with ``config_paths``). If not provided, names will be inferred from config paths.')
    argparser.add_argument('-b', '--baselines',  nargs='+', default=None, help='Models to treat as baselines.')
    argparser.add_argument('-B', '--baseline_names',  nargs='+', default=None, help='Names of baselines (should be in 1-1 alignment with ``baselines``. If not provided, names will be inferred from baselines.')
    argparser.add_argument('-s', '--systems',  nargs='+', default=None, help='Models to treat as (non-baseline) systems.')
    argparser.add_argument('-S', '--system_names',  nargs='+', default=None, help='Names of systems (should be in 1-1 alignment with ``systems``. If not provided, names will be inferred from systems.')
    argparser.add_argument('-p', '--partitions',  nargs='+', default=None, help='Names of partitions to evaluate. If not provided, defaults to ``"train"``, ``"dev"``, ``"test"``.')
    argparser.add_argument('-a', '--agg', type=str, default='median', help='Aggregation function to use over ensembles. E.g., ``"mean"``, ``"median"``, ``"min"``, ``"max"``.')
    argparser.add_argument('-C', '--collapse', action='store_true', help='Collapse (sum across) all response variables for a given model.')
    argparser.add_argument('-c', '--csv', action='store_true', help='Output to CSV.')
    args = argparser.parse_args()

    response = args.response

    if args.metric.lower() in ['err', 'mse', 'loss']:
        metric = 'MSE'
    elif args.metric.lower() in ['loglik', 'll', 'likelihood']:
        metric = 'Loglik'
    elif args.metric.lower() in ['pve', 'r', 'r2']:
        metric = '% var expl'
    elif args.metric.lower() in ['n', 'iter', 'niter', 'n_iter']:
        metric = 'Training iterations completed'
    elif args.metric.lower() in ['t', 'time', 'walltime', 'wall_time']:
        metric = 'Training wall time'
    else:
        raise ValueError('Unrecognized metric: %s.' % args.metric)

    if args.collapse:
        task_names = ['Response']
    else:
        if args.task_names is None:
            task_names = [os.path.splitext(os.path.basename(p))[0] for p in args.config_paths]
        else:
            task_names = args.task_names[:]
        assert len(args.config_paths) == len(task_names)

    if args.baselines is None:
        baselines = []
    else:
        baselines = args.baselines[:]
    if args.baseline_names is None:
        baseline_names = baselines[:]
    else:
        baseline_names = args.baseline_names[:]
    assert len(baselines) == len(baseline_names)

    systems = args.systems
    system_names = args.system_names
    base_partitions = args.partitions
    if base_partitions is None:
        base_partitions = ['CVtrain', 'train', 'CVdev', 'CVtest', 'dev', 'test']
    partitions_found = set()

    results = {}
    system_names_all = []
    for i, path in enumerate(args.config_paths):
        p = Config(path)
        if systems is None:
            _systems = [x for x in p.model_names + p.ensemble_names + p.crossval_family_names if x not in baselines]
        if system_names is None:
            _system_names = _systems[:]
        else:
            _system_names = args.system_names[:]
        for s in _system_names:
            if s not in system_names_all:
                system_names_all.append(s)
        assert len(_systems) == len(_system_names)
        for j, b in enumerate(baselines):
            if b in p.model_names:
                b_path = b.replace(':', '+')
                for partition in base_partitions:
                    if os.path.exists(p.outdir + '/' + b_path):
                        for path in os.listdir(p.outdir + '/' + b_path):
                            if (not response or response in path) and path.startswith('eval') and path.endswith('_%s.txt' % partition):
                                eval_path = p.outdir + '/' + b_path + '/' + path
                                _response = path.split('_')[1]
                                if args.collapse:
                                    _task_name = task_names[0]
                                else:
                                    _task_name = task_names[i] + ' ' + _response
                                converged = True
                                if b.startswith('LME'):
                                    converged = False
                                with open(eval_path, 'r') as f:
                                    line = f.readline()
                                    while line:
                                        if line.strip().startswith(metric):
                                            val = float(line.strip().split()[-1].replace('%', ''))
                                            if _task_name not in results:
                                                results[_task_name] = {}
                                            if baseline_names[j] not in results[_task_name]:
                                                results[_task_name][baseline_names[j]] = {}
                                            if partition not in results[_task_name][baseline_names[j]]:
                                                results[_task_name][baseline_names[j]][partition] = {}
                                            if 'loss' not in results[_task_name][baseline_names[j]][partition]:
                                                results[_task_name][baseline_names[j]][partition]['loss'] = val
                                            else:
                                                results[_task_name][baseline_names[j]][partition]['loss'] += val
                                            if 'converged' not in results[_task_name][baseline_names[j]][partition]:
                                                results[_task_name][baseline_names[j]][partition]['converged'] = converged
                                            else:
                                                results[_task_name][baseline_names[j]][partition]['converged'] += converged
                                            partitions_found.add(partition)
                                        if line.strip() == 'No convergence warnings.':
                                            converged = True
                                        line = f.readline()
        for j, s in enumerate(_systems):
            if s in p.model_names:
                s_path = s.replace(':', ':')
                for partition in base_partitions:
                    if os.path.exists(p.outdir + '/' + s_path):
                        for path in os.listdir(p.outdir + '/' + s_path):
                            if (not response or response in path) and path.startswith('eval') and path.endswith('_%s.txt' % partition):
                                eval_path = p.outdir + '/' + s_path + '/' + path
                                _response = path.split('_')[1]
                                if args.collapse:
                                    _task_name = task_names[0]
                                else:
                                    _task_name = task_names[i] + ' ' + _response
                                converged = True
                                if s.startswith('LME'):
                                    converged = False
                                with open(eval_path, 'r') as f:
                                    line = f.readline()
                                    while line:
                                        if line.strip().startswith(metric):
                                            val = float(line.strip().split()[-1].replace('%', '').replace('s', ''))
                                            if _task_name not in results:
                                                results[_task_name] = {}
                                            if _system_names[j] not in results[_task_name]:
                                                results[_task_name][_system_names[j]] = {}
                                            if partition not in results[_task_name][_system_names[j]]:
                                                results[_task_name][_system_names[j]][partition] = {}
                                            if 'loss' not in results[_task_name][_system_names[j]][partition]:
                                                results[_task_name][_system_names[j]][partition]['loss'] = val
                                            else:
                                                results[_task_name][_system_names[j]][partition]['loss'] += val
                                            if 'converged' not in results[_task_name][_system_names[j]][partition]:
                                                results[_task_name][_system_names[j]][partition]['converged'] = converged
                                            else:
                                                results[_task_name][_system_names[j]][partition]['converged'] += converged
                                            partitions_found.add(partition)
                                        if line.strip() == 'No convergence warnings.':
                                            converged = True
                                        line = f.readline()

        base_partitions = list(filter(lambda x: x in partitions_found, base_partitions))

        # Aggregate over any ensembles
        agg_fn = getattr(np, args.agg)
        for j, system_name in enumerate(system_names_all):
            for task_name in results:
                if system_name not in results[task_name]:
                    submodels = []
                    for x in results[task_name]:
                        re_match = ENSEMBLE.match(x)
                        if re_match and re_match.group(1) == system_name:
                            submodels.append(x)
                    n_submodels = len(submodels)
                    for k, submodel in enumerate(submodels):
                        if submodel in results[task_name]:
                            for partition in results[task_name][submodel]:
                                if system_name not in results[task_name]:
                                    results[task_name][system_name] = {}
                                if partition not in results[task_name][system_name]:
                                    results[task_name][system_name][partition] = {'loss': [], 'converged': []}
                                results[task_name][system_name][partition]['loss'].append(results[task_name][submodel][partition]['loss'])
                                results[task_name][system_name][partition]['converged'].append(results[task_name][submodel][partition]['converged'])
                    if submodels:
                        for partition in results[task_name][system_name]:
                            results[task_name][system_name][partition]['loss'] = agg_fn(
                                results[task_name][system_name][partition]['loss']
                            )
                            results[task_name][system_name][partition]['converged'] = np.mean(
                                results[task_name][system_name][partition]['converged']
                            )

        # Aggregate over any cross-validation
        agg_fn = getattr(np, args.agg)
        for j, system_name in enumerate(system_names_all):
            for task_name in results:
                if system_name not in results[task_name]:
                    submodels = {}
                    for x in results[task_name]:
                        re_match = CROSSVAL.match(x)
                        if re_match and re_match.group(1) == system_name:
                            fold = re_match.group(3)
                            ensemble_id = re_match.group(5)
                            if ensemble_id not in submodels:
                                submodels[ensemble_id] = []
                            submodels[ensemble_id].append(x)
                    _results = {}
                    # Collect stats by ensemble_id
                    for k, ensemble_id in enumerate(submodels):
                        for submodel in submodels[ensemble_id]:
                            if submodel in results[task_name]:
                                for partition in results[task_name][submodel]:
                                    if task_name not in _results:
                                        _results[task_name] = {}
                                    if system_name not in _results[task_name]:
                                        _results[task_name][system_name] = {}
                                    if partition not in _results[task_name][system_name]:
                                        _results[task_name][system_name][partition] = {}
                                    if ensemble_id not in _results[task_name][system_name][partition]:
                                        _results[task_name][system_name][partition][ensemble_id] = {'loss': [], 'converged': []}
                                    _results[task_name][system_name][partition][ensemble_id]['loss'].append(results[task_name][submodel][partition]['loss'])
                                    _results[task_name][system_name][partition][ensemble_id]['converged'].append(results[task_name][submodel][partition]['converged'])
                    # Aggregate within ensemble_id
                    for task_name in _results:
                        for system_name in _results[task_name]:
                            for partition in _results[task_name][system_name]:
                                for ensemble_id in _results[task_name][system_name][partition]:
                                    loss = _results[task_name][system_name][partition][ensemble_id]['loss']
                                    n_loss = len(loss)
                                    loss = sum(loss)
                                    if metric != 'Loglik':
                                        loss /= n_loss
                                    converged = sum(_results[task_name][system_name][partition][ensemble_id]['converged']) / n_loss
                                    if system_name not in results[task_name]:
                                        results[task_name][system_name] = {}
                                    if partition not in results[task_name][system_name]:
                                        results[task_name][system_name][partition] = {'loss': [], 'converged': []}
                                    results[task_name][system_name][partition]['loss'].append(loss)
                                    results[task_name][system_name][partition]['converged'].append(converged)
                    if submodels:
                        for partition in results[task_name][system_name]:
                            results[task_name][system_name][partition]['loss'] = agg_fn(
                                results[task_name][system_name][partition]['loss']
                            )
                            results[task_name][system_name][partition]['converged'] = np.mean(
                                results[task_name][system_name][partition]['converged']
                            )

    if args.csv:
        out_fn = results_to_csv
    else:
        out_fn = results_to_table

    sys.stdout.write(
        out_fn(
            results,
            system_names_all,
            baselines=baseline_names,
            base_partitions=base_partitions
        )
    )

