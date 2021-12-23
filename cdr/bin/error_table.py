import sys
import os
import argparse
from cdr.config import Config
from cdr.util import sn

def new_row(system, results, tasks):
    s = system
    out = s
    for t in tasks:
        if t in results and s in results[t]:
            if 'train' in results[t][s]:
                train = '%.4f' % results[t][s]['train']['loss']
                if len(train.split('.')[0]) > 3:
                    train = '%d' % round(float(train))
                train = str(train)
                if not results[t][s]['train']['converged']:
                    train += '\\textsuperscript{\\textdagger}'
            else:
                train = '---'
            if 'dev' in results[t][s]:
                dev = '%.4f' % results[t][s]['dev']['loss']
                if len(dev.split('.')[0]) > 3:
                    dev = '%d' % round(float(dev))
                dev = str(dev)
                if not results[t][s]['dev']['converged']:
                    dev += '\\textsuperscript{\\textdagger}'
            else:
                dev = '---'
            if 'test' in results[t][s]:
                test = '%.4f' % results[t][s]['test']['loss']
                if len(test.split('.')[0]) > 3:
                    test = '%d' % round(float(test))
                test = str(test)
                if not results[t][s]['test']['converged']:
                    test += '\\textsuperscript{\\textdagger}'
            else:
                test = '---'
            out += ' & ' + ' & '.join([train, dev, test])
        else:
            out += ' & ' + ' & '.join(['---'] * 3)
    out += '\\\\\n'
    return out
    

def results_to_table(results, systems, baselines=None, indent=4):
    tasks = results.keys()
    out = ''
    out += '\\begin{table}\n'
    out += ' ' * indent + '\\begin{tabular}{r|%s}\n' % ('|'.join(['ccc'] * len(results)))
   
    out += ' ' * (indent * 2) + ' & '.join(['Model'] + ['\\multicolumn{3}{|c}{%s}' % t for t in tasks]) + '\\\\\n'
    out += ' ' * (indent * 2) + '& ' + ' & '.join(['Train', 'Dev', 'Test'] * len(tasks)) + '\\\\\n'
    out += ' ' * (indent * 2) + '\\hline\n'

    if baselines is None:
        baselines = []

    for b in baselines:
        out += ' ' * (indent * 2) + new_row(b, results, tasks)
    if len(baselines) > 0:
        out += ' ' * (indent * 2) + '\\hline\n'
    for s in systems:
        out += ' ' * (indent * 2) + new_row(s, results, tasks)           
 
    out += ' ' * indent + '\\end{tabular}\n'
    out += '\\end{table}\n'

    return out


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Generate a LaTeX table summarizing results from CDR vs. baseline models in some output directory.
    Tasks are defined as sets of experiments within the same config file (because they are constrained to use the same data).
    ''')
    argparser.add_argument('config_paths', nargs='+', help='Path(s) to config files defining models to compare.')
    argparser.add_argument('-r', '--response', default=None, help='Name of response to evaluate.')
    argparser.add_argument('-t', '--task_names', nargs='+', default=None, help='Task names to use (should be in 1-1 alignment with ``config_paths``). If not provided, names will be inferred from config paths.')
    argparser.add_argument('-b', '--baselines',  nargs='+', default=None, help='Models to treat as baselines.')
    argparser.add_argument('-B', '--baseline_names',  nargs='+', default=None, help='Names of baselines (should be in 1-1 alignment with ``baselines``. If not provided, names will be inferred from baselines.')
    argparser.add_argument('-s', '--systems',  nargs='+', default=None, help='Models to treat as (non-baseline) systems.')
    argparser.add_argument('-S', '--system_names',  nargs='+', default=None, help='Names of systems (should be in 1-1 alignment with ``systems``. If not provided, names will be inferred from systems.')
    args = argparser.parse_args()

    response = args.response
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

    systems = args.systems[:]
    if args.system_names is None:
        system_names = systems[:]
    else:
        system_names = args.system_names[:]
    assert len(systems) == len(system_names)

    results = {}
    for i, path in enumerate(args.config_paths):
        p = Config(path)
        if systems is None:
            _systems = [x for x in p.model_list if x not in baselines]
        for j, b in enumerate(baselines):
            if b in p.model_list:
                b_path = b.replace(':', '+')
                for partition in ['train', 'dev', 'test']:
                    for path in os.listdir(p.outdir + '/' + b_path):
                        if (not response or response in path) and path.startswith('eval') and path.endswith('%s.txt' % partition):
                            eval_path = p.outdir + '/' + b_path + '/' + path
                            _response = path.split('_')[1]
                            _task_name = task_names[i] + ' ' + _response
                            converged = True
                            if b.startswith('LME'):
                                converged = False
                            with open(eval_path, 'r') as f:
                                line = f.readline()
                                while line:
                                    if line.strip().startswith('MSE'):
                                        val = float(line.strip().split()[1])
                                        if _task_name not in results:
                                            results[_task_name] = {}
                                        if baseline_names[j] not in results[_task_name]:
                                            results[_task_name][baseline_names[j]] = {}
                                        if partition not in results[_task_name][baseline_names[j]]:
                                            results[_task_name][baseline_names[j]][partition] = {'loss': val, 'converged': converged}
                                    if line.strip() == 'No convergence warnings.':
                                        converged = True
                                    line = f.readline()
        for j, s in enumerate(_systems):
            if s in p.model_list:
                s_path = s.replace(':', ':')
                for partition in ['train', 'dev', 'test']:
                    for path in os.listdir(p.outdir + '/' + s_path):
                        if (not response or response in path) and path.startswith('eval') and path.endswith('%s.txt' % partition):
                            eval_path = p.outdir + '/' + s_path + '/' + path
                            _response = path.split('_')[1]
                            _task_name = task_names[i] + ' ' + _response
                            converged = True
                            if s.startswith('LME'):
                                converged = False
                            with open(eval_path, 'r') as f:
                                line = f.readline()
                                while line:
                                    if line.strip().startswith('MSE'):
                                        val = float(line.strip().split()[1])
                                        if _task_name not in results:
                                            results[_task_name] = {}
                                        if system_names[j] not in results[_task_name]:
                                            results[_task_name][system_names[j]] = {}
                                        if partition not in results[_task_name][system_names[j]]:
                                            results[_task_name][system_names[j]][partition] = {'loss': val, 'converged': converged}
                                    if line.strip() == 'No convergence warnings.':
                                        converged = True
                                    line = f.readline()


    sys.stdout.write(results_to_table(results, system_names, baselines=baseline_names))

