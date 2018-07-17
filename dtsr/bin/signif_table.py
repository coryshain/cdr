import sys
import re
import os
import numpy as np
import argparse

from dtsr.config import Config

def extract_signif_2step(path):
    signif = np.nan
    converged = True
    with open(path, 'r') as f:
        in_signif_table = 0
        for line in f.readlines():
            l = line.strip()
            if l.endswith('had the following convergence warnings:'):
                converged = False
            if l.endswith('Pr(>Chisq)'):
                in_signif_table = 1
            elif in_signif_table:
                in_signif_table += 1
                if in_signif_table == 3:
                    signif = line.strip().split()[8:]
                    if signif[0] == '<':
                        signif = float(signif[1])
                    else:
                        signif = float(signif[0])
    return signif, converged

def extract_signif_pt(path):
    signif = np.nan
    converged = True
    with open(path, 'r') as f:
        for line in f.readlines():
            l = line.strip()
            if l.startswith('p: '):
                signif = l.split()[1]
                signif = re.match(' *([^*]+)\**', signif).group(1)
                signif = float(signif)
    return signif, converged

def comparison2str(comparison):
    return '_v_'.join([''.join(['!' + x for x in y]) if len(y) > 0 else 'FULL' for y in c])

def extract_comparisons(paths):
    models = []
    comparisons = {}
    comparisons_converged = {}
    for path in paths:
        m = '_'.join(os.path.basename(path).split('_')[:-2]).split('_v_')
        m_split = [x.split('!') for x in m]
        m_base = [x[0] for x in m_split]
        m_ablated = [set(x[1:]) for x in m_split]
        a = 0 if len(m_ablated[0]) < len(m_ablated[1]) else 1
        b = 1 - a
        if m_base[a] == m_base[b] and len(m_ablated[b] - m_ablated[a]) == 1 and len(m_ablated[a] - m_ablated[b]) == 0:
            if m_base[0] not in models:
                models.append(m_base[0])

            if args.mode == '2step':
                signif, converged = extract_signif_2step(path)
            else:
                signif, converged = extract_signif_pt(path)

            comparison = (tuple(sorted(list(m_ablated[a]))), tuple(sorted(list(m_ablated[b]))))

            if comparison not in comparisons:
                comparisons[comparison] = {}

            comparisons[comparison][m_base[0]] = signif
            if comparison not in comparisons_converged:
                comparisons_converged[comparison] = {}
            comparisons_converged[comparison][m_base[0]] = converged

    return models, comparisons, comparisons_converged

# Thanks to Daniel Sparks on StackOverflow for this one (post available at
# http://stackoverflow.com/questions/5084743/how-to-print-pretty-string-output-in-python)
def pretty_table(row_collection, key_list, field_sep=' '):
  return '\n'.join([field_sep.join([str(row[col]).ljust(width)
    for (col, width) in zip(key_list, [max(map(len, column_vector))
      for column_vector in [ [v[k]
        for v in row_collection if k in v]
          for k in key_list ]])])
            for row in row_collection])



if __name__ == '__main__':

    argparser = argparse.ArgumentParser('''
        Generates table of significances of ablative model comparisons
    ''')
    argparser.add_argument('config_path', help='Path to configuration (*.ini) file')
    argparser.add_argument('-M', '--mode', type=str, default='2step', help='Type of significance test to use (one of ["pt", "2step"] for permutation testing or 2-step LME/LRT, respectively).')
    argparser.add_argument('-p', '--partition', type=str, default='dev', help='Name of partition to use (one of "train", "dev", "test")')
    argparser.add_argument('-H', '--human_readable', action='store_true', help='Return table in human readable format (otherwise return as CSV)')
    args, unknown = argparser.parse_known_args()

    p = Config(args.config_path)
    suffix = '_2stepLRT_%s.txt' %args.partition if args.mode == '2step' else '_PT_%s.txt' %args.partition
    paths = [p.outdir + '/' + x for x in os.listdir(p.outdir) if x.endswith(suffix)]

    models, comparisons, comparisons_converged = extract_comparisons(paths)
    comparison_keys = sorted(list(comparisons.keys()), key= lambda x: len(x[0]))
    models = sorted(models)

    cols = ['model']
    for c in comparison_keys:
        cols.append(comparison2str(c))

    if args.mode == '2step':
        cols.append('converged')

    header = {}
    for col in cols:
        header[col] = col

    data = [header]
    for m in models:
        converged_str = ''
        row = {'model': m}
        for c in comparison_keys:
            c_str = comparison2str(c)
            cur_signif = comparisons[c].get(m, np.nan)
            cur_signif_str = str(cur_signif)
            if args.human_readable:
                cur_signif_str += '' if cur_signif > 0.05 else '*' if cur_signif > 0.01 else '**' if cur_signif > 0.001 else '***'
            converged_str += str(int(comparisons_converged[c].get(m, True)))
            row[c_str] = cur_signif_str
        if args.mode == '2step':
            row['converged'] = converged_str
        data.append(row)

    if args.human_readable:
        print(pretty_table(data, cols))
    else:
        for row in data:
            print(', '.join([row[col] for col in cols]))
