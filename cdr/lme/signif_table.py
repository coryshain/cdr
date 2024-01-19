import os
import numpy as np
import pandas as pd
import argparse

from cdr.config import Config
from cdr.util import get_irf_name

def parse_summary(path):
    with open(path, 'r') as f:
        effects = {}
        signif = None
        in_effect = False
        in_signif = False
        for line in f:
            line = line.strip()
            pieces = line.split()
            if line.startswith('Fixed effects:'):
                in_effect = True
            elif 'Pr(>Chisq)' in line or line.startswith('Model comparison:'):
                in_signif = True
            elif in_effect:
                if not (line.startswith('Estimate') or
                        line.startswith('(Intercept)') or
                        line.startswith('scale')):
                    in_effect = False
                else:
                    vals = pieces[0:2]
                    if vals[0] != 'Estimate':
                        effects[vals[0]] = float(vals[1])
            elif in_signif:
                if line.startswith('m2'):
                   ix = -1
                   if '*' in pieces[ix]:
                      ix = -2
                   signif = pieces[ix]
                elif line.startswith('p:'):
                   signif = pieces[1].replace('*', '')
        if signif is not None:
            signif = float(signif)
        return effects, signif


def p2str(p):
    eps = 1e-4
    alpha = 0.05
    p = np.round(p, 4)
    out = '%0.4f' % p
    if p < eps:
        p = eps
        out = '$<$ %0.4f' % p
    if p < alpha:
        out = '\\textbf{%s}' % out
    return out

argparser = argparse.ArgumentParser('Compile LaTeX table of test results')
args = argparser.parse_args()

dvs = {
    'brown': ['fdur'],
    'dundee': ['fdurSPsummed', 'fdurFP', 'fdurGP'],
    'geco': ['fdurFP', 'fdurGP'],
    'natstor': ['fdur'],
    'natstormaze': ['rt'],
    'provo': ['fdurSPsummed', 'fdurFP', 'fdurGP']
}

dataset2name = {
    'brown': 'Brown SPR',
    'dundee': 'Dundee ET',
    'geco': 'GECO ET',
    'natstor': 'Natural Stories SPR',
    'natstormaze': 'Natural Stories Maze',
    'provo': 'Provo ET',
}

comparisons = {
    'null_v_freqonly': ('scale(unigramsurpOWT)', 'Freq'),
    'null_v_predonly': ('scale(gpt)', 'Pred'),
    'predonly_v_both': ('scale(unigramsurpOWT)', 'Freq over Pred'),
    'freqonly_v_both': ('scale(gpt)', 'Pred over Freq'),
    'both_v_interaction': ('scale(unigramsurpOWT):scale(gpt)', 'Interaction'),
}

results_path = '../results/cdrnn_freq_pred_owt/lme/'

out = '''
\\begin{table}
  \\centering
  \\tiny
  \\begin{tabular}{rrr|r|rr|rr|rr|rr|rr}
    & & & & '''
out += ' & '.join(['\\multicolumn{2}{|c}{%s}' % comparisons[comparison][1] for comparison in comparisons]) + '\\\\\n'
out += '& Dataset & Response & Test LL &' + ' & '.join([' $\\beta$ & $p$'] * 5) + '\\\\\n'
for eval_type in ('insample', 'outofsample'):
    out += '    \\hline\n'
    out += '    \\hline\n'
    n = 0
    if eval_type == 'insample':
        eval_type_name = 'In-sample (LRT)'
    else:
        eval_type_name = 'Out-of-sample (PT)'
    for i, dataset in enumerate(dvs):
        if i > 0:
            # out += '    \\hline\n'
            out += '    \\cdashline{2-14}[0.5pt/1pt]\n'
        p = Config('ini_owt/%s.ini' % dataset)
        for j, response in enumerate(dvs[dataset]):
            n += 1
            if j == len(dvs[dataset]) - 1:
                dataset_name = '& \\multirow{-%d}{*}{%s}' % (len(dvs[dataset]), dataset2name[dataset])
                if i == len(dvs) - 1:
                    dataset_name = '\\multirow{-%d}{*}{\\rotatebox{90}{%s}} ' % (n, eval_type_name) + dataset_name
            else:
                dataset_name = '& '
            path = os.path.join(results_path, dataset, '%s_%s_interaction_%s_eval.txt' % (dataset, response, eval_type))
            with open(path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('Loglik:'):
                        LL = int(np.round(float(line.split()[-1])))
            response_name = get_irf_name(response, p.irf_name_map)
            out += '    %s & %s & %d' % (dataset_name, response_name, LL)
            for comparison in comparisons:
                effect_name, comparison_name = comparisons[comparison]
                path = os.path.join(results_path, dataset, '%s_%s_test_%s_%s.txt' % (dataset, response, comparison, eval_type))
                if os.path.exists(path):
                    effects, signif = parse_summary(path)
                    effect = '%0.2f' % np.round(effects[effect_name], 2)
                    signif = p2str(signif)
                else:
                    effect = signif = '---'
                out += ' & %s & %s' % (
                    effect, signif
                )
            out += '\\\\\n'
out += '''
  \\end{tabular}
  \\caption{}
\\end{table}
'''
print(out)


