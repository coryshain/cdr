import os
import re
import argparse

if __name__ == '__main__':

    argparser = argparse.ArgumentParser('''
    Generate a LaTeX table summarizing results from DTSR vs. baseline models in some output directory
    ''')
    argparser.add_argument('dir', type=str, help='Path to directory containing model summary files.')
    args = argparser.parse_args()
    dir = args.dir

    loss_val = re.compile('  MSE: (.+)')

    systems = ['LMEnoS', 'LMEoptS', 'LMEfullS', 'GAMnoS', 'GAMoptS', 'GAMfullS', 'DTSR']
    tasks = ['noRE', 'si', 'ss']

    table_str = '''
    \\begin{table}
    \\begin{tabular}{r|ccc|ccc|ccc}
    & \\multicolumn{9}{c}{Random effects structure} \\\\
    & \\multicolumn{3}{c}{$\\emptyset$} & \\multicolumn{3}{c}{I\\textsubscript{subj}} & \\multicolumn{3}{c}{S\\textsubscript{subj}} \\\\
    System & Train & Dev & Test & Train & Dev & Test & Train & Dev & Test \\\\
    \\hline
    '''

    for s in systems:
        if s == 'DTSR':
            table_str += '\\hline\n'
        row = s
        for t in tasks:
            converged = True
            if s.startswith('LME') and t == 'noRE':
                s_name= s.replace('LME', 'LM')
            else:
                s_name = s
            cur_dir = dir + '/' + s_name + '_' + t + '/'
            if os.path.exists(cur_dir + 'summary.txt'):
                with open(cur_dir + 'summary.txt', 'r') as s_file:
                    for line in s_file.readlines():
                        if 'failed to converge' in line:
                            converged = False
                        if line.startswith('  MSE'):
                            MSE = float(loss_val.match(line).group(1))
                    if converged:
                        row += ' & ' + str(MSE)
                    else:
                        row += ' & ' + str(MSE) + '\\textsuperscript{\\textdagger}'
            else:
                row += ' & ---'
            if os.path.exists(cur_dir + 'eval_dev.txt'):
                with open(cur_dir + 'eval_dev.txt', 'r') as s_file:
                    for line in s_file.readlines():
                        if 'failed to converge' in line:
                            converged = False
                        if line.startswith('  MSE'):
                            MSE = float(loss_val.match(line).group(1))
                    if converged:
                        row += ' & ' + str(MSE)
                    else:
                        row += ' & ' + str(MSE) + '\\textsuperscript{\\textdagger}'
            else:
                row += ' & ---'
            if os.path.exists(cur_dir + 'eval_test.txt'):
                with open(cur_dir + 'eval_test.txt', 'r') as s_file:
                    for line in s_file.readlines():
                        if 'failed to converge' in line:
                            converged = False
                        if line.startswith('  MSE'):
                            MSE = float(loss_val.match(line).group(1))
                    if converged:
                        row += ' & ' + str(MSE)
                    else:
                        row += ' & ' + str(MSE) + '\\textsuperscript{\\textdagger}'
            else:
                row += ' & ---'

        table_str += row + '\\\\\n'


    table_str += '\\end{tabular}\n\\end{table}'

    print(table_str)
