import sys
import os
import re
from numpy import rint

loss_val = re.compile('  MSE: (.+)')

dir = sys.argv[1]

systems = ['LMEnoS', 'LMEoptS', 'LMEfullS', 'GAMnoS', 'GAMoptS', 'GAMfullS', 'DTSR']
tasks = ['noRE', 'si', 'ss']

table_str = '''
\\begin{table}
\\begin{tabular}{r|cc|cc|cc}
& \\multicolumn{6}{c}{Random effects structure} \\\\
& \\multicolumn{2}{c}{$\\emptyset$} & \\multicolumn{2}{c}{I\\textsubscript{subj}} & \\multicolumn{2}{c}{S\\textsubscript{subj}} \\\\
System & Train & Test & Train & Test & Train & Test \\\\
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
                        MSE = int(rint(float(loss_val.match(line).group(1))))
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
                        MSE = int(rint(float(loss_val.match(line).group(1))))
                if converged:
                    row += ' & ' + str(MSE)
                else:
                    row += ' & ' + str(MSE) + '\\textsuperscript{\\textdagger}'
        else:
            row += ' & ---'

    table_str += row + '\\\\\n'


table_str += '\\end{tabular}\n\\end{table}'

print(table_str)
