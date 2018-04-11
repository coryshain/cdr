import os
import re
import argparse

def print_table(dataset, systems, system_map, tasks, loss_val):
    table_str = '''
    %s
    \\begin{table}
    \\begin{tabular}{r|cc|cc|cc}
    & \\multicolumn{6}{c}{Random effects structure} \\\\
    & \\multicolumn{2}{c}{$\\emptyset$} & \\multicolumn{2}{c}{I\\textsubscript{subj}} & \\multicolumn{2}{c}{S\\textsubscript{subj}} \\\\
    Inference type & Train & Test & Train & Test & Train & Test \\\\
    \\hline
    ''' % dataset

    dir = dataset + '_'

    for s in systems:
        row = system_map[s]
        for t in tasks:
            s_name = s
            cur_dir = dir + s_name + '/DTSR_' + t + '/'
            if os.path.exists(cur_dir + 'summary.txt'):
                with open(cur_dir + 'summary.txt', 'r') as s_file:
                    for line in s_file.readlines():
                        if line.startswith('  MSE'):
                            MSE = float(loss_val.match(line).group(1))
                    row += ' & ' + str(MSE)
            else:
                row += ' & ---'
            if os.path.exists(cur_dir + 'eval_dev.txt'):
                with open(cur_dir + 'eval_dev.txt', 'r') as s_file:
                    for line in s_file.readlines():
                        if line.startswith('  MSE'):
                            MSE = float(loss_val.match(line).group(1))
                    row += ' & ' + str(MSE)
            else:
                row += ' & ---'

        table_str += row + '\\\\\n'

    table_str += '\\end{tabular}\n\\end{table}'
    print(table_str)

if __name__ == '__main__':

    argparser = argparse.ArgumentParser('''
    Generate a LaTeX table summarizing results from DTSR models fit to Natural Stories, UCL, and Dundee.
    ''')
    args = argparser.parse_args()

    loss_val = re.compile('  MSE: (.+)')

    datasets = ['natstor', 'dundee', 'ucl']
    systems = ['nn', 'nn_reg', 'prior_loose']
    system_map = {
        'nn': 'SGD',
        'nn_reg': 'Regularized SGD',
        'prior_loose': 'BBVI'
    }

    tasks = ['noRE', 'si', 'ss']

    for d in datasets:
        print_table(d, systems, system_map, tasks, loss_val)
