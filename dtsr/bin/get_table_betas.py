import argparse
from dtsr.config import Config

def print_table(beta_summaries, names, beta_names):
    table_str = '''
    \\begin{table}
    \\begin{tabular}{r|%s}
    & %s \\\\
    & %s \\\\
    \\hline
    ''' %(
        '|'.join(['l'*3 for x in names]),
        ' & '.join(['\\multicolumn{3}{c}{%s}'%x for x in names]),
        ' & '.join(['Mean & 2.5\\% & 97.5\\%' for x in names])
    )

    for x in sorted(list(beta_names)):
        row = x.split('-')[0]
        for m in names:
            data = beta_summaries[m].get(x, None)
            if data is not None:
                row += ' & ' + ' & '.join([
                    '%.2e' % beta_summaries[m][x]['mean'],
                    '%.2e' % beta_summaries[m][x]['lower'],
                    '%.2e' % beta_summaries[m][x]['upper']
                ])
            else:
                row += ' & ---' * 3
        table_str += row + '\\\\\n'


    table_str += '\\end{tabular}\n\\end{table}'
    print(table_str)

if __name__ == '__main__':

    argparser = argparse.ArgumentParser('''
    Generates a LaTeX table of beta summaries for one or more DTSR models
    ''')
    argparser.add_argument('config', help='Path to config file defining models')
    argparser.add_argument('-m', '--models', nargs='+', default=[], help='Folder name(s) containing model eval summaries (if blank applies to all models described in config file).')
    argparser.add_argument('-n', '--names', nargs='*', default=[], help='Model names to print in table (must be omitted or same length as --models).')
    args = argparser.parse_args()

    p = Config(args.config)
    if len(args.models) == 0:
        models = p.model_list[:]
    else:
        models = args.models[:]

    if len(args.names) == 0:
        names = p.model_list[:]
    else:
        assert len(args.names) == len(models), 'Length mismatch between number of models and number of model names'
        names = args.names

    beta_summaries = {}
    beta_names = set()

    for i in range(len(models)):
        m = models[i]
        name = names[i]
        beta_summaries[name] = {}
        with open(p.outdir + '/' + m + '/summary.txt', 'r') as f:
            l = f.readline()
            while l and not l.startswith('Posterior integral summaries by predictor'):
                l = f.readline()
            f.readline()
            l = f.readline()
            while l and len(l.strip()) > 0:
                row = l.strip().split()
                assert len(row) == 4, 'Ill-formed row in effect table: "%s"' %l.strip()
                beta_names.add(row[0])
                beta_summaries[name][row[0]] = {
                    'mean': float(row[1]),
                    'lower': float(row[2]),
                    'upper': float(row[3])
                }
                l = f.readline()

    print_table(beta_summaries, names, beta_names)


