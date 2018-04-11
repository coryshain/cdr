from numpy import inf
import argparse

# Thanks to Daniel Sparks on StackOverflow for this one (post available at
# http://stackoverflow.com/questions/5084743/how-to-print-pretty-string-output-in-python)
def getPrintTable(row_collection, key_list, field_sep=' '):
  return '\n'.join([field_sep.join([str(row[col]).ljust(width)
    for (col, width) in zip(key_list, [max(map(lambda x: len(str(x)), column_vector))
      for column_vector in [ [v[k]
        for v in row_collection if k in v]
          for k in key_list ]])])
            for row in row_collection])

if __name__ == 'main':

    argparser = argparse.ArgumentParser('''
    Generate a summary table from a batch of spillover optimization runs.
    ''')
    argparser.add_argument('paths', nargs='+', help='Path(s) to model fit summary file(s).')
    args = argparser.parse_args()

    fit_list = args.paths

    rows = []

    for path in fit_list:
        loglik = inf
        with open(path, 'r') as f:
            l = f.readline()
            while l:
                if l.strip().startswith('Model name:'):
                    name = l.strip()[12:]
                elif l.strip().startswith('MSE:'):
                    loss = l.strip()[5:]
                l = f.readline()
        rows.append({'Name': name, 'Loss': loss})

    headers = ['Name', 'Loss']
    header_row = {}
    for h in headers:
        header_row[h] = h

    converged = rows[:]
    converged.sort(key = lambda x: x['Loss'])
    converged.insert(0, header_row)

    if len(converged) > 1:
        print('===================================')
        print('Spillover optimization summary')
        print('===================================')

        print(getPrintTable(converged, headers))