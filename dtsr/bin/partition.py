import sys
import argparse
import pandas as pd
from dtsr.data import compute_splitID, compute_partition

if __name__ == 'main':
    argparser = argparse.ArgumentParser('''
        A utility for splitting a dataset into train, test, and dev sets given splitting criteria.
    ''')
    argparser.add_argument('path', help='Path to full data set')
    argparser.add_argument('-m', '--mod', type=int, default=4, help='Modulus to use for splitting')
    argparser.add_argument('-n', '--n', type=int, default=3, help='Arity of partition')
    argparser.add_argument('-f', '--fields', nargs='+', default=['subject', 'sentid'], help='Field names to use as split IDs')
    args, unknown = argparser.parse_known_args()

    df = pd.read_csv(args.path, sep=' ', skipinitialspace=True)
    for f in args.fields:
        df[f] = df[f].astype('category')
    cols = df.columns
    df['splitID'] = compute_splitID(df, args.fields)

    select = compute_partition(df, args.mod, args.n)

    if args.n == 3:
        names = ['train', 'dev', 'test']
        for i in range(len(names)):
            df[select[i]].to_csv(args.path + '.' + names[i], sep=' ', index=False, na_rep='nan', columns=cols)
    elif args.n == 2:
        names = ['train', 'test']
        for i in range(len(names)):
            df[select[i]].to_csv(args.path + '.' + names[i], sep=' ', index=False, na_rep='nan', columns=cols)
    else:
        names = range(args.n)
        for i in range(len(names)):
            df[select[i]].to_csv(args.path + '.' + str(names[i]), sep=' ', index=False, na_rep='nan', columns=cols)