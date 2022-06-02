import sys
import os
import argparse
import pandas as pd
from cdr.data import compute_splitID, compute_partition

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
        A utility for splitting a dataset into train, test, and dev sets given splitting criteria.
    ''')
    argparser.add_argument('path', help='Path to full data set')
    argparser.add_argument('-m', '--mod', type=int, default=4, help='Modulus to use for splitting')
    argparser.add_argument('-n', '--n', type=int, default=3, help='Arity of partition')
    argparser.add_argument('-f', '--fields', nargs='+', default=['subject', 'sentid'], help='Field names to use as split IDs')
    argparser.add_argument('-p', '--partition', type=str, default=None, help='ID of partition to send to stdout ("train", "dev", "test", or integer). If unspecified, saves all elements of the partition to separate files in the source directory.')
    argparser.add_argument('-s', '--sep', type=str, default=None, help='Column separator. If unspecified, assumes comma-delimited.')
    args = argparser.parse_args()

    if args.sep:
        sep = args.sep
    else:
        sep = ','
    df = pd.read_csv(args.path, sep=sep, skipinitialspace=True)
    for f in args.fields:
        df[f] = df[f].astype('category')
    cols = df.columns
    df['splitID'] = compute_splitID(df, args.fields)

    select = compute_partition(df, args.mod, args.n)

    if args.n == 3:
        names = ['train', 'dev', 'test']
    elif args.n == 2:
        names = ['train', 'test']
    else:
        names = [str(x) for x in range(args.n)]

    if args.partition is None:
        for i in range(len(names)):
            outpath, ext = os.path.splitext(args.path)
            outpath += '_' + names[i] + ext
            df[select[i]].to_csv(outpath, sep=sep, index=False, na_rep='nan', columns=cols)
    else:
        try:
            i = int(args.partition)
        except:
            i = names.index(args.partition)
        df[select[i]].to_csv(sys.stdout, sep=sep, index=False, na_rep='nan', columns=cols)
