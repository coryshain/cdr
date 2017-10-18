import sys
import argparse
import pandas as pd
from dtsr import compute_splitID, compute_partition

argparser = argparse.ArgumentParser('''
    A utility for splitting a dataset into train, test, and dev sets given splitting criteria.
''')
argparser.add_argument('path', help='Path to full data set')
argparser.add_argument('-m', '--mod', type=int, default=4, help='Modulus to use for splitting')
argparser.add_argument('-f', '--fields', nargs='+', default=['subject', 'sentid'], help='Field names to use as split IDs')
args, unknown = argparser.parse_known_args()

df = pd.read_csv(args.path, sep=' ', skipinitialspace=True)
for f in args.fields:
    df[f] = df[f].astype('category')
cols = df.columns
df['splitID'] = compute_splitID(df, args.fields)

for p in ['train', 'dev', 'test']:
    select = compute_partition(df, p, args.mod)
    df[select].to_csv(args.path + '.' + p, sep=' ', index=False, na_rep='nan', columns=cols)