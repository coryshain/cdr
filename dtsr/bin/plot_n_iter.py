import sys
import os
import re
import argparse
import numpy as np
from matplotlib import pyplot as plt

def get_n_iter(path):
    with open(path, 'r') as f:
        line = f.readline()
        while line and not line.startswith('    Training iterations completed:'):
            line = f.readline()
        if line.strip():
            n_iter = int(line.strip().split()[-1])
        else:
            n_iter = None
    return n_iter
            
        

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
        Plot effect estimates from LME models fitted to DTSR-convolved predictors.
        Optimal estimates should be close to 1.
    ''')
    argparser.add_argument('paths', nargs='+', help='Space-delimited list of paths, supports regex. Individual elements can also contain ";"-delimited lists of paths, in which case all n_iters in the element will be collapsed in plots.')
    argparser.add_argument('-p', '--partition', default='train', help='Partition name (default: "train").')
    argparser.add_argument('-d', '--search_directory', default='./', help='Root directory (default: "./").')
    argparser.add_argument('-e', '--exclude', nargs='*', default=[], help='Exclude data from paths containing any of the strings in ``exclude``.')
    argparser.add_argument('-n', '--names', nargs='*', help='Names of elements in ``paths``.')
    argparser.add_argument('-o', '--outpath', default='dtsr_n_iters.png', help='Filename to use for saving plot.')
    args = argparser.parse_args()

    paths = [x.split(';') for x in args.paths]

    if args.names is None:
        names = args.paths
    else:
        assert len(args.names) == len(args.paths), 'If path element names are specified, there must be an equal number of names and elements in ``paths``.'
        names = args.names

    summaries = []
    for root, _, files in os.walk(args.search_directory):
        for f in files:
            if f == 'summary.txt':
                summaries.append(os.path.join(root, f))
    summaries = sorted(summaries)

    n_iters = []
    for e in paths:
        n_iters_cur = []
        for p in e:
            f = re.compile(p)
            for i in range(len(summaries)):
                x = summaries[i]
                if x == p or f.search(x):
                    excluded = False
                    for exclude in args.exclude:
                        if exclude in x:
                            excluded = True
                            break
                    if not excluded:
                        n_iters_cur.append(get_n_iter(x))
        n_iters.append(n_iters_cur)
       
    n_iters = [np.array(n) for n in n_iters]
    medians = [np.percentile(n, 50) for n in n_iters]
    uq = [np.percentile(n,75) for n in n_iters]
    lq = [np.percentile(n,25) for n in n_iters]

    plt.rcParams["font.family"] = "sans-serif"
    plt.gca().spines['top'].set_visible(False) 
    plt.gca().spines['right'].set_visible(False) 
    plt.gca().spines['bottom'].set_visible(False) 
    plt.gca().spines['left'].set_visible(False)
    plt.gca().tick_params(top='off', bottom='off', left='off', right='off', labelleft='on', labelbottom='on') 
    plt.grid(b=True, which='major', axis='both', ls='--', lw=.5, c='k', alpha=.3)

    for i in range(len(n_iters)):
        yerr = np.stack([medians[i]-lq[i],uq[i]-medians[i]], axis=0)[..., None]
        plt.bar(i, medians[i], yerr=yerr, capsize=10)
    plt.xticks(range(len(n_iters)), names)
    plt.ylabel('Num Iter', weight='bold')
    plt.gcf().set_size_inches(4,2.5)
    plt.tight_layout()
    plt.savefig(args.outpath, height=2.5, width=4, dpi=150)

