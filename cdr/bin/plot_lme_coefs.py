import sys
import os
import re
import argparse
import numpy as np
from matplotlib import pyplot as plt

def get_coefs(path):
    coefs = []
    with open(path, 'r') as f:
        line = f.readline()
        n_fixed = 0
        while line:
            starts_coefs = line.startswith('Fixed effects') or line.startswith('Coefficients')
            n_fixed += (starts_coefs or n_fixed > 0)
            if n_fixed > 3:
                if line.strip() and not (line.strip() == '---' or line.startswith('convergence')):
                    coefs.append(float(line.strip().split()[1]))
                else:
                    break 
            line = f.readline()
    return coefs
            
        

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
        Plot effect estimates from LME models fitted to CDR-convolved predictors.
        Optimal estimates should be close to 1.
    ''')
    argparser.add_argument('paths', nargs='+', help='Space-delimited list of paths, supports regex. Individual elements can also contain ";"-delimited lists of paths, in which case all coefs in the element will be collapsed in plots.')
    argparser.add_argument('-p', '--partition', default='train', help='Partition name (default: "train").')
    argparser.add_argument('-d', '--search_dir', nargs='+', default=['./'], help='Directory/directories to search (default: "./").')
    argparser.add_argument('-e', '--exclude', nargs='*', default=[], help='Exclude data from paths containing any of the strings in ``exclude``.')
    argparser.add_argument('-n', '--names', nargs='*', help='Names of elements in ``paths``.')
    argparser.add_argument('-o', '--outpath', default='lme_coefs.png', help='Filename to use for saving plot.')
    args = argparser.parse_args()

    paths = [x.split(';') for x in args.paths]

    if args.names is None:
        names = args.paths
    else:
        assert len(args.names) == len(args.paths), 'If path element names are specified, there must be an equal number of names and elements in ``paths``.'
        names = args.names

    summaries = []
    for directory in args.search_dir:
        for root, _, files in os.walk(directory):
            for f in files:
                if f == 'lm_%s_summary.txt' % args.partition:
                    summaries.append(os.path.join(root, f))

    included = []
    coefs = []
    for e in paths:
        included_cur = []
        coefs_cur = []
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
                        coefs_new = get_coefs(x)
                        coefs_cur += coefs_new
                        included_cur.append(x)
#                        print(x)
#                        print(coefs_new)
#                        input()
        coefs.append(coefs_cur)
        included.append(included_cur)
      
#    for x in included:
#        for y in x:
#            print(y)
#        print() 
#    for x in coefs:
#        print(len(x))
#    print()

    coefs = [np.array(c) for c in coefs]
    medians = [np.percentile(c, 50) for c in coefs]
    uq = [np.percentile(c,75) for c in coefs]
    lq = [np.percentile(c,25) for c in coefs]

    plt.rcParams["font.family"] = "sans-serif"
    plt.gca().spines['top'].set_visible(False) 
    plt.gca().spines['right'].set_visible(False) 
    plt.gca().spines['bottom'].set_visible(False) 
    plt.gca().spines['left'].set_visible(False)
    plt.gca().tick_params(top='off', bottom='off', left='off', right='off', labelleft='on', labelbottom='on') 
    plt.grid(b=True, which='major', axis='both', ls='--', lw=.5, c='k', alpha=.3)

#    plt.boxplot(coefs, flierprops=dict(markersize=1, marker='.'))
    for i in range(len(coefs)):
        print(medians[i])
        print(lq[i])
        print(uq[i])
        print()
        yerr = np.stack([medians[i]-lq[i],uq[i]-medians[i]], axis=0)[..., None]
        plt.bar(i, medians[i], yerr=yerr, capsize=10)
    plt.xticks(range(1, len(coefs)+1), names)
    plt.ylabel('LME Estimate', weight='bold')
    plt.gcf().set_size_inches(4,2.5)
    plt.tight_layout()
    plt.savefig(args.outpath, height=2.5, width=4, dpi=150)

