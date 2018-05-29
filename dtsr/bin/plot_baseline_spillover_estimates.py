import sys
import os
import re
import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from dtsr.config import Config

spillover_pos = re.compile('S([0-9]+)')

def parse_coefficients(path):
    coefs = {}
    with open(path, 'r') as f:
        line = f.readline()
        while line and not line.strip().startswith('Estimate'):
            line = f.readline()
        if line:
            line = f.readline()
        while line and not (line.strip() in ['', '---'] or line.startswith('convergence')):
            row = line.strip().split()
            spill = 0
            spill_search = spillover_pos.search(row[0])
            if spill_search:
                spill = int(spill_search.group(1))

            varname = re.sub(r'S[0-9]+', '', row[0])

            if varname in coefs:
                coefs[varname][spill] = float(row[1])
            else:
                coefs[varname] = {spill: float(row[1])}

            line = f.readline()

    return coefs


if __name__ == '__main__':

    argparser = argparse.ArgumentParser('''
    Generates plots of fitted spillover estimates for "fullS" baseline models, visualizing those models' estimates of temporal diffusion.
    ''')
    argparser.add_argument('paths', nargs='+', help='Paths to config files defining baseline models to plot')
    argparser.add_argument('-m', '--models', nargs='*', default=None, help='List of models to plot. If none specified, plots all relevant models in the config files')
    argparser.add_argument('-x', '--x', type=float, default=6, help='Width of plots in inches')
    argparser.add_argument('-X', '--xlab', type=str, default=None, help='x-axis label (if default -- None -- no label)')
    argparser.add_argument('-y', '--y', type=float, default=4, help='Height of plots in inches')
    argparser.add_argument('-Y', '--ylab', type=str, default=None, help='y-axis label (if default -- None -- no label)')
    argparser.add_argument('-c', '--cmap', type=str, default='gist_rainbow', help='Name of matplotlib colormap library to use for curves')
    argparser.add_argument('-l', '--nolegend', action='store_true', help='Omit legend from figure')
    args = argparser.parse_args()

    configs = []
    for c in args.paths:
        configs.append(Config(c))

    if args.models is None:
        models = []
    else:
        models = [x for x in args.models if ('LMfullS' in x or 'LMEfullS' in x)]
        assert len(models > 0), 'No valid models in script call. Models must be linear fullS models.'

    cm = plt.get_cmap(args.cmap)
    legend = not args.nolegend

    for p in configs:
        if len(models) == 0:
            models = [x for x in p.model_list[:] if ('LMfullS' in x or 'LMEfullS' in x)]
        for name in models:
            dirpath = p.logdir + '/' + name + '/'
            summary_path = dirpath + 'summary.txt'
            if os.path.exists(summary_path):
                coefs = parse_coefficients(summary_path)
                varname = []
                spill = []
                estimate = []
                for var in coefs:
                    if len(coefs[var]) > 1:
                        for s in coefs[var]:
                            varname.append(p.irf_name_map.get(var, var))
                            spill.append(s)
                            estimate.append(coefs[var][s])

                d = pd.DataFrame(
                    {
                        'Predictor': varname,
                        'Spillover Position': spill,
                        'Estimate': estimate
                    }
                )

                predictors = sorted(list(d.Predictor.unique()))
                d = d.pivot(index='Spillover Position', columns='Predictor', values='Estimate')

                fig, ax = plt.subplots()
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                ax.set_prop_cycle(color=[cm(1. * i / len(predictors)) for i in range(len(predictors))])
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_visible(False)
                ax.spines['left'].set_visible(False)
                ax.grid(b=True, which='major', axis='y', ls='--', lw=.5, c='k', alpha=.3)
                ax.axhline(y=0, lw=1, c='gray', alpha=1)
                ax.axvline(x=0, lw=1, c='gray', alpha=1)

                for c in predictors:
                    ax.plot(d.index, d[c], marker='o', label=c, lw=2, alpha=0.8, solid_capstyle='butt')

                if args.xlab:
                    ax.set_xlabel(args.xlab, weight='bold')
                if args.ylab:
                    ax.set_ylabel(args.ylab, weight='bold')
                if legend:
                    plt.legend(fancybox=True, framealpha=0.75, frameon=True, facecolor='white', edgecolor='gray')

                fig.set_size_inches(args.x, args.y)
                fig.tight_layout()
                fig.savefig(dirpath + p.logdir.split('/')[-1] + '_' + name + '_spillover_plot.png', dpi=600)

                plt.close(fig)





