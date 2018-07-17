import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_irf(
        plot_x,
        plot_y,
        features,
        uq=None,
        lq=None,
        dir='.',
        filename='irf_plot.png',
        irf_name_map=None,
        plot_x_inches=6,
        plot_y_inches=4,
        cmap='gist_rainbow',
        legend=True,
        xlab=None,
        ylab=None,
        transparent_background=False
):
    cm = plt.get_cmap(cmap)
    plt.rcParams["font.family"] = "sans-serif"
    plt.gca().set_prop_cycle(color=[cm(1. * i / len(features)) for i in range(len(features))])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.grid(b=True, which='major', axis='both', ls='--', lw=.5, c='k', alpha=.3)
    plt.axhline(y=0, lw=1, c='gray', alpha=1)
    plt.axvline(x=0, lw=1, c='gray', alpha=1)

    feats = features[:]
    if irf_name_map is not None:
        for i in range(len(feats)):
            feats[i] = ':'.join([irf_name_map.get(x, x) for x in feats[i].split(':')])
    sort_ix = [i[0] for i in sorted(enumerate(feats), key=lambda x:x[1])]
    for i in range(len(sort_ix)):
        if plot_y[1:,sort_ix[i]].sum() == 0:
            plt.plot(plot_x[:2], plot_y[:2,sort_ix[i]], label=feats[sort_ix[i]], lw=2, alpha=0.8, solid_capstyle='butt')
        else:
            plt.plot(plot_x, plot_y[:,sort_ix[i]], label=feats[sort_ix[i]], lw=2, alpha=0.8, solid_capstyle='butt')
        if uq is not None and lq is not None:
            plt.fill_between(plot_x[:,0], lq[:,sort_ix[i]], uq[:,sort_ix[i]], alpha=0.25)

    if xlab:
        plt.xlabel(xlab, weight='bold')
    if ylab:
        plt.ylabel(ylab, weight='bold')
    if legend:
        plt.legend(fancybox=True, framealpha=0.75, frameon=True, facecolor='white', edgecolor='gray')

    plt.gcf().set_size_inches(plot_x_inches, plot_y_inches)
    plt.tight_layout()
    try:
        plt.savefig(dir+'/'+filename, dpi=600, transparent=transparent_background)
    except:
        sys.stderr.write('Error saving plot to file %s. Skipping...' %(dir+'/'+filename))
    plt.close('all')

def plot_legend(
        features,
        irf_name_map=None,
        cmap='gist_earth'
):
    plt.gca().set_prop_cycle(color=[cm(1. * i / len(features)) for i in range(len(features))])



def plot_heatmap(m, row_names, col_names, dir='.', filename='eigenvectors.png', plot_x_inches=7, plot_y_inches=5, cmap='Blues'):
    cm = plt.get_cmap(cmap)
    plt.gca().set_xticklabels(col_names)
    plt.gca().set_xticks(np.arange(len(col_names)) + 0.5, minor=False)
    plt.gca().set_yticklabels(row_names)
    plt.gca().set_yticks(np.arange(len(row_names)) + 0.5, minor=False)
    heatmap = plt.pcolor(m, cmap=cm)
    plt.colorbar(heatmap)
    plt.gcf().set_size_inches(plot_x_inches, plot_y_inches)
    plt.gcf().subplots_adjust(bottom=0.25,left=0.25)
    try:
        plt.savefig(dir+'/'+filename)
    except:
        sys.stderr.write('Error saving plot to file %s. Skipping...' %(dir+'/'+filename))
    plt.close('all')



