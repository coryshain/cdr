import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style='white', rc={'axes.facecolor': (0, 0, 0, 0)})

def plot_convolutions(
        plot_x,
        plot_y,
        features,
        uq=None,
        lq=None,
        dir='.',
        filename='convolution_plot.png',
        irf_name_map=None,
        plot_x_inches=7,
        plot_y_inches=5,
        cmap='gist_earth',
        legend=True,
        xlab=None,
        ylab=None
):
    cm = plt.get_cmap(cmap)
    plt.gca().set_prop_cycle(color=[cm(1. * i / len(features)) for i in range(len(features))])
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    feats = features[:]
    if irf_name_map is not None:
        for i in range(len(feats)):
            feats[i] = ':'.join([irf_name_map.get(x, x) for x in feats[i].split(':')])
    sort_ix = [i[0] for i in sorted(enumerate(feats), key=lambda x:x[1])]
    for i in range(len(sort_ix)):
        if plot_y[1:,sort_ix[i]].sum() == 0:
            plt.plot(plot_x[:2], plot_y[:2,sort_ix[i]], label=feats[sort_ix[i]])
        else:
            plt.plot(plot_x, plot_y[:,sort_ix[i]], label=feats[sort_ix[i]])
        if uq is not None and lq is not None:
            plt.fill_between(plot_x[:,0], lq[:,sort_ix[i]], uq[:,sort_ix[i]], alpha=0.25)
    if xlab:
        plt.xlabel(xlab)
    if ylab:
        plt.ylabel(ylab)
    if legend:
        plt.legend(fancybox=True, framealpha=0.5)
    plt.gcf().set_size_inches(plot_x_inches, plot_y_inches)
    plt.savefig(dir+'/'+filename)
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
    plt.savefig(dir+'/'+filename)
    plt.close('all')



