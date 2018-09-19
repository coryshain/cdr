import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import markers

def plot_irf(
        plot_x,
        plot_y,
        irf_names,
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
        use_line_markers=False,
        transparent_background=False
):
    """
    Plot N impulse response functions at T time points.

    :param plot_x: ``numpy`` array with shape (T,); time points for which to plot the response. For example, if the plots contain 1000 points from 0s to 10s, **plot_x** could be generated as ``np.linspace(0, 10, 1000)``.
    :param plot_y: ``numpy`` array with shape (T, N); response of each IRF at each time point.
    :param irf_names: ``list`` of ``str``; DTSR ID's of IRFs in the same order as they appear in axis 1 of **plot_y**.
    :param uq: ``numpy`` array with shape (T, N), or ``None``; upper bound of credible interval for each time point. If ``None``, no credible interval will be plotted.
    :param lq: ``numpy`` array with shape (T, N), or ``None``; lower bound of credible interval for each time point. If ``None``, no credible interval will be plotted.
    :param dir: ``str``; output directory.
    :param filename: ``str``; filename.
    :param irf_name_map: ``dict`` of ``str`` to ``str``; map from DTSR IRF ID's to more readable names to appear in legend. Any plotted IRF whose ID is not found in **irf_name_map** will be represented with the DTSR IRF ID.
    :param plot_x_inches: ``float``; width of plot in inches.
    :param plot_y_inches: ``float``; height of plot in inches.
    :param cmap: ``str``; name of ``matplotlib`` ``cmap`` object (determines colors of plotted IRF).
    :param legend: ``bool``; include a legend.
    :param xlab: ``str`` or ``None``; x-axis label. If ``None``, no label.
    :param ylab: ``str`` or ``None``; y-axis label. If ``None``, no label.
    :param use_line_markers: ``bool``; add markers to IRF lines.
    :param transparent_background: ``bool``; use a transparent background. If ``False``, uses a white background.
    :return: ``None``
    """

    cm = plt.get_cmap(cmap)
    plt.rcParams["font.family"] = "sans-serif"
    prop_cycle_kwargs = {'color': [cm(1. * i / len(irf_names)) for i in range(len(irf_names))]}
    if use_line_markers:
        prop_cycle_kwargs['marker'] = list(markers.MarkerStyle.markers.keys())[:-3][:len(irf_names)]
    plt.gca().set_prop_cycle(**prop_cycle_kwargs)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().tick_params(top='off', bottom='off', left='off', right='off', labelleft='on', labelbottom='on')
    plt.grid(b=True, which='major', axis='both', ls='--', lw=.5, c='k', alpha=.3)
    plt.axhline(y=0, lw=1, c='gray', alpha=1)
    plt.axvline(x=0, lw=1, c='gray', alpha=1)

    irf_names_processed = irf_names[:]
    if irf_name_map is not None:
        for i in range(len(irf_names_processed)):
            irf_names_processed[i] = ':'.join([irf_name_map.get(x, x) for x in irf_names_processed[i].split(':')])
    sort_ix = [i[0] for i in sorted(enumerate(irf_names_processed), key=lambda x:x[1])]
    for i in range(len(sort_ix)):
        if plot_y[1:,sort_ix[i]].sum() == 0:
            plt.plot(plot_x[:2], plot_y[:2,sort_ix[i]], label=irf_names_processed[sort_ix[i]], lw=2, alpha=0.8, linestyle='-', solid_capstyle='butt')
        else:
            markevery = int(len(plot_y) / 10)
            plt.plot(plot_x, plot_y[:,sort_ix[i]], label=irf_names_processed[sort_ix[i]], lw=2, alpha=0.8, linestyle='-', markevery=markevery, solid_capstyle='butt')
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

def plot_heatmap(
        m,
        row_names,
        col_names,
        dir='.',
        filename='eigenvectors.png',
        plot_x_inches=7,
        plot_y_inches=5,
        cmap='Blues'
):
    """
    Plot a heatmap. Generally used in DTSR for visualizing eigenvector matrices in principal components models.

    :param m: 2D ``numpy`` array; source data for plot.
    :param row_names: ``list`` of ``str``; row names.
    :param col_names: ``list`` of ``str``; column names.
    :param dir: ``str``; output directory.
    :param filename: ``str``; filename.
    :param plot_x_inches: ``float``; width of plot in inches.
    :param plot_y_inches: ``float``; height of plot in inches.
    :param cmap: ``str``; name of ``matplotlib`` ``cmap`` object (determines colors of plotted IRF).
    :return: ``None``
    """

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



