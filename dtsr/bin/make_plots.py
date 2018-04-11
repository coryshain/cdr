import argparse
import sys
import pickle
from dtsr.config import Config
from dtsr.dtsr import load_dtsr

if __name__ == '__main__':

    argparser = argparse.ArgumentParser('''
        Generates predictions from data given saved model(s)
    ''')
    argparser.add_argument('models', nargs='+', help='Names of models to plot')
    argparser.add_argument('-d', '--dir', type=str, default='./', help='Path to directory that contains saved models')
    argparser.add_argument('-p', '--prefix', type=str, default='', help='Filename prefix to use for outputs')
    argparser.add_argument('-u', '--ntimeunits', type=float, default=5.0, help='Number of time units on x-axis')
    argparser.add_argument('-r', '--resolution', type=float, default=500, help='Number of points per time unit')
    argparser.add_argument('-x', '--x', type=float, default=10.0, help='Width of plot in inches')
    argparser.add_argument('-y', '--y', type=float, default=6.0, help='Height of plot in inches')
    argparser.add_argument('-c', '--cmap', type=str, default='gist_rainbow', help='Name of matplotlib colormap library to use for curves')
    argparser.add_argument('-n', '--namemap', type=str, default=None, help='Path to config file with IRF name map information for legend (if blank, no name map is used)')
    argparser.add_argument('-l', '--nolegend', action='store_true', help='Omit legend from figure')
    args, unknown = argparser.parse_known_args()

    models = args.models
    dir = args.dir
    prefix = args.prefix
    if prefix != '':
        prefix += '_'
    n_time_units = args.ntimeunits
    resolution = args.resolution
    x_inches = args.x
    y_inches = args.y
    cmap = args.cmap
    name_map = args.namemap
    if name_map is not None:
        name_map = Config(name_map).irf_name_map
    legend = not args.nolegend

    for m in models:

        sys.stderr.write('Retrieving saved model %s...\n' % m)
        dtsr_model = load_dtsr(dir + '/' + m)

        dtsr_model.make_plots(
            irf_name_map=name_map,
            plot_n_time_units=n_time_units,
            plot_n_points_per_time_unit=resolution,
            plot_x_inches=x_inches,
            plot_y_inches=y_inches,
            cmap=cmap,
            prefix=prefix + m,
            legend=legend
        )
        if hasattr(dtsr_model, 'inference_name'):
            dtsr_model.make_plots(
                irf_name_map=name_map,
                plot_n_time_units=n_time_units,
                plot_n_points_per_time_unit=resolution,
                plot_x_inches=x_inches,
                plot_y_inches=y_inches,
                cmap=cmap,
                mc=True,
                prefix=prefix + m,
                legend=legend
            )


