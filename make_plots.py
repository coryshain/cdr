import argparse
import os
import sys
import pickle
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None

from dtsr import Config, read_data, Formula, preprocess_data, print_tee, mse, mae, compute_splitID, compute_partition

if __name__ == '__main__':

    argparser = argparse.ArgumentParser('''
        Generates predictions from data given saved model(s)
    ''')
    argparser.add_argument('config_path', help='Path to configuration (*.ini) file')
    argparser.add_argument('-m', '--models', nargs='*', default=[], help='Path to configuration (*.ini) file')
    argparser.add_argument('-p', '--partition', type=str, default='dev', help='Name of partition to use (one of "train", "dev", "test")')
    args, unknown = argparser.parse_known_args()

    p = Config(args.config_path)
    if len(args.models) > 0:
        models = args.models
    else:
        models = p.model_list[:]

    run_baseline = False
    run_dtsr = False
    for m in models:
        if not run_baseline and m.startswith('LM') or m.startswith('GAM'):
            run_baseline = True
        elif not run_dtsr and m.startswith('DTSR'):
            run_dtsr = True

    dtsr_formula_list = [Formula(p.models[m]['formula']) for m in p.model_list if m.startswith('DTSR')]
    dtsr_formula_name_list = [m for m in p.model_list if m.startswith('DTSR')]

    for m in models:
        formula = p.models[m]['formula']
        if m.startswith('DTSR'):
            dv = formula.strip().split('~')[0].strip()

            sys.stderr.write('Retrieving saved model %s...\n' % m)
            with open(p.logdir + '/' + m + '/m.obj', 'rb') as m_file:
                dtsr_model = pickle.load(m_file)

            dtsr_model.make_plots(
                irf_name_map=p.irf_name_map,
                plot_n_time_units=p.plot_n_time_units,
                plot_n_points_per_time_unit=p.plot_n_points_per_time_unit,
                plot_x_inches=p.plot_x_inches,
                plot_y_inches=p.plot_y_inches,
                cmap=p.cmap
            )
            if p.network_type.startswith('bayes'):
                dtsr_model.make_plots(
                    irf_name_map=p.irf_name_map,
                    plot_n_time_units=p.plot_n_time_units,
                    plot_n_points_per_time_unit=p.plot_n_points_per_time_unit,
                    plot_x_inches=p.plot_x_inches,
                    plot_y_inches=p.plot_y_inches,
                    cmap=p.cmap,
                    mc=True
                )


