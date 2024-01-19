import os
import numpy as np
import argparse

from cdr.config import Config
from cdr.model import CDREnsemble
from cdr.util import filter_models, stderr, get_irf_name
from cdr.plot import plot_irf

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('Plot interactions for main paper figure.')
    argparser.add_argument('paths', nargs='+', help='Path(s) to config file(s) defining experiments')
    args = argparser.parse_args()

    for path in args.paths:
        p = Config(path)

        models = ['freqIRF', 'predIRF', 'main', 'bigram', 'trigram']
        _model = None
        for m_ix, model in enumerate(models):
            if model not in ('predIRF', 'main'):
                if model.endswith('IRF'):
                    _model = 'main'
                else:
                    _model = model
                model_path = _model.replace(':', '+')
                p.set_model(_model)
    
                stderr('Retrieving saved model %s...\n' % _model)
                cdr_model = CDREnsemble(p.outdir, model_path)
                cdr_model.set_weight_type('uniform')
                cdr_model.set_predict_mode(True)
                irf_name_map = cdr_model.irf_name_map

            stderr('Plotting...\n')

            if model == 'main':
                main = 'unigramsurpOWT'
                control = 'gpt'
            else:
                control = None
                if model == 'bigram':
                    main = 'fwprobopenwebtext2surp'
                elif model == 'trigram':
                    main = 'fwprobopenwebtext3surp'
                elif model == 'freqIRF':
                    main = 'unigramsurpOWT'
                else:  # model == 'predIRF'
                    main = 'gpt'
            responses = cdr_model.response_names
            params = ['mean', 'mu', 'sigma', 'beta']
            for i in range(1 + (control is not None)):
                # Get curvature plots at different control values
                x = None
                if control is not None:
                    control_name = get_irf_name(control, cdr_model.irf_name_map)
                    control_mean = cdr_model.impulse_means[control]
                    control_sd = cdr_model.impulse_sds[control]
                    control_below = control_mean - control_sd
                    control_above = control_mean + control_sd
                    X_refs = {
                        '%s = %0.1f' % (control_name, control_mean): {control: control_mean},
                        '%s = %0.1f' % (control_name, control_below): {control: control_below},
                        '%s = %0.1f' % (control_name, control_above): {control: control_above},
                    }
                else:
                    control_name = ''
                    X_refs = {control_name: None}

                if model.endswith('IRF'):
                    xvar = 't_delta'
                    manipulations = [{main: cdr_model.impulse_sds[main]}]
                    pair_manipulations = False
                    ref_varies_with_x = True
                else:
                    xvar = main
                    manipulations = None
                    pair_manipulations = True
                    ref_varies_with_x = False

                plot_data = {}
                for X_ref in X_refs:
                    _x, m, lq, uq, _ = cdr_model.get_plot_data(
                        xvar=xvar,
                        responses=responses,
                        response_params=params,
                        ref_varies_with_x=ref_varies_with_x,
                        manipulations=manipulations,
                        pair_manipulations=pair_manipulations,
                        X_ref = X_refs[X_ref],
                        n_samples = 1000,
                        level = 95
                    )
                    if x is None:
                        x = _x
                    if model.endswith('IRF'):
                        for response in m:
                            for param in m[response]:
                                m[response][param] = m[response][param][:, 1:]
                                lq[response][param] = lq[response][param][:, 1:]
                                uq[response][param] = uq[response][param][:, 1:]
                    plot_data[X_ref] = (m, lq, uq)
                for response in responses:
                    for param in params:
                        names = list(X_refs.keys())
                        y = np.concatenate(
                            [plot_data[name][0][response][param] for name in names],
                            axis=-1
                        )
                        lq = np.concatenate(
                            [plot_data[name][1][response][param] for name in names],
                            axis=-1
                        )
                        uq = np.concatenate(
                            [plot_data[name][2][response][param] for name in names],
                            axis=-1
                        )
                        ymin, ymax = lq.min(), uq.max()
                        yrange = ymax - ymin
                        pad = 0.05 * yrange
                        ymin -= pad
                        ymax += pad
                        ylim = (ymin, ymax)
                        if model == 'main':
                            plot_x_inches = 2.8
                            plot_y_inches = 3
                            legend = True
                            prop_cycle_length = 5
                            prop_cycle_map = [1, 2, 3]
                            if i == 0:
                                cmap='winter_r'
                            else:
                                cmap='autumn_r'
                            xlab = main
                        else:
                            plot_x_inches = 2
                            plot_y_inches = 2
                            legend = False
                            prop_cycle_length = 10
                            if model == 'bigram':
                                cmap = 'Purples'
                                xlab = main
                                prop_cycle_map = [9]
                            elif model == 'trigram':
                                cmap = 'Greens'
                                xlab = main
                                prop_cycle_map = [9]
                            elif model == 'freqIRF':
                                cmap='winter_r'
                                xlab = 'Delay (s)'
                                prop_cycle_map = [5]
                            else:  # model == 'predIRF'
                                cmap='autumn_r'
                                xlab = 'Delay (s)'
                                prop_cycle_map = [5]
                        
                        plot_irf(
                            x,
                            y,
                            names,
                            lq=lq,
                            uq=uq,
                            outdir=os.path.join(p.outdir, _model),
                            filename='%s_%s_%s_%s_mainfig.pdf' % (os.path.basename(path)[:-4], response, model, param),
                            irf_name_map=cdr_model.irf_name_map,
                            prop_cycle_length=prop_cycle_length,
                            prop_cycle_map=prop_cycle_map,
                            cmap=cmap,
                            xlab=xlab,
                            ylab=response,
                            ylim=ylim,
                            use_grid=False,
                            use_fill=False,
                            use_bottom_spine=True,
                            use_left_spine=True,
                            plot_x_inches=plot_x_inches,
                            plot_y_inches=plot_y_inches,
                            legend=legend,
                            legend_above=True
                        )
                if control is not None:
                    main, control = control, main


            if model not in ['freqIRF', 'predIRF'] or m_ix == len(models) - 1:
                cdr_model.finalize()

