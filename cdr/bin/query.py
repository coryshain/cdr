import argparse
import os
import numpy as np
import pandas as pd
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

from cdr.config import Config
from cdr.model import CDREnsemble
from cdr.plot import plot_irf, plot_surface
from cdr.util import filter_models, stderr, sn


def run_contrast(contrast, estimates):
    if isinstance(contrast, str):
        out = {}
        for response in estimates:
            out[response] = {}
            for dim in estimates[response]:
                out[response][dim] = estimates[response][dim][contrast]
        return out

    assert len(contrast) == 2, 'Must be exactly two items to compare. Got %d.' % len(contrast)
    out = {}
    val1 = run_contrast(contrast[0], estimates)
    val2 = run_contrast(contrast[1], estimates)
    for response in val1:
        out[response] = {}
        for dim in val1[response]:
            out[response][dim] = val1[response][dim] - val2[response][dim]

    return out


if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
        Query arbitrary scalar estimates from saved model(s).
        **query_path** points to a YAML file that contains at minimum the fields ``estimates`` and ``contrasts``,
        where ``estimates`` is a dictionary mapping short names for estimates to dictionaries defining input
        manipulations to query, in the format required by the **manipulations** kwarg of ``get_plot_data``.
        ``contrasts`` is a dictionary mapping keys to binary trees of effect contrasts represented in YAML
        as nested pairs, where the second child of each branch is subtracted from the first.
        For example, in the following definition::
          contrasts:
            effect1:
              - - query1a
                - query1b
              - - query2a
                - query2b
        the leaves of the tree stand for names of estimates in the ``estimates`` dictionary, and the nesting structure
        defines subtractive contrasts. First, query1b is subtracted from query1a, and query2b is subtracted from query2a.
        Then the result of the second subtraction is subtracted from the result of the first, yielding the contrast
        named ``effect1``. Any remaining material in the YAML file will be passed as kwargs to ``get_plot_data``.
    ''')
    argparser.add_argument('config_paths', nargs='+', help='Path(s) to config file(s) defining experiments.')
    argparser.add_argument('query_path', help='Path to YAML file defining query parameters (i.e., kwargs for the get_plot_data method).')
    argparser.add_argument('-m', '--models', nargs='*', default = [], help='Model names to plot. Regex permitted. If unspecified, plots all CDR models.')
    argparser.add_argument('-p', '--plot', action='store_true', help='Additionally dump the resulting plots for each contrast.')
    argparser.add_argument('-C', '--cpu_only', action='store_true', help='Use CPU implementation even if GPU is available.')
    args = argparser.parse_args()

    config_paths = args.config_paths
    query_path = args.query_path
    plot = args.plot

    with open(query_path, 'r') as f:
        query = yaml.load(f, Loader=Loader)
    assert 'estimates' in query, 'query must contain a top-level field named "estimates"'
    estimates = query['estimates']
    del query['estimates']
    if 'contrasts' in query:
        contrasts = query['contrasts']
        del query['contrasts']
    else:
        contrasts = {}
    kwargs = query
    query_name = os.path.basename(query_path)[:-4]

    manip_names = sorted(list(estimates.keys()))
    manipulations = [estimates[x] for x in manip_names]
    kwargs['manipulations'] = manipulations
    is3d = 'yvar' in kwargs and kwargs['yvar'] is not None
    if 'level' in query:
        level = query['level']
    else:
        level = 95
    alpha = 1 - level / 100
    lq = alpha / 2
    uq = 1 - alpha / 2

    for config_path in config_paths:
        p = Config(config_path)

        if not p.use_gpu_if_available or args.cpu_only:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        model_list = sorted(set(p.model_names) | set(p.ensemble_names))
        models = filter_models(model_list, args.models, cdr_only=True)

        for m in models:
            m_path = m.replace(':', '+')
            p.set_model(m)

            stderr('Retrieving saved model %s...\n' % m)
            cdr_model = CDREnsemble(p.outdir, m_path)
            cdr_model.set_predict_mode(True)
            outdir = os.path.normpath(cdr_model.outdir)
            if not os.path.exists(outdir):
                os.makedirs(outdir)

            stderr('Querying...\n')

            plot_ax, _, _, _, samples = cdr_model.get_plot_data(**kwargs)
            for response in samples:
                for dim in samples[response]:
                    s = samples[response][dim][..., 1:]
            if is3d:
                plot_x, plot_y = plot_ax
                step_x = plot_x[1, 0] - plot_x[0, 0]
                step_y = plot_y[0, 1] - plot_y[0, 0]
            else:
                plot_x = plot_ax
                step_x = plot_x[1] - plot_x[0]
                plot_y = None
                step_y = 0.
            for response in samples:
                for dim in samples[response]:
                    samples[response][dim] = {
                        x: samples[response][dim][..., i+1] for i, x in enumerate(manip_names) #  i+1 skips the baseline query
                    }

            out = []
            contrast_plots = {}
            for contrast in contrasts:
                val = run_contrast(contrasts[contrast], samples)
                for response in val:
                    if response not in contrast_plots:
                        contrast_plots[response] = {}
                    for dim in val[response]:
                        if dim not in contrast_plots[response]:
                            contrast_plots[response][dim] = {}
                        x = val[response][dim]
                        contrast_plots[response][dim][contrast] = x
                        x = x.sum(axis=1) * step_x
                        if is3d:
                            x = x.sum(axis=1) * step_y
                        lower, upper = np.quantile(x, [lq, uq])
                        mean = x.mean()
                        out.append((contrast, response, dim, mean, lower, upper))

            out = pd.DataFrame(
                out,
                columns=['Name', 'Response', 'ResponseParam', 'Mean', '%0.1f%%' % ((alpha / 2) * 100), '%0.1f%%' % (100 - (alpha / 2) * 100)]
            )

            outpath = os.path.join(outdir, query_name + '.csv')
            out.to_csv(outpath, index=False)
            print(out)

            if plot:
                for response in contrast_plots:
                    for dim in contrast_plots[response]:
                        x = []
                        lower = []
                        upper = []
                        contrast_names = sorted(contrast_plots[response][dim].keys())
                        for contrast in contrast_names:
                            _x = contrast_plots[response][dim][contrast]
                            x.append(_x.mean(axis=0))
                            _lower, _upper = np.quantile(_x, [lq, uq], axis=0)
                            lower.append(_lower)
                            upper.append(_upper)
                        if len(x):
                            x = np.stack(x, axis=-1)
                            lower = np.stack(lower, axis=-1)
                            upper = np.stack(upper, axis=-1)
                            if is3d:
                                plot_surface(
                                    plot_x,
                                    plot_y,
                                    x,
                                    lq=lower,
                                    uq=upper,
                                    outdir=outdir,
                                    filename='%s_%s_%s.png' % (query_name, sn(response), sn(dim))
                                )
                            else:
                                plot_irf(
                                    plot_x,
                                    x,
                                    contrast_names,
                                    lq=lower,
                                    uq=upper,
                                    sort_names=True,
                                    outdir=outdir,
                                    filename='%s_%s_%s.png' % (query_name, sn(response), sn(dim))
                                )

            cdr_model.set_predict_mode(False)
            cdr_model.finalize()




