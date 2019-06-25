import argparse
import sys
import os
import pickle
from dtsr.config import Config
from dtsr.util import load_dtsr, filter_models, stderr

if __name__ == '__main__':

    argparser = argparse.ArgumentParser('''
        Compute root mean squared deviation of fitted IRFs from gold synthetic IRFs.
    ''')
    argparser.add_argument('paths', nargs='+', help='Path(s) to config file(s) defining experiments')
    argparser.add_argument('-m', '--models', nargs='*', default = [], help='Model names for which to compute RMSD. Regex permitted. If unspecified, uses all DTSR models.')
    argparser.add_argument('-S', '--summed', action='store_true', help='Use summed rather than individual IRFs.')
    argparser.add_argument('-u', '--ntimeunits', type=float, default=None, help='Number of time units over which to compute RMSD.')
    argparser.add_argument('-r', '--resolution', type=float, default=1000, help='Number of points to use for computing RMSD.')
    argparser.add_argument('-a', '--algorithm', type=str, default='MAP', help='Algorithm ("sampling" or "MAP") to use for extracting predictions from DTSRBayes. Ignored for DTSRMLE.')
    args, unknown = argparser.parse_known_args()

    for path in args.paths:
        p = Config(path)

        if not p.use_gpu_if_available:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        models = filter_models(p.model_list, args.models, dtsr_only=True)

        synth_path = os.path.dirname(os.path.dirname(p.X_train)) + '/d.obj'
        if not os.path.exists(synth_path):
            raise ValueError('Path to synth data %s does not exist. Check to make sure that model is fitted to synthetic data and that paths are correct in the config file.')
        with open(synth_path, 'rb') as f:
            d = pickle.load(f)
        def gold_irf_lambda(x):
            return d.irf(x, coefs=True)

        for m in models:
            p.set_model(m)
            formula = p.models[m]['formula']

            stderr('Retrieving saved model %s...\n' % m)
            dtsr_model = load_dtsr(p.outdir + '/' + m)

            stderr('Computing RMSD...\n')

            rmsd = dtsr_model.irf_rmsd(
                gold_irf_lambda,
                summed=args.summed,
                n_time_units=args.ntimeunits,
                n_time_points=args.resolution,
                algorithm=args.algorithm
            )

            summary = '=' * 50 + '\n'
            summary += 'DTSR regression\n\n'
            summary += 'Model name: %s\n\n' % m
            summary += 'Formula:\n'
            summary += '  ' + formula + '\n\n'
            summary += 'Path to synth model:\n'
            summary += '  ' + synth_path + '\n\n'
            summary += 'RMSD from gold: %s\n\n' % rmsd
            summary += '=' * 50 + '\n'

            if args.summed:
                out_name = 'synth_summed_rmsd'
            else:
                out_name = 'synth_rmsd'

            with open(p.outdir + '/' + m + '/' + out_name + '.txt', 'w') as f:
                f.write(summary)
            sys.stdout.write(summary)



