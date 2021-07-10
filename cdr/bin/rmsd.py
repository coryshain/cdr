import argparse
import sys
import os
import pickle
from cdr.config import Config
from cdr.util import load_cdr, filter_models, stderr

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
        Compute root mean squared deviation of fitted IRFs from gold synthetic IRFs.
    ''')
    argparser.add_argument('paths', nargs='+', help='Path(s) to config file(s) defining experiments')
    argparser.add_argument('-m', '--models', nargs='*', default = [], help='Model names for which to compute RMSD. Regex permitted. If unspecified, uses all CDR models.')
    argparser.add_argument('-S', '--summed', action='store_true', help='Use summed rather than individual IRFs.')
    argparser.add_argument('-u', '--ntimeunits', type=float, default=None, help='Number of time units over which to compute RMSD.')
    argparser.add_argument('-r', '--resolution', type=float, default=1000, help='Number of points to use for computing RMSD.')
    argparser.add_argument('-a', '--algorithm', type=str, default='MAP', help='Algorithm ("sampling" or "MAP") to use for extracting predictions from CDRBayes. Ignored for CDRMLE.')
    argparser.add_argument('--cpu_only', action='store_true', help='Use CPU implementation even if GPU is available.')
    args, unknown = argparser.parse_known_args()

    for path in args.paths:
        p = Config(path)

        if not p.use_gpu_if_available or args.cpu_only:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        models = filter_models(p.model_list, args.models, cdr_only=True)

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
            m_path = m.replace(':', '+')

            stderr('Retrieving saved model %s...\n' % m)
            cdr_model = load_cdr(p.outdir + '/' + m_path)

            stderr('Computing RMSD...\n')

            rmsd = cdr_model.irf_rmsd(
                gold_irf_lambda,
                summed=args.summed,
                n_time_units=args.ntimeunits,
                n_time_points=args.resolution,
                algorithm=args.algorithm
            )

            summary = '=' * 50 + '\n'
            summary += 'CDR regression\n\n'
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

            with open(p.outdir + '/' + m_path + '/' + out_name + '.txt', 'w') as f:
                f.write(summary)
            sys.stdout.write(summary)



