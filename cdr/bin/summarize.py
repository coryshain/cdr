import argparse
import os
from cdr.config import Config
from cdr.model import CDREnsemble
from cdr.util import filter_models, stderr

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
        Generate summary of saved model(s)
    ''')
    argparser.add_argument('paths', nargs='+', help='Path(s) to config file(s) defining experiments')
    argparser.add_argument('-m', '--models', nargs='*', default=[], help='List of models for which to generate summaries. Regex permitted. If not, generates summaries for all CDR models.')
    argparser.add_argument('-r', '--random', action='store_true', help='Report random effects.')
    argparser.add_argument('-l', '--level', type=float, default=95., help='Level (in percent) for any credible intervals.')
    argparser.add_argument('-n', '--n_samples', default='default', help='Number of MC samples to use for computing statistics. If unspecified, uses model default (**n_samples_eval** parameter).')
    argparser.add_argument('-t', '--timeunits', type=float, default=None, help='Number of time units over which to compute effect size integrals. If unspecified, uses longest timespan attested in training.')
    argparser.add_argument('-T', '--save_table', action='store_true', help='Save CSV table of model parameters.')
    argparser.add_argument('-p', '--prefix', type=str, default=None, help='String to prepend to output file.')
    argparser.add_argument('--cpu_only', action='store_true', help='Use CPU implementation even if GPU is available.')
    args = argparser.parse_args()

    if args.n_samples == 'default':
        n_samples = args.n_samples
    elif args.n_samples == 'None':
        n_samples = None
    else:
        n_samples = int(args.n_samples)

    for path in args.paths:
        p = Config(path)

        if not p.use_gpu_if_available or args.cpu_only:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        model_list = sorted(set(p.model_names) | set(p.ensemble_names))
        models = filter_models(model_list, args.models)

        for m in models:
            m_path = m.replace(':', '+')

            stderr('Retrieving saved model %s...\n' % m)
            cdr_model = CDREnsemble(p.outdir, m_path)

            stderr('Resampling summary statistics...\n')
            summary = cdr_model.summary(
                random=args.random,
                level=args.level,
                n_samples=n_samples,
                integral_n_time_units=args.timeunits
            )

            outdir = os.path.join(os.path.normpath(p.outdir), m_path)
            if args.prefix:
                filename = args.prefix + '_summary.txt'
            else:
                filename = 'summary.txt'
            outname = os.path.join(outdir, filename)

            stderr('Saving summary to %s' % outname)
            if not os.path.exists(outdir):
                os.makedirs(outdir)
            with open(outname, 'w') as f:
                f.write(summary)
            stderr(summary)

            if args.save_table:
                if args.prefix:
                    outname = p.outdir + '/' + m_path + '/' + args.prefix + '_cdr_parameters.csv'
                else:
                    outname = p.outdir + '/' + m_path + '/cdr_parameters.csv'

                cdr_model.save_parameter_table(level=args.level, n_samples=n_samples, outfile=outname)
                
                if args.prefix:
                    outname = p.outdir + '/' + m_path + '/' + args.prefix + '_cdr_irf_integrals.csv'
                else:
                    outname = p.outdir + '/' + m_path + '/cdr_irf_integrals.csv'

                cdr_model.save_integral_table(level=args.level, n_samples=n_samples, outfile=outname)

            cdr_model.finalize()
