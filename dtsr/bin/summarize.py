import argparse
import sys
import os
from dtsr.config import Config
from dtsr.util import load_dtsr, filter_models

if __name__ == '__main__':

    argparser = argparse.ArgumentParser('''
        Generate summary of saved model(s)
    ''')
    argparser.add_argument('paths', nargs='+', help='Path(s) to config file(s) defining experiments')
    argparser.add_argument('-m', '--models', nargs='*', default = [], help='List of models for which to generate summaries. Regex permitted. If not, generates summaries for all DTSR models.')
    argparser.add_argument('-r', '--random', action='store_true', help='Report random effects (default True)')
    argparser.add_argument('-l', '--level', type=float, default = 95., help='Level (in percent) for any credible intervals (DTSRBayes only).')
    argparser.add_argument('-n', '--nsamples', type=int, default = None, help='Number of MC samples to use for computing statistics (DTSRBayes only). If unspecified, uses model default (**n_samples_eval** parameter).')
    argparser.add_argument('-t', '--timeunits', type=float, default = None, help='Number of time units over which to compute effect size integrals. If unspecified, uses longest timespan attested in training.')
    argparser.add_argument('-p', '--prefix', type=str, default = None, help='String to prepend to output file.')
    args, unknown = argparser.parse_known_args()

    for path in args.paths:
        p = Config(path)

        if not p.use_gpu_if_available:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        models = filter_models(p.model_list, args.models, dtsr_only=True)

        for m in models:

            sys.stderr.write('Retrieving saved model %s...\n' % m)
            dtsr_model = load_dtsr(p.outdir + '/' + m)

            summary = dtsr_model.summary(
                random=args.random,
                level=args.level,
                n_samples=args.nsamples,
                integral_n_time_units=args.timeunits
            )

            if args.prefix:
                outname = p.outdir + '/' + m + '/' + args.prefix + '_summary.txt'
            else:
                outname = p.outdir + '/' + m + '/summary.txt'

            sys.stderr.write('Saving summary to %s' %outname)
            with open(outname, 'w') as f:
                f.write(summary)
            sys.stderr.write(summary)

            dtsr_model.finalize()



