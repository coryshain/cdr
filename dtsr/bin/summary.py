import argparse
import sys
import os
from dtsr.config import Config
from dtsr.util import load_dtsr, filter_models

if __name__ == '__main__':

    argparser = argparse.ArgumentParser('''
        Generates predictions from data given saved model(s)
    ''')
    argparser.add_argument('paths', nargs='+', help='Path(s) to config file(s) defining experiments')
    argparser.add_argument('-m', '--models', nargs='*', default = [], help='List of models for which to generate summaries. Regex permitted. If not, generates summaries for all DTSR models.')
    argparser.add_argument('-f', '--fixed', type=bool, default = True, help='Report fixed effects (default True)')
    argparser.add_argument('-r', '--random', type=bool, default = False, help='Report random effects (default True)')
    argparser.add_argument('-n', '--nsample', type=int, default = 1024, help='Number of MC samples to use for computing statistics (DTSRBayes only)')
    args, unknown = argparser.parse_known_args()

    for path in args.paths:
        p = Config(path)

        if not p.use_gpu_if_available:
            os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

        models = filter_models(p.model_list, args.models, dtsr_only=True)

        for m in models:

            sys.stderr.write('Retrieving saved model %s...\n' % m)
            dtsr_model = load_dtsr(p.outdir + '/' + m)

            summary = dtsr_model.summary(fixed=args.fixed, random=args.random)

            with open(p.outdir + '/' + m + '/summary.txt', 'w') as f:
                f.write(summary)
            sys.stderr.write(summary)

            dtsr_model.finalize()



