import sys
import argparse
try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources
from .. import templates

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Initialize a CDR(NN) config file template.
    ''')
    argparser.add_argument('-t', '--type', type=str, default='cdr', help='Type of model. One of ``["cdr", "plot"]``.')
    argparser.add_argument('-a', '--annotate', action='store_true', help='Print annotation comments in the output config file.')
    args = argparser.parse_args()

    if args.type.lower() == 'cdr':
        src_path = 'cdr_model_template.ini'
    elif args.type.lower() == 'plot':
        src_path = 'cdr_plot_template.ini'
    else:
        raise ValueError('Unrecognized config type: %s' % args.type)

    src_config = pkg_resources.open_text(templates, src_path)

    for line in src_config:
        if args.annotate or not line.startswith('#'):
            sys.stdout.write(line)