import sys
import os
import argparse

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Print usage for CDR executables.
    ''')
    argparser.add_argument('names', default=[], nargs='*', help='Names of executables to show.')
    args = argparser.parse_args()
    bin_dir = os.path.dirname(os.path.realpath(__file__))
    all_scripts = [x[:-3] for x in os.listdir(bin_dir) if x.endswith('py') and not (x.endswith('__init__.py') or x.endswith('help.py'))]

    if not args.names:
        scripts = all_scripts
    else:
        missing = []
        scripts = []
        for x in args.names:
            if x in all_scripts:
                scripts.append(x)
            else:
                missing.append(x)

        if missing:
            sys.stderr.write('Some requested scripts do not exist: %s. Skipping...\n' % ', '.join(missing))

    for s in scripts:
        print('='*50)
        print('PROGRAM NAME: %s' %s)
        print('')
        print('To run, use command:')
        print('python -m cdr.bin.%s [args]' %s)
        print('')
        os.system('python -m cdr.bin.%s -h' %s)
        print('')
