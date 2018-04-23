import os

if __name__ == '__main__':

    bin_dir = os.path.dirname(os.path.realpath(__file__))
    scripts = [x[:-3] for x in os.listdir(bin_dir) if x.endswith('py') and not (x.endswith('__init__.py') or x.endswith('help.py'))]


    for s in scripts:
        print('='*50)
        print('PROGRAM NAME: %s' %s)
        print('')
        print('To run, use command:')
        print('python -m dtsr.bin.%s [args]' %s)
        print('')
        os.system('python -m dtsr.bin.%s -h' %s)
        print('')
