import sys
import argparse
from cdr.config import Config

base = """
#PBS -l walltime=%d:00:00
#PBS -l nodes=1:ppn=8
#PBS -l mem=%dGB

module load %s
source activate %s
cd %s
"""

 
if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Generate PBS batch jobs to run CDR models specified in one or more config files.
    ''')
    argparser.add_argument('paths', nargs='+', help='Path(s) to CDR config file(s).')
    argparser.add_argument('-f', '--fit', action='store_true', help='Whether to fit the model to the training set')
    argparser.add_argument('-p', '--partition', nargs='+', help='Partition(s) over which to predict/evaluate')
    argparser.add_argument('-d', '--working_dir', type=str, default='/fs/project/schuler.77/shain.3/cdrnn', help='CDR working directory.')
    argparser.add_argument('-P', '--python_module', type=str, default='python/3.7-conda4.5', help='Python module to load')
    argparser.add_argument('-c', '--conda', type=str, default='cdr', help='Name of conda environment to load')
    argparser.add_argument('-m', '--memory', type=int, default=64, help='Number of GB of memory to request')
    args = argparser.parse_args()

    paths = args.paths
    fit = args.fit
    partitions = args.partition
    working_dir = args.working_dir
    python_module = args.python_module
    conda = args.conda
    memory = args.memory
   
    for path in paths:
        c = Config(path)
        outdir = c.outdir
    
        models = c.model_list
    
        for m in models:
            if 'synth' in path:
                start_ix = -2
            else:
                start_ix = -1
            basename = '_'.join(path[:-4].split('/')[start_ix:] + [m])
            filename = basename + '.pbs'
            with open(filename, 'w') as f:
                if 'synth' in path:
                    time = 12
                else:
                    time = 48
                f.write('#PBS -N %s\n' % basename)
                f.write(base % (time, memory, python_module, conda, working_dir))
                if fit:
                    f.write('python3 -m cdr.bin.train %s -m %s\n' % (path, m))
                if partitions:
                    f.write('python3 -m cdr.bin.predict %s -p %s -m %s\n' % (path, ' '.join(partitions), m))
    
