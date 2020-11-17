import sys
import argparse
from cdr.config import Config

base = """
#PBS -l walltime=%d:00:00
#PBS -l nodes=1:ppn=8
#PBS -l mem=64GB

module load python/3.7-conda4.5
source activate tf1.11
cd /fs/project/schuler.77/shain.3/cdrnn
"""

 
if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Generate PBS batch jobs to run CDR models specified in one or more config files.
    ''')
    argparser.add_argument('paths', nargs='+', help='Path(s) to CDR config file(s).')
    argparser.add_argument('-f', '--fit', action='store_true', help='Whether to fit the model to the training set')
    argparser.add_argument('-p', '--partition', nargs='+', help='Partition(s) over which to predict/evaluate')
    args = argparser.parse_args()

    paths = args.paths
    partitions = args.partition
   
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
            filename = basename + '_predict.pbs'
            with open(filename, 'w') as f:
                if 'synth' in path:
                    time = 12
                else:
                    time = 48
                f.write('#PBS -N %s\n' % basename)
                f.write(base % time)
                if args.fit:
                    f.write('python3 -m cdr.bin.train %s -m %s\n' % (path, m))
                if partitions:
                    f.write('python3 -m cdr.bin.predict %s -p %s -m %s\n' % (path, ' '.join(partitions), m))
    
