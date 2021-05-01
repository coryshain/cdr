import sys
import argparse
from cdr.config import Config

base = """#!/bin/bash
#
#SBATCH --job-name=%s
#SBATCH --output=%s.out
#SBATCH --time=%d:00:00
#SBATCH --ntasks=%d
#SBATCH --mem=%dgb
"""

 
if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Generate SLURM batch jobs to run CDR models specified in one or more config files.
    ''')
    argparser.add_argument('paths', nargs='+', help='Path(s) to CDR config file(s).')
    argparser.add_argument('-j', '--job_type', default='fit', help='Type of job to run. One of ``["fit", "predict", "summarize", "plot", "save_and_exit"]``')
    argparser.add_argument('-p', '--partition', nargs='+', help='Partition(s) over which to predict/evaluate')
    argparser.add_argument('-t', '--time', type=int, default=48, help='Maximum number of hours to train models')
    argparser.add_argument('-n', '--n_cores', type=int, default=8, help='Number of cores to request')
    argparser.add_argument('-m', '--memory', type=int, default=64, help='Number of GB of memory to request')
    argparser.add_argument('-P', '--slurm_partition', default=None, help='Value for SLURM --partition setting, if applicable')
    argparser.add_argument('-c', '--cli_args', default='', help='Command line arguments to pass into call')
    args = argparser.parse_args()

    paths = args.paths
    job_type = args.job_type
    partitions = args.partition
    time = args.time
    n_cores = args.n_cores
    memory = args.memory
    slurm_partition = args.slurm_partition
    cli_args = args.cli_args
   
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
            job_name = '_'.join([basename, job_type])
            filename = job_name + '.pbs'
            with open(filename, 'w') as f:
                f.write(base % (job_name, job_name, time, n_cores, memory))
                if slurm_partition:
                    f.write('#SBATCH --partition=%s\n' % slurm_partition)
                f.write('\n')
                if job_type.lower() == 'save_and_exit':
                    f.write('python3 -m cdr.bin.train %s -m %s -s -S %s\n' % (path, m, cli_args))
                elif job_type.lower() == 'fit':
                    f.write('python3 -m cdr.bin.train %s -m %s %s\n' % (path, m, cli_args))
                elif partitions and job_type.lower() in ['fit', 'predict']:
                    f.write('python3 -m cdr.bin.predict %s -p %s -m %s %s\n' % (path, ' '.join(partitions), m, cli_args))
                elif job_type.lower() == 'summarize':
                    f.write('python3 -m cdr.bin.summarize %s -m %s %s\n' % (path, m, cli_args))
                elif job_type.lower() == 'plot':
                    f.write('python3 -m cdr.bin.plot %s -m %s %s\n' % (path, m, cli_args))
                else:
                    raise ValueError('Unrecognized job type: %s.' % job_type)

    
