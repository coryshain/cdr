import sys
import argparse
from cdr.config import Config

base = """#!/bin/bash
#
#SBATCH --job-name=%s
#SBATCH --time=%d:00:00
#SBATCH --ntasks=8
#SBATCH --mem=%dgb
"""

 
if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Generate SLURM batch jobs to run CDR models specified in one or more config files.
    ''')
    argparser.add_argument('paths', nargs='+', help='Path(s) to CDR config file(s).')
    argparser.add_argument('-j', '--job_type', default='fit', help='Type of job to run. One of ``["fit", "predict", "plot", "save_and_exit"]``')
    argparser.add_argument('-p', '--partition', nargs='+', help='Partition(s) over which to predict/evaluate')
    argparser.add_argument('-t', '--time', type=int, default=48, help='Maximum number of hours to train models')
    argparser.add_argument('-m', '--memory', type=int, default=64, help='Number of GB of memory to request')
    argparser.add_argument('-P', '--slurm_partition', default=None, help='Value for SLURM --partition setting, if applicable')
    argparser.add_argument('-C', '--plot_cli', default='', help='CLI args to add to any plotting calls')
    args = argparser.parse_args()

    paths = args.paths
    job_type = args.job_type
    partitions = args.partition
    time = args.time
    memory = args.memory
    slurm_partition = args.slurm_partition
    plot_cli = args.plot_cli
   
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
                f.write(base % (job_name, time, memory))
                if slurm_partition:
                    f.write('#SBATCH --partition=%s\n' % slurm_partition)
                f.write('\n')
                if job_type.lower() == 'save_and_exit':
                    f.write('python3 -m cdr.bin.train %s -m %s -s\n' % (path, m))
                if job_type.lower() == 'fit':
                    f.write('python3 -m cdr.bin.train %s -m %s\n' % (path, m))
                if partitions and job_type.lower() in ['fit', 'predict']:
                    f.write('python3 -m cdr.bin.predict %s -p %s -m %s\n' % (path, ' '.join(partitions), m))
                if job_type.lower() == 'plot':
                    f.write('python3 -m cdr.bin.plot %s -m %s -p %s %s\n' % (path, basename, m, plot_cli))

    
