import sys
import os
import argparse
from cdr.config import Config

base = """#!/bin/bash
#
#SBATCH --job-name=%s
#SBATCH --output="%s-%%N-%%j.out"
#SBATCH --time=%d:00:00
#SBATCH --mem=%dgb
#SBATCH --ntasks=%d
"""

 
if __name__ == '__main__':
    argparser = argparse.ArgumentParser('''
    Generate SLURM batch jobs to run CDR models specified in one or more config files.
    ''')
    argparser.add_argument('paths', nargs='+', help='Path(s) to CDR config file(s).')
    argparser.add_argument('-j', '--job_types', nargs='+', default=['fit'], help='Type of job to run. List of ``["fit", "predict", "summarize", "plot", "save_and_exit"]``')
    argparser.add_argument('-p', '--partition', nargs='+', help='Partition(s) over which to predict/evaluate')
    argparser.add_argument('-t', '--time', type=int, default=48, help='Maximum number of hours to train models')
    argparser.add_argument('-n', '--n_cores', type=int, default=4, help='Number of cores to request')
    argparser.add_argument('-g', '--use_gpu', action='store_true', help='Whether to request a GPU node')
    argparser.add_argument('-m', '--memory', type=int, default=64, help='Number of GB of memory to request')
    argparser.add_argument('-P', '--slurm_partition', default=None, help='Value for SLURM --partition setting, if applicable')
    argparser.add_argument('-e', '--exclude', nargs='+', help='Nodes to exclude')
    argparser.add_argument('-c', '--cli_args', default='', help='Command line arguments to pass into call')
    argparser.add_argument('-s', '--singularity_path', default='', help='Path to singularity image to invoke before running')
    argparser.add_argument('-o', '--outdir', default='./', help='Directory in which to place generated batch scripts.')
    args = argparser.parse_args()

    paths = args.paths
    job_types = args.job_types
    partitions = args.partition
    time = args.time
    n_cores = args.n_cores
    use_gpu = args.use_gpu
    memory = args.memory
    slurm_partition = args.slurm_partition
    if args.exclude:
        exclude = ','.join(args.exclude)
    else:
        exclude = []
    cli_args = args.cli_args.replace('\\', '') # Delete escape characters
    singularity_path = args.singularity_path
    outdir = args.outdir

    if not os.path.exists(outdir):
        os.makedirs(outdir)
   
    for path in paths:
        c = Config(path)

        models = c.model_names

        for m in models:
            start_ix = -1
            basename = '_'.join(path[:-4].split('/')[start_ix:] + [m])
            job_name = '_'.join([basename, ''.join(job_types)])
            filename = outdir + '/' + job_name + '.pbs'
            with open(filename, 'w') as f:
                f.write(base % (job_name, job_name, time, memory, n_cores))
                if use_gpu:
                    f.write('#SBATCH --gres=gpu:1\n')
                if slurm_partition:
                    f.write('#SBATCH --partition=%s\n' % slurm_partition)
                if exclude:
                    f.write('#SBATCH --exclude=%s\n' % exclude)
                wrapper = '%s'
                if singularity_path:
                    if use_gpu:
                        wrapper = wrapper % ('singularity exec --nv %s bash -c "cd %s; %%s"\n' % (singularity_path, os.getcwd()))
                    else:
                        wrapper = wrapper % ('singularity exec %s bash -c "cd %s; %%s"\n' % (singularity_path, os.getcwd()))
                for job_type in job_types:
                    if job_type.lower() == 'save_and_exit':
                        job_str = wrapper % ('python3 -m cdr.bin.train %s -m %s -s -S %s' % (path, m, cli_args))
                    elif job_type.lower() == 'fit':
                        job_str = wrapper % ('python3 -m cdr.bin.train %s -m %s %s' % (path, m, cli_args))
                    elif partitions and job_type.lower() in ['fit', 'predict']:
                        job_str = wrapper % ('python3 -m cdr.bin.predict %s -p %s -m %s %s' % (path, ' '.join(partitions), m, cli_args))
                    elif job_type.lower() == 'summarize':
                        job_str = wrapper % ('python3 -m cdr.bin.summarize %s -m %s %s' % (path, m, cli_args))
                    elif job_type.lower() == 'plot':
                        job_str = wrapper % ('python3 -m cdr.bin.plot %s -m %s %s' % (path, m, cli_args))
                    else:
                        raise ValueError('Unrecognized job type: %s.' % job_type)
                    f.write(job_str)

