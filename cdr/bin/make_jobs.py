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
    argparser.add_argument('-j', '--job_type', default='fit', help='Type of job to run. One of ``["fit", "predict", "plot"]``')
    argparser.add_argument('-p', '--partition', nargs='+', help='Partition(s) over which to predict/evaluate')
    argparser.add_argument('-d', '--working_dir', type=str, default='/fs/project/schuler.77/shain.3/cdrnn', help='CDR working directory.')
    argparser.add_argument('-M', '--python_module', type=str, default='python/3.7-conda4.5', help='Python module to load')
    argparser.add_argument('-c', '--conda', type=str, default='cdr', help='Name of conda environment to load')
    argparser.add_argument('-t', '--time', type=int, default=48, help='Number of hours to train non-synth models.')
    argparser.add_argument('-T', '--time_synth', type=int, default=12, help='Number of hours to train synth models.')
    argparser.add_argument('-m', '--memory', type=int, default=64, help='Number of GB of memory to request')
    argparser.add_argument('-P', '--plot_cli', default='', help='CLI args to add to any plotting calls.')
    args = argparser.parse_args()

    paths = args.paths
    job_type = args.job_type
    partitions = args.partition
    working_dir = args.working_dir
    python_module = args.python_module
    conda = args.conda
    time = args.time
    time_synth = args.time_synth
    memory = args.memory
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
                if 'synth' in path:
                    time_cur = time_synth
                else:
                    time_cur = time
                f.write('#PBS -N %s\n' % job_name)
                f.write(base % (time_cur, memory, python_module, conda, working_dir))
                if job_type.lower() == 'fit':
                    f.write('python3 -m cdr.bin.train %s -m %s\n' % (path, m))
                if partitions and job_type.lower() in ['fit', 'predict']:
                    f.write('python3 -m cdr.bin.predict %s -p %s -m %s\n' % (path, ' '.join(partitions), m))
                if job_type.lower() == 'plot':
                    f.write('python3 -m cdr.bin.plot %s -m %s -p %s %s\n' % (path, basename, m, plot_cli))

    
