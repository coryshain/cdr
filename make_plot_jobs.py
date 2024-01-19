import os
import argparse
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('Generate batch scripts for plotting')
    argparser.add_argument('-i', '--ini_dir', default=None, help='Path to directory containing model config (*.ini) files.')
    args = argparser.parse_args()

    assert os.path.exists('config.yml'), 'Repository has not yet been initialized. First run `python -m initialize`.'
    with open('config.yml', 'r') as f:
        repo_cfg = yaml.load(f, Loader=Loader)

    ini_dir = args.ini_dir
    if ini_dir is None:
        ini_dir = repo_cfg['ini_path']

    base = """#!/bin/bash
#
#SBATCH --job-name=%s
#SBATCH --output="%s-%%N-%%j.out"
#SBATCH --time=1:00:00
#SBATCH --mem=8gb
#SBATCH --ntasks=4
#SBATCH --partition=evlab
"""
    
    # Change this to reflect how your system runs bash executables
    if repo_cfg['singularity_path']:
        wrapper = '\nsingularity exec --nv %s bash -c "%%s"' % repo_cfg['singularity_path']
    else:
        wrapper = '\n%s'
    
    for cfg in ('brown.ini', 'dundee.ini', 'geco.ini', 'natstor.ini', 'natstormaze.ini', 'provo.ini'):
        for plot in ('', '_surf'):
            for response in ('', '_mu', '_sigma', '_beta'):
                job_name = '%s_main_plot%s%s' % (cfg[:-4], plot, response)
                job_str = 'python3 -m cdr.bin.plot %s/%s -m main -c plot_config%s%s.ini' % (ini_dir, cfg, plot, response)
                job_str = wrapper % job_str
                job_str = base % (job_name, job_name) + job_str
                with open(job_name + '.pbs', 'w') as f:
                    f.write(job_str)
    
                if not plot:
                    for iv in ('_freq', '_pred'):
                        job_name = '%s_main_plot%s%s%s' % (cfg[:-4], plot, response, iv)
                        job_str = 'python3 -m cdr.bin.plot %s/%s -m main -c plot_config%s%s%s.ini' % (ini_dir, cfg, plot, response, iv)
                        job_str = wrapper % job_str
                        job_str = base % (job_name, job_name) + job_str
                        with open(job_name + '.pbs', 'w') as f:
                            f.write(job_str)

            _plot = plot
            if not _plot:
                _plot = '2'    

            job_name = '%s_bigram_plot%s' % (cfg[:-4], plot)
            job_str = 'python3 -m cdr.bin.plot %s/%s -m bigram -c plot_config%s2.ini' % (ini_dir, cfg, _plot)
            job_str = wrapper % job_str
            job_str = base % (job_name, job_name) + job_str
            with open(job_name + '.pbs', 'w') as f:
                f.write(job_str)
        
            job_name = '%s_trigram_plot%s' % (cfg[:-4], plot)
            job_str = 'python3 -m cdr.bin.plot %s/%s -m trigram -c plot_config%s2.ini' % (ini_dir, cfg, _plot)
            job_str = wrapper % job_str
            job_str = base % (job_name, job_name) + job_str
            with open(job_name + '.pbs', 'w') as f:
                f.write(job_str)
    
            if not plot:
                job_name = '%s_bigram_plot%s_bigram' % (cfg[:-4], plot)
                job_str = 'python3 -m cdr.bin.plot %s/%s -m bigram -c plot_config%s_bigram.ini' % (ini_dir, cfg, plot)
                job_str = wrapper % job_str
                job_str = base % (job_name, job_name) + job_str
                with open(job_name + '.pbs', 'w') as f:
                    f.write(job_str)
            
                job_name = '%s_trigram_plot%s_trigram' % (cfg[:-4], plot)
                job_str = 'python3 -m cdr.bin.plot %s/%s -m trigram -c plot_config%s_trigram.ini' % (ini_dir, cfg, plot)
                job_str = wrapper % job_str
                job_str = base % (job_name, job_name) + job_str
                with open(job_name + '.pbs', 'w') as f:
                    f.write(job_str)

        
        job_name = '%s_mainfig_plot' % (cfg[:-4])
        job_str = 'python3 -m cdr.bin.plot_mainfig %s/%s' % (ini_dir, cfg)
        job_str = wrapper % job_str
        job_str = base % (job_name, job_name) + job_str
        with open(job_name + '.pbs', 'w') as f:
            f.write(job_str)
     
