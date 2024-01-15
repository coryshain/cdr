import argparse

base = """#!/bin/bash
#
#SBATCH --job-name=%s
#SBATCH --output="%s-%%N-%%j.out"
#SBATCH --time=1:00:00
#SBATCH --mem=8gb
#SBATCH --ntasks=4
"""

# Wrapper is useful for running in singularity environments (you'll have to change the path to the image).
# If not using singularity, set wrapper to '%s' (no-op)
wrapper = '\nsingularity exec --nv ../singularity_images/tf-latest-gpu.simg bash -c "%s"'

if __name__ == '__main__':
    argparser = argparse.ArgumentParser('Generate SLURM batch jobs for plotting')
    argparser.add_argument('-i', '--ini_dir', default='pred_fn_ini', help='Path to directory containing model config (*ini) files')
    args = argparser.parse_args()

    ini_dir = args.ini_dir

    for cfg in ('brown.ini', 'dundee.ini', 'geco.ini', 'natstor.ini', 'natstormaze.ini', 'provo.ini'):
        for plot in ('', '_surf'):
            job_name = '%s_nosurp_plot%s' % (cfg[:-4], plot)
            job_str = 'python3 -m cdr.bin.plot %s/%s -m CDR_nosurp -c plot_config%s.ini' % (ini_dir, cfg, plot)
            job_str = wrapper % job_str
            job_str = base % (job_name, job_name) + job_str
            with open(job_name + '.pbs', 'w') as f:
                f.write(job_str)
        
            job_name = '%s_prob2surp_plot%s' % (cfg[:-4], plot)
            job_str = 'python3 -m cdr.bin.plot %s/%s -m CDR_gptprob_invert -c plot_config_prob2surp%s.ini' % (ini_dir, cfg, plot)
            job_str = wrapper % job_str
            job_str = base % (job_name, job_name) + job_str
            with open(job_name + '.pbs', 'w') as f:
                f.write(job_str)
        
            job_name = '%s_prob_plot%s' % (cfg[:-4], plot)
            job_str = 'python3 -m cdr.bin.plot %s/%s -m CDR_gptprob_invert -c plot_config%s.ini' % (ini_dir, cfg, plot)
            job_str = wrapper % job_str
            job_str = base % (job_name, job_name) + job_str
            with open(job_name + '.pbs', 'w') as f:
                f.write(job_str)
        
            job_name = '%s_square2surp_plot%s' % (cfg[:-4], plot)
            job_str = 'python3 -m cdr.bin.plot %s/%s -m CDR_gpt2.00_invert -c plot_config_square2surp%s.ini' % (ini_dir, cfg, plot)
            job_str = wrapper % job_str
            job_str = base % (job_name, job_name) + job_str
            with open(job_name + '.pbs', 'w') as f:
                f.write(job_str)
        
            job_name = '%s_square_plot%s' % (cfg[:-4], plot)
            job_str = 'python3 -m cdr.bin.plot %s/%s -m CDR_gpt2.00_invert -c plot_config%s.ini' % (ini_dir, cfg, plot)
            job_str = wrapper % job_str
            job_str = base % (job_name, job_name) + job_str
            with open(job_name + '.pbs', 'w') as f:
                f.write(job_str)
        
            job_name = '%s_surpprob_plot%s' % (cfg[:-4], plot)
            job_str = 'python3 -m cdr.bin.plot %s/%s -m CDR_gptsurpprob -c plot_config%s.ini' % (ini_dir, cfg, plot)
            job_str = wrapper % job_str
            job_str = base % (job_name, job_name) + job_str
            with open(job_name + '.pbs', 'w') as f:
                f.write(job_str)
    
            job_name = '%s_surpproblin_plot%s' % (cfg[:-4], plot)
            job_str = 'python3 -m cdr.bin.plot %s/%s -m CDR_gptsurpproblin -c plot_config%s.ini' % (ini_dir, cfg, plot)
            job_str = wrapper % job_str
            job_str = base % (job_name, job_name) + job_str
            with open(job_name + '.pbs', 'w') as f:
                f.write(job_str)
        
            job_name = '%s_gptpcfg_plot%s' % (cfg[:-4], plot)
            job_str = 'python3 -m cdr.bin.plot %s/%s -m CDR_gptpcfg -c plot_config%s.ini' % (ini_dir, cfg, plot)
            job_str = wrapper % job_str
            job_str = base % (job_name, job_name) + job_str
            with open(job_name + '.pbs', 'w') as f:
                f.write(job_str)
        
        for surp in ('ngram', 'pcfg', 'gpt', 'gptj', 'gpt3', 'cloze'):
            if cfg.startswith('provo') or surp != 'cloze':
                for suff in ('',):  #, 'prob_h0', '0.50_h0', '0.75_h0', '1.00_h0', '1.33_h0', '2.00_h0'):
                    for plot in ('', '_surf'):
                        job_name = '%s_%s%s_plot%s' % (cfg[:-4], surp, suff, plot)
                        job_str = 'python3 -m cdr.bin.plot %s/%s -m CDR_%s -c plot_config%s.ini' % (ini_dir, cfg, surp + suff, plot)
                        job_str = wrapper % job_str
                        job_str = base % (job_name, job_name) + job_str
                        with open(job_name + '.pbs', 'w') as f:
                            f.write(job_str)
    
                        job_name = '%s_%s%s_plot_alls' % (cfg[:-4], surp, suff)
                        job_str = 'python3 -m cdr.bin.plot %s/%s -m CDR_%s -c plot_config_alls_%s.ini' % (ini_dir, cfg, surp + suff, cfg[:-4])
                        job_str = wrapper % job_str
                        job_str = base % (job_name, job_name) + job_str
                        with open(job_name + '.pbs', 'w') as f:
                            f.write(job_str)
    
        # Normal error
        for surp in ('gpt',):
            for suff in ('',):
                for plot in ('', '_surf'):
                    job_name = '%s_normal_%s%s_plot%s' % (cfg[:-4], surp, suff, plot)
                    job_str = 'python3 -m cdr.bin.plot %s/%s_normal.ini -m CDR_%s -c plot_config%s.ini' % (ini_dir, cfg[:-4], surp + suff, plot)
                    job_str = wrapper % job_str
                    job_str = base % (job_name, job_name) + job_str
                    with open(job_name + '.pbs', 'w') as f:
                        f.write(job_str)
    
        # Word skipping
        if cfg in ('dundee.ini', 'geco.ini', 'provo.ini'):
            for surp in ('gpt',):
                for suff in ('',):
                    for plot in ('', '_surf'):
                        job_name = '%s_skip_%s%s_plot%s' % (cfg[:-4], surp, suff, plot)
                        job_str = 'python3 -m cdr.bin.plot %s/%s_skip.ini -m CDR_%s -c plot_config%s.ini' % (ini_dir, cfg[:-4], surp + suff, plot)
                        job_str = wrapper % job_str
                        job_str = base % (job_name, job_name) + job_str
                        with open(job_name + '.pbs', 'w') as f:
                            f.write(job_str)
    
