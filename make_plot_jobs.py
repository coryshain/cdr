base = """#!/bin/bash
#
#SBATCH --job-name=%s
#SBATCH --output="%s-%%N-%%j.out"
#SBATCH --time=1:00:00
#SBATCH --mem=8gb
#SBATCH --ntasks=4
"""

# Change if needed in order to work in your local runtime environment
wrapper = '\nsingularity exec --nv ../singularity_images/tf-latest-gpu.simg bash -c "%s"'

for cfg in ('brown.ini', 'dundee.ini', 'geco.ini', 'natstor.ini', 'natstormaze.ini', 'provo.ini'):
    for plot in ('', '_surf'):
        job_name = '%s_nosurp_plot%s' % (cfg[:-4], plot)
        job_str = 'python3 -m cdr.bin.plot pred_fn_ini/%s -m CDR_nosurp -c plot_config%s.ini' % (cfg, plot)
        job_str = wrapper % job_str
        job_str = base % (job_name, job_name) + job_str
        with open(job_name + '.pbs', 'w') as f:
            f.write(job_str)
    
        job_name = '%s_prob2surp_plot%s' % (cfg[:-4], plot)
        job_str = 'python3 -m cdr.bin.plot pred_fn_ini/%s -m CDR_gptprob_invert -c plot_config_prob2surp%s.ini' % (cfg, plot)
        job_str = wrapper % job_str
        job_str = base % (job_name, job_name) + job_str
        with open(job_name + '.pbs', 'w') as f:
            f.write(job_str)
    
        job_name = '%s_prob_plot%s' % (cfg[:-4], plot)
        job_str = 'python3 -m cdr.bin.plot pred_fn_ini/%s -m CDR_gptprob_invert -c plot_config%s.ini' % (cfg, plot)
        job_str = wrapper % job_str
        job_str = base % (job_name, job_name) + job_str
        with open(job_name + '.pbs', 'w') as f:
            f.write(job_str)
    
        job_name = '%s_square2surp_plot%s' % (cfg[:-4], plot)
        job_str = 'python3 -m cdr.bin.plot pred_fn_ini/%s -m CDR_gpt2.00_invert -c plot_config_square2surp%s.ini' % (cfg, plot)
        job_str = wrapper % job_str
        job_str = base % (job_name, job_name) + job_str
        with open(job_name + '.pbs', 'w') as f:
            f.write(job_str)
    
        job_name = '%s_square_plot%s' % (cfg[:-4], plot)
        job_str = 'python3 -m cdr.bin.plot pred_fn_ini/%s -m CDR_gpt2.00_invert -c plot_config%s.ini' % (cfg, plot)
        job_str = wrapper % job_str
        job_str = base % (job_name, job_name) + job_str
        with open(job_name + '.pbs', 'w') as f:
            f.write(job_str)
    
        job_name = '%s_surpprob_plot%s' % (cfg[:-4], plot)
        job_str = 'python3 -m cdr.bin.plot pred_fn_ini/%s -m CDR_gptsurpprob -c plot_config%s.ini' % (cfg, plot)
        job_str = wrapper % job_str
        job_str = base % (job_name, job_name) + job_str
        with open(job_name + '.pbs', 'w') as f:
            f.write(job_str)
    
        job_name = '%s_gptpcfg_plot%s' % (cfg[:-4], plot)
        job_str = 'python3 -m cdr.bin.plot pred_fn_ini/%s -m CDR_gptpcfg -c plot_config%s.ini' % (cfg, plot)
        job_str = wrapper % job_str
        job_str = base % (job_name, job_name) + job_str
        with open(job_name + '.pbs', 'w') as f:
            f.write(job_str)
    
    for surp in ('ngram', 'pcfg', 'gpt', 'gptj', 'gpt3', 'cloze'):
        if cfg.startswith('provo') or surp != 'cloze':
            for suff in ('',):  #, 'prob_h0', '0.50_h0', '0.75_h0', '1.00_h0', '1.33_h0', '2.00_h0'):
                for plot in ('', '_surf'):
                    job_name = '%s_%s%s_plot%s' % (cfg[:-4], surp, suff, plot)
                    job_str = 'python3 -m cdr.bin.plot pred_fn_ini/%s -m CDR_%s -c plot_config%s.ini' % (cfg, surp + suff, plot)
                    job_str = wrapper % job_str
                    job_str = base % (job_name, job_name) + job_str
                    with open(job_name + '.pbs', 'w') as f:
                        f.write(job_str)

                    job_name = '%s_%s%s_plot_alls' % (cfg[:-4], surp, suff)
                    job_str = 'python3 -m cdr.bin.plot pred_fn_ini/%s -m CDR_%s -c plot_config_alls_%s.ini' % (cfg, surp + suff, cfg[:-4])
                    job_str = wrapper % job_str
                    job_str = base % (job_name, job_name) + job_str
                    with open(job_name + '.pbs', 'w') as f:
                        f.write(job_str)

    # Normal error
    for surp in ('gpt',):
        for suff in ('',):
            for plot in ('', '_surf'):
                job_name = '%s_normal_%s%s_plot%s' % (cfg[:-4], surp, suff, plot)
                job_str = 'python3 -m cdr.bin.plot pred_fn_ini/%s_normal.ini -m CDR_%s -c plot_config%s.ini' % (cfg[:-4], surp + suff, plot)
                job_str = wrapper % job_str
                job_str = base % (job_name, job_name) + job_str
                with open(job_name + '.pbs', 'w') as f:
                    f.write(job_str)


