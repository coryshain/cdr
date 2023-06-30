base = """#!/bin/bash
#
#SBATCH --job-name=%s
#SBATCH --output="%s-%%N-%%j.out"
#SBATCH --time=1:00:00
#SBATCH --mem=8gb
#SBATCH --ntasks=4
#SBATCH --partition=use-everything
"""

# Change this to reflect how your system runs bash executables
wrapper = '\nsingularity exec --nv ../singularity_images/tf-latest-gpu.simg bash -c "%s"'

for cfg in ('brown.ini', 'dundee.ini', 'geco.ini', 'natstor.ini', 'natstormaze.ini', 'provo.ini'):
    for plot in ('', '_surf'):
        for response in ('', '_mu', '_sigma', '_beta'):
            job_name = '%s_main_plot%s%s' % (cfg[:-4], plot, response)
            job_str = 'python3 -m cdr.bin.plot freq_pred_ini/%s -m main -c plot_config%s%s.ini' % (cfg, plot, response)
            job_str = wrapper % job_str
            job_str = base % (job_name, job_name) + job_str
            with open(job_name + '.pbs', 'w') as f:
                f.write(job_str)

    job_name = '%s_bigram_plot' % cfg[:-4]
    job_str = 'python3 -m cdr.bin.plot freq_pred_ini/%s -m bigram -c plot_config_surf.ini' % cfg
    job_str = wrapper % job_str
    job_str = base % (job_name, job_name) + job_str
    with open(job_name + '.pbs', 'w') as f:
        f.write(job_str)

    job_name = '%s_trigram_plot' % cfg[:-4]
    job_str = 'python3 -m cdr.bin.plot freq_pred_ini/%s -m trigram -c plot_config_surf.ini' % cfg
    job_str = wrapper % job_str
    job_str = base % (job_name, job_name) + job_str
    with open(job_name + '.pbs', 'w') as f:
        f.write(job_str)
 
