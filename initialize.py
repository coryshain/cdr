import os
import yaml
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper
try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources
from cdr import templates

# Create config

CONFIG_PATH = 'config.yml'
CONFIG_DEFAULT = {
    'ini_path': 'ini',
    'ini_cutoff_path': 'ini_cutoff95',
    'data_path': 'data',
    'results_path': 'results',
    'singularity_path': ''
}

if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, 'r') as f:
        cfg = yaml.load(f, Loader=Loader)
        if cfg is None:
            cfg = {}
else:
    cfg = {}
for key in CONFIG_DEFAULT:
    if key not in cfg:
        cfg[key] = CONFIG_DEFAULT[key]

prompts = {
    'ini_path': 'Type path in which to place model config (*.ini) files (key: {key}), ' + \
                'or press Enter to use the default path ("{default}")',
    'ini_cutoff_path': 'Type path in which to place model config (*.ini) files using a 95% data cutoff (key: {key}), ' + \
                'or press Enter to use the default path ("{default}")',
    'data_path': 'Type path to reading data from https://osf.io/8v5qb/ ' + \
                '(key: {key}), or press Enter to use default path ("{default}")',
    'results_path': 'Type path in which to place modeling results (key: {key}), ' + \
                    'or press Enter to use default path ("{default}")',
    'singularity_path': 'Type path to singularity image to use for running/evaluating models (key: {key}), ' + \
                        'or press Enter to use default ("{default}"). If empty, singularity will not be used'
}

cutoffs = {
    'brown': '; fdur <= 587.74',
    'dundee': '; fdurFP <= 501.0; fdurGP <= 706.0; fdurSPsummed <= 501.0',
    'geco': '; fdurFP <= 456.0; fdurGP <= 830.0',
    'natstor': '; fdur <= 590.0',
    'natstormaze': '; rt <= 1628.0',
    'provo': '; fdurFP <= 653.0; fdurGP <= 1132.199999999997; fdurSPsummed <= 644.0'
}

for key in cfg:
    prompt = prompts[key] + ' >>> '
    prompt = prompt.format(key=key, default=cfg[key])
    ans = input(prompt)
    if ans.strip():
        cfg[key] = ans

print('Saving config file config.yml. To change the config, re-run `python -m initialize`.')

with open(CONFIG_PATH, 'w') as f:
    yaml.dump(cfg, f, Dumper=Dumper)

# Create any needed directories

if not os.path.exists(cfg['ini_path']):
    os.makedirs(cfg['ini_path'])
if not os.path.exists(cfg['ini_cutoff_path']):
    os.makedirs(cfg['ini_cutoff_path'])
if not os.path.exists(cfg['results_path']):
    os.makedirs(cfg['results_path'])
if not os.path.exists(cfg['results_path'] + '_cutoff95'):
    os.makedirs(cfg['results_path'] + '_cutoff95')

# Create model *.ini files

for dataset in cutoffs:
    ini = dataset + '.ini'
    ini_src = pkg_resources.open_text(templates, ini).read()
    ini_out = ini_src.format(
        data_path=cfg['data_path'],
        results_path=cfg['results_path'],
        cutoff_filter='',
        cutoff_suffix=''
    )
    ini_out = ini_out.split('# NAACL19')[0]
    ini_out_path = os.path.join(cfg['ini_path'], ini)
    with open(ini_out_path, 'w') as f:
        f.write(ini_out)

    ini_out = ini_src.format(
        data_path=cfg['data_path'],
        results_path=cfg['results_path'],
        cutoff_filter=cutoffs[dataset],
        cutoff_suffix='_cutoff95'
    )
    ini_out = ini_out.split('# NAACL19')[0]
    ini_out_path = os.path.join(cfg['ini_cutoff_path'], ini)
    with open(ini_out_path, 'w') as f:
        f.write(ini_out)


