import os
import configparser
import pandas as pd

ini_path = 'ini_owt'

out = []
for cfg_path in os.listdir(ini_path):
    dataset = cfg_path[:-4]
    c = configparser.ConfigParser()
    c.read(os.path.join(ini_path, cfg_path))
    data_path = c['data'].get('X_train')
    dvs = c['model_main'].get('formula').split(' ~ ')[0].strip().split(' + ')
    df = pd.read_csv(data_path, sep=' ')
    for dv in dvs:
        cutoff95 = df[dv].quantile(0.95)
        out.append(
            dict(
                dataset=dataset,
                dv=dv,
                cutoff95=cutoff95
            )
        )

out = pd.DataFrame(out)
out.to_csv('tail_cutoff.csv', index=None)
