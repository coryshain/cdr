import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

for network in ['LANG', 'MD', 'LANG_MD', 'MDSPWM']:
    df = pd.read_csv('../data/fMRI_ns/%s_y.csv' % network, sep=' ')
    df = df.sort_values(['subject', 'fROI', 'docid', 'time'])
    df = df[df.time < df.maxstimtime]

    gb = {}
    for x, y in df[['subject', 'fROI', 'docid', 'BOLD']].groupby(['subject', 'fROI', 'docid']):
        k1 = x[1:]
        k2 = x[0]
        val = y
        if not k1 in gb:
            gb[k1] = {}
        gb[k1][k2] = y

    out = []
    for x in gb:
        fROI, docid = x
        for subject in gb[x]:
            BOLD = gb[x][subject].BOLD
            others = []
            stim_len = None
            for other in gb[x]:
                if other != subject:
                    new = gb[x][other].BOLD
                    if stim_len is None:
                        stim_len = len(new)
                    if len(new) > stim_len:
                        new = new.head(stim_len)
                    others.append(new)
            if len(BOLD) > stim_len:
                BOLD = BOLD.head(stim_len)
            others = np.nanmean(others, axis=0)

            out.append(
                pd.DataFrame(
                    {
                        'subject': subject,
                        'fROI': fROI,
                        'docid': docid,
                        'BOLD': BOLD,
                        'others': others
                    }
                )
            )

    out = pd.concat([x.reset_index(drop=True) for x in out], axis=0)
    out = out.reset_index(drop=True)

    results = smf.ols('BOLD ~ others', data=out).fit()

    r = np.corrcoef(out.BOLD, out.others)[0, 1]

    if not os.path.exists('../results/fMRI_ns_WM/%s_ceil.txt' % network):
        os.makedirs('../results/fMRI_ns_WM/%s_ceil.txt' % network)

    with open('../results/fMRI_ns_WM/%s_ceil.txt' % network, 'w') as f:
        f.write('Network: %s\n\n' % network)
        f.write(str(results.summary()) + '\n\n')
        f.write('Correlation with mean of others: %s\n' % r)

# out.to_csv(sys.stdout, sep=' ', index=False, na_rep='NaN')
