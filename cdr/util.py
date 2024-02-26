from __future__ import print_function
import sys
import os
import re
import math
import pickle
import numpy as np
import pandas as pd
from scipy import linalg, special

file_re = re.compile('output_([^_]+)_f(\d+)_([^_]+).csv')


def stderr(s):
    sys.stderr.write(s)
    sys.stderr.flush()


def extract_cdr_prediction_data(dirpath, metric='mse'):
    twostep_ix = 0
    filetype_ix = 1
    response_ix = 3
    filenum_ix = 5
    partition_ix = 6

    parser = re.compile(
        '(LM_2STEP_)?'
        '(CDRpreds|squared_error|losses_mse|mse_losses|loglik|preds|preds_table|obs|output)'
        '(_(.+?))?(_f([0-9]+))?_([^_]*).(csv|txt)'
    )
    out = {}
    # Might be an ensemble, so check
    basename = os.path.basename(dirpath)
    parentdir = os.path.normpath(os.path.dirname(dirpath))
    modeldirs = []
    use_crossval = False
    folds = set()
    ensembleids = set()
    modeldir2fold = {}
    modeldir2ensembleid = {}
    for x in os.listdir(parentdir):
        if os.path.isdir(os.path.join(parentdir, x)):
            ensemble = re.match('\.m\d+', x[len(basename):])
            cv = re.match('\.CV[^.~]+~([^.~]+)(\.m(\d+))?', x[len(basename):])
            if x == basename or \
                    (x.startswith(basename) and (ensemble or cv)):
                x_full = os.path.join(parentdir, x)

                if cv:
                    fold = cv.group(1)
                    modeldir2fold[x] = fold
                    folds.add(fold)

                    ensembleid = cv.group(3)
                    modeldir2ensembleid[x] = ensembleid
                    ensembleids.add(ensembleid)

                    use_crossval = True

                modeldirs.append(x_full)
    folds = sorted(list(folds))
    ensembleids = sorted(list(ensembleids))
    if len(modeldirs) > 1: # Ensemble
        try:
            modeldirs.remove(os.path.normpath(dirpath))
        except ValueError:
            pass
    for modeldir in modeldirs:
        model_basename = os.path.basename(modeldir)
        for path in os.listdir(modeldir):
            parsed = parser.match(path)
            if parsed:
                parsed = parsed.groups()
                if parsed[twostep_ix]:
                    predtype = '2step'
                else:
                    predtype = 'direct'
                filetype = parsed[filetype_ix]
                if filetype in ['CDRpreds', 'preds_table', 'output']:
                    filetype = 'table'
                elif filetype in ['squared_error', 'losses_mse', 'mse_losses', 'obs', 'preds']:
                    filetype = 'mse'
                response = parsed[response_ix]
                if response is None:
                    response = 'y'
                filenum = parsed[filenum_ix]
                if filenum is None:
                    filenum = 0
                else:
                    filenum = int(filenum)
                partition = parsed[partition_ix]

                data_path = os.path.join(modeldir, path)

                if filetype == 'table':
                    a = pd.read_csv(
                        data_path,
                        sep=' ',
                        skipinitialspace=True
                    )
                    if metric == 'mse':
                        a = (a['CDRobs'] - a['CDRpreds']) ** 2
                    elif metric == 'corr':
                        a = a[['CDRobs', 'CDRpreds']]
                    elif metric == 'loglik':
                        a = a['CDRloglik']
                    else:
                        raise ValueError('Unrecognized metric: %s' % metric)
                else:
                    a = pd.read_csv(
                        data_path,
                        sep=' ',
                        header=None,
                        skipinitialspace=True
                    )

                if response not in out:
                    out[response] = {}
                if filenum not in out[response]:
                    out[response][filenum] = {}
                if partition not in out[response][filenum]:
                    out[response][filenum][partition] = {}
                if predtype not in out[response][filenum][partition]:
                    out[response][filenum][partition][predtype] = {}
                if model_basename in out[response][filenum][partition][predtype]:
                    raise ValueError('Found redundant results file: %s.' % data_path)
                else:
                    out[response][filenum][partition][predtype][model_basename] = a

    # Concatenate across crossval folds if needed
    if use_crossval:
        _out = {}
        for response in out:
            if response not in _out:
                _out[response] = {}
            for filenum in out[response]:
                if filenum not in _out[response]:
                    _out[response][filenum] = {}
                for partition in out[response][filenum]:
                    if partition not in _out[response][filenum]:
                        _out[response][filenum][partition] = {}
                    for predtype in out[response][filenum][partition]:
                        if predtype not in _out[response][filenum][partition]:
                            _out[response][filenum][partition][predtype] = {}
                        ensembles = {}
                        for model_basename in out[response][filenum][partition][predtype]:
                            ensembleid = modeldir2ensembleid[model_basename]
                            if ensembleid not in ensembles:
                                ensembles[ensembleid] = {}
                            fold = modeldir2fold[model_basename]
                            if fold not in ensembles[ensembleid]:
                                ensembles[ensembleid][fold] = out[response][filenum][partition][predtype][model_basename]
                            else:
                                raise ValueError('Found redundant results file: %s.' % model_basename)
                        for ensembleid in ensembleids:
                            outname = basename
                            if ensembleid is not None:
                                outname += '.m%s' % ensembleid
                            ensemble = []
                            for fold in folds:
                                ensemble.append(ensembles[ensembleid][fold])
                            ensemble = pd.concat(ensemble, axis=0).reset_index(drop=True)
                            _out[response][filenum][partition][predtype][outname] = ensemble

        out = _out

    return out


def names2ix(names, l, dtype=np.int32):
    """
    Generate 1D numpy array of indices in **l** corresponding to names in **names**

    :param names: ``list`` of ``str``; names to look up in **l**
    :param l: ``list`` of ``str``; list of names from which to extract indices
    :param dtype: ``numpy`` dtype object; return dtype
    :return: ``numpy`` array; indices of **names** in **l**
    """

    if type(names) is not list:
        names = [names]
    ix = []
    for n in names:
        ix.append(l.index(n))
    return np.array(ix, dtype=dtype)


def mse(true, preds):
    """
    Compute mean squared error (MSE).

    :param true: True values
    :param preds: Predicted values
    :return: ``float``; MSE
    """

    return ((true-preds)**2).mean()


def mae(true, preds):
    """
    Compute mean absolute error (MAE).

    :param true: True values
    :param preds: Predicted values
    :return: ``float``; MAE
    """

    return (true-preds).abs().mean()


def percent_variance_explained(true, preds):
    """
    Compute percent variance explained.

    :param true: True values
    :param preds: Predicted values
    :return: ``float``; percent variance explained
    """

    num = mse(true, preds)
    denom = np.std(true) ** 2
    return max(0., (1 - num / denom) * 100)


def get_random_permutation(n):
    """
    Draw a random permutation of integers 0 to **n**.
    Used to shuffle arrays of length **n**.
    For example, a permutation and its inverse can be generated by calling ``p, p_inv = get_random_permutation(n)``.
    To randomly shuffle an **n**-dimensional vector ``x``, call ``x[p]``.
    To un-shuffle ``x`` after it has already been shuffled, call ``x[p_inv]``.

    :param n: maximum value
    :return: 2-tuple of ``numpy`` arrays; the permutation and its inverse
    """

    p = np.random.permutation(np.arange(n))
    p_inv = np.zeros_like(p)
    p_inv[p] = np.arange(n)
    return p, p_inv


def sn(string):
    """
    Compute a valid scope name version of a string.

    :param string: ``str``; input string
    :return: ``str``; transformed string
    """

    return re.sub('[^A-Za-z0-9_.\\-/]', '.', string)


def reg_name(string):
    """
    Standardize a variable name for regularization

    :param string: ``str``; input string
    :return: ``str``; transformed string
    """

    name = string.split(':')[0]
    name = name.replace('/', '_')

    return name


def pca(X, n_dim=None, dtype=np.float32):
    """
    Perform principal components analysis on a data table.

    :param X: ``numpy`` or ``pandas`` array; the input data
    :param n_dim: ``int`` or ``None``; maximum number of principal components. If ``None``, all components are retained.
    :param dtype: ``numpy`` dtype; return dtype
    :return: 5-tuple of ``numpy`` arrays; transformed data, eigenvectors, eigenvalues, input means, and input standard deviations
    """

    X = np.array(X, dtype=dtype)
    assert len(X.shape) == 2, 'Wrong dimensionality for PCA (X must be rank 2).'
    means = X.mean(0, keepdims=True)
    sds = X.std(0, keepdims=True)
    X -= means
    X /= sds
    C = np.cov(X, rowvar=False)
    eigenval, eigenvec = linalg.eigh(C)
    sorted_id = np.argsort(eigenval)[::-1]
    eigenval = eigenval[sorted_id]
    eigenvec = eigenvec[:,sorted_id]
    if n_dim is not None and n_dim < eigenvec.shape[1]:
        eigenvec = eigenvec[:,:n_dim]
    Xpc = np.dot(X, eigenvec)
    return Xpc, eigenvec, eigenval, means, sds


def logsumexp(a):
    return np.exp(a - special.logsumexp(a))


def nested(model_name_1, model_name_2):
    """
    Check whether two CDR models are nested with 1 degree of freedom

    :param model_name_1: ``str``; name of first model
    :param model_name_2: ``str``; name of second model
    :return: ``bool``; ``True`` if models are nested with 1 degree of freedom, ``False`` otherwise
    """
    split = (model_name_1.split('!'), model_name_2.split('!'))
    m_base = [x[0] for x in split]
    m_ablated = [set(x[1:]) for x in split]
    a = 0 if len(m_ablated[0]) < len(m_ablated[1]) else 1
    b = 1 - a

    return m_base[a] == m_base[b] and len(m_ablated[b] - m_ablated[a]) == 1 and len(m_ablated[a] - m_ablated[b]) == 0


def filter_names(names, filters):
    """
    Return elements of **names** permitted by **filters**, preserving order in which filters were matched.
    Filters can be ordinary strings, regular expression objects, or string representations of regular expressions.
    For a regex filter to be considered a match, the expression must entirely match the name.

    :param names: ``list`` of ``str``; pool of names to filter.
    :param filters: ``list`` of ``{str, SRE_Pattern}``; filters to apply in order
    :return: ``list`` of ``str``; names in **names** that pass at least one filter
    """

    filters_regex = [re.compile(f if f.endswith('$') else f + '$') for f in filters]

    out = []

    for i in range(len(filters)):
        filter = filters[i]
        filter_regex = filters_regex[i]
        for name in names:
            if name not in out:
                if name == filter:
                    out.append(name)
                elif filter_regex.match(name):
                    out.append(name)

    return out


def filter_models(names, filters=None, cdr_only=False):
    """
    Return models contained in **names** that are permitted by **filters**, preserving order in which filters were matched.
    Filters can be ordinary strings, regular expression objects, or string representations of regular expressions.
    For a regex filter to be considered a match, the expression must entirely match the name.
    If ``filters`` is zero-length, returns **names**.

    :param names: ``list`` of ``str``; pool of model names to filter.
    :param filters: ``list`` of ``{str, SRE_Pattern}`` or ``None``; filters to apply in order. If ``None``, no additional filters.
    :param cdr_only: ``bool``; if ``True``, only returns CDR models. If ``False``, returns all models admitted by **filters**.
    :return: ``list`` of ``str``; names in **names** that pass at least one filter, or all of **names** if no filters are applied.
    """

    if cdr_only:
        names = [name for name in names if not (name.startswith('LM') or name.startswith('GAM'))]

    if filters is None:
        filters = []

    if len(filters) > 0:
        out = filter_names(names, filters)
    else:
        out = names
    return out


def get_partition_list(partition):
    if not isinstance(partition, list):
        partition = partition.strip().split()
    if ':' in partition[0]:
        partition = partition[0].split(':')
    else:
        if len(partition) == 1:
            partition = partition[0].split('-')
        if len(partition) == 1:
            partition = partition[0].split('+')
    return partition


def paths_from_partition_cliarg(partition, config):
    partition = get_partition_list(partition)
    X_paths = []
    y_paths = []

    X_map = {
        'train': config.X_train,
        'dev': config.X_dev,
        'test': config.X_test
    }

    Y_map = {
        'train': config.Y_train,
        'dev': config.Y_dev,
        'test': config.Y_test
    }

    for p in partition:
        _X_paths = X_map[p]
        _y_paths = Y_map[p]

        assert _X_paths is not None, 'No data path provided for partition %s. If you did not intend to require ' + \
                                     'dev evaluation, set ``eval_freq`` to ``0`` in your model config.' % p
        assert _y_paths is not None, 'No data path provided for partition %s. If you did not intend to require ' + \
                                     'dev evaluation, set ``eval_freq`` to ``0`` in your model config.' % p

        for _X_path in _X_paths:
            if _X_path not in X_paths:
                X_paths.append(_X_path)
        for _y_path in _y_paths:
            if _y_path not in y_paths:
                y_paths.append(_y_path)

    return X_paths, y_paths


def get_irf_name(x, irf_name_map):
    k = None
    for y in sorted(list(irf_name_map.keys())):
        if y == x:
            k = y
        elif y in x:
            if k is None or len(y) > len(k):
                k = y

    if k is None:
        out = x
    else:
        out = irf_name_map[k]

    return out


def get_numerical_sd(sd, in_dim=1, out_dim=1):
    in_dim = float(int(in_dim))
    out_dim = float(int(out_dim))
    if isinstance(sd, str):
        if sd.lower().startswith('xavier') or sd.lower().startswith('glorot'):
            factor = sd[6:]
            if factor:
                factor = int(factor)
            else:
                factor = 1
            out = math.sqrt(2. / (in_dim + out_dim))
        elif sd.lower().startswith('he'):
            factor = sd[2:]
            if factor:
                factor = int(factor)
            else:
                factor = 1
            out = math.sqrt(2. / in_dim)
        else:
            raise ValueError('Unrecognized variance initializer: %s' % sd)
    else:
        factor = 1
        out = sd

    out *= factor

    return out


def load_cdr(dir_path, suffix=''):
    """
    Convenience method for reconstructing a saved CDR object. First loads in metadata from ``m.obj``, then uses
    that metadata to construct the computation graph. Then, if saved weights are found, these are loaded into the
    graph.

    :param dir_path: Path to directory containing the CDR checkpoint files.
    :param suffix: ``str``; file suffix.
    :return: The loaded CDR instance.
    """

    with open(dir_path + '/m%s.obj' % suffix, 'rb') as f:
        m = pickle.load(f)
    m.build(outdir=dir_path, verbose=False)
    m.load(outdir=dir_path, suffix=suffix)
    return m


def flatten_dict(d, keys=None):
    if keys is None:
        keys = []
    out = []
    for k in sorted(d.keys()):
        v = d[k]
        if isinstance(v, dict):
            out += flatten_dict(v, keys + [k])
        else:
            out.append((tuple(keys + [k]), v))
    return out


def pretty_print_time(s):
    days, s = divmod(s, 24*60*60)
    hours, s = divmod(s, 60*60)
    minutes, s = divmod(s, 60)
    seconds = s
    vals = []
    for i, val in enumerate((days, hours, minutes, seconds)):
        if i == 0:
            suffix = 'd'
        elif i == 1:
            suffix = 'h'
        elif i == 2:
            suffix = 'm'
        else:
            suffix = 's'
        if len(vals):
            vals.append(('%02d%s' % (val, suffix)))
        elif val > 0:
            vals.append('%d%s' % (val, suffix))

    if len(vals):
        out = '-'.join(vals)
    else:
        out = '0'
    return out

