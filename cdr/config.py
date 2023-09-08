import sys
import os
import shutil
from itertools import chain, combinations
if sys.version_info[0] == 2:
    import ConfigParser as configparser
else:
    import configparser

from .formula import Formula
from .kwargs import MODEL_INITIALIZATION_KWARGS, BAYES_KWARGS, NN_BAYES_KWARGS, \
    PLOT_KWARGS_CORE, PLOT_KWARGS_OTHER


PLOT_KEYS_CORE = [x.key for x in PLOT_KWARGS_CORE]
PLOT_KEYS_OTHER = [x.key for x in PLOT_KWARGS_OTHER]


# Thanks to Brice (https://stackoverflow.com/users/140264/brice) at Stack Overflow for this
def powerset(iterable):
    xs = list(iterable)
    return chain.from_iterable(combinations(xs,n) for n in range(1, len(xs)+1))

class Config(object):
    """
    Parses an \*.ini file and stores settings needed to define a set of CDR experiments.

    :param path: Path to \*.ini file
    """

    def __init__(self, path):
        self.current_model = None

        config = configparser.ConfigParser()
        config.optionxform = str
        assert os.path.exists(path), 'Config file %s does not exist' %path
        config.read(path)

        data = config['data']
        global_settings = config['global_settings']
        if 'cdr_settings' in config:
            cdr_settings = config['cdr_settings']
        elif 'dtsr_settings' in config:
            cdr_settings = config['dtsr_settings']
        else:
            config['cdr_settings'] = {}
            cdr_settings = config['cdr_settings']

        ########
        # Data #
        ########

        self.X_train = data.get('X_train')
        self.X_dev = data.get('X_dev', None)
        self.X_test = data.get('X_test', None)

        self.Y_train = data.get('Y_train', data.get('y_train', None))
        assert self.Y_train, 'Y_train must be provided'
        self.Y_dev = data.get('Y_dev', data.get('y_dev', None))
        self.Y_test = data.get('Y_test', data.get('y_test', None))

        sep = data.get('sep', ',')
        if sep.lower() in ['', "' '", '" "', 's', 'space']:
            sep = ' '
        self.sep = sep

        series_ids = data.get('series_ids')
        if series_ids:
            self.series_ids = series_ids.strip().split()
        else:
            self.series_ids = []
        self.modulus = data.getint('modulus', 4)
        split_ids = data.get('split_ids', '')
        self.split_ids = split_ids.strip().split()
        filters = data.get('filters', None)
        if filters is None:
            filters = []
        else:
            filters = filters.split(';')
            for i in range(len(filters)):
                f = filters[i].strip().split()
                k = f[0]
                v = ' '.join(f[1:])
                filters[i] = (k, v)
        self.filters = filters

        self.history_length = data.getint('history_length', 128)
        self.future_length = data.getint('future_length', 0)
        self.t_delta_cutoff = data.getfloat('t_delta_cutoff', None)

        self.merge_cols = data.get('merge_cols', None)
        if self.merge_cols is not None:
            self.merge_cols = self.merge_cols.split()

        ###################
        # Global Settings #
        ###################

        self.outdir = global_settings.get('outdir', None)
        if self.outdir is None:
            self.outdir = global_settings.get('logdir', None)
        if self.outdir is None:
            self.outdir = './cdr_model/'
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        if not os.path.normpath(os.path.realpath(path)) == os.path.normpath(os.path.realpath(self.outdir + '/config.ini')):
            shutil.copy2(path, self.outdir + '/config.ini')
        self.use_gpu_if_available = global_settings.getboolean('use_gpu_if_available', True)

        #################
        # CDR Settings  #
        #################

        self.global_cdr_settings = self.build_cdr_settings(cdr_settings, add_defaults=False)

        ############
        # Model(s) #
        ############

        # Add ablations and crossval folds
        self.models = {}
        self.model_list = []
        self.ensembles = {}
        self.ensemble_list = []
        for model_field in [m for m in config.keys() if m.startswith('model_')]:
            model_name = model_field[6:]
            formula = Formula(config[model_field]['formula'])
            is_cdrnn = len(formula.nns_by_id) > 0
            if not (model_name.startswith('LM') or model_name.startswith('GAM')):
                reg_type = 'cdr'
            else:
                reg_type = model_name.split('_')[0]
            use_crossval = 'crossval_factor' in config[model_field]
            if use_crossval:
                model_configs = {}
                folds = config[model_field]['crossval_folds'].split()
                for fold in folds:
                    levels = fold.split(';')
                    fold_name = model_name + '_CV%s~%s' % (config[model_field]['crossval_factor'], '~'.join(levels))
                    model_config = configparser.ConfigParser()
                    model_config[model_field] = config[model_field]
                    model_config = model_config[model_field]
                    del model_config['crossval_folds']
                    model_config['crossval_fold'] = fold
                    model_configs[fold_name] = model_config
            else:
                model_configs = {model_name: config[model_field]}
            _models = {}
            _model_list = []
            for model_name in model_configs:
                model_config = model_configs[model_name]
                model_settings = self.build_cdr_settings(
                    model_config,
                    global_settings=self.global_cdr_settings,
                    is_cdr=reg_type=='cdr',
                    is_cdrnn=is_cdrnn
                )
                _models[model_name] = model_settings
                if reg_type == 'lme':
                    _models[model_name]['correlated'] = config[model_field].getboolean('correlated', True)
                _model_list.append(model_name)
                if 'ablate' in config[model_field]:
                    for ablated in powerset(config[model_field]['ablate'].strip().split()):
                        ablated = list(ablated)
                        new_name = model_name + '!' + '!'.join(ablated)
                        formula = Formula(config[model_field]['formula'])
                        formula.ablate_impulses(ablated)
                        new_model = _models[model_name].copy()
                        if reg_type == 'cdr':
                            new_model['formula'] = str(formula)
                        elif reg_type == 'lme':
                            new_model['formula'] = formula.to_lmer_formula_string(
                                z=False,
                                correlated=_models[model_name]['correlated'],
                                transform_dirac=False
                            )
                        else:
                            raise ValueError('Ablation with reg_type "%s" not currently supported.' % reg_type)
                        new_model['ablated'] = set(ablated)
                        _models[new_name] = new_model
                        _model_list.append(new_name)
                if config[model_field].get('n_ensemble', config[model_field].get('n_ensembles', None)) \
                        or 'n_ensemble' in cdr_settings:
                    if 'n_ensemble' in config[model_field]:
                        n_ensemble = config[model_field].getint('n_ensemble')
                    else:
                        n_ensemble = cdr_settings.getint('n_ensemble')
                    __models = {}
                    __model_list = []
                    for _m in _models:
                        for i in range(n_ensemble):
                            __models[_m + '.m%d' % i] = _models[_m]
                    for _m in _model_list:
                        for i in range(n_ensemble):
                            __model_list.append(_m + '.m%d' % i)
                    _models.update(__models)
                    _model_list = __model_list
                self.ensemble_list.append(model_name)
                self.ensembles[model_name] = model_settings

                self.models.update(_models)
                self.model_list += _model_list

        self.irf_name_map = {
            't_delta': 'Delay (s)',
            'time_X': 'Timestamp (s)',
            'X_time': 'Timestamp (s)',
            'rate': 'Rate'
        }
        if 'irf_name_map' in config:
            for x in config['irf_name_map']:
                self.irf_name_map[x] = config['irf_name_map'][x]

    def __getitem__(self, item):
        if self.current_model is None:
            return self.global_cdr_settings[item]
        if self.current_model in self.models:
            return self.models[self.current_model][item]
        raise ValueError('There is no model named "%s" defined in the config file.' %self.current_model)

    def get(self, item, default=None):
        if (self.current_model is None and item in self.global_cdr_settings) or \
                (self.current_model in self.models and item in self.models[self.current_model]):
            return self[item]
        return default

    def __str__(self):
        out = ''
        V = vars(self)
        for x in V:
            out += '%s: %s\n' %(x, V[x])
        return out

    def set_model(self, model_name=None):
        """
        Change internal state to that of model named **model_name**.
        ``Config`` instances can store settings for multiple models.
        ``set_model()`` determines which model's settings are returned by ``Config`` getter methods.

        :param model_name: ``str``; name of target model
        :return: ``None``
        """
        if model_name is None or model_name in self.models:
            self.current_model = model_name
        else:
            raise ValueError('There is no model named "%s" defined in the config file.' %model_name)

    def build_cdr_settings(self, settings, add_defaults=True, global_settings=None, is_cdr=True, is_cdrnn=False):
        """
        Given a settings object parsed from a config file, compute CDR parameter dictionary.

        :param settings: settings from a ``ConfigParser`` object.
        :param add_defaults: ``bool``; whether to add default settings not explicitly specified in the config.
        :param global_settings: ``dict`` or ``None``; dictionary of global defaults for parameters missing from **settings**.
        :param is_cdr: ``bool``; whether this is a CDR(NN) model.
        :param is_cdrnn: ``bool``; whether this is a CDRNN model.
        :return: ``dict``; dictionary of settings key-value pairs.
        """

        if global_settings is None:
            global_settings = {}
        out = {}

        # Core fields
        out['formula'] = settings.get('formula', None)
        if is_cdr and out['formula']:
            # Standardize the model string
            out['formula'] = str(Formula(out['formula']))

        # Model initialization keyword arguments
        if is_cdr:
            # Allowing settings to propagate for the wrong model type if specified
            # allows the global config to specify defaults for multiple model types.
            # Cross-type settings will only propagate if they are explicitly defined
            # in the config (defaults are ignored).

            # General initialization keyword arguments
            for kwarg in MODEL_INITIALIZATION_KWARGS:
                if add_defaults:
                    if kwarg.in_settings(settings) or kwarg.key not in global_settings:
                        out[kwarg.key] = kwarg.kwarg_from_config(settings, is_cdrnn=is_cdrnn)
                    else:
                        out[kwarg.key] = global_settings[kwarg.key]
                elif kwarg.in_settings(settings):
                    out[kwarg.key] = kwarg.kwarg_from_config(settings, is_cdrnn=is_cdrnn)
                if kwarg.key == 'plot_interactions' and kwarg.key in out and isinstance(out[kwarg.key], str):
                    out[kwarg.key] = out[kwarg.key].split()

        out['ablated'] = set()

        # Cross validation settings
        out['crossval_factor'] = settings.get('crossval_factor', global_settings.get('crossval_factor', None))
        if 'crossval_fold' in settings:
            crossval_fold = settings['crossval_fold'].split(';')
        elif 'crossval_fold' in global_settings:
            crossval_fold = global_settings['crossval_fold']
        else:
            crossval_fold = []
        out['crossval_fold'] = crossval_fold

        return out


class PlotConfig(object):
    """
    Parses an \*.ini file and stores settings needed to define CDR plots

    :param path: Path to \*.ini file
    """

    def __init__(self, path=None):
        if path is None:
            self.settings_core, self.settings_other = self.build_plot_settings({})
        else:
            config = configparser.ConfigParser()
            config.optionxform = str
            assert os.path.exists(path), 'Config file %s does not exist' %path
            config.read(path)

            plot_settings = config['plot']
            self.settings_core, self.settings_other = self.build_plot_settings(plot_settings)

    def __getitem__(self, item):
        if item in self.settings_core:
            return self.settings_core[item]
        return self.settings_other[item]

    def __setitem__(self, key, value):
        if key in PLOT_KEYS_CORE:
            self.settings_core[key] = value
        elif key in PLOT_KEYS_OTHER:
            self.settings_other[key] = value
        else:
            raise ValueError('Attempted to set value for unrecognized plot kwarg %s' % key)
        

    def get(self, item, default=None):
        if item in self.settings_core:
            return self.settings_core.get(item, default)
        return self.settings_other.get(item, default)

    def build_plot_settings(self, settings):
        """
        Given a settings object parsed from a config file, compute plot parameters.

        :param settings: settings from a ``ConfigParser`` object.
        :return: ``dict``; dictionary of settings key-value pairs.
        """

        out_core = {}
        out_other = {}

        for kwarg in PLOT_KWARGS_CORE:
            if kwarg.in_settings(settings):
                val = kwarg.kwarg_from_config(settings)
                if kwarg.key in ['responses', 'response_params', 'pred_names'] and val is not None:
                    val = val.split()
                elif kwarg.key == 'prop_cycle_map' and val is not None:
                    val = val.split()
                    is_dict = len(val[-1].split(';')) == 2
                    if is_dict:
                        val_tmp = val
                        val = {}
                        for x in val_tmp:
                            k, v = x.split(';')
                            val[k] = int(v)
                    else:
                        val = [int(x) for x in val]
                elif kwarg.key == 'ylim' and val is not None:
                    val = tuple(float(x) for x in val.split())
                out_core[kwarg.key] = val
            else:
                out_core[kwarg.key] = kwarg.default_value

        for kwarg in PLOT_KWARGS_OTHER:
            if kwarg.in_settings(settings):
                val = kwarg.kwarg_from_config(settings)
                out_other[kwarg.key] = val
            else:
                out_other[kwarg.key] = kwarg.default_value

        return out_core, out_other
