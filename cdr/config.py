import sys
import os
import shutil
from itertools import chain, combinations
if sys.version_info[0] == 2:
    import ConfigParser as configparser
else:
    import configparser

from .formula import Formula
from .kwargs import MODEL_INITIALIZATION_KWARGS, \
    CDR_INITIALIZATION_KWARGS, CDRMLE_INITIALIZATION_KWARGS, CDRBAYES_INITIALIZATION_KWARGS, \
    CDRNN_INITIALIZATION_KWARGS, CDRNNMLE_INITIALIZATION_KWARGS, CDRNNBAYES_INITIALIZATION_KWARGS


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

        ########
        # Data #
        ########

        self.X_train = data.get('X_train')
        self.X_dev = data.get('X_dev', None)
        self.X_test = data.get('X_test', None)

        self.y_train = data.get('y_train')
        self.y_dev = data.get('y_dev', None)
        self.y_test = data.get('y_test', None)

        sep = data.get('sep', ' ')
        if sep.lower() in ['', "' '", '" "', 's', 'space']:
            sep = ' '
        self.sep = sep

        series_ids = data.get('series_ids')
        self.series_ids = series_ids.strip().split()
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

        self.global_cdr_settings = self.build_cdr_settings(cdr_settings)

        ############
        # Model(s) #
        ############

        # Add ablations
        self.models = {}
        self.model_list = []
        for model_field in [m for m in config.keys() if m.startswith('model_')]:
            model_name = model_field[6:]
            reg_type = None
            is_cdrnn = False
            if model_name.startswith('CDR') or model_name.startswith('DTSR'):
                reg_type = 'cdr'
                if model_name.startswith('CDRNN'):
                    is_cdrnn = True
            elif model_name.startswith('LME'):
                reg_type = 'lme'
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
            for model_name in model_configs:
                model_config = model_configs[model_name]
                model_settings = self.build_cdr_settings(
                    model_config,
                    global_settings=self.global_cdr_settings,
                    is_cdr=reg_type=='cdr',
                    is_cdrnn=is_cdrnn
                )
                self.models[model_name] = model_settings
                if reg_type == 'lme':
                    self.models[model_name]['correlated'] = config[model_field].getboolean('correlated', True)
                self.model_list.append(model_name)
                if 'ablate' in config[model_field]:
                    for ablated in powerset(config[model_field]['ablate'].strip().split()):
                        ablated = list(ablated)
                        new_name = model_name + '!' + '!'.join(ablated)
                        formula = Formula(config[model_field]['formula'])
                        formula.ablate_impulses(ablated)
                        new_model = self.models[model_name].copy()
                        if reg_type == 'cdr':
                            new_model['formula'] = str(formula)
                        elif reg_type == 'lme':
                            new_model['formula'] = formula.to_lmer_formula_string(
                                z=False,
                                correlated=self.models[model_name]['correlated'],
                                transform_dirac=False
                            )
                        else:
                            raise ValueError('Ablation with reg_type "%s" not currently supported.' % reg_type)
                        new_model['ablated'] = set(ablated)
                        self.models[new_name] = new_model
                        self.model_list.append(new_name)

        if 'irf_name_map' in config:
            self.irf_name_map = {}
            for x in config['irf_name_map']:
                self.irf_name_map[x] = config['irf_name_map'][x]
        else:
            self.irf_name_map = {}

    def __getitem__(self, item):
        if self.current_model is None:
            return self.global_cdr_settings[item]
        if self.current_model in self.models:
            return self.models[self.current_model][item]
        raise ValueError('There is no model named "%s" defined in the config file.' %self.current_model)

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

    def build_cdr_settings(self, settings, global_settings=None, is_cdr=True, is_cdrnn=False):
        """
        Given a settings object parsed from a config file, compute CDR parameter dictionary.

        :param settings: settings from a ``ConfigParser`` object.
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
        is_bayes = False
        if is_cdr and out['formula']:
            # Standardize the model string
            out['formula'] = str(Formula(out['formula']))
        out['network_type'] = settings.get('network_type', global_settings.get('network_type', 'bayes'))
        is_bayes = out['network_type'] == 'bayes'

        # Model initialization keyword arguments
        if is_cdr:
            # Allowing settings to propagate for the wrong model type if specified
            # allows the global config to specify defaults for multiple model types.
            # Cross-type settings will only propagate if they are explicitly defined
            # in the config (defaults are ignored).

            for kwarg in MODEL_INITIALIZATION_KWARGS:
                if kwarg.in_settings(settings) or kwarg.key not in global_settings:
                    out[kwarg.key] = kwarg.kwarg_from_config(settings)
                else:
                    out[kwarg.key] = global_settings[kwarg.key]

            # CDRNN initialization keyword arguments
            for kwarg in CDRNN_INITIALIZATION_KWARGS:
                if is_cdrnn:
                    if kwarg.in_settings(settings) or kwarg.key not in global_settings:
                        out[kwarg.key] = kwarg.kwarg_from_config(settings)
                    else:
                        out[kwarg.key] = global_settings[kwarg.key]
                elif kwarg.in_settings(settings) and not kwarg.key in out:
                    out[kwarg.key] = kwarg.kwarg_from_config(settings)

            # CDRNNBayes initialization keyword arguments
            for kwarg in CDRNNBAYES_INITIALIZATION_KWARGS:
                if is_cdrnn and is_bayes:
                    if kwarg.in_settings(settings) or kwarg.key not in global_settings:
                        out[kwarg.key] = kwarg.kwarg_from_config(settings)
                    else:
                        out[kwarg.key] = global_settings[kwarg.key]
                elif kwarg.in_settings(settings) and not kwarg.key in out:
                    out[kwarg.key] = kwarg.kwarg_from_config(settings)

            # CDRNNMLE initialization keyword arguments
            for kwarg in CDRNNMLE_INITIALIZATION_KWARGS:
                if is_cdrnn and not is_bayes:
                    if kwarg.in_settings(settings) or kwarg.key not in global_settings:
                        out[kwarg.key] = kwarg.kwarg_from_config(settings)
                    else:
                        out[kwarg.key] = global_settings[kwarg.key]
                elif kwarg.in_settings(settings) and not kwarg.key in out:
                    out[kwarg.key] = kwarg.kwarg_from_config(settings)

            # CDR initialization keyword arguments
            for kwarg in CDR_INITIALIZATION_KWARGS:
                if not is_cdrnn:
                    if kwarg.in_settings(settings) or kwarg.key not in global_settings:
                        out[kwarg.key] = kwarg.kwarg_from_config(settings)
                    else:
                        out[kwarg.key] = global_settings[kwarg.key]
                elif kwarg.in_settings(settings) and not kwarg.key in out:
                    out[kwarg.key] = kwarg.kwarg_from_config(settings)

            # CDRBayes initialization keyword arguments
            for kwarg in CDRBAYES_INITIALIZATION_KWARGS:
                if not is_cdrnn and is_bayes:
                    if kwarg.in_settings(settings) or kwarg.key not in global_settings:
                        out[kwarg.key] = kwarg.kwarg_from_config(settings)
                    else:
                        out[kwarg.key] = global_settings[kwarg.key]
                elif kwarg.in_settings(settings) and not kwarg.key in out:
                    out[kwarg.key] = kwarg.kwarg_from_config(settings)

            # CDRMLE initialization keyword arguments
            for kwarg in CDRMLE_INITIALIZATION_KWARGS:
                if not is_cdrnn and not is_bayes:
                    if kwarg.in_settings(settings) or kwarg.key not in global_settings:
                        out[kwarg.key] = kwarg.kwarg_from_config(settings)
                    else:
                        out[kwarg.key] = global_settings[kwarg.key]
                elif kwarg.in_settings(settings) and not kwarg.key in out:
                    out[kwarg.key] = kwarg.kwarg_from_config(settings)

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

        # Plotting defaults
        out['plot_n_time_units'] = settings.getfloat('plot_n_time_units', global_settings.get('plot_n_time_units', 2.5))
        out['plot_n_time_points'] = settings.getfloat('plot_n_time_points', global_settings.get('plot_n_time_points', 1000))
        out['surface_plot_n_time_points'] = settings.getfloat('surface_plot_n_time_points', global_settings.get('surface_plot_n_time_points', 1024))
        out['generate_irf_surface_plots'] = settings.getboolean('generate_irf_surface_plots', global_settings.get('generate_irf_surface_plots', False))
        out['generate_interaction_surface_plots'] = settings.getboolean('generate_interaction_surface_plots', global_settings.get('generate_interaction_surface_plots', False))
        out['generate_curvature_plots'] = settings.getboolean('generate_curvature_plots', global_settings.get('generate_curvature_plots', False))
        plot_interactions = settings.get('plot_interactions', global_settings.get('plot_interactions', ''))
        if isinstance(plot_interactions, str):
            plot_interactions = plot_interactions.split()
        out['plot_interactions'] = plot_interactions
        out['reference_time'] = settings.get('reference_time', settings.get('plot_t_interaction', global_settings.get('reference_time', 0.)))
        out['plot_x_inches'] = settings.getfloat('plot_x_inches', global_settings.get('plot_x_inches', 6))
        out['plot_y_inches'] = settings.getfloat('plot_y_inches', global_settings.get('plot_y_inches', 4))
        out['plot_legend'] = settings.getboolean('plot_legend', global_settings.get('plot_legend', True))
        out['cmap'] = settings.get('cmap', global_settings.get('cmap', 'gist_rainbow'))
        out['dpi'] = settings.get('dpi', global_settings.get('dpi', 300))



        return out


