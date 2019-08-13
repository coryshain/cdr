import sys
import os
import shutil
from itertools import chain, combinations
if sys.version_info[0] == 2:
    import ConfigParser as configparser
else:
    import configparser

from .formula import Formula
from .kwargs import CDR_INITIALIZATION_KWARGS, CDRMLE_INITIALIZATION_KWARGS, CDRBAYES_INITIALIZATION_KWARGS


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
        if 'cdr_settings' not in config:
            config['cdr_settings'] = {}
        cdr_settings = config['cdr_settings']

        ########
        # Data #
        ########

        self.X_train = data.get('X_train')
        self.X_dev = data.get('X_dev', None)
        self.X_test = data.get('X_test', None)

        self.y_train = data.get('y_train')
        self.y_dev = data.get('y_dev', None)
        self.y_test = data.get('y_test', None)

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
        if os.path.realpath(path) != os.path.realpath(self.outdir + '/config.ini'):
            shutil.copy2(path, self.outdir + '/config.ini')
        self.use_gpu_if_available = global_settings.getboolean('use_gpu_if_available', True)

        #################
        # CDR Settings #
        #################

        self.global_cdr_settings = self.build_cdr_settings(cdr_settings)

        ############
        # Model(s) #
        ############

        self.models = {}
        self.model_list = []
        for model_field in [m for m in config.keys() if m.startswith('model_')]:
            model_name = model_field[6:]
            is_cdr = model_name.startswith('CDR') or model_name.startswith('DTSR')
            self.models[model_name] = self.build_cdr_settings(config[model_field], add_defaults=False, is_cdr=is_cdr)
            self.model_list.append(model_name)
            if 'ablate' in config[model_field]:
                for ablated in powerset(config[model_field]['ablate'].strip().split()):
                    ablated = list(ablated)
                    new_name = model_name + '!' + '!'.join(ablated)
                    formula = Formula(config[model_field]['formula'])
                    formula.ablate_impulses(ablated)
                    new_model = self.models[model_name].copy()
                    new_model['formula'] = str(formula)
                    self.models[new_name] = new_model
                    self.model_list.append(new_name)

        if 'irf_name_map' in config:
            self.irf_name_map = {}
            for x in config['irf_name_map']:
                self.irf_name_map[x] = config['irf_name_map'][x]
        else:
            self.irf_name_map = None

    def __getitem__(self, item):
        if self.current_model is None:
            return self.global_cdr_settings[item]
        if self.current_model in self.models:
            return self.models[self.current_model].get(item, self.global_cdr_settings[item])
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

    def build_cdr_settings(self, settings, add_defaults=True, is_cdr=True):
        """
        Given a settings object parsed from a config file, compute CDR parameter dictionary.

        :param settings: settings from a ``ConfigParser`` object.
        :param add_defaults: ``bool``; whether to supply defaults for parameters missing from **settings**.
        :param is_cdr: ``bool``; whether this is a CDR model.
        :return: ``dict``; dictionary of settings key-value pairs.
        """

        out = {}

        # Core fields
        out['formula'] = settings.get('formula', None)
        if is_cdr and out['formula']:
            # Standardize the model string
            out['formula'] = str(Formula(out['formula']))
        if 'network_type' in settings or add_defaults:
            out['network_type'] = settings.get('network_type', 'mle')

        # CDR initialization keyword arguments
        for kwarg in CDR_INITIALIZATION_KWARGS:
            if kwarg.in_settings(settings) or add_defaults:
                out[kwarg.key] = kwarg.kwarg_from_config(settings)

        # CDRMLE initialization keyword arguments
        for kwarg in CDRMLE_INITIALIZATION_KWARGS:
            if kwarg.in_settings(settings) or add_defaults:
                out[kwarg.key] = kwarg.kwarg_from_config(settings)

        # CDRBayes initialization keyword arguments
        for kwarg in CDRBAYES_INITIALIZATION_KWARGS:
            if kwarg.in_settings(settings) or add_defaults:
                out[kwarg.key] = kwarg.kwarg_from_config(settings)

        # Plotting defaults
        if 'plot_n_time_units' in settings or add_defaults:
            out['plot_n_time_units'] = settings.getfloat('plot_n_time_units', 2.5)
        if 'plot_n_time_points' in settings or add_defaults:
            out['plot_n_time_points'] = settings.getfloat('plot_n_time_points', 1000)
        if 'plot_x_inches' in settings or add_defaults:
            out['plot_x_inches'] = settings.getfloat('plot_x_inches', 6)
        if 'plot_y_inches' in settings or add_defaults:
            out['plot_y_inches'] = settings.getfloat('plot_y_inches', 4)
        if 'cmap' in settings or add_defaults:
            out['cmap'] = settings.get('cmap', 'gist_rainbow')
        if 'dpi' in settings or add_defaults:
            out['dpi'] = settings.get('dpi', 300)

        return out


