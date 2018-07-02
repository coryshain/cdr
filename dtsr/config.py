import sys
import os
import shutil
from itertools import chain, combinations
import configparser
from numpy import inf

from dtsr.formula import Formula
from dtsr.kwargs import DTSR_INITIALIZATION_KWARGS, DTSRMLE_INITIALIZATION_KWARGS, DTSRBAYES_INITIALIZATION_KWARGS


# Thanks to Brice (https://stackoverflow.com/users/140264/brice) at Stack Overflow for this
def powerset(iterable):
    xs = list(iterable)
    return chain.from_iterable(combinations(xs,n) for n in range(1, len(xs)+1))

class Config(object):
    """
    Parses an *.ini file and stores settings needed to define a set of DTSR experiments.

    :param path: Path to *.ini file
    """

    def __init__(self, path):
        self.current_model = None

        config = configparser.ConfigParser()
        config.optionxform = str
        config.read(path)

        data = config['data']
        global_settings = config['global_settings']
        dtsr_settings = config['dtsr_settings']

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

        self.history_length = data.getint('history_length', 128)

        ###################
        # Global Settings #
        ###################

        self.outdir = global_settings.get('outdir', None)
        if self.outdir is None:
            self.outdir = global_settings.get('logdir', None)
        if self.outdir is None:
            self.outdir = './dtsr_model/'
        if not os.path.exists(self.outdir):
            os.makedirs(self.outdir)
        if os.path.realpath(path) != os.path.realpath(self.outdir + '/config.ini'):
            shutil.copy2(path, self.outdir + '/config.ini')
        self.use_gpu_if_available = global_settings.getboolean('use_gpu_if_available', True)

        #################
        # DTSR Settings #
        #################

        self.global_dtsr_settings = self.build_dtsr_settings(dtsr_settings)

        ###########
        # Filters #
        ###########

        if 'filters' in config:
            filters = config['filters']
            self.filter_map = {}
            for f in filters:
                self.filter_map[f] = [x.strip() for x in filters[f].strip().split(',')]

        ############
        # Model(s) #
        ############

        self.models = {}
        self.model_list = []
        for model_field in [m for m in config.keys() if m.startswith('model_')]:
            self.models[model_field[6:]] = self.build_dtsr_settings(config[model_field], add_defaults=False)
            self.model_list.append(model_field[6:])
            if 'ablate' in config[model_field]:
                for ablated in powerset(config[model_field]['ablate'].strip().split()):
                    ablated = list(ablated)
                    name = model_field[6:] + '!' + '!'.join(ablated)
                    formula = Formula(config[model_field]['formula'])
                    formula.ablate_impulses(ablated + ['rate'])
                    new_model = self.models[model_field[6:]].copy()
                    new_model['formula'] = str(formula)
                    self.models[name] = new_model
                    self.model_list.append(name)

        if 'irf_name_map' in config:
            self.irf_name_map = {}
            for x in config['irf_name_map']:
                self.irf_name_map[x] = config['irf_name_map'][x]
        else:
            self.irf_name_map = None

    def __getitem__(self, item):
        if self.current_model is None:
            return self.global_dtsr_settings[item]
        if self.current_model in self.models:
            return self.models[self.current_model].get(item, self.global_dtsr_settings[item])
        raise ValueError('There is no model named "%s" defined in the config file.' %self.current_model)

    def __str__(self):
        out = ''
        V = vars(self)
        for x in V:
            out += '%s: %s\n' %(x, V[x])
        return out

    def set_model(self, model_name=None):
        if model_name is None or model_name in self.models:
            self.current_model = model_name
        else:
            raise ValueError('There is no model named "%s" defined in the config file.' %model_name)

    def build_dtsr_settings(self, settings, add_defaults=True):
        out = {}

        # Core fields
        out['formula'] = settings.get('formula', None)
        if 'network_type' in settings or add_defaults:
            out['network_type'] = settings.get('network_type', 'bayes')

        # DTSR initialization keyword arguments
        for kwarg in DTSR_INITIALIZATION_KWARGS:
            if kwarg.in_settings(settings) or add_defaults:
                out[kwarg.key] = kwarg.kwarg_from_config(settings)

        # DTSRMLE initialization keyword arguments
        for kwarg in DTSRMLE_INITIALIZATION_KWARGS:
            if kwarg.in_settings(settings) or add_defaults:
                out[kwarg.key] = kwarg.kwarg_from_config(settings)

        # DTSRBayes initialization keyword arguments
        for kwarg in DTSRBAYES_INITIALIZATION_KWARGS:
            if kwarg.in_settings(settings) or add_defaults:
                out[kwarg.key] = kwarg.kwarg_from_config(settings)

        # Plotting defaults
        if 'plot_n_time_units' in settings or add_defaults:
            out['plot_n_time_units'] = settings.getfloat('plot_n_time_units', 2.5)
        if 'plot_n_points_per_time_unit' in settings or add_defaults:
            out['plot_n_points_per_time_unit'] = settings.getfloat('plot_n_points_per_time_unit',
                                                                                          500)
        if 'plot_x_inches' in settings or add_defaults:
            out['plot_x_inches'] = settings.getfloat('plot_x_inches', 7)
        if 'plot_y_inches' in settings or add_defaults:
            out['plot_y_inches'] = settings.getfloat('plot_y_inches', 5)
        if 'cmap' in settings or add_defaults:
            out['cmap'] = settings.get('cmap', 'gist_rainbow')

        return out


