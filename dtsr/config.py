import sys
import os
import shutil
import configparser
from numpy import inf

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
        self.model_list = [m[6:] for m in config.sections() if m.startswith('model_')]
        for model_field in [m for m in config.keys() if m.startswith('model_')]:
            self.models[model_field[6:]] = self.build_dtsr_settings(config[model_field], add_defaults=False)

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

        # a. Formula
        out['formula'] = settings.get('formula', None)

        # b. Implementation
        if 'float_type' in settings or add_defaults:
            out['float_type'] = settings.get('float_type', 'float32')
        if 'int_type' in settings or add_defaults:
            out['int_type'] = settings.get('int_type', 'int32')
        if 'n_iter' in settings or add_defaults:
            out['n_iter'] = settings.getint('n_iter', 1000)
        if 'minibatch_size' in settings or add_defaults:
            out['minibatch_size'] = settings.get('minibatch_size', 1024)
            if out['minibatch_size'] in ['None', 'inf']:
                out['minibatch_size'] = inf
            else:
                try:
                    out['minibatch_size'] = int(out['minibatch_size'])
                except:
                    raise ValueError('minibatch_size parameter invalid: %s' % out['minibatch_size'])
        if 'eval_minibatch_size' in settings or add_defaults:
            out['eval_minibatch_size'] = settings.get('eval_minibatch_size', 100000)
            if out['eval_minibatch_size'] in ['None', 'inf']:
                out['eval_minibatch_size'] = inf
            else:
                try:
                    out['eval_minibatch_size'] = int(out['eval_minibatch_size'])
                except:
                    raise ValueError(
                        'eval_minibatch_size parameter invalid: %s' % out['eval_minibatch_size'])
        if 'n_interp' in settings or add_defaults:
            out['n_interp'] = settings.getint('n_interp', 64)
        if 'optim_name' in settings or add_defaults:
            out['optim_name'] = settings.get('optim_name', 'Nadam')
            if out['optim_name'] == 'None':
                out['optim_name'] = None
        if 'optim_epsilon' in settings or add_defaults:
            out['optim_epsilon'] = settings.get('optim_epsilon', 0.01)
        if 'learning_rate' in settings or add_defaults:
            out['learning_rate'] = settings.getfloat('learning_rate', 0.001)
        if 'learning_rate_min' in settings or add_defaults:
            out['learning_rate_min'] = settings.get('learning_rate_min', 1e-5)
            if out['learning_rate_min'] in ['None', '-inf']:
                out['learning_rate_min'] = -inf
            else:
                try:
                    out['learning_rate_min'] = float(out['learning_rate_min'])
                except:
                    raise ValueError(
                        'learning_rate_min parameter invalid: %s' % out['learning_rate_min'])
        if 'lr_decay_family' in settings or add_defaults:
            out['lr_decay_family'] = settings.get('lr_decay_family', None)
            if out['lr_decay_family'] == 'None':
                out['lr_decay_family'] = None
        if 'lr_decay_steps' in settings or add_defaults:
            out['lr_decay_steps'] = settings.getint('lr_decay_steps', 100)
        if 'lr_decay_rate' in settings or add_defaults:
            out['lr_decay_rate'] = settings.getfloat('lr_decay_rate', .5)
        if 'lr_decay_staircase' in settings or add_defaults:
            out['lr_decay_staircase'] = settings.getboolean('lr_decay_staircase', False)
        if 'ema_decay' in settings or add_defaults:
            out['ema_decay'] = settings.getfloat('ema_decay', 0.999)

        # c. Model hyperparameters
        if 'network_type' in settings or add_defaults:
            out['network_type'] = settings.get('network_type', 'bayes')
        if 'pc' in settings or add_defaults:
            out['pc'] = settings.getboolean('pc', False)
        if 'init_sd' in settings or add_defaults:
            out['init_sd'] = settings.getfloat('init_sd', 0.01)
        if 'intercept_init' in settings or add_defaults:
            out['intercept_init'] = settings.get('intercept_init', None)
        if 'intercept_init' in settings or add_defaults:
            if out['intercept_init'] in [None, 'None']:
                out['intercept_init'] = None
            else:
                try:
                    out['intercept_init'] = float(out['intercept_init'])
                except:
                    raise ValueError('intercept_init parameter invalid: %s' % out['intercept_init'])

        # d. Logging
        if 'log_freq' in settings or add_defaults:
            out['log_freq'] = settings.getint('log_freq', 1)
        if 'log_random' in settings or add_defaults:
            out['log_random'] = settings.getboolean('log_random', True)
        if 'save_freq' in settings or add_defaults:
            out['save_freq'] = settings.getint('save_freq', 1)

        # e. Plotting
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

        # f. MLE implementation
        if 'loss_type' in settings or add_defaults:
            out['loss_type'] = settings.get('loss_type', 'MSE')
        if 'regularizer_name' in settings or add_defaults:
            out['regularizer_name'] = settings.get('regularizer_name', None)
            if out['regularizer_name'] == 'None':
                out['regularizer_name'] = None
        if 'regularizer_scale' in settings or add_defaults:
            out['regularizer_scale'] = settings.getfloat('regularizer_scale', 0.01)

        # g. Bayes net implementation
        if 'inference_name' in settings or add_defaults:
            out['inference_name'] = settings.get('inference_name', None)
        if 'declare_priors' in settings or add_defaults:
            out['declare_priors'] = settings.getboolean('declare_priors', True)
        if 'n_samples' in settings or add_defaults:
            out['n_samples'] = settings.getint('n_samples', 1)
        if 'n_samples_eval' in settings or add_defaults:
            out['n_samples_eval'] = settings.getint('n_samples_eval', 128)

        if 'intercept_prior_sd' in settings or add_defaults:
            out['intercept_prior_sd'] = settings.get('intercept_prior_sd', None)
            if out['intercept_prior_sd'] in [None, 'None']:
                out['intercept_prior_sd'] = None
            else:
                try:
                    out['intercept_prior_sd'] = float(out['intercept_prior_sd'])
                except:
                    raise ValueError(
                        'intercept_prior_sd parameter invalid: %s' % out['intercept_prior_sd'])

        if 'coef_prior_sd' in settings or add_defaults:
            out['coef_prior_sd'] = settings.get('coef_prior_sd', None)
            if out['coef_prior_sd'] in [None, 'None']:
                out['coef_prior_sd'] = None
            else:
                try:
                    out['coef_prior_sd'] = float(out['coef_prior_sd'])
                except:
                    raise ValueError('coef_prior_sd parameter invalid: %s' % out['coef_prior_sd'])

        if 'prior_sd_scaling_coefficient' in settings or add_defaults:
            out['prior_sd_scaling_coefficient'] = settings.getfloat('prior_sd_scaling_coefficient', 1.)

        if 'conv_prior_sd' in settings or add_defaults:
            out['conv_prior_sd'] = settings.getfloat('conv_prior_sd', 1.)

        if 'y_scale_init' in settings or add_defaults:
            out['y_scale_init'] = settings.get('y_scale_init', None)
            if out['y_scale_init'] in [None, 'None']:
                out['y_scale_init'] = None
            else:
                try:
                    out['y_scale_init'] = float(out['y_scale_init'])
                except:
                    raise ValueError('y_scale_init parameter invalid: %s' % out['y_scale_init'])

        if 'y_scale_trainable' in settings or add_defaults:
            out['y_scale_trainable'] = settings.getboolean('y_scale_trainable', True)
        if 'y_scale_prior_sd' in settings or add_defaults:
            out['y_scale_prior_sd'] = settings.get('y_scale_prior_sd', None)
            if out['y_scale_prior_sd'] in [None, 'None']:
                out['y_scale_prior_sd'] = None
            else:
                try:
                    out['y_scale_prior_sd'] = float(out['y_scale_prior_sd'])
                except:
                    raise ValueError(
                        'y_scale_prior_sd parameter invalid: %s' % out['y_scale_prior_sd'])

        if 'y_scale_prior_sd_scaling_coefficient' in settings or add_defaults:
            out['y_scale_prior_sd_scaling_coefficient'] = settings.getfloat('y_scale_prior_sd_scaling_coefficient', 1.)
        if 'ranef_to_fixef_prior_sd_ratio' in settings or add_defaults:
            out['ranef_to_fixef_prior_sd_ratio'] = settings.getfloat('ranef_to_fixef_prior_sd_ratio', 1.)
        if 'posterior_to_prior_sd_ratio' in settings or add_defaults:
            out['posterior_to_prior_sd_ratio'] = settings.getfloat('posterior_to_prior_sd_ratio', 0.01)

        if 'mh_proposal_sd' in settings or add_defaults:
            out['mh_proposal_sd'] = settings.get('mh_proposal_sd', None)
            if out['mh_proposal_sd'] in [None, 'None']:
                out['mh_proposal_sd'] = None
            else:
                try:
                    out['mh_proposal_sd'] = float(out['mh_proposal_sd'])
                except:
                    raise ValueError('mh_proposal_sd parameter invalid: %s' % out['mh_proposal_sd'])

        if 'mv' in settings or add_defaults:
            out['mv'] = settings.getboolean('mv', False)
        if 'mv_ran' in settings or add_defaults:
            out['mv_ran'] = settings.getboolean('mv_ran', False)
        if 'asymmetric_error' in settings or add_defaults:
            out['asymmetric_error'] = settings.getboolean('asymmetric_error', False)

        return out

