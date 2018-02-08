import sys
import os
import shutil
import configparser
from numpy import inf

class Config(object):
    def __init__(self, path):
        config = configparser.ConfigParser()
        config.optionxform = str
        config.read(path)

        data = config['data']
        settings = config['settings']
        filters = config['filters']

        ## Data
        self.X_train = data.get('X_train', None)
        self.X_dev = data.get('X_dev', None)
        self.X_test = data.get('X_test', None)

        self.y_train = data.get('y_train', None)
        self.y_dev = data.get('y_dev', None)
        self.y_test = data.get('y_test', None)

        split_ids = data.get('split_ids')
        self.split_ids = split_ids.strip().split()
        series_ids = data.get('series_ids')
        self.series_ids = series_ids.strip().split()

        ## Settings
        self.logdir = settings.get('logdir', 'log')
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        if not os.path.exists(self.logdir + '/config.ini'):
            shutil.copy2(path, self.logdir + '/config.ini')
        self.pc = settings.getboolean('pc', False)
        self.modulus = settings.getint('modulus', 4)
        self.network_type = settings.get('network_type', 'nn')
        self.float_type = settings.get('float_type', 'float32')
        self.int_type = settings.get('int_type', 'int32')
        self.history_length = settings.get('history_length', 128)
        if self.history_length in ['None', 'inf']:
            self.history_length = inf
        else:
            try:
                self.history_length = int(self.history_length)
            except:
                raise ValueError('history_length parameter invalid: %s' %self.history_length)
        self.low_memory = settings.getboolean('low_memory', False)
        self.n_iter = settings.getint('n_iter', 200)
        self.minibatch_size = settings.get('minibatch_size', 128)
        if self.minibatch_size in ['None', 'inf']:
            self.minibatch_size = inf
        else:
            try:
                self.minibatch_size = int(self.minibatch_size)
            except:
                raise ValueError('minibatch_size parameter invalid: %s' %self.minibatch_size)
        self.eval_minibatch_size = settings.get('eval_minibatch_size', 100000)
        if self.eval_minibatch_size in ['None', 'inf']:
            self.eval_minibatch_size = inf
        else:
            try:
                self.eval_minibatch_size = int(self.eval_minibatch_size)
            except:
                raise ValueError('eval_minibatch_size parameter invalid: %s' % self.eval_minibatch_size)
        self.log_freq = settings.getint('log_freq', 1)
        self.log_random = settings.getboolean('log_random', False)
        self.save_freq = settings.getint('save_freq', 1)
        self.plot_x_inches = settings.getfloat('plot_x_inches', 7)
        self.plot_y_inches = settings.getfloat('plot_y_inches', 5)
        self.cmap = settings.get('cmap', 'gist_rainbow')
        self.use_gpu_if_available = settings.getboolean('use_gpu_if_available', True)
        self.validate_delta_t = settings.getboolean('validate_delta_t', True)
        self.optim = settings.get('optim', 'Adam')
        if self.optim == 'None':
            self.optim = None
        self.learning_rate = settings.getfloat('learning_rate', 0.01)
        self.learning_rate_min = settings.get('learning_rate_min', 1e-4)
        if self.learning_rate_min in ['None', '-inf']:
            self.learning_rate_min = -inf
        else:
            try:
                self.learning_rate_min = float(self.learning_rate_min)
            except:
                raise ValueError('learning_rate_min parameter invalid: %s' %self.learning_rate_min)
        self.lr_decay_family = settings.get('lr_decay_family', None)
        if self.lr_decay_family == 'None':
            self.lr_decay_family = None
        self.lr_decay_steps = settings.getint('lr_decay_steps', 25)
        self.lr_decay_rate = settings.getfloat('lr_decay_rate', .1)
        self.lr_decay_staircase = settings.getboolean('lr_decay_staircase', False)
        self.init_sd = settings.getfloat('init_sd', 1.)

        ## NN settings
        self.loss = settings.get('loss', 'MSE')
        self.regularizer = settings.get('regularizer', None)
        if self.regularizer == 'None':
            self.regularizer = None
        self.regularizer_scale = settings.getfloat('regularizer_scale', 0.01)

        ## Bayes net settings
        self.inference_name = settings.get('inference_name', None)
        self.n_samples = settings.getint('n_samples', 1)
        self.n_samples_eval = settings.getint('n_samples_eval', 128)
        self.intercept_prior_sd = settings.getfloat('intercept_prior_sd', 1.)
        self.coef_prior_sd = settings.getfloat('coef_prior_sd', 1.)
        self.conv_prior_sd = settings.getfloat('conv_prior_sd', 1.)
        self.y_scale_prior_sd = settings.getfloat('y_scale_prior_sd', 1.)
        self.mv = settings.getboolean('mv', False)
        self.mv_ran = settings.getboolean('mv_ran', False)
        self.y_scale = settings.get('y_scale', None)
        if self.y_scale == 'None':
            self.y_scale = None
        if self.y_scale is not None:
            self.y_scale = float(self.y_scale)
        self.mh_proposal_sd = settings.getfloat('mh_proposal_sd', 1.)


        ## Filters
        self.filter_map = {}
        for f in filters:
            self.filter_map[f] = [x.strip() for x in filters[f].strip().split(',')]

        ## Model(s)
        self.models = {}
        self.model_list = [m[6:] for m in config.sections() if m.startswith('model_')]
        for model_field in [m for m in config.keys() if m.startswith('model_')]:
            self.models[model_field[6:]] = {}
            for f in config[model_field]:
                self.models[model_field[6:]][f] = config[model_field][f]

        if 'fixef_name_map' in config:
            self.fixef_name_map = {}
            for x in config['fixef_name_map']:
                self.fixef_name_map[x] = config['fixef_name_map'][x]
        else:
            self.fixef_name_map = None
