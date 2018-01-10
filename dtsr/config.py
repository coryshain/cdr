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
        self.modulus = settings.getint('modulus', 4)
        self.network_type = settings.get('network_type', 'nn')
        self.float_type = settings.get('float_type', 'float32')
        self.int_type = settings.get('int_type', 'int32')
        self.history_length = settings.get('history_length', 100)
        try:
            self.history_length = int(self.history_length)
        except:
            pass
        self.low_memory = settings.getboolean('low_memory', False)
        self.n_epoch_train = settings.getint('n_epoch_train', 100)
        self.minibatch_size = settings.get('minibatch_size', 128)
        if self.minibatch_size == 'inf':
            self.minibatch_size = inf
        else:
            self.minibatch_size = int(self.minibatch_size)
        self.log_freq = settings.getint('log_freq', 1)
        self.log_random = settings.getboolean('log_random', False)
        self.save_freq = settings.getint('save_freq', 1)
        self.plot_x_inches = settings.getfloat('plot_x_inches', 7)
        self.plot_y_inches = settings.getfloat('plot_y_inches', 5)
        self.cmap = settings.get('cmap', 'gist_rainbow')
        self.use_gpu_if_available = settings.getboolean('use_gpu_if_available', True)
        self.validate_delta_t = settings.getboolean('validate_delta_t', True)
        ## NN settings
        self.optim = settings.get('optim', 'Adam')
        self.learning_rate = settings.getfloat('learning_rate', 0.001)
        self.learning_rate_decay_factor = settings.getfloat('learning_rate_decay_factor', 0.)
        self.learning_rate_decay_family = settings.get('learning_rate_decay_family', None)
        self.learning_rate_min = settings.getfloat('learning_rate_min', 1e-4)
        self.init_sd = settings.getfloat('init_sd', .1)
        self.loss = settings.get('loss', 'MSE')
        ## Bayes net settings
        self.inference_name = settings.get('inference_name', None)
        self.n_samples = settings.getint('n_samples', 1)
        self.n_samples_eval = settings.getint('n_samples_eval', 100)
        self.conv_prior_sd = settings.getfloat('conv_prior_sd', 1.)
        self.coef_prior_sd = settings.getfloat('coef_prior_sd', 1.)
        self.y_sigma_scale = settings.getfloat('y_sigma_scale', 0.5)


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
