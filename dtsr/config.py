import sys
import os
import shutil
import configparser

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
        self.network_type = settings.get('network_type', 'mle')
        self.conv_func = settings.get('conv_func', 'gamma')
        self.loss = settings.get('loss', 'MSE')
        self.modulus = settings.getint('modulus', 4)
        self.log_random = settings.getboolean('log_random', False)
        self.log_convolution_plots = settings.getboolean('log_convolution_plots', False)
        self.n_epoch_train = settings.getint('n_epoch_train', 100)
        self.n_epoch_tune = settings.getint('n_epoch_tune', 100)
        self.minibatch_size = settings.getint('minibatch_size', 128)
        self.plot_x_inches = settings.getfloat('plot_x_inches', 7)
        self.plot_y_inches = settings.getfloat('plot_y_inches', 5)
        self.cmap = settings.get('cmap', 'gist_earth')
        self.use_gpu_if_available = settings.getboolean('use_gpu_if_available', True)

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
