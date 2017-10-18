import sys
import os
import shutil
import configparser
from numpy import inf

class Config(object):
    def __init__(self, path):
        config = configparser.ConfigParser()
        config.read(path)

        data = config['data']
        settings = config['settings']
        filters = config['filters']

        ## Data (Required)
        self.X_train = data.get('X_train')
        self.X_dev = data.get('X_dev')
        self.X_test = data.get('X_test')

        self.y_train = data.get('y_train')
        self.y_dev = data.get('y_dev')
        self.y_test = data.get('y_test')

        series_ids = data.get('series_ids')
        self.series_ids = series_ids.strip().split()

        ## Settings (Required)
        self.logdir = settings.get('logdir', 'log')
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)
        shutil.copy2(sys.argv[1], self.logdir + '/config.ini')
        self.network_type = settings.get('network_type', 'mle')
        self.conv_func = settings.get('conv_func', 'gamma')
        self.loss = settings.get('loss', 'MSE')
        self.modulus = settings.getint('modulus', 4)
        self.log_random = settings.getboolean('log_random', False)
        self.log_convolution_plots = settings.getboolean('log_convolution_plots', False)
        self.n_epoch_train = settings.getint('n_epoch_train', 50)
        self.n_epoch_finetune = settings.getint('n_epoch_finetune', 250)

        ## Filters (optional)
        self.filter_map = {}
        for f in filters:
            self.filter_map[f] = [x.strip() for x in filters[f].strip().split(',')]

        ## Model(s) (at least one required)
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
