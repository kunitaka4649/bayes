import configparser
import sys
import csv

class Configure:
    def __init__(self, config_file):
        config = configparser.SafeConfigParser()
        with open(config_file) as f:
            config.readfp(f)
            self.config = config

    def get_is_train(self):
        return self.config.getboolean('Files', 'is_train')

    def get_load_path(self):
        return self.config['Files']['load_path']

    def get_save_path(self):
        return self.config['Files']['save_path']

    def get_input_path(self):
        return self.config['Files']['input_path']

    def input_parser(self, input_path):
        targets, tokens = [], []
        with open(input_path, encoding='utf-8', newline='') as f:
            for cols in csv.reader(f, delimiter='\t'):
                targets.append(cols[0])
                tokens.append(cols[1])
        return targets, tokens