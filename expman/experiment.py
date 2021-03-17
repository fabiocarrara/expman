import argparse
import ast
import os
import hashlib
import shutil
import numbers
from glob import glob
from io import StringIO

import numpy as np
import pandas as pd


hash_naming = False

def use_hash_naming(use_hashes=True):
    global hash_naming
    assert isinstance(use_hashes, bool), "Value must be a boolean."
    hash_naming = use_hashes

def _guessed_cast(x):
    try:
        return ast.literal_eval(x)
    except:
        return x

def exp_filter(string):
    if '=' not in string:
        raise argparse.ArgumentTypeError(
            'Filter {} is not in format <param1>=<value1>[, <param2>=<value2>[, ...]]'.format(string))
    filters = string.split(',')
    filters = map(lambda x: x.split('='), filters)
    filters = {k: _guessed_cast(v) for k, v in filters}
    return filters


class Experiment:

    PARAM_FILENAME = 'params.json'

    @staticmethod
    def _abbr(name, value, params):

        def prefix_len(a, b):
            return len(os.path.commonprefix((a, b)))

        prefix = [name[:prefix_len(p, name) + 1] for p in params.keys() if p != name]
        prefix = max(prefix, key=len) if len(prefix) > 0 else name

        sep = ''
        if isinstance(value, str):
            sep = '-'
        elif isinstance(value, numbers.Number):
            value = '{:g}'.format(value)
            sep = '-' if prefix[-1].isdigit() else ''
        elif isinstance(value, (list, tuple)):
            value = map(str, value)
            value = map(lambda v: v.replace(os.sep, '|'), value)
            value = ','.join(list(value))
            sep = '-'

        return prefix, sep, value

    @classmethod
    def abbreviate(cls, params):
        if isinstance(params, pd.DataFrame):
            params = params.iloc[0]
            params = params.replace({np.nan: None})

        if hash_naming:
            exp_name = hashlib.md5(str(sorted(params.items())).encode()).hexdigest()
        else:
            abbrev_params = {k: '{}{}{}'.format(*cls._abbr(k, v, params)) for k, v in params.items()}
            abbrev = sorted(abbrev_params.values())
            exp_name = '_'.join(abbrev)
        
        return exp_name

    @classmethod
    def from_dir(cls, exp_dir):
        root = os.path.dirname(exp_dir.rstrip('/'))
        params = os.path.join(exp_dir, cls.PARAM_FILENAME)

        assert os.path.exists(exp_dir), "Experiment directory not found: '{}'".format(exp_dir)
        assert os.path.exists(params), "Empty run directory found: '{}'".format(params)

        params = cls._read_params(params)
        exp = cls(params, root=root, create=False)
        return exp

    @classmethod
    def is_exp_dir(cls, exp_dir):
        if os.path.isdir(exp_dir):
            params = os.path.join(exp_dir, cls.PARAM_FILENAME)
            if os.path.exists(params):
                return True

        return False
    
    @classmethod
    def update_exp_dir(cls, exp_dir):
        exp_dir = exp_dir.rstrip('/')
        root = os.path.dirname(exp_dir)
        name = os.path.basename(exp_dir)
        params = os.path.join(exp_dir, cls.PARAM_FILENAME)

        assert os.path.exists(exp_dir), "Experiment directory not found: '{}'".format(exp_dir)
        assert os.path.exists(params), "Empty run directory found: '{}'".format(params)

        params = cls._read_params(params)
        new_name = cls.abbreviate(params)
        
        if name != new_name:
            new_exp_dir = os.path.join(root, new_name)
            assert not os.path.exists(new_exp_dir), \
                "Destination experiment directory already exists: '{}'".format(new_exp_dir)
            
            print('Renaming:\n  {} into\n  {}'.format(exp_dir, new_exp_dir))
            shutil.move(exp_dir, new_exp_dir)

    def __init__(self, params, root='runs/', ignore=(), create=True):
        # relative dir containing this run
        self.root = root
        # params to be ignored in the run naming
        self.ignore = ignore
        # parameters of this run
        if isinstance(params, argparse.Namespace):
            params = vars(params)
        
        def _sanitize(v):
            return tuple(v) if isinstance(v, list) else v
            
        params = {k: _sanitize(v) for k, v in params.items() if k not in self.ignore}
        self.params = pd.Series(params, name='params')

        # whether to create the run directory if not exists
        self.create = create

        self.name = self.abbreviate(self.params)
        self.path = os.path.join(self.root, self.name)
        self.existing = os.path.exists(self.path)
        self.found = self.existing

        if not self.existing:
            if self.create:
                os.makedirs(self.path)
                self.write_params()
                self.existing = True
            else:
                print("Run directory '{}' not found, but not created.".format(self.path))

        else:
            param_fname = self.path_to(self.PARAM_FILENAME)
            assert os.path.exists(param_fname), "Empty run, parameters not found: '{}'".format(param_fname)
            self.params = self._read_params(param_fname)


    def __str__(self):
        s = StringIO()
        print('Experiment Dir: {}'.format(self.path), file=s)
        print('Params:', file=s)
        with pd.option_context('display.width', None), pd.option_context('max_columns', None):
            self.params.to_string(s)

        return s.getvalue()

    def __repr__(self):
        return self.__str__()

    def path_to(self, path):
        path = os.path.join(self.path, path)
        return path

    def add_parameter(self, key, value):
        assert key not in self.params, "Parameter already exists: '{}'".format(key)
        self.params[key] = value
        self._update_run_dir()
        self.write_params()

    def rename_parameter(self, key, new_key):
        assert key in self.params, "Cannot rename non-existent parameter: '{}'".format(key)
        assert new_key not in self.params, "Destination name for parameter exists: '{}'".format(key)

        self.params[new_key] = self.params[key]
        del self.params[key]

        self._update_run_dir()
        self.write_params()

    def remove_parameter(self, key):
        assert key in self.params, "Cannot remove non-existent parameter: '{}'".format(key)
        del self.params[key]
        self._update_run_dir()
        self.write_params()

    def _update_run_dir(self):
        old_run_dir = self.path
        if self.existing:
            self.name = self.abbreviate(self.params)
            self.path = os.path.join(self.root, self.name)
            assert not os.path.exists(self.path), "Cannot rename run, new name exists: '{}'".format(self.path)
            shutil.move(old_run_dir, self.path)
    
    @staticmethod
    def _read_params(path):
        # read json to pd.Series
        params = pd.read_json(path, typ='series')
        # transform lists to tuples (for hashability)
        params = params.apply(lambda x: tuple(x) if isinstance(x, list) else x)
        return params
    
    def write_params(self):
        # write Series as json
        self.params.to_json(self.path_to(self.PARAM_FILENAME))

def test():
    parser = argparse.ArgumentParser(description='Experiment Manager Test')
    parser.add_argument('-e', '--epochs', type=int, default=70)
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-m', '--model', choices=('1d-conv', 'paper'), default='1d-conv')
    parser.add_argument('-s', '--seed', type=int, default=23)
    parser.add_argument('--no-cuda', action='store_true')
    parser.set_defaults(no_cuda=False)
    args = parser.parse_args()

    run = Experiment(args, root='prova', ignore=['no_cuda'])
    print(run)
    print(run.path_to('ckpt/best.h5'))
