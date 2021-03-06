import argparse
import os
import shutil
import numbers
from glob import glob
from io import StringIO

import numpy as np
import pandas as pd


hash_naming = False

def use_hash_naming(use_hashes):
    assert isinstance(use_hashes, bool), "Value must be a boolean."
    hash_naming = use_hashes
    

def exp_filter(string):
    if '=' not in string:
        raise argparse.ArgumentTypeError(
            'Filter {} is not in format <param1>=<value1>[, <param2>=<value2>[, ...]]'.format(string))
    filters = string.split(',')
    filters = map(lambda x: x.split('='), filters)
    filters = {k: v for k, v in filters}
    return filters


class Experiment:

    filenames = {
        'params': 'params.json',
        'log': 'log.csv',
        'results': 'results.csv',
        'ckpt': 'ckpt',
        'last': 'last.pth',
        'best': 'best.pth',
    }

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
            exp_name = str(hash(frozenset(params.items())))
        else:
            abbrev_params = {k: '{}{}{}'.format(*cls._abbr(k, v, params)) for k, v in params.items()}
            abbrev = sorted(abbrev_params.values())
            exp_name = '_'.join(abbrev)
        
        return exp_name

    @classmethod
    def collect_all(cls, exps, what, index=None):

        def collect(exp):
            params = exp.params.to_frame().transpose().infer_objects() # as DataFrame
            what_csv = exp.path_to(what)

            if os.path.exists(what_csv):
                stuff = pd.read_csv(what_csv, index_col=index)
            else:  # try globbing
                stuff = os.path.join(exp.path, what)
                stuff = list(glob(stuff))
                if len(stuff) == 0:
                    return pd.DataFrame()

                stuff = map(lambda x: pd.read_csv(x, index_col=index, float_precision='round_trip'), stuff)
                stuff = pd.concat(stuff, ignore_index=True)

            params['exp_id'] = collect.exp_id
            stuff['exp_id'] = collect.exp_id
            collect.exp_id += 1
            
            return pd.merge(params, stuff, on='exp_id')
            
        collect.exp_id = 0
        
        results = map(collect, exps)
        results = pd.concat(results, ignore_index=True, sort=False)
        
        # build minimal exp_name
        params = results.loc[:,:'exp_id'].drop('exp_id', axis=1)
        varying_params = params.loc[:, params.nunique() > 1]
        exp_name = varying_params.apply(cls.abbreviate, axis=1)
        idx = results.columns.get_loc('exp_id') + 1
        results.insert(idx, 'exp_name', exp_name)
        
        return results

    @classmethod
    def filter(cls, filters, exps):

        def __filter_exp(e):
            for param, value in filters.items():
                try:
                    p = e.params[param]
                    ptype = type(p)
                    if p != ptype(value):
                        return False
                except:
                    return False

            return True

        return filter(__filter_exp, exps)

    @classmethod
    def from_dir(cls, exp_dir):
        root = os.path.dirname(exp_dir.rstrip('/'))
        params = os.path.join(exp_dir, cls.filenames['params'])

        assert os.path.exists(exp_dir), "Experiment directory not found: '{}'".format(exp_dir)
        assert os.path.exists(params), "Empty run directory found: '{}'".format(params)

        params = cls._read_params(params)
        exp = cls(params, root=root, create=False)
        return exp

    @classmethod
    def gather(cls, root='runs/'):
        if cls.is_exp_dir(root):
            exps = [root, ]
        else:
            exps = glob(os.path.join(root, '*'))
            exps = filter(cls.is_exp_dir, exps)

        exps = map(lambda x: cls.from_dir(x), exps)
        exps = filter(lambda x: x.existing, exps)
        return exps

    @classmethod
    def is_exp_dir(cls, exp_dir):
        if os.path.isdir(exp_dir):
            params = os.path.join(exp_dir, cls.filenames['params'])
            if os.path.exists(params):
                return True

        return False
    
    @classmethod
    def update_exp_dir(cls, exp_dir):
        exp_dir = exp_dir.rstrip('/')
        root = os.path.dirname(exp_dir)
        name = os.path.basename(exp_dir)
        params = os.path.join(exp_dir, cls.filenames['params'])

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
            self.log = pd.DataFrame()
            self.results = pd.DataFrame()

            if self.create:
                os.makedirs(self.path)
                self.write_params()
                self.existing = True
            else:
                print("Run directory '{}' not found, but not created.".format(self.path))

        else:
            log_fname = self.path_to('log')
            param_fname = self.path_to('params')
            results_fname = self.path_to('results')

            assert os.path.exists(param_fname), "Empty run, parameters not found: '{}'".format(param_fname)

            self.params = self._read_params(param_fname)
            self.log = pd.read_csv(log_fname, index_col=0) if os.path.exists(log_fname) else pd.DataFrame()
            self.results = pd.read_csv(results_fname) if os.path.exists(results_fname) else pd.DataFrame()


    def __str__(self):
        s = StringIO()
        print('Experiment Dir: {}'.format(self.path), file=s)
        print('Params:', file=s)
        with pd.option_context('display.width', None), pd.option_context('max_columns', None):
            self.params.to_string(s)

        # if not self.results.empty:
        #     print('\nResults:', file=s)
        #     with pd.option_context('display.width', None), pd.option_context('max_columns', None):
        #         self.results.to_string(s, index=False)

        return s.getvalue()

    def __repr__(self):
        return self.__str__()

    def path_to(self, what):
        # assert what in self.filenames, "Unknown run resource: '{}'".format(what)
        basename = self.filenames.get(what, what)
        path = os.path.join(self.path, basename)
        return path

    def ckpt(self, which='best'):
        ckpt_path = os.path.join(self.path_to('ckpt'), self.filenames.get(which, which))
        return ckpt_path

    def push_log(self, metrics):
        ts = pd.to_datetime('now')
        metrics = pd.DataFrame(metrics, index=(ts,))
        self.log = pd.concat((self.log, metrics))
        self.log.to_csv(self.path_to('log'))

    def write_results(self, metrics):
        ts = pd.to_datetime('now')
        self.results = pd.DataFrame(metrics, index=(ts,))
        self.results.to_csv(self.path_to('results'))
        with pd.option_context('display.width', None), pd.option_context('max_columns', None):
            print(self.results)

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

    def require_csv(self, path, index=None):
        csv_path = self.path_to(path)
        if os.path.exists(csv_path):
            data = pd.read_csv(csv_path)
            data = data.set_index(index) if index else data
            return data, csv_path
        elif isinstance(index, str):
            index = pd.Index([], name=index)
        elif isinstance(index, (list, tuple)):
            empty = [[]] * len(index)
            index = pd.MultiIndex(levels=empty, labels=empty, names=index)
        
        return pd.DataFrame([], index=index), csv_path

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
        self.params.to_json(self.path_to('params'))

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
    print(run.ckpt('best'))
