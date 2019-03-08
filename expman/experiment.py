import argparse
import os
import shutil
from glob import glob
from io import StringIO

import pandas as pd


class Experiment:
    filenames = {
        'params': 'params.csv',
        'log': 'log.csv',
        'results': 'results.csv',
        'ckpt': 'ckpt',
        'last': 'last.pth',
        'best': 'best.pth',
    }

    @staticmethod
    def _abbr(name, value, params, main):
        is_main_param = main == name

        def prefix_len(a, b):
            return len(os.path.commonprefix((a, b)))

        if is_main_param:
            prefix = ''
        else:
            prefix = [name[:prefix_len(p, name) + 1] for p in params.keys() if p != name]
            prefix = max(prefix, key=lambda x: len(x)) if len(prefix) > 0 else name

        sep = '-' if (type(value) == str and not is_main_param) else ''
        return prefix, sep, str(value)

    @classmethod
    def abbreviate(cls, params, main=None):
        if isinstance(params, pd.DataFrame):
            params = params.iloc[0]

        if main not in params:
            main = None

        abbrev_params = {k: '{}{}{}'.format(*cls._abbr(k, v, params, main)) for k, v in params.items()}

        if main:
            abbrev = []
            main_param = abbrev_params[main]
            secondary_params = sorted(v for k, v in abbrev_params.items() if k != main)
            abbrev.append(main_param)
            abbrev.extend(secondary_params)
        else:
            abbrev = abbrev_params.values()

        abbrev = '_'.join(abbrev)
        return abbrev

    @classmethod
    def collect_all(cls, exps, what, index=None):

        def collect(exp):
            params = pd.read_csv(exp.path_to('params'))
            what_csv = exp.path_to(what)
            if not os.path.exists(what_csv):
                return pd.DataFrame()

            stuff = pd.read_csv(what_csv, index_col=index)
            params['__tmp_key'] = 0
            stuff['__tmp_key'] = 0
            return pd.merge(params, stuff, on='__tmp_key').drop('__tmp_key', axis=1)

        results = map(collect, exps)
        results = pd.concat(results, ignore_index=True, sort=False)
        return results

    @classmethod
    def filter(cls, filters, exps):

        def __filter_exp(e):
            for param, value in filters:
                p = e.params.loc[0, param]
                ptype = type(p)

                if p != ptype(value):
                    return False

            return True

        return filter(__filter_exp, exps)

    @classmethod
    def from_dir(cls, exp_dir, main=None):
        root = os.path.dirname(exp_dir.rstrip('/'))
        params = os.path.join(exp_dir, cls.filenames['params'])

        assert os.path.exists(exp_dir), "Experiment directory not found: '{}'".format(exp_dir)
        assert os.path.exists(params), "Empty run directory found: '{}'".format(params)

        params = pd.read_csv(params).to_dict(orient='record')[0]
        exp = cls(params, root=root, main=main, create=False)
        return exp

    @classmethod
    def gather(cls, root='runs/', main=None):
        if cls.is_exp_dir(root):
            exps = [root, ]
        else:
            exps = glob(os.path.join(root, '*'))
            exps = filter(cls.is_exp_dir, exps)

        exps = map(lambda x: cls.from_dir(x, main=main), exps)
        exps = filter(lambda x: x.existing, exps)
        return exps

    @classmethod
    def is_exp_dir(cls, exp_dir):
        if os.path.isdir(exp_dir):
            params = os.path.join(exp_dir, cls.filenames['params'])
            if os.path.exists(params):
                return True

        return False

    def __init__(self, params, root='runs/', ignore=(), main=None, create=True):
        # relative dir containing this run
        self.root = root
        # params to be ignored in the run naming
        self.ignore = ignore
        # parameters of this run
        if isinstance(params, argparse.Namespace):
            params = vars(params)
        params = {k: v for k, v in params.items() if k not in self.ignore}
        self.params = pd.DataFrame(params, index=[0])

        # main param (for naming the run)
        assert main is None or main in self.params, "'main' should be one of: ({}), got {}".format(
            ','.join(self.params.keys()), main)
        self.main = main

        # whether to create the run directory if not exists
        self.create = create

        self.name = self.abbreviate(self.params, self.main)
        self.path = os.path.join(self.root, self.name)
        self.existing = os.path.exists(self.path)

        if not self.existing:
            self.log = pd.DataFrame()
            self.results = pd.DataFrame()

            if self.create:
                os.makedirs(self.path)
                os.makedirs(self.path_to('ckpt'))
                self.params.to_csv(self.path_to('params'), index=False)
                self.existing = True
            else:
                print("Run directory '{}' not found, but not created.".format(self.path))

        else:
            log_fname = self.path_to('log')
            param_fname = self.path_to('params')
            results_fname = self.path_to('results')

            assert os.path.exists(param_fname), "Empty run, parameters not found: '{}'".format(param_fname)

            self.param = pd.read_csv(param_fname)
            self.log = pd.read_csv(log_fname, index_col=0) if os.path.exists(log_fname) else pd.DataFrame()
            self.results = pd.read_csv(results_fname) if os.path.exists(results_fname) else pd.DataFrame()

    def __str__(self):
        s = StringIO()
        print('Experiment Dir: {}'.format(self.path), file=s)
        print('Params:', file=s)
        with pd.option_context('display.width', None), pd.option_context('max_columns', None):
            self.params.to_string(s, index=False)

        # if not self.results.empty:
        #     print('\nResults:', file=s)
        #     with pd.option_context('display.width', None), pd.option_context('max_columns', None):
        #         self.results.to_string(s, index=False)

        return s.getvalue()

    def __repr__(self):
        return self.__str__()

    def path_to(self, what):
        # assert what in self.filenames, "Unknown run resource: '{}'".format(what)
        basename = self.filenames[what] if what in self.filenames else what
        path = os.path.join(self.path, basename)
        return path

    def ckpt(self, which='best'):
        ckpt_path = os.path.join(self.path_to('ckpt'), self.filenames[which])
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
        self.params.to_csv(self.path_to('params'), index=False)

    def rename_parameter(self, key, new_key):
        assert key in self.params, "Cannot rename non-existent parameter: '{}'".format(key)
        assert new_key not in self.params, "Destination name for parameter exists: '{}'".format(key)

        self.params[new_key] = self.params[key]
        del self.params[key]

        self._update_run_dir()
        self.params.to_csv(self.path_to('params'), index=False)

    def remove_parameter(self, key):
        assert key in self.params, "Cannot remove non-existent parameter: '{}'".format(key)
        del self.params[key]
        self._update_run_dir()
        self.params.to_csv(self.path_to('params'), index=False)

    def _update_run_dir(self):
        old_run_dir = self.path
        if self.existing:
            self.name = self.abbreviate(self.params, self.main)
            self.path = os.path.join(self.root, self.name)
            assert not os.path.exists(self.path), "Cannot rename run, new name exists: '{}'".format(self.path)
            shutil.move(old_run_dir, self.path)


def test():
    parser = argparse.ArgumentParser(description='Experiment Manager Test')
    parser.add_argument('-e', '--epochs', type=int, default=70)
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-m', '--model', choices=('1d-conv', 'paper'), default='1d-conv')
    parser.add_argument('-s', '--seed', type=int, default=23)
    parser.add_argument('--no-cuda', action='store_true')
    parser.set_defaults(no_cuda=False)
    args = parser.parse_args()

    run = Experiment(args, root='prova', ignore=['no_cuda'], main='model')
    print(run)
    print(run.ckpt('best'))
