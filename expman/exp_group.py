import os
import pandas as pd

from glob import glob
from .experiment import Experiment


class ExpGroup:
    @classmethod
    def gather(cls, root='runs/'):
        if Experiment.is_exp_dir(root):
            exps = (root,)
        else:
            exps = glob(os.path.join(root, '*'))
            exps = filter(Experiment.is_exp_dir, exps)

        exps = map(Experiment.from_dir, exps)
        exps = filter(lambda x: x.existing, exps)
        exps = tuple(exps)
        return cls(exps)

    def __init__(self, experiments=()):
        assert isinstance(experiments, (list, tuple)), "'experiments' must be a list or tuple"
        self.experiments = experiments

    @staticmethod
    def _collect_one(exp_id, exp, csv=None, index_col=None):
        params = exp.params.to_frame().transpose().infer_objects()  # as DataFrame
        params['exp_id'] = exp_id

        if csv is None:
            return params

        csv_path = exp.path_to(csv)
        if os.path.exists(csv_path):
            stuff = pd.read_csv(csv_path, index_col=index_col)
        else:  # try globbing
            csv_files = os.path.join(exp.path, csv)
            csv_files = list(glob(csv_files))
            if len(csv_files) == 0:
                return pd.DataFrame()

            stuff = map(lambda x: pd.read_csv(x, index_col=index_col, float_precision='round_trip'), csv_files)
            stuff = pd.concat(stuff, ignore_index=True)

        stuff['exp_id'] = exp_id
        return pd.merge(params, stuff, on='exp_id')

    def collect(self, csv=None, index_col=None, prefix=''):
        results = [self._collect_one(exp_id, exp, csv=csv, index_col=index_col) for exp_id, exp in enumerate(self.experiments)]
        results = pd.concat(results, ignore_index=True, sort=False)

        if len(results):
            # build minimal exp_name
            exp_name = ''
            params = results.loc[:, :'exp_id'].drop('exp_id', axis=1)
            if len(params) > 1:
                varying_params = params.loc[:, params.nunique() > 1]
                exp_name = varying_params.apply(Experiment.abbreviate, axis=1)
            idx = results.columns.get_loc('exp_id') + 1
            results.insert(idx, 'exp_name', prefix + exp_name)

        return results

    def filter(self, filters):
        if isinstance(filters, str):
            filters = string.split(',')
            filters = map(lambda x: x.split('='), filters)
            filters = {k: v for k, v in filters}

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

        filtered_exps = filter(__filter_exp, self.experiments)
        filtered_exps = tuple(filtered_exps)
        return ExpGroup(filtered_exps)

    def items(self, short_names=True, prefix=''):
        if short_names:
            params = self.collect(prefix=prefix)
            exp_names = params['exp_name'].values
            return zip(exp_names, self.experiments)

        return self.experiments

    def __iter__(self):
        return iter(self.experiments)