import os
from glob import glob

import pandas as pd
from .experiment import Experiment


class ExpGroup:
    @classmethod
    def gather(cls, root="runs/", include_parents=False):
        """Gather experiments recursively.

        By default, returns only leaf experiments. Set include_parents=True
        to include parent experiments with children.
        """

        def _gather_recursive(path):
            if not Experiment.is_exp_dir(path):
                return []

            exp = Experiment.from_dir(path)
            exps = (
                [exp]
                if exp.existing and (include_parents or Experiment.is_child_exp_dir(path))
                else []
            )

            subdirs = [d for d in glob(os.path.join(path, "*")) if os.path.isdir(d)]
            for subdir in subdirs:
                exps.extend(_gather_recursive(subdir))
            return exps

        # Always use recursive approach
        if Experiment.is_exp_dir(root):
            exps = _gather_recursive(root)
        else:
            exps = []
            for item in glob(os.path.join(root, "*")):
                if os.path.isdir(item):
                    exps.extend(_gather_recursive(item))

        return cls(tuple(exps))

    def __init__(self, experiments=()):
        assert isinstance(experiments, (list, tuple)), "'experiments' must be a list or tuple"
        self.experiments = experiments

    @staticmethod
    def _collect_one(exp_id, exp, csv=None, index_col=None):
        params = exp.full_params.to_frame().transpose().infer_objects()
        params["exp_id"] = exp_id

        if csv is None:
            return params

        csv_path = exp.path_to(csv)
        if os.path.exists(csv_path):
            stuff = pd.read_csv(csv_path, index_col=index_col)
        else:
            csv_files = glob(os.path.join(exp.path, csv))
            if not csv_files:
                return pd.DataFrame()
            stuff = [
                pd.read_csv(f, index_col=index_col, float_precision="round_trip") for f in csv_files
            ]
            stuff = pd.concat(stuff, ignore_index=True)

        stuff["exp_id"] = exp_id
        return pd.merge(params, stuff, on="exp_id")

    def collect(self, csv=None, index_col=None, prefix=""):
        results = [
            self._collect_one(exp_id, exp, csv=csv, index_col=index_col)
            for exp_id, exp in enumerate(self.experiments)
        ]
        results = pd.concat(results, ignore_index=True, sort=False)

        if len(results):
            # Build minimal exp_name from varying parameters
            params = results.loc[:, :"exp_id"].drop("exp_id", axis=1)
            exp_name = ""
            if len(params) > 1:
                varying = params.loc[:, params.nunique() > 1]
                exp_name = varying.apply(Experiment.abbreviate, axis=1)
            results.insert(results.columns.get_loc("exp_id") + 1, "exp_name", prefix + exp_name)

        return results

    def filter(self, filters):
        if isinstance(filters, str):
            filters = {k: v for k, v in (x.split("=") for x in filters.split(","))}

        def matches(exp):
            for param, value in filters.items():
                try:
                    if exp.params[param] != type(exp.params[param])(value):
                        return False
                except (KeyError, ValueError, TypeError):
                    return False
            return True

        return ExpGroup(tuple(e for e in self.experiments if matches(e)))

    def items(self, short_names=True, prefix=""):
        if not short_names:
            return self.experiments
        params = self.collect(prefix=prefix)
        return zip(params["exp_name"].values, self.experiments)

    def sort(self, key=None, reverse=False):
        sorted_exps = sorted(self.experiments, key=key, reverse=reverse)
        return ExpGroup(sorted_exps)

    def __iter__(self):
        return iter(self.experiments)

    def __len__(self):
        return len(self.experiments)
