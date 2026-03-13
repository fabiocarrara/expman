import argparse
import ast
import hashlib
import numbers
import os
import shutil
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
    if "=" not in string:
        raise argparse.ArgumentTypeError(
            f"Filter {string} is not in format <param1>=<value1>[, <param2>=<value2>[, ...]]"
        )
    return {k: _guessed_cast(v) for k, v in (x.split("=") for x in string.split(","))}


class Experiment:

    PARAM_FILENAME = "params.json"

    @staticmethod
    def _abbr(name, value, params):

        def prefix_len(a, b):
            return len(os.path.commonprefix((a, b)))

        prefix = [name[: prefix_len(p, name) + 1] for p in params.keys() if p != name]
        prefix = max(prefix, key=len) if len(prefix) > 0 else name

        sep = ""
        if isinstance(value, str):
            value = "".join(c if c not in '<>:"|?*\\/' else "-" for c in value)
            sep = "-"
        elif isinstance(value, numbers.Number):
            value = "{:g}".format(value)
            sep = "-" if prefix[-1].isdigit() else ""
        elif isinstance(value, (list, tuple)):
            value = map(str, value)
            value = map(lambda v: v.replace(os.sep, "|"), value)
            value = ",".join(list(value))
            sep = "-"

        return prefix, sep, value

    @classmethod
    def abbreviate(cls, params):
        if isinstance(params, pd.DataFrame):
            params = params.iloc[0]
            params = params.replace({np.nan: None})

        if hash_naming:
            exp_name = hashlib.md5(str(sorted(params.items())).encode()).hexdigest()
        else:
            abbrev_params = {
                k: "{}{}{}".format(*cls._abbr(k, v, params)) for k, v in params.items()
            }
            abbrev = sorted(abbrev_params.values())
            exp_name = "_".join(abbrev)

        return exp_name

    @classmethod
    def from_dir(cls, exp_dir):
        exp_dir = exp_dir.rstrip("/")
        root = os.path.dirname(exp_dir)
        params_path = os.path.join(exp_dir, cls.PARAM_FILENAME)

        assert os.path.exists(exp_dir), "Experiment directory not found: '{}'".format(exp_dir)
        assert os.path.exists(params_path), "Empty run directory found: '{}'".format(params_path)

        params = cls._read_params(params_path)
        exp = cls(params, root=root, create=False)

        return exp

    @classmethod
    def is_exp_dir(cls, exp_dir):
        return os.path.isdir(exp_dir) and os.path.exists(os.path.join(exp_dir, cls.PARAM_FILENAME))

    @classmethod
    def is_child_exp_dir(cls, exp_dir):
        if not cls.is_exp_dir(exp_dir):
            return False
        subdirs = [
            os.path.join(exp_dir, d)
            for d in os.listdir(exp_dir)
            if os.path.isdir(os.path.join(exp_dir, d))
        ]
        return not any(cls.is_exp_dir(d) for d in subdirs)

    def __init__(self, params, root="runs/", ignore=(), create=True):
        self.root = root  # dir containing this run
        self.ignore = ignore  # params to be ignored in the run naming
        self.parent = None  # parent experiment, if any
        self.create = create  # whether to create the run directory if not exists

        # Normalize parameters
        if isinstance(params, argparse.Namespace):
            params = vars(params)

        def _sanitize(v):
            return tuple(v) if isinstance(v, list) else v

        params = {k: _sanitize(v) for k, v in params.items() if k not in self.ignore}
        self.params = pd.Series(params, name="params")

        # Set up directory
        self.name = self.abbreviate(self.full_params)
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
            assert os.path.exists(param_fname), f"Empty run, parameters not found: '{param_fname}'"
            self.params = self._read_params(param_fname)

        # Detect parent from filesystem
        if Experiment.is_exp_dir(self.root):
            self.parent = Experiment.from_dir(self.root)

    def __str__(self):
        s = StringIO()
        print(f"Experiment Dir: {self.path}", file=s)
        print("Params:", file=s)
        with pd.option_context("display.width", None), pd.option_context(
            "display.max_columns", None
        ):
            self.params.to_string(s)

        return s.getvalue()

    def __repr__(self):
        return self.__str__()

    @property
    def full_params(self):
        """Return combined params from parent hierarchy and this experiment."""
        if self.parent is None:
            return self.params.copy()
        else:
            full = self.parent.full_params.copy()
            full = pd.concat([full, self.params])
            return full

    def path_to(self, path):
        path = os.path.join(self.path, path)
        return path

    def add_parameter(self, key, value):
        assert key not in self.params, f"Parameter already exists: '{key}'"
        self.params[key] = value
        self._update_run_dir()
        self.write_params()

    def rename_parameter(self, key, new_key):
        assert key in self.params, f"Cannot rename non-existent parameter: '{key}'"
        assert new_key not in self.params, f"Destination name for parameter exists: '{new_key}'"
        self.params[new_key] = self.params.pop(key)
        self._update_run_dir()
        self.write_params()

    def remove_parameter(self, key):
        assert key in self.params, f"Cannot remove non-existent parameter: '{key}'"
        del self.params[key]
        self._update_run_dir()
        self.write_params()

    def _update_run_dir(self):
        """Rename directory if parameter names change."""
        if not self.existing:
            return
        old_path = self.path
        self.name = self.abbreviate(self.params)
        self.path = os.path.join(self.root, self.name)
        if old_path != self.path:
            assert not os.path.exists(
                self.path
            ), f"Cannot rename run, new name exists: '{self.path}'"
            shutil.move(old_path, self.path)

    @staticmethod
    def _read_params(path):
        # read json to pd.Series
        params = pd.read_json(path, typ="series")
        # transform lists to tuples (for hashability)
        params = params.apply(lambda x: tuple(x) if isinstance(x, list) else x)
        return params

    def write_params(self):
        # write Series as json
        self.params.to_json(self.path_to(self.PARAM_FILENAME))
