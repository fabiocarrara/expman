from .experiment import Experiment, exp_filter, use_hash_naming
from .exp_group import ExpGroup

abbreviate = Experiment.abbreviate
from_dir = Experiment.from_dir
gather = ExpGroup.gather
is_exp_dir = Experiment.is_exp_dir
