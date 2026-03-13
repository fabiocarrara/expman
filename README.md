# expman

A lightweight experiment manager that makes it easy to organize, track, and compare ML experiments. It gives each experiment its own directory, automatically names runs based on parameters, and lets you quickly gather results into a table for analysis.

**Use expman when you:**
- Run many experiments with different hyperparameters
- Want results organized automatically on disk
- Need to compare metrics across runs
- Have nested experiments (e.g., trying different learning rates for each model)

## Installation

```bash
pip install git+https://github.com/fabiocarrara/expman.git
```

Requires: numpy, pandas

## Basic Usage

#### Create an experiment
Each experiment gets its own folder automatically named from parameters:

```python
import expman

exp = expman.Experiment({'lr': 0.01, 'batch_size': 32}, root='runs/')
print(exp.path)  # runs/b32_l0.01
```

#### Or use argparse
You can pass command-line arguments directly to `Experiment`:

```python
import argparse
import expman

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--batch-size', type=int, default=32)
parser.add_argument('--model', default='resnet')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--no-cuda', action='store_true')

args = parser.parse_args()

# Experiment automatically converts argparse Namespace
# Use 'ignore' to exclude parameters from the directory name
exp = expman.Experiment(args, root='runs/', ignore=['no_cuda'])
print(exp.path)  # runs/b32_l0.01_m-resnet_s42
```

#### Save your results
After training, save metrics to a CSV in the experiment folder:

```python
import pandas as pd

metrics = pd.DataFrame({
    'epoch': [1, 2, 3],
    'loss': [0.5, 0.3, 0.2]
})

metrics_path = exp.path_to('metrics.csv')  # returns path inside exp folder
metrics.to_csv(metrics_path, index=False)
```

#### Compare across experiments
Gather all your runs and collect metrics into a table:

```python
group = expman.gather('runs/')
df = group.collect('metrics.csv')  # columns: all params + all metrics files
print(df)
```

#### Modify parameters
You can add, rename, or remove parameters after creating an experiment. The directory is automatically renamed:

```python
exp = expman.Experiment({'lr': 0.01, 'batch_size': 32}, root='runs/')
print(exp.path)  # runs/b32_l0.01

# Add a new parameter
exp.add_parameter('weight_decay', 0.0001)
print(exp.path)  # runs/b32_l0.01_w0.0001

# Rename a parameter
exp.rename_parameter('batch_size', 'mini_batch')
print(exp.path)  # runs/m32_l0.01_w0.0001

# Remove a parameter
exp.remove_parameter('weight_decay')
print(exp.path)  # runs/m32_l0.01
```

## Nested Experiments

If you're testing multiple learning rates *for each model*, use parent-child experiments:

```python
# Parent: fixed model config
parent = expman.Experiment({'model': 'cnn', 'dataset': 'cifar10'}, root='runs/')

# Children: different learning rates (automatically placed under parent)
exp1 = expman.Experiment({'lr': 0.01}, root=parent.path)
exp2 = expman.Experiment({'lr': 0.001}, root=parent.path)

# Child inherits parent params
print(exp1.full_params)  # {'model': 'cnn', 'dataset': 'cifar10', 'lr': 0.01}

# When you gather, leaf experiments (the children) are returned with full params
df = expman.gather('runs/').collect()
# Automatically organized:
# runs/
# └── m-cnn_d-cifar10/
#     ├── lr0.01/
#     └── lr0.001/
```

## How It Works

**Directory naming:** Experiments are automatically named from their parameters, making it easy to find them back later.

**Inheritance:** When you create child experiments inside a parent's directory, they automatically get the parent's parameters. You only store the differences in `params.json`.

**Gathering:** Call `gather()` to recursively find all experiments and `collect()` to turn them into a DataFrame. By default it returns leaf experiments (the actual runs with results) to avoid duplicating parent data.
