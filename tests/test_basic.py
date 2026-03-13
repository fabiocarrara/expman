import shutil
import tempfile
import unittest
import os
import json

import expman


class TestBasic(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()

    # ===== BASIC EXPERIMENT CREATION AND PROPERTIES =====

    def test_main(self):
        """Test basic experiment creation with various parameter types."""
        params = {'a': 1, 'b': 2.0, 'c': 'asd', 'd': (1, 2.0, 'bsd')}
        exp = expman.Experiment(params, root=self.root)
        self.assertTrue(os.path.exists(exp.path))
        self.assertEqual(exp.params['a'], 1)
        self.assertEqual(exp.params['b'], 2.0)

    def test_parameter_types(self):
        """Test that various parameter types are correctly stored and retrieved."""
        params = {
            'int_param': 42,
            'float_param': 3.14,
            'str_param': 'test',
            'tuple_param': (1, 2, 3),
            'list_param': [4, 5, 6],
            'dict_param': {'nested': 'value'}
        }
        exp = expman.Experiment(params, root=self.root)
        
        self.assertEqual(exp.params['int_param'], 42)
        self.assertEqual(exp.params['float_param'], 3.14)
        self.assertEqual(exp.params['str_param'], 'test')
        self.assertEqual(exp.params['tuple_param'], (1, 2, 3))
        # Note: lists are converted to tuples by Experiment class
        self.assertEqual(exp.params['list_param'], (4, 5, 6))
        self.assertEqual(exp.params['dict_param'], {'nested': 'value'})

    def test_experiment_path_and_directory(self):
        """Test that experiment creates proper directory structure."""
        params = {'model': 'cnn', 'lr': 0.01}
        exp = expman.Experiment(params, root=self.root)
        
        # Check that experiment directory exists
        self.assertTrue(os.path.isdir(exp.path))
        
        # Check that params.json exists
        params_file = os.path.join(exp.path, 'params.json')
        self.assertTrue(os.path.isfile(params_file))

    def test_abbreviation(self):
        """Test parameter abbreviation for directory naming."""
        params = {'learning_rate': 0.01, 'batch_size': 32}
        exp = expman.Experiment(params, root=self.root)
        
        # Experiment name should be in the path
        self.assertTrue(os.path.exists(exp.path))
        
        # All params should be abbreviated in directory name
        dir_name = os.path.basename(exp.path)
        self.assertIn('l', dir_name)  # abbreviation of learning_rate
        self.assertIn('b', dir_name)  # abbreviation of batch_size

    def test_string_representation(self):
        """Test __str__ and __repr__ methods."""
        params = {'a': 1}
        exp = expman.Experiment(params, root=self.root)
        
        str_repr = str(exp)
        self.assertIsInstance(str_repr, str)
        self.assertTrue(len(str_repr) > 0)

    def test_path_to_method(self):
        """Test path_to() method for file path resolution."""
        params = {'a': 1}
        exp = expman.Experiment(params, root=self.root)
        
        test_file = exp.path_to('output.txt')
        self.assertTrue(test_file.startswith(exp.path))
        self.assertTrue(test_file.endswith('output.txt'))

    # ===== PARAMETER MODIFICATION =====

    def test_add_parameter(self):
        """Test adding a new parameter to an existing experiment."""
        params = {'a': 1}
        exp = expman.Experiment(params, root=self.root)
        
        exp.add_parameter('b', 2)
        self.assertEqual(exp.params['b'], 2)
        self.assertEqual(exp.params['a'], 1)
        
        # Verify it's persisted to disk
        reloaded = expman.Experiment.from_dir(exp.path)
        self.assertEqual(reloaded.params['b'], 2)

    def test_rename_parameter(self):
        """Test renaming an existing parameter."""
        params = {'old_name': 42}
        exp = expman.Experiment(params, root=self.root)
        
        exp.rename_parameter('old_name', 'new_name')
        self.assertEqual(exp.params['new_name'], 42)
        self.assertNotIn('old_name', exp.params.index)
        
        # Verify it's persisted to disk
        reloaded = expman.Experiment.from_dir(exp.path)
        self.assertEqual(reloaded.params['new_name'], 42)

    def test_remove_parameter(self):
        """Test removing a parameter from an experiment."""
        params = {'a': 1, 'b': 2}
        exp = expman.Experiment(params, root=self.root)
        
        exp.remove_parameter('a')
        self.assertNotIn('a', exp.params.index)
        self.assertEqual(exp.params['b'], 2)
        
        # Verify it's persisted to disk
        reloaded = expman.Experiment.from_dir(exp.path)
        self.assertNotIn('a', reloaded.params.index)

    # ===== DIRECTORY UTILITIES =====

    def test_is_exp_dir(self):
        """Test that is_exp_dir correctly identifies experiment directories."""
        params = {'a': 1}
        exp = expman.Experiment(params, root=self.root)
        
        # Experiment directory should be recognized
        self.assertTrue(expman.Experiment.is_exp_dir(exp.path))
        
        # Non-experiment directory should not be recognized
        other_dir = os.path.join(self.root, 'not_exp')
        os.makedirs(other_dir)
        self.assertFalse(expman.Experiment.is_exp_dir(other_dir))

    def test_is_child_exp_dir(self):
        """Test that is_child_exp_dir correctly identifies leaf experiments."""
        parent = expman.Experiment({'a': 1}, root=self.root)
        child = expman.Experiment({'b': 2}, root=parent.path)
        
        # Parent has children, so should not be a child exp dir (it's a parent)
        self.assertFalse(expman.Experiment.is_child_exp_dir(parent.path))
        
        # Child has no children, so should be a child exp dir (it's a leaf)
        self.assertTrue(expman.Experiment.is_child_exp_dir(child.path))

    # ===== PARENT-CHILD RELATIONSHIPS =====

    def test_parent_child_basic(self):
        """Test basic sub-experiment creation with parent parameter inheritance."""
        parent_params = {'a': 1, 'b': 2}
        parent = expman.Experiment(parent_params, root=self.root)
        
        # Child should only have its own params, created in parent's directory
        child_params = {'c': 3}
        child = expman.Experiment(child_params, root=parent.path)
        
        # Check child has only its own params
        self.assertEqual(set(child.params.index), {'c'})
        self.assertEqual(child.params['c'], 3)
        
        # Check full_params includes both parent and child
        self.assertEqual(set(child.full_params.index), {'a', 'b', 'c'})
        self.assertEqual(child.full_params['a'], 1)
        self.assertEqual(child.full_params['b'], 2)
        self.assertEqual(child.full_params['c'], 3)
        
        # Check parent relationship is set
        self.assertIsNotNone(child.parent)
        self.assertEqual(child.parent.params.to_dict(), parent.params.to_dict())
    
    def test_parent_child_persistence(self):
        """Test that parent-child relationships are persisted via filesystem."""
        parent_params = {'a': 1, 'b': 2}
        parent = expman.Experiment(parent_params, root=self.root)
        parent_path = parent.path
        
        child_params = {'c': 3}
        child = expman.Experiment(child_params, root=parent_path)
        child_path = child.path
        
        # Check that child is in a subdirectory of parent
        self.assertEqual(os.path.dirname(child_path), parent_path)
        
        # Check that child's params.json contains only child params (no parent reference)
        params_file = os.path.join(child_path, 'params.json')
        with open(params_file, 'r') as f:
            saved_data = json.load(f)
        
        # Only child params should be saved
        self.assertEqual(set(saved_data.keys()), {'c'})
    
    def test_parent_child_reload(self):
        """Test that parent-child relationships are correctly reconstructed from disk."""
        parent_params = {'a': 1, 'b': 2}
        parent = expman.Experiment(parent_params, root=self.root)
        
        child_params = {'c': 3}
        child = expman.Experiment(child_params, root=parent.path)
        child_path = child.path
        
        # Reload the child from disk
        reloaded_child = expman.Experiment.from_dir(child_path)
        
        # Check that the reloaded child has the correct parent
        self.assertIsNotNone(reloaded_child.parent)
        self.assertEqual(reloaded_child.parent.params.to_dict(), parent.params.to_dict())
        
        # Check full params
        self.assertEqual(set(reloaded_child.full_params.index), {'a', 'b', 'c'})
        self.assertEqual(reloaded_child.full_params['a'], 1)
        self.assertEqual(reloaded_child.full_params['b'], 2)
        self.assertEqual(reloaded_child.full_params['c'], 3)
    
    def test_nested_hierarchy(self):
        """Test multi-level hierarchy (grandparent -> parent -> child)."""
        grandparent_params = {'a': 1}
        grandparent = expman.Experiment(grandparent_params, root=self.root)
        
        parent_params = {'b': 2}
        parent = expman.Experiment(parent_params, root=grandparent.path)
        
        child_params = {'c': 3}
        child = expman.Experiment(child_params, root=parent.path)
        
        # Child should have all params from the hierarchy
        self.assertEqual(set(child.full_params.index), {'a', 'b', 'c'})
        self.assertEqual(child.full_params['a'], 1)
        self.assertEqual(child.full_params['b'], 2)
        self.assertEqual(child.full_params['c'], 3)

    # ===== EXPGROUP GATHERING AND COLLECTION =====

    def test_gather_nested_experiments(self):
        """Test that gather by default only collects leaf experiments."""
        # Create parent experiment
        parent_params = {'a': 1}
        parent = expman.Experiment(parent_params, root=self.root)
        
        # Create two child experiments
        child1_params = {'b': 2}
        child1 = expman.Experiment(child1_params, root=parent.path)
        
        child2_params = {'b': 3}
        child2 = expman.Experiment(child2_params, root=parent.path)
        
        # Gather should find only the two children (leaves), not the parent
        group = expman.ExpGroup.gather(root=self.root)
        self.assertEqual(len(group.experiments), 2)
        
        # Gather with include_parents=True should find all three
        group_all = expman.ExpGroup.gather(root=self.root, include_parents=True)
        self.assertEqual(len(group_all.experiments), 3)
    
    def test_collect_nested_experiments(self):
        """Test that collect includes full params for nested experiments."""
        # Create parent and child experiments
        parent_params = {'a': 1}
        parent = expman.Experiment(parent_params, root=self.root)
        
        child_params = {'b': 2}
        child = expman.Experiment(child_params, root=parent.path)
        
        # By default, gather only finds leaf experiments
        group = expman.ExpGroup.gather(root=self.root)
        df = group.collect()
        
        # Should have only 1 row (the child, which is a leaf)
        self.assertEqual(len(df), 1)
        self.assertTrue((df['a'] == 1).all())  # Child has parent's 'a' in full_params
        self.assertTrue((df['b'] == 2).all())  # Child has its own 'b'
        
        # With include_parents=True, should see both parent and child
        group_all = expman.ExpGroup.gather(root=self.root, include_parents=True)
        df_all = group_all.collect()
        self.assertEqual(len(df_all), 2)

    def test_collect_simple_experiments(self):
        """Test collect() for simple (non-nested) experiments."""
        # Create two simple experiments
        exp1 = expman.Experiment({'model': 'cnn', 'lr': 0.01}, root=self.root)
        exp2 = expman.Experiment({'model': 'mlp', 'lr': 0.001}, root=self.root)
        
        # Gather and collect
        group = expman.ExpGroup.gather(root=self.root)
        df = group.collect()
        
        # Should have 2 rows
        self.assertEqual(len(df), 2)
        # Columns: model, lr, exp_id, exp_name
        self.assertIn('model', df.columns)
        self.assertIn('lr', df.columns)
        self.assertIn('exp_id', df.columns)
        
        # Check that both experiments are present
        models = sorted(df['model'].tolist())
        self.assertEqual(models, ['cnn', 'mlp'])

    def test_collect_without_csv(self):
        """Test collect() returns only parameters when no CSV metrics are present."""
        # Create experiments
        exp = expman.Experiment({'a': 1, 'b': 2}, root=self.root)
        
        # Collect without CSV metrics
        group = expman.ExpGroup.gather(root=self.root)
        df = group.collect()
        
        # Should have parameters plus exp_id
        self.assertIn('a', df.columns)
        self.assertIn('b', df.columns)
        self.assertEqual(len(df), 1)
        self.assertEqual(df['a'].iloc[0], 1)
        self.assertEqual(df['b'].iloc[0], 2)

    # ===== CSV METRICS TESTS =====

    def test_collect_with_csv_simple(self):
        """Test collect() with CSV metrics in simple (non-nested) experiments."""
        import pandas as pd
        
        # Create two experiments
        exp1 = expman.Experiment({'model': 'cnn'}, root=self.root)
        exp2 = expman.Experiment({'model': 'mlp'}, root=self.root)
        
        # Create metrics CSV for each experiment
        metrics1 = pd.DataFrame({'loss': [0.5, 0.3, 0.2], 'accuracy': [0.8, 0.85, 0.9]})
        metrics1.to_csv(os.path.join(exp1.path, 'metrics.csv'), index=False)
        
        metrics2 = pd.DataFrame({'loss': [0.6, 0.4, 0.25], 'accuracy': [0.75, 0.8, 0.88]})
        metrics2.to_csv(os.path.join(exp2.path, 'metrics.csv'), index=False)
        
        # Gather and collect with CSV
        group = expman.ExpGroup.gather(root=self.root)
        df = group.collect(csv='metrics.csv')
        
        # Should have 6 rows total (3 metrics rows per experiment)
        self.assertEqual(len(df), 6)
        
        # Check that params are included
        self.assertIn('model', df.columns)
        self.assertIn('loss', df.columns)
        self.assertIn('accuracy', df.columns)
        
        # Check that models are correctly matched with their metrics
        cnn_rows = df[df['model'] == 'cnn']
        mlp_rows = df[df['model'] == 'mlp']
        
        self.assertEqual(len(cnn_rows), 3)
        self.assertEqual(len(mlp_rows), 3)
        
        # Verify accuracy values match
        self.assertListEqual(cnn_rows['accuracy'].tolist(), [0.8, 0.85, 0.9])
        self.assertListEqual(mlp_rows['accuracy'].tolist(), [0.75, 0.8, 0.88])
    
    def test_collect_with_csv_nested(self):
        """Test collect() with CSV metrics in nested (parent-child) experiments."""
        import pandas as pd
        
        # Create parent experiment
        parent = expman.Experiment({'model': 'cnn'}, root=self.root)
        
        # Create child experiments with different learning rates
        child1 = expman.Experiment({'lr': 0.01}, root=parent.path)
        child2 = expman.Experiment({'lr': 0.001}, root=parent.path)
        
        # Create metrics CSV for each child
        metrics1 = pd.DataFrame({'loss': [0.5, 0.3], 'accuracy': [0.8, 0.85]})
        metrics1.to_csv(os.path.join(child1.path, 'metrics.csv'), index=False)
        
        metrics2 = pd.DataFrame({'loss': [0.6, 0.4], 'accuracy': [0.75, 0.8]})
        metrics2.to_csv(os.path.join(child2.path, 'metrics.csv'), index=False)
        
        # Gather and collect with CSV (should only get leaves by default)
        group = expman.ExpGroup.gather(root=self.root)
        df = group.collect(csv='metrics.csv')
        
        # Should have 4 rows total (2 metrics rows per child)
        self.assertEqual(len(df), 4)
        
        # Check that full params are included (model from parent, lr from child)
        self.assertIn('model', df.columns)
        self.assertIn('lr', df.columns)
        self.assertIn('loss', df.columns)
        self.assertIn('accuracy', df.columns)
        
        # All rows should have model='cnn' (from parent)
        self.assertTrue((df['model'] == 'cnn').all())
        
        # Check learning rates are correctly matched
        lr_001_rows = df[df['lr'] == 0.01]
        lr_0001_rows = df[df['lr'] == 0.001]
        
        self.assertEqual(len(lr_001_rows), 2)
        self.assertEqual(len(lr_0001_rows), 2)
        
        # Verify metrics are correctly associated
        self.assertListEqual(lr_001_rows['accuracy'].tolist(), [0.8, 0.85])
        self.assertListEqual(lr_0001_rows['accuracy'].tolist(), [0.75, 0.8])
    
    def test_collect_with_csv_nested_with_parents(self):
        """Test collect() with CSV metrics including parent experiments."""
        import pandas as pd
        
        # Create parent experiment
        parent = expman.Experiment({'model': 'cnn'}, root=self.root)
        
        # Create child experiments
        child = expman.Experiment({'lr': 0.01}, root=parent.path)
        
        # Create metrics CSV for both parent and child
        parent_metrics = pd.DataFrame({'loss': [0.8], 'accuracy': [0.7]})
        parent_metrics.to_csv(os.path.join(parent.path, 'metrics.csv'), index=False)
        
        child_metrics = pd.DataFrame({'loss': [0.5, 0.3], 'accuracy': [0.8, 0.85]})
        child_metrics.to_csv(os.path.join(child.path, 'metrics.csv'), index=False)
        
        # Gather with parents included
        group = expman.ExpGroup.gather(root=self.root, include_parents=True)
        df = group.collect(csv='metrics.csv')
        
        # Should have 3 rows: 1 for parent, 2 for child
        self.assertEqual(len(df), 3)
        
        # Parent row should have model='cnn' but no 'lr'
        parent_row = df[pd.isna(df['lr'])]
        self.assertEqual(len(parent_row), 1)
        self.assertEqual(parent_row['model'].iloc[0], 'cnn')
        self.assertEqual(parent_row['loss'].iloc[0], 0.8)
        self.assertEqual(parent_row['accuracy'].iloc[0], 0.7)
        
        # Child rows should have both model and lr
        child_rows = df[df['lr'] == 0.01]
        self.assertEqual(len(child_rows), 2)
        self.assertTrue((child_rows['model'] == 'cnn').all())
        self.assertListEqual(child_rows['accuracy'].tolist(), [0.8, 0.85])

    # ===== FILTERING AND ITERATION =====

    def test_filter_basic(self):
        """Test filtering experiments by parameter values."""
        exp1 = expman.Experiment({'model': 'cnn', 'lr': 0.01}, root=self.root)
        exp2 = expman.Experiment({'model': 'mlp', 'lr': 0.01}, root=self.root)
        exp3 = expman.Experiment({'model': 'cnn', 'lr': 0.001}, root=self.root)
        
        group = expman.ExpGroup.gather(root=self.root)
        
        # Filter by model
        cnn_group = group.filter('model=cnn')
        self.assertEqual(len(cnn_group.experiments), 2)
        
        # Filter by learning rate
        lr_001_group = group.filter('lr=0.01')
        self.assertEqual(len(lr_001_group.experiments), 2)

    def test_filter_multiple_params(self):
        """Test filtering experiments by multiple parameters."""
        exp1 = expman.Experiment({'model': 'cnn', 'lr': 0.01}, root=self.root)
        exp2 = expman.Experiment({'model': 'mlp', 'lr': 0.01}, root=self.root)
        exp3 = expman.Experiment({'model': 'cnn', 'lr': 0.001}, root=self.root)
        
        group = expman.ExpGroup.gather(root=self.root)
        
        # Filter by multiple parameters
        filtered = group.filter('model=cnn,lr=0.01')
        self.assertEqual(len(filtered.experiments), 1)
        
        df = filtered.collect()
        self.assertEqual(df['model'].iloc[0], 'cnn')
        self.assertEqual(df['lr'].iloc[0], 0.01)

    def test_items_iteration(self):
        """Test iterating over experiments with items()."""
        exp1 = expman.Experiment({'a': 1}, root=self.root)
        exp2 = expman.Experiment({'a': 2}, root=self.root)
        
        group = expman.ExpGroup.gather(root=self.root)
        
        # Test iteration with items(short_names=False) returns just experiments
        items_list = list(group.items(short_names=False))
        self.assertEqual(len(items_list), 2)
        
        # Each item should be an Experiment instance
        for exp in items_list:
            self.assertIsInstance(exp, expman.Experiment)
        
        # Test iteration with items(short_names=True) returns (name, exp) tuples
        named_items = list(group.items(short_names=True))
        self.assertEqual(len(named_items), 2)
        for name, exp in named_items:
            self.assertIsInstance(name, str)
            self.assertIsInstance(exp, expman.Experiment)

    def test_group_iteration(self):
        """Test iterating over experiments directly."""
        exp1 = expman.Experiment({'a': 1}, root=self.root)
        exp2 = expman.Experiment({'a': 2}, root=self.root)
        
        group = expman.ExpGroup.gather(root=self.root)
        
        # Test direct iteration
        exp_list = list(group)
        self.assertEqual(len(exp_list), 2)
        
        # All items should be Experiment instances
        for exp in exp_list:
            self.assertIsInstance(exp, expman.Experiment)

    def test_group_length(self):
        """Test len() on ExpGroup."""
        exp1 = expman.Experiment({'a': 1}, root=self.root)
        exp2 = expman.Experiment({'a': 2}, root=self.root)
        
        group = expman.ExpGroup.gather(root=self.root)
        
        self.assertEqual(len(group), 2)

    def test_sort_experiments(self):
        """Test sorting experiments by parameter values."""
        exp1 = expman.Experiment({'value': 30}, root=self.root)
        exp2 = expman.Experiment({'value': 10}, root=self.root)
        exp3 = expman.Experiment({'value': 20}, root=self.root)
        
        group = expman.ExpGroup.gather(root=self.root)
        
        # Sort by parameter value
        sorted_group = group.sort(key=lambda e: e.params['value'])
        values = [e.params['value'] for e in sorted_group]
        self.assertEqual(values, [10, 20, 30])
        
        # Sort in reverse
        sorted_group_rev = group.sort(key=lambda e: e.params['value'], reverse=True)
        values_rev = [e.params['value'] for e in sorted_group_rev]
        self.assertEqual(values_rev, [30, 20, 10])

    # ===== EDGE CASES AND ERROR HANDLING =====

    def test_reload_experiment_from_disk(self):
        """Test reloading experiment from disk preserves all data."""
        params = {'a': 1, 'b': 2.5, 'c': 'test'}
        exp = expman.Experiment(params, root=self.root)
        exp_path = exp.path
        
        # Reload and verify
        reloaded = expman.Experiment.from_dir(exp_path)
        self.assertEqual(reloaded.params['a'], 1)
        self.assertEqual(reloaded.params['b'], 2.5)
        self.assertEqual(reloaded.params['c'], 'test')

    def test_empty_group(self):
        """Test behavior with empty experiment group."""
        group = expman.ExpGroup.gather(root=self.root)
        
        # Empty group should have length 0
        self.assertEqual(len(group), 0)
        
        # Note: collect() on empty group will raise ValueError from pd.concat
        # This is expected behavior - empty groups have nothing to collect
        with self.assertRaises(ValueError):
            group.collect()

    def test_multiple_parameter_modifications(self):
        """Test multiple parameter modifications in sequence."""
        exp = expman.Experiment({'a': 1, 'b': 2}, root=self.root)
        
        # Modify multiple times
        exp.add_parameter('c', 3)
        exp.rename_parameter('a', 'x')
        exp.remove_parameter('b')
        
        # Verify final state
        self.assertIn('x', exp.params.index)
        self.assertIn('c', exp.params.index)
        self.assertNotIn('a', exp.params.index)
        self.assertNotIn('b', exp.params.index)
        
        # Verify persistence
        reloaded = expman.Experiment.from_dir(exp.path)
        self.assertEqual(reloaded.params['x'], 1)
        self.assertEqual(reloaded.params['c'], 3)
        self.assertEqual(len(reloaded.params), 2)

    def tearDown(self):
        shutil.rmtree(self.root)