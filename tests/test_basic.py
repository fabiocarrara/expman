import shutil
import tempfile
import unittest
import os
import json

import expman


class TestBasic(unittest.TestCase):

    def setUp(self):
        self.root = tempfile.mkdtemp()

    def test_main(self):
        params = {'a': 1, 'b': 2.0, 'c': 'asd', 'd': (1, 2.0, 'bsd')}
        exp = expman.Experiment(params, root=self.root)
        print(exp.path)
        self.assertTrue(True)

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

    def tearDown(self):
        shutil.rmtree(self.root)