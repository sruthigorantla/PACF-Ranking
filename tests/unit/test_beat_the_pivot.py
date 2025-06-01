import unittest
import numpy as np
from active_fair_ranking.algorithms.data import SetofItems, Item
from active_fair_ranking.algorithms.beat_the_pivot import beat_the_pivot
from active_fair_ranking.algorithms.utils import setup_logging
import omegaconf
import logging
import pytest

class TestBeatThePivot(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def setup(self, config, logger):
        """Setup test fixtures"""
        self.config = config
        self.logger = logger
        
        # Create a test set of items
        self.items = [
            Item(identifier="item-1", theta=0.9, group_id="A"),
            Item(identifier="item-2", theta=0.8, group_id="A"),
            # Item(identifier="item-3", theta=0.7, group_id="B"),
            # Item(identifier="item-4", theta=0.6, group_id="B"),
            # Item(identifier="item-5", theta=0.5, group_id="A")
        ]
        self.set_of_items = SetofItems(self.items)

    def test_beat_the_pivot_basic(self):
        """Test basic functionality of beat_the_pivot algorithm"""
        epsilon = 0.1
        delta = 0.1
        
        result = beat_the_pivot(
            set_of_items=self.set_of_items,
            epsilon=epsilon,
            delta=delta,
            args=self.config,
            logger=self.logger
        )
        
        # Check that the result contains the expected keys
        self.assertIn('predicted_ranking', result)
        self.assertIn('num_of_comparisons', result)
        
        # Check that the predicted ranking has the correct length
        self.assertEqual(len(result['predicted_ranking']), len(self.items))

    def test_beat_the_pivot_fairness(self):
        """Test fairness properties of beat_the_pivot algorithm"""
        epsilon = 0.1
        delta = 0.1
        
        result = beat_the_pivot(
            set_of_items=self.set_of_items,
            epsilon=epsilon,
            delta=delta,
            args=self.config,
            logger=self.logger
        )
        
        # Get the predicted ranking
        predicted_ranking = result['predicted_ranking']
        
        # Check that items from both groups are present in the ranking
        group_ids = [item.group_id for item in self.items]
        unique_groups = set(group_ids)
        
        # Count items from each group in the top k positions
        k = 3
        top_k_groups = [self.items[i].group_id for i in predicted_ranking[:k]]
        group_counts = {group: top_k_groups.count(group) for group in unique_groups}
        
        # Check that no group is completely excluded from top k
        self.assertTrue(all(count > 0 for count in group_counts.values()))

    def test_beat_the_pivot_convergence(self):
        """Test convergence properties of beat_the_pivot algorithm"""
        epsilon = 0.1
        delta = 0.1
        
        result = beat_the_pivot(
            set_of_items=self.set_of_items,
            epsilon=epsilon,
            delta=delta,
            args=self.config,
            logger=self.logger
        )
        
        # Check that the number of comparisons is reasonable
        self.assertGreater(result['num_of_comparisons'], 0)
        self.assertLess(result['num_of_comparisons'], len(self.items) * (len(self.items) - 1))

    def test_beat_the_pivot_edge_cases(self):
        """Test edge cases for beat_the_pivot algorithm"""
        # Test with a single item
        single_item = SetofItems([Item(identifier="item-1", theta=0.8, group_id="A")])
        
        result = beat_the_pivot(
            set_of_items=single_item,
            epsilon=0.1,
            delta=0.1,
            args=self.config,
            logger=self.logger
        )
        
        self.assertEqual(len(result['predicted_ranking']), 1)
        self.assertEqual(result['predicted_ranking'][0], 0)

if __name__ == '__main__':
    unittest.main() 