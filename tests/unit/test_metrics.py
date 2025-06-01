import unittest
import numpy as np
from active_fair_ranking.algorithms.data import SetofItems, Item
from active_fair_ranking.algorithms.metrics import kendal_tau_ranking_groupwise
from active_fair_ranking.algorithms.utils import setup_logging
import omegaconf
import logging

class TestMetrics(unittest.TestCase):
    def setUp(self):
        # Create a basic config for testing
        self.config = omegaconf.OmegaConf.create({
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': 'test.log'
            },
            'metrics': {
                'lpq_norm': {
                    'p_norm': 2,
                    'q_norm': 2
                }
            }
        })
        self.logger = setup_logging(self.config)
        
        # Create test items with known rankings
        self.items = [
            Item(identifier="item-1", theta=0.9, group_id="A"),
            Item(identifier="item-2", theta=0.8, group_id="A"),
            Item(identifier="item-3", theta=0.7, group_id="B"),
            Item(identifier="item-4", theta=0.6, group_id="B"),
            Item(identifier="item-5", theta=0.5, group_id="A")
        ]
        self.set_of_items = SetofItems(self.items)

    def test_kendal_tau_perfect_ranking(self):
        """Test Kendall Tau with perfect ranking"""
        # Perfect ranking (items ordered by theta)
        perfect_ranking = [0, 1, 2, 3, 4]
        
        result = kendal_tau_ranking_groupwise(
            self.set_of_items,
            perfect_ranking,
            self.config,
            p_norm=2,
            q_norm=2,
            logger=self.logger
        )
        
        # Perfect ranking should have high Kendall Tau
        self.assertGreater(result, 0.8)

    def test_kendal_tau_reversed_ranking(self):
        """Test Kendall Tau with reversed ranking"""
        # Reversed ranking (worst possible)
        reversed_ranking = [4, 3, 2, 1, 0]
        
        result = kendal_tau_ranking_groupwise(
            self.set_of_items,
            reversed_ranking,
            self.config,
            p_norm=2,
            q_norm=2,
            logger=self.logger
        )
        
        # Reversed ranking should have low Kendall Tau
        self.assertLess(result, 0.2)

    def test_kendal_tau_groupwise(self):
        """Test Kendall Tau with group-wise considerations"""
        # Create a ranking that's good within groups but bad across groups
        groupwise_ranking = [0, 2, 1, 3, 4]  # A, B, A, B, A
        
        result = kendal_tau_ranking_groupwise(
            self.set_of_items,
            groupwise_ranking,
            self.config,
            p_norm=2,
            q_norm=2,
            logger=self.logger
        )
        
        # Should have moderate Kendall Tau
        self.assertGreater(result, 0.4)
        self.assertLess(result, 0.8)

    def test_kendal_tau_edge_cases(self):
        """Test Kendall Tau with edge cases"""
        # Test with single item
        single_item = SetofItems([Item(identifier="item-1", theta=0.8, group_id="A")])
        single_ranking = [0]
        
        result = kendal_tau_ranking_groupwise(
            single_item,
            single_ranking,
            self.config,
            p_norm=2,
            q_norm=2,
            logger=self.logger
        )
        
        # Single item should have perfect Kendall Tau
        self.assertEqual(result, 1.0)

    def test_kendal_tau_norm_parameters(self):
        """Test Kendall Tau with different norm parameters"""
        ranking = [0, 1, 2, 3, 4]
        
        # Test with different p and q norms
        result1 = kendal_tau_ranking_groupwise(
            self.set_of_items,
            ranking,
            self.config,
            p_norm=1,
            q_norm=1,
            logger=self.logger
        )
        
        result2 = kendal_tau_ranking_groupwise(
            self.set_of_items,
            ranking,
            self.config,
            p_norm=2,
            q_norm=2,
            logger=self.logger
        )
        
        # Results should be different but both valid
        self.assertNotEqual(result1, result2)
        self.assertTrue(0 <= result1 <= 1)
        self.assertTrue(0 <= result2 <= 1)

if __name__ == '__main__':
    unittest.main() 