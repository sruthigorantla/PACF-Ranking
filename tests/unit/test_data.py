import unittest
import numpy as np
from active_fair_ranking.algorithms.data import SetofItems, Item, create_ranking
from active_fair_ranking.algorithms.utils import setup_logging
import omegaconf
import logging

class TestDataStructures(unittest.TestCase):
    def setUp(self):
        # Create a basic config for testing
        self.config = omegaconf.OmegaConf.create({
            'logging': {
                'level': 'INFO',
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                'file': 'test.log'
            }
        })
        self.logger = setup_logging(self.config)

    def test_item_creation(self):
        """Test creation of Item objects"""
        item = Item(identifier="item-1", theta=0.8, group_id="A")
        self.assertEqual(item.identifier, "item-1")
        self.assertEqual(item.theta, 0.8)
        self.assertEqual(item.group_id, "A")

    def test_set_of_items_creation(self):
        """Test creation of SetofItems object"""
        items = [
            Item(identifier="item-1", theta=0.8, group_id="A"),
            Item(identifier="item-2", theta=0.6, group_id="B"),
            Item(identifier="item-3", theta=0.9, group_id="A")
        ]
        set_of_items = SetofItems(items)
        
        self.assertEqual(len(set_of_items.items), 3)
        self.assertEqual(set_of_items.get_group_ids(), ["A", "B"])
        self.assertEqual(len(set_of_items.get_item_ids_with_group_id("A")), 2)

    def test_get_probabilities(self):
        """Test probability calculation for items"""
        items = [
            Item(identifier="item-1", theta=0.8, group_id="A"),
            Item(identifier="item-2", theta=0.6, group_id="B"),
            Item(identifier="item-3", theta=0.9, group_id="A")
        ]
        set_of_items = SetofItems(items)
        probabilities = set_of_items.get_probabilities()
        
        self.assertEqual(len(probabilities), 3)
        self.assertTrue(all(0 <= p <= 1 for p in probabilities))

    def test_create_ranking(self):
        """Test creation of ranking with different parameters"""
        # Test with random protected attributes
        ranking = create_ranking(
            num_of_items=5,
            num_of_groups=2,
            dataset=None,
            prot_attrs=["A", "B", "A", "B", "A"],
            args=self.config,
            logger=self.logger
        )
        
        self.assertEqual(len(ranking.items), 5)
        self.assertEqual(len(ranking.get_group_ids()), 2)

    def test_get_items_by_ids(self):
        """Test retrieving items by their IDs"""
        items = [
            Item(identifier="item-1", theta=0.8, group_id="A"),
            Item(identifier="item-2", theta=0.6, group_id="B"),
            Item(identifier="item-3", theta=0.9, group_id="A")
        ]
        set_of_items = SetofItems(items)
        
        selected_items = set_of_items.get_items_by_ids(["item-1", "item-3"])
        self.assertEqual(len(selected_items.items), 2)
        self.assertTrue(all(item.group_id == "A" for item in selected_items.items))

if __name__ == '__main__':
    unittest.main() 