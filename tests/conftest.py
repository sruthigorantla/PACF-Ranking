import pytest
import omegaconf
import logging
import math
from active_fair_ranking.algorithms.data import SetofItems, Item
from active_fair_ranking.algorithms.utils import setup_logging

@pytest.fixture
def config():
    """Create a basic configuration for testing"""
    num_of_items = 2
    subset_size = 2
    num_of_rounds = 100
    
    # Calculate final_sample_size based on the formula
    final_sample_size = math.ceil((num_of_items - 1) / (subset_size - 1)) * num_of_rounds
    
    return omegaconf.OmegaConf.create({
        'logging': {
            'level': 'INFO',
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'file': 'test.log'
        },
        'num_of_rounds': num_of_rounds,
        'num_of_items': num_of_items,
        'final_sample_size': final_sample_size,
        'beat_the_pivot': {
            'subset_size': subset_size
        },
        'metrics': {
            'lpq_norm': {
                'p_norm': 2,
                'q_norm': 2
            }
        }
    })

@pytest.fixture
def logger(config):
    """Create a logger for testing"""
    return setup_logging(config)

@pytest.fixture
def sample_items():
    """Create a sample set of items for testing"""
    items = [
        Item(identifier="item-1", theta=0.9, group_id="A"),
        Item(identifier="item-2", theta=0.8, group_id="A"),
        Item(identifier="item-3", theta=0.7, group_id="B"),
        Item(identifier="item-4", theta=0.6, group_id="B"),
        Item(identifier="item-5", theta=0.5, group_id="A")
    ]
    return SetofItems(items)

@pytest.fixture
def single_item():
    """Create a single item for testing edge cases"""
    return SetofItems([Item(identifier="item-1", theta=0.8, group_id="A")]) 