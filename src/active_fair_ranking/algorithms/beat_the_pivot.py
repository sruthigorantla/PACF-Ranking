import argparse
import logging
import math
import random
from collections import defaultdict

import numpy as np

from active_fair_ranking.algorithms.data import SetofItems
from active_fair_ranking.algorithms.find_the_pivot import find_the_pivot, play_optimized


def initialize_pivot(
    set_of_items: SetofItems,
    epsilon: float,
    delta: float,
    args: argparse.Namespace,
    logger: logging.Logger,
):
    assert len(set_of_items) == args.num_of_items

    pivot_item_id, num_of_find_the_pivot_loops = find_the_pivot(
        set_of_items=set_of_items,
        num_of_items=args.num_of_items,
        subset_size=args.beat_the_pivot.subset_size,
        epsilon=min(epsilon / 2, 1 / 2),
        delta=delta / 2,
        args=args,
        logger=logger,
    )
    logger.info("\n\nPivot item: %s", pivot_item_id)

    assert (
        args.final_sample_size
        == math.ceil((len(set_of_items) - 1) / (args.beat_the_pivot.subset_size - 1))
        * args.num_of_rounds
    )

    set_S_item_ids = set(set_of_items.get_item_ids()) - set([pivot_item_id])
    num_of_sets = math.ceil(
        (args.num_of_items - 1) / (args.beat_the_pivot.subset_size - 1)
    )

    # split set_S_item_ids into num_of_sets subsets
    set_S_item_ids_subsets = np.array_split(sorted(set_S_item_ids), num_of_sets)

    # convert set_S_item_ids_subsets to subset of sets
    set_S_item_ids_subsets = [set(subset_ids) for subset_ids in set_S_item_ids_subsets]

    # if the last subset's size is less than k-1, then randomly add items ids from set_S_item_ids that is not already in this subset
    if len(set_S_item_ids_subsets[-1]) < args.beat_the_pivot.subset_size - 1:
        set_S_item_ids_subsets[-1] = set_S_item_ids_subsets[-1].union(
            random.sample(
                sorted(set_S_item_ids - set_S_item_ids_subsets[-1]),
                k=args.beat_the_pivot.subset_size - 1 - len(set_S_item_ids_subsets[-1]),
            )
        )

    # For all the subsets, excluding last one, the size should be equal to k-1
    for subset in set_S_item_ids_subsets:
        assert len(subset) == args.beat_the_pivot.subset_size - 1

    # Add pivot_item_id to all the subsets
    set_S_item_ids_subsets = [
        subset.union([pivot_item_id]) for subset in set_S_item_ids_subsets
    ]

    return (
        set_S_item_ids_subsets,
        pivot_item_id,
        num_of_find_the_pivot_loops,
    )


def beat_the_pivot(
    set_of_items: SetofItems,
    epsilon: float,
    delta: float,
    args: argparse.Namespace,
    logger: logging.Logger,
):
    (
        subset_G_item_ids_subsets,
        pivot_item_id,
        num_of_find_the_pivot_loops,
    ) = initialize_pivot(
        set_of_items=set_of_items,
        epsilon=epsilon,
        delta=delta,
        args=args,
        logger=logger,
    )

    # win_count is a dictionary of item indentifiers with their win counts
    win_count = defaultdict(lambda: 0)
    p = defaultdict(lambda: 0)
    oracle_sample_complexity = num_of_find_the_pivot_loops

    for subset_id, subset_item_ids in enumerate(subset_G_item_ids_subsets):
        oracle_sample_complexity += 1
        logger.info("Subset %s: %s", subset_id, subset_item_ids)
        logger.info("Playing the game for %d rounds ...", args.num_of_rounds)

        # Current group's set of items
        set_of_items_g = set_of_items.get_items_by_ids(subset_item_ids)

        result = play_optimized(
            probabilities=set_of_items_g.get_probabilities(),
            num_of_rounds=args.num_of_rounds,
            args=args,
        )

        # win_count is a dictionary of item indentifiers with their win counts
        win_count = defaultdict(lambda: 0)
        for item in set_of_items_g.items:
            win_count[item.identifier] = result[set_of_items_g.items.index(item)]

        # print win percentages of each item in set_A along with their original probabilities
        logger.debug("Win percentages:")
        for item in set_of_items_g.items:
            logger.debug(
                "Item %s: %f (original probability: %f)",
                item.identifier,
                win_count[item.identifier] / args.num_of_rounds,
                set_of_items_g.get_probabilities()[set_of_items_g.items.index(item)],
            )

        # define a matrix of p_i,j where p_i,j is win[i] / (win[i] + win[j])
        # where i and j are indices of items in set_A and i != j

        for item_1 in set_of_items_g.items:
            if win_count[item_1.identifier] + win_count[pivot_item_id] == 0:
                p[item_1.identifier] = 0
            else:
                p[item_1.identifier] = win_count[item_1.identifier] / (
                    win_count[item_1.identifier] + win_count[pivot_item_id]
                )

    # get a ordering of item_ids such that pivot_item_id is at the beginning
    # the remaining item_ids are sorted in descending order of p_i,j
    sorted_item_ids = sorted(
        p.keys(),
        key=lambda item_id: p[item_id],
        reverse=True,
    )
    # put pivot_item_id at the beginning
    sorted_item_ids.insert(0, sorted_item_ids.pop(sorted_item_ids.index(pivot_item_id)))

    return sorted_item_ids, args.final_sample_size, oracle_sample_complexity
