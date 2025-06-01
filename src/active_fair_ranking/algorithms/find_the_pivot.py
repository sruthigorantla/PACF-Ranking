import argparse
import logging
import math
import random
from collections import defaultdict
from typing import List

import numpy as np

from active_fair_ranking.algorithms.data import SetofItems


def initialize(
    set_of_items: SetofItems,
    subset_size: int,
    epsilon: float,
    delta: float,
    args: argparse.Namespace,
    logger: logging.Logger,
):
    # select a random item from num_of_items

    logger.info("\n\nInitializing for finding the pivot sub-routine ...")

    # choose a random item from set_of_items
    r_item_id = random.choice(set_of_items.items).identifier

    logger.info(f"picked item id = {r_item_id} from {len(set_of_items)} items")

    # randomly select (subset_size - 1) items from the remaining (num_of_items - 1) items
    # without replacement
    local_item_ids = [
        item.identifier for item in set_of_items.items if item.identifier != r_item_id
    ]
    assert len(local_item_ids) >= subset_size - 1

    set_A_item_ids = set(
        random.sample(
            local_item_ids,
            k=subset_size - 1,
        )
    )
    assert r_item_id not in set_A_item_ids
    assert len(set_A_item_ids) == subset_size - 1

    # Add r to set_A_indices
    set_A_item_ids.add(r_item_id)

    # set_S is the complement of set_A_indices in the set of all items
    set_S_item_ids = (
        set([item.identifier for item in set_of_items.items]) - set_A_item_ids
    )

    logger.info("item ids for Set A: %s", set_A_item_ids)
    logger.info("item ids for Set S: %s", set_S_item_ids)

    logger.info("Initialization complete.\n")

    return (
        r_item_id,
        set_of_items.get_items_by_ids(set_A_item_ids),
        set_of_items.get_items_by_ids(set_S_item_ids),
    )


def play_optimized(
    probabilities: List[float],
    num_of_rounds: int,
    args: argparse.Namespace,
):
    assert len(probabilities) == 2, "we only support 2 items for now"
    # play the game for num_rounds
    # return the result of the game

    # winning criteria: sample item with highest probability using softmax of theta
    # sample the winner from the categorical distribution
    # args.final_sample_size += 1

    # p_i_j = probabilities[0] / (probabilities[0] + probabilities[1])
    # if random.rand() < p_i_j:
    #     # return [1, 0] if item 1 wins
    #     return [1, 0]
    # else:
    #     return [0, 1]

    args.final_sample_size += num_of_rounds
    answer = np.random.multinomial(num_of_rounds, probabilities)

    # with 10% chance, switch the answer
    # if np.random.rand() < 0.2:
    #     answer[0], answer[1] = answer[1], answer[0]

    return answer


def find_the_pivot(
    set_of_items: SetofItems,
    num_of_items: int,
    subset_size: int,
    epsilon: float,
    delta: float,
    args: argparse.Namespace,
    logger: logging.Logger,
):
    # _set_seed(args.seed)
    assert len(set_of_items.items) == num_of_items

    r_item_id, set_of_items_A, set_of_items_S = initialize(
        set_of_items=set_of_items,
        subset_size=subset_size,
        epsilon=epsilon,
        delta=delta,
        args=args,
        logger=logger,
    )

    # run a while loop until the pivot is found
    # l keeps increasing from 1 onwards
    num_of_find_the_pivot_loops = 0
    while True:
        num_of_find_the_pivot_loops += 1
        logger.info("Playing the game for %d rounds ...", args.num_of_rounds)

        result = play_optimized(
            probabilities=set_of_items_A.get_probabilities(),
            num_of_rounds=args.num_of_rounds,
            args=args,
        )

        # if set_of_items_A has items item-1 to item-5, then switch the result
        # for item_id in ["item-1", "item-2", "item-3", "item-4", "item-5"]:
        #     if item_id in set_of_items_A.get_item_ids():
        #         #swap results
        #         result[0], result[1] = result[1], result[0]

        # count number of times each item in set_A won in play() function
        # win_count is a dictionary of item indentifiers with their win counts
        win_count = defaultdict(lambda: 0)
        for item in set_of_items_A.items:
            win_count[item.identifier] = result[set_of_items_A.items.index(item)]

        # print win percentages of each item in set_A along with their original probabilities
        logger.debug("Win percentages:")
        for item in set_of_items_A.items:
            logger.debug(
                "Item %s: %f (original probability: %f)",
                item.identifier,
                win_count[item.identifier] / args.num_of_rounds,
                set_of_items_A.get_probabilities()[set_of_items_A.items.index(item)],
            )

        # c_item_id is the item in set_A with the highest win percentage
        c_item_id = max(win_count, key=lambda x: win_count[x])

        # define a matrix of p_i,j where p_i,j is win[i] / (win[i] + win[j])
        # where i and j are indices of items in set_A and i != j

        # p is a dictionary of dictionaries
        p = defaultdict(lambda: defaultdict(lambda: 0))
        for item_1 in set_of_items_A.items:
            for item_2 in set_of_items_A.items:
                if item_1.identifier != item_2.identifier:
                    total_wins = (
                        win_count[item_1.identifier] + win_count[item_2.identifier]
                    )
                    if (total_wins) == 0:
                        p[item_1.identifier][item_2.identifier] = 0
                    else:
                        p[item_1.identifier][item_2.identifier] = (
                            win_count[item_1.identifier] / total_wins
                        )

        if p[c_item_id][r_item_id] > (0.5 + (epsilon / 2)):
            r_item_id = c_item_id

        # if set_S is null, then break
        if len(set_of_items_S) == 0:
            break
        elif len(set_of_items_S) < (subset_size - 1):
            # get ordered set of item ids
            set_A_item_ids = set(set_of_items_A.get_item_ids())
            set_S_item_ids = set(set_of_items_S.get_item_ids())

            # remove r_item_id from set_A_item_ids
            set_A_item_ids.remove(r_item_id)

            # uniformly at random sample (subset_size - 1 - len(set_S)) items from set_A_indices
            num_ids_to_sample = subset_size - 1 - len(set_of_items_S)
            assert len(set_A_item_ids) >= num_ids_to_sample
            set_A_item_ids = set(
                random.sample(
                    sorted(set_A_item_ids),
                    k=num_ids_to_sample,
                )
            )

            set_A_item_ids = set_A_item_ids.union(set_S_item_ids)
            set_A_item_ids.add(r_item_id)

            # redefine set_A based on new set_A_indices
            logger.info(
                "Old set_of_items_A: %s",
                set_of_items_A.get_item_ids(),
            )
            set_of_items_A = set_of_items.get_items_by_ids(set_A_item_ids)
            logger.info(
                "New set_of_items_A: %s",
                set_of_items_A.get_item_ids(),
            )
            assert len(set_of_items_A) == len(set_A_item_ids)
            set_of_items_S = SetofItems([])

        else:
            # remove r_index from set_A_indices
            set_A_item_ids = set(set_of_items_A.get_item_ids())
            set_S_item_ids = set(set_of_items_S.get_item_ids())

            # select (k-1) items from set_S_indices
            set_A_item_ids = set(
                random.sample(sorted(set_S_item_ids), k=(subset_size - 1))
            )
            set_A_item_ids.add(r_item_id)
            set_S_item_ids = set_S_item_ids - set_A_item_ids

            set_of_items_A = set_of_items.get_items_by_ids(set_A_item_ids)
            set_of_items_S = set_of_items.get_items_by_ids(set_S_item_ids)

            assert len(set_of_items_A.items) == len(set_A_item_ids)
            assert len(set_of_items_S.items) == len(set_S_item_ids)

    return r_item_id, num_of_find_the_pivot_loops
