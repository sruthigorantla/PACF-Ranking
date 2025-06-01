import argparse
import logging
import math
import random
import copy
from collections import defaultdict
import time
import numpy as np

from active_fair_ranking.algorithms.data import SetofItems
from active_fair_ranking.algorithms.find_the_pivot import find_the_pivot, play_optimized


def baseline(
    set_of_items: SetofItems,
    epsilon: float,
    delta: float,
    args: argparse.Namespace,
    logger: logging.Logger,
):
    set_S_item_ids = set(set_of_items.get_item_ids())

    # create n partitions of items. Initialize each partitition to be empty
    partitions = [[] for _ in range(args.num_of_items)]

    # maintain estimate of scores for each item. Initialized to 0
    scores = defaultdict(int)

    iteration_id = 0
    while len(set_S_item_ids) > 1:

        iteration_id += 1
        alpha = math.sqrt(
            math.log2(125 * args.num_of_items * math.log2(1.12 * iteration_id) / delta)
            / iteration_id
        )
        if iteration_id % 1000 == 0:
            print(
                f"Iteration {iteration_id}, len(set_S_item_ids): {len(set_S_item_ids)}"
            )
        for item_id in set_S_item_ids:
            # choose another item uniformly at random from the complement set
            complement_set = set_S_item_ids - {item_id}
            other_item_id = random.choice(list(complement_set))

            # do pairwise comparison between the two items
            set_of_paired_items = set_of_items.get_items_by_ids(
                [item_id, other_item_id], do_sort=False
            )

            result = play_optimized(
                probabilities=set_of_paired_items.get_probabilities(),
                num_of_rounds=2**5,
                args=args,
            )

            for item in set_of_paired_items.items:
                if set_of_paired_items.items.index(item) == np.argmax(result):
                    winner = item.identifier
                    break

            assert winner in set_of_paired_items.get_item_ids()

            # if item_id wins, increment its score
            scores[item_id] = scores[item_id] * ((iteration_id - 1) / iteration_id)

            if winner == item_id:
                scores[item_id] += 1 / iteration_id

        # sort the items in set_S_item_ids based on their scores in decreasing order
        sorted_items = sorted(set_S_item_ids, key=lambda x: scores[x], reverse=True)
        sorted_items = [item_id for item_id in sorted_items]

        # initialize k partitions
        k_vals = [val for val in range(args.num_of_items + 1)]

        to_remove_item_ids = []
        update_k_vals = np.array([0 for _ in range(args.num_of_items + 1)])

        for ind, (item_id) in enumerate(sorted_items):

            ell = 0
            while ind + 1 > k_vals[ell]:
                ell += 1

                is_cond_1 = False
                is_cond_2 = False

                # check condition 1
                if k_vals[ell - 1] == 0 or scores[item_id] < scores[
                    sorted_items[k_vals[ell - 1] - 1]
                ] - (4 * alpha):
                    is_cond_1 = True

                # check condition 2
                if k_vals[ell] == len(set_S_item_ids) or scores[item_id] > scores[
                    sorted_items[k_vals[ell]]
                ] + (4 * alpha):
                    is_cond_2 = True

                if is_cond_1 and is_cond_2:
                    partitions[ell - 1].append(item_id)
                    to_remove_item_ids.append(item_id)

                    update_k_vals[ell] += 1
                    break

        for ind, val in enumerate(update_k_vals):
            if ind == 0:
                continue
            k_vals[ind:] -= val

        for item_id in to_remove_item_ids:
            set_S_item_ids.remove(item_id)

    print("Partitions:")
    print(partitions)

    exit()
