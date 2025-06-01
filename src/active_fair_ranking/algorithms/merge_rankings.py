import argparse
import copy
import logging
from collections import defaultdict
from typing import Dict, List

import numpy as np

from active_fair_ranking.algorithms.data import SetofItems
from active_fair_ranking.algorithms.find_the_pivot import find_the_pivot, play_optimized
from active_fair_ranking.algorithms.utils import Graph


def heap_merge(group_ids_ordered, predicted_ranking_group_wise, total_num_of_items, set_of_items, args, logger):
    positions = [0] * len(group_ids_ordered)
    is_group_done = [False] * len(group_ids_ordered)

    final_ranking = []
    count = 0
    oracle_complexity = 0

    initial_candidates = []
    for idx, group_id in enumerate(group_ids_ordered):
        if not is_group_done[idx]:
            initial_candidates.append(predicted_ranking_group_wise[group_id][positions[idx]])

    while count != total_num_of_items:
        # choose the first available item from each available group
        # and play a match between them to choose a winner
        # match can be played using `play_optimized()` function
        # After choosing a winner, update the candidate list with the next item from that group

        temp_set_of_items = set_of_items.get_items_by_ids(initial_candidates)

        if len(temp_set_of_items) > 1:
            if len(temp_set_of_items) > 2:
                raise ValueError("More than 2 items in the candidate list is not allowed")

            result = play_optimized(
                temp_set_of_items.get_probabilities(),
                num_of_rounds=args.num_of_rounds,
                args=args,
            )
            oracle_complexity += 1

            win_count = defaultdict(lambda: 0)
            for item in temp_set_of_items.items:
                win_count[item.identifier] = result[temp_set_of_items.items.index(item)]

            winner_item_id = max(win_count, key=lambda x: win_count[x])
            winner_item_group = set_of_items.get_item_by_id(winner_item_id).group_id
        else:
            winner_item_id = temp_set_of_items.items[0].identifier
            winner_item_group = temp_set_of_items.items[0].group_id

        logger.info(f"Candidate list: {initial_candidates} with winner: {winner_item_id} from {winner_item_group}")

        # update position and available arrays
        positions[group_ids_ordered.index(winner_item_group)] += 1
        if positions[group_ids_ordered.index(winner_item_group)] == len(predicted_ranking_group_wise[winner_item_group]):
            is_group_done[group_ids_ordered.index(winner_item_group)] = True

        # update final ranking
        final_ranking.append(winner_item_id)
        count += 1

        # update candidate list
        initial_candidates = []
        for idx, group_id in enumerate(group_ids_ordered):
            if not is_group_done[idx]:
                initial_candidates.append(predicted_ranking_group_wise[group_id][positions[idx]])
    return final_ranking, oracle_complexity


def heap_merge_two(group_ids_ordered: List[str], predicted_ranking_group_wise: dict, total_num_of_items: int, set_of_items: SetofItems, args, logger):
    set_of_items_copy = copy.deepcopy(set_of_items)
    oracle_complexity = 0

    # inefficient implementation of heap merge with two groups in sequential
    for group_id in group_ids_ordered[1:]:
        temp_group_ids_ordered = [group_ids_ordered[0], group_id]

        temp_predicted_ranking_group_wise = {}
        temp_total_num_of_items = 0
        for group_id in temp_group_ids_ordered:
            temp_predicted_ranking_group_wise[group_id] = predicted_ranking_group_wise[group_id]
            temp_total_num_of_items += len(predicted_ranking_group_wise[group_id])

        ranking, temp_oracle_complexity = heap_merge(temp_group_ids_ordered, temp_predicted_ranking_group_wise, temp_total_num_of_items, set_of_items_copy, args, logger)
        oracle_complexity += temp_oracle_complexity

        # update the first group, with the merged ranking
        predicted_ranking_group_wise[group_ids_ordered[0]] = ranking
        # this also requires to set the merged ranking group id to the first group
        for item_id in ranking:
            set_of_items_copy.items[set_of_items_copy.get_item_index_by_id(item_id)].group_id = group_ids_ordered[0]

    final_ranking = predicted_ranking_group_wise[group_ids_ordered[0]]

    return final_ranking, oracle_complexity


def heap_merge_efficient(group_ids_ordered: List[str], predicted_ranking_group_wise: dict, total_num_of_items: int, set_of_items: SetofItems, args, logger):
    set_of_items_copy = copy.deepcopy(set_of_items)
    oracle_complexity = 0

    total_num_of_items = sum([len(predicted_ranking_group_wise[group_id]) for group_id in group_ids_ordered])

    num_of_groups = len(group_ids_ordered)
    # outer loop
    for idx in range(int(np.log2(num_of_groups)) + 1):
        # inner loop
        groups_to_pop = []
        for idy in range(0, len(group_ids_ordered), 2):
            if idy + 1 >= len(group_ids_ordered):
                # this means that there is only one group left
                # so we can just return the ranking for that group
                pass
            else:
                # merge the two group lists
                temp_group_ids_ordered = group_ids_ordered[idy : idy + 2]

                temp_predicted_ranking_group_wise = {}
                temp_total_num_of_items = 0
                for group_id in temp_group_ids_ordered:
                    temp_predicted_ranking_group_wise[group_id] = predicted_ranking_group_wise[group_id]
                    temp_total_num_of_items += len(predicted_ranking_group_wise[group_id])

                ranking, temp_oracle_complexity = heap_merge(temp_group_ids_ordered, temp_predicted_ranking_group_wise, temp_total_num_of_items, set_of_items_copy, args, logger)
                oracle_complexity += temp_oracle_complexity

                # update the first group, with the merged ranking
                predicted_ranking_group_wise[group_ids_ordered[idy]] = copy.deepcopy(ranking)
                # this also requires to set the merged ranking group id to the first group
                for item_id in ranking:
                    set_of_items_copy.items[set_of_items_copy.get_item_index_by_id(item_id)].group_id = group_ids_ordered[idy]

                # remove the second group from the list
                groups_to_pop.append(idy + 1)

        # remove elements from indices in groups_to_pop from group_ids_ordered
        for idx in sorted(groups_to_pop, reverse=True):
            group_ids_ordered.pop(idx)

    assert len(group_ids_ordered) == 1
    final_ranking = predicted_ranking_group_wise[group_ids_ordered[0]]

    assert len(final_ranking) == total_num_of_items

    # shuffle the final ranking
    return final_ranking, oracle_complexity


def merge_rankings(
    set_of_items: SetofItems,
    set_of_items_group_wise: Dict[str, List[SetofItems]],
    predicted_ranking_group_wise: Dict[str, List[str]],
    group_ids: List[int],
    args: argparse.Namespace,
    merge_mode="ftp",
    logger: logging.Logger = None,
):
    assert merge_mode in ["ftp", "play", "heap"]

    # get group ids based on decreasing order of their sizes
    group_weights = {}
    for g_id in group_ids:
        group_weights[g_id] = len(set_of_items_group_wise[g_id])
    group_ids_ordered = sorted(
        group_ids,
        key=lambda x: group_weights[x],
        reverse=True,
    )

    oracle_complexity = 0
    # total number of items across all groups
    total_num_of_items = sum([len(set_of_items_group_wise[group_id]) for group_id in group_ids_ordered])

    if merge_mode == "heap":
        final_ranking, oracle_complexity = heap_merge_efficient(group_ids_ordered, predicted_ranking_group_wise, total_num_of_items, set_of_items, args, logger)
        return final_ranking, oracle_complexity
    else:
        raise ValueError("Only heap merge is supported for now")
        # calculate pairwise item ordering
        graph = Graph(vertices=set_of_items.get_item_ids(), logger=logger)

        # for all items within a group, their pairwise item ordering is already determined
        for group_id in group_ids_ordered:
            item_ids = predicted_ranking_group_wise[group_id]
            for idx in range(len(item_ids)):
                for idy in range(idx + 1, len(item_ids)):
                    graph.addEdge(item_ids[idx], item_ids[idy])

        # calculate pairwise group ordering across groups
        visited_group_ids = set()
        for group_id in group_ids_ordered:
            visited_group_ids.add(group_id)
            for group_id_other in set(group_ids_ordered) - visited_group_ids:
                item_ids = set_of_items_group_wise[group_id].get_item_ids()
                item_ids_other = set_of_items_group_wise[group_id_other].get_item_ids()

                idx, idy = 0, 0
                while idx < len(item_ids) and idy < len(item_ids_other):
                    # get winner and loser based on play_optimized
                    temp_set_of_items = set_of_items.get_items_by_ids(
                        [item_ids[idx], item_ids_other[idy]],
                        do_sort=False,
                    )
                    assert temp_set_of_items.items[0].group_id != temp_set_of_items
                    assert len(temp_set_of_items) == 2

                    if merge_mode == "ftp":
                        oracle_complexity += 1  # Since only two items are sent to find the pivot, it will call oracle once
                        (pivot_item_id, _) = find_the_pivot(
                            set_of_items=temp_set_of_items,
                            num_of_items=2,
                            num_of_rounds=args.num_of_rounds,
                            subset_size=2,
                            epsilon=min(args.dataset_config[args.dataset].epsilon / 2, 1 / 2),
                            delta=args.dataset_config[args.dataset].delta / 2,
                            args=args,
                            logger=logger,
                        )
                        logger.debug(f"Chose pivot item: {pivot_item_id}")
                        if pivot_item_id == item_ids[idx]:
                            # For all remaining items in item_ids_other, they are all losers
                            for idz in range(idy, len(item_ids_other)):
                                graph.addEdge(item_ids[idx], item_ids_other[idz])

                            # next compare with the next item in item_ids
                            idx += 1
                        else:
                            # For all remaining items in item_ids, they are all losers
                            for idz in range(idx, len(item_ids)):
                                graph.addEdge(item_ids_other[idy], item_ids[idz])
                            # next compare with the next item in item_ids_other
                            idy += 1

                    elif merge_mode == "play":
                        pass
                        # result = play_optimized(
                        #     probabilities=temp_set_of_items.get_probabilities(),
                        #     num_of_rounds=args.num_of_rounds,
                        #     args=args,
                        # )

                        # logger.debug(f"Original probs: {temp_set_of_items.get_probabilities()}")
                        # logger.debug(f"Predicted probs: {result/sum(result)}")

                        # if temp_set_of_items.items[0].group_id != group_id:
                        #     # switch the result ordering since we expect id 0 in result to be of first group
                        #     result = [result[1], result[0]]

                        # # winner is
                        # if result[0] >= result[1]:
                        #     # For all remaining items in item_ids_other, they are all losers
                        #     for idz in range(idy, len(item_ids_other)):
                        #         graph.addEdge(item_ids[idx], item_ids_other[idz])

                        #     # next compare with the next item in item_ids
                        #     idx += 1
                        # else:
                        #     # For all remaining items in item_ids, they are all losers
                        #     for idz in range(idx, len(item_ids)):
                        #         graph.addEdge(item_ids_other[idy], item_ids[idz])
                        #     # next compare with the next item in item_ids_other
                        #     idy += 1

        # create a final ordering such that the pairwise items are satisfied
        final_ranking = graph.topologicalSort()
        return final_ranking, oracle_complexity
