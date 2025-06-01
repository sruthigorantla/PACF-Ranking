import argparse
import logging
from collections import defaultdict
from typing import List

import numpy as np
from numpy import linalg as LA

from active_fair_ranking.algorithms.data import SetofItems


def g(a, b, theta_a, theta_b, epsilon):
    if (theta_a > theta_b + epsilon) and (a > b):
        return 1.0
    else:
        return 0.0


def g_diff(a: int, b: int, theta_a: float, theta_b: float, epsilon: float):
    """_summary_

    Args:
        a (_type_): _description_
        b (_type_): _description_
        theta_a (_type_): _description_
        theta_b (_type_): _description_
        epsilon (_type_): _description_

    Returns:
        float: theta of a - theta of b
    """

    ## the following is for what is in the paper
    if (theta_a > theta_b + epsilon) and (a > b):
        return theta_a - theta_b
    else:
        return 0.0

    ## the following is for what is in the code
    # if (theta_a > theta_b) and (a > b):
    #     return theta_a - theta_b
    # else:
    #     return 0.0


# def kendal_tau_ranking(
#     set_of_items: SetofItems,
#     predicted_ranking: List[str],
#     args: argparse.Namespace,
# ):
#     logger.info(f"\n\nCalculating metrics...")
#     num_of_items = len(set_of_items)
#     metric = 0.0
#     for idx_i in range(num_of_items):
#         for idx_j in range(idx_i + 1, num_of_items):
#             item_id_i = predicted_ranking[idx_i]
#             item_id_j = predicted_ranking[idx_j]
#             theta_i = set_of_items.get_item_by_id(item_id_i).theta
#             theta_j = set_of_items.get_item_by_id(item_id_j).theta

#             metric += g(idx_j, idx_i, theta_j, theta_i, args.dataset_config[args.dataset].epsilon)

#     # normalize the metric by divinding by 1 / (num_of_items choose 2)
#     metric = metric / (num_of_items * (num_of_items - 1) / 2)

#     return metric


def get_lp_lq_norm(
    itemwise_metric,
    set_of_items,
    args,
    p_norm=1,
    q_norm=1,
    logger: logging.Logger = None,
):
    assert len(itemwise_metric) == args.num_of_items

    groupwise_costs = defaultdict(list)
    for item_id in itemwise_metric.keys():
        logger.debug(
            f"Item: {item_id}, Cost: {itemwise_metric[item_id]} from group: {set_of_items.get_item_by_id(item_id).group_id}"
        )
        groupwise_costs[set_of_items.get_item_by_id(item_id).group_id].append(
            itemwise_metric[item_id]
        )

    l_p_norms = []
    l_p_norms_groupwise = defaultdict(float)
    count = 0
    for group_id, val in groupwise_costs.items():
        group_wt = 1.0 / len(groupwise_costs[group_id])
        l_p_norm = group_wt * LA.norm(groupwise_costs[group_id], ord=p_norm)
        l_p_norms.append(l_p_norm)
        l_p_norms_groupwise[group_id] = l_p_norm
        count += len(groupwise_costs[group_id])
    assert count == args.num_of_items

    # take l_q norm across groups
    l_q_norm = LA.norm(l_p_norms, ord=q_norm)
    return l_q_norm, l_p_norms_groupwise


def kendal_tau_ranking_groupwise(
    set_of_items: SetofItems,
    predicted_ranking: List[str],
    args: argparse.Namespace,
    p_norm=None,
    q_norm=None,
    logger: logging.Logger = None,
):
    logger.info(f"\n\nCalculating groupwise metrics...")
    num_of_items = len(set_of_items)
    assert len(set_of_items) == len(predicted_ranking)

    metric = defaultdict(float)
    itemwise_metric = defaultdict(lambda: 0.0)
    cnt = defaultdict(int)

    # ensure that predicted_ranking[0] is part of the key in itemwise_metric
    itemwise_metric[predicted_ranking[0]] = 0.0

    for idx_i in range(num_of_items):
        for idx_j in range(idx_i + 1, num_of_items):
            item_id_i = predicted_ranking[idx_i]
            item_id_j = predicted_ranking[idx_j]
            item_i = set_of_items.get_item_by_id(item_id_i)
            item_j = set_of_items.get_item_by_id(item_id_j)
            theta_i = item_i.theta
            theta_j = item_j.theta

            group_id_j = item_j.group_id

            # since item idx_i is ranked higher, g() should return 0.0 by default
            if "compas" in args.datasets:
                eps = args.dataset_config["compas"].epsilon
            elif "german" in args.datasets:
                eps = args.dataset_config["german"].epsilon
            else:
                eps = args.dataset_config[args.dataset].epsilon

            assert (
                g(
                    idx_i,
                    idx_j,
                    theta_i,
                    theta_j,
                    eps,
                )
                == 0.0
            )

            itemwise_metric[item_id_j] = max(
                g_diff(
                    idx_j,
                    idx_i,
                    theta_j,
                    theta_i,
                    eps,
                ),
                itemwise_metric[item_id_j],
            )

            metric[group_id_j] += g(
                idx_j,
                idx_i,
                theta_j,
                theta_i,
                eps,
            )
            cnt[group_id_j] += 1

    # normalize the metric by divinding by cnt for each group id
    # sanity
    total = 0.0
    for g_id in metric.keys():
        total += metric[g_id]
        metric[g_id] = metric[g_id] / cnt[g_id]

    assert (num_of_items * (num_of_items - 1) / 2) == sum(cnt.values())
    kendall_tau = total / sum(cnt.values())

    lpq_norm, l_p_norms_groupwise = get_lp_lq_norm(
        itemwise_metric,
        set_of_items,
        args,
        p_norm=p_norm,
        q_norm=q_norm,
        logger=logger,
    )

    return metric, kendall_tau, lpq_norm, l_p_norms_groupwise
