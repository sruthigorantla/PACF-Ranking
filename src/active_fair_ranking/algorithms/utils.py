import concurrent.futures
import logging
import math
import random
import sys

# Python program to print topological sorting of a DAG
from collections import Counter, defaultdict
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def setup_logging(config):
    # Extract the logger configuration from the loaded configuration
    logging_config = config.logging

    # Set up the logger
    logging.basicConfig(
        level=logging_config.level,
        format=logging_config.format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(logging_config.file, mode="w", encoding="utf-8"),
        ],
    )

    # Use the logger
    logger = logging.getLogger(__name__)
    logger.info("Logging initialized.")
    return logger


# Class to represent a graph for merge ranking
class Graph:
    def __init__(self, vertices, logger):
        self.logger = logger
        self.graph = defaultdict(list)  # dictionary containing adjacency List
        self.V = vertices  # No. of vertices

    # function to add an edge to graph
    def addEdge(self, u, v):
        self.graph[u].append(v)

    # A recursive function used by topologicalSort
    def topologicalSortUtil(self, item, visited, stack):
        # Mark the current node as visited.
        visited[item] = True

        # Recur for all the vertices adjacent to this vertex
        for item_j in self.graph[item]:
            if visited[item_j] == False:
                self.topologicalSortUtil(item_j, visited, stack)

        # Push current vertex to stack which stores result
        stack.insert(0, item)

    # neighbors generator given key
    def neighbor_gen(self, v):
        for k in self.graph[v]:
            yield k

    # non recursive topological sort
    def nonRecursiveTopologicalSortUtil(self, v, visited, stack):
        # working stack contains key and the corresponding current generator
        working_stack = [(v, self.neighbor_gen(v))]

        while working_stack:
            # get last element from stack
            v, gen = working_stack.pop()
            visited[v] = True

            # run through neighbor generator until it's empty
            for next_neighbor in gen:
                if not visited[next_neighbor]:  # not seen before?
                    # remember current work
                    working_stack.append((v, gen))
                    # restart with new neighbor
                    working_stack.append(
                        (
                            next_neighbor,
                            self.neighbor_gen(next_neighbor),
                        )
                    )
                    break
            else:
                # no already-visited neighbor (or no more of them)
                stack.append(v)

    def topologicalSort(self):
        # Mark all the vertices as not visited
        visited = {}
        for node in self.V:
            visited[node] = False

        stack = []

        # calculate in-degrees of all vertices
        in_degree = defaultdict(int)
        for item in self.V:
            in_degree[item] = 0

        for item in self.V:
            for item_j in self.graph[item]:
                in_degree[item_j] += 1

        # Call the recursive helper function to store Topological
        # Sort starting from all vertices one by one
        for item in self.V:
            if visited[item] == False:
                self.nonRecursiveTopologicalSortUtil(item, visited, stack)

        # Print contents of stack
        stack.reverse()
        self.logger.info("Topological sort: {}".format(stack))
        assert len(stack) == len(self.V)
        return stack


def softmax(x: List[float]) -> List[float]:
    # Numerically stable implementation of softmax

    # Subtract the maximum value for numerical stability
    x = x - np.max(x)

    # Compute the numerator
    exp_x = np.exp(x)

    # Compute the denominator
    sum_exp_x = np.sum(exp_x)

    # Compute the softmax values
    softmax_x = exp_x / sum_exp_x

    assert np.isclose(np.sum(softmax_x), 1.0), "Softmax values do not sum to 1"

    return softmax_x


def _set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)


# write a decorator that converts a function call into multithreded call
def multithreaded_execution(func):
    def wrapper(*args, **kwargs):
        # Create a progress bar
        progress_bar = tqdm(total=len(args), file=sys.stdout) if len(args) > 2 else None

        # Function to update the progress bar
        def update_progress(_):
            progress_bar.update(1)
            progress_bar.refresh()

        with concurrent.futures.ThreadPoolExecutor() as executor:
            results = [
                executor.submit(func, arg1, args[-1], **kwargs) for arg1 in args[:-1]
            ]

            # Add the update_progress function as a callback for each future
            if len(args) > 2:
                for result in results:
                    result.add_done_callback(update_progress)

                progress_bar.close()

            concurrent.futures.wait(results)
            return [future.result() for future in results]

    return wrapper


def plot_groupwise_lineplot(results, dim, args, axs, show_legend=True):
    mean_results = {}
    std_results = {}
    sample_sizes_dict = {}

    group_to_linestyle = {
        "g-0": "solid",
        "g-1": "dashed",
        "g-2": "dotted",
        "g-3": "dashdot",
    }
    group_to_marker = {
        "g-0": "o",
        "g-1": "v",
        "g-2": "+",
        "g-3": "x",
    }
    color_mapper = {
        "color-blind-btp": "blue",
        "color-aware-btp": "red",
        "baseline": "green",
    }
    label_map = {
        "color-aware-btp": "Fair-BTP",
        "color-blind-btp": "BTP",
        "baseline": "AR",
    }
    for group_wise_mode in [
        "color-blind-btp",
        "color-aware-btp",
        "baseline",
    ]:
        if group_wise_mode not in results:
            continue

        # get group ids
        set_of_items = results[group_wise_mode]["set_of_items"]
        group_ids = set_of_items.get_group_ids()

        orig_sample_sizes = list(results[group_wise_mode]["result"].keys())
        orig_sample_sizes = sorted(orig_sample_sizes)
        sample_sizes = np.array(
            [
                np.mean(
                    results[group_wise_mode]["result"][num_of_rounds]["sample_size"]
                )
                for num_of_rounds in list(results[group_wise_mode]["result"].keys())
            ]
        )
        sample_sizes = sorted(sample_sizes)

        # get the mean and std of the results

        mean_results[group_wise_mode] = defaultdict(list)
        std_results[group_wise_mode] = defaultdict(list)

        sample_sizes_dict[group_wise_mode] = sample_sizes

        for sample_size, new_sample_size in zip(orig_sample_sizes, sample_sizes):
            for g_id in group_ids:
                mean_results[group_wise_mode][f"{dim}_{g_id}"].append(
                    np.mean(
                        results[group_wise_mode]["result"][sample_size][f"{dim}_{g_id}"]
                    )
                )
                std_results[group_wise_mode][f"{dim}_{g_id}"].append(
                    np.std(
                        results[group_wise_mode]["result"][sample_size][f"{dim}_{g_id}"]
                    )
                )

    # plot a single line plot where all (group, group_wise_mode) pairs are plotted
    # color-blind should be solid lines and color-aware should be dotted lines
    for _, g_id in enumerate(group_ids):
        for group_wise_mode in [
            "color-blind-btp",
            "color-aware-btp",
            "baseline",
        ]:
            if group_wise_mode not in sample_sizes_dict:
                continue
            axs[0].plot(
                sample_sizes_dict[group_wise_mode],
                mean_results[group_wise_mode][f"{dim}_{g_id}"],
                label=label_map[group_wise_mode] + f"-{g_id}",
                color=color_mapper[group_wise_mode],
                linestyle=group_to_linestyle[g_id],
                marker=group_to_marker[g_id],
                markersize=8,
            )
            axs[0].set_xscale("log")
        if show_legend:
            axs[0].legend(fontsize=15, loc="upper right", bbox_to_anchor=(1, 1))

    # increase font size of x and y axis
    axs[0].tick_params(axis="x", labelsize=20)
    axs[0].tick_params(axis="y", labelsize=20)

    # add dataset name for this plot
    axs[0].set_title(
        f"Dataset={args.dataset}",
        fontsize=20,
    )


def plot_barchart(results, group_ids, dim, args, axs):
    # for all group_ids plot a bar chart with error bars
    # x-axis is the sample size options with every sample size having a bar for every group
    # y-axis is the kendall tau value

    for i_prime, group_wise_mode in enumerate(["color-blind", "color-aware"]):
        # plot the results

        orig_sample_sizes = list(results[group_wise_mode].keys())
        orig_sample_sizes = sorted(orig_sample_sizes)
        sample_sizes = np.array(
            [
                np.mean(results[group_wise_mode][num_of_rounds]["sample_size"])
                for num_of_rounds in list(results[group_wise_mode].keys())
            ]
        )

        # sort the results by sample size
        sorted_indices = np.argsort(sample_sizes)
        sample_sizes = sample_sizes[sorted_indices]
        # get the mean and std of the results
        mean_results = {}
        std_results = {}

        for sample_size, new_sample_size in zip(orig_sample_sizes, sample_sizes):
            mean_results[new_sample_size] = {}
            std_results[new_sample_size] = {}
            for g_id in group_ids:
                mean_results[new_sample_size][f"{dim}_{g_id}"] = np.mean(
                    results[group_wise_mode][sample_size][f"{dim}_{g_id}"]
                )
                std_results[new_sample_size][f"{dim}_{g_id}"] = np.std(
                    results[group_wise_mode][sample_size][f"{dim}_{g_id}"]
                )

        width = 0.2
        for i, g_id in enumerate(group_ids):
            axs[i_prime].bar(
                np.arange(len(sample_sizes)) + (i * width),
                [
                    mean_results[sample_size][f"{dim}_{g_id}"]
                    for sample_size in sample_sizes
                ],
                width,
                yerr=[
                    std_results[sample_size][f"{dim}_{g_id}"]
                    for sample_size in sample_sizes
                ],
                label=f"group {g_id}",
            )

        axs[i_prime].set_ylabel(f"lp={args.metrics.lpq_norm.p_norm}-norm")
        # axs[i_prime].set_ylim([0, 0.1])
        axs[i_prime].set_xlabel("Sample size")
        axs[i_prime].set_xticks(np.arange(len(sample_sizes)) + width / 2)
        axs[i_prime].set_xticklabels(sample_sizes)
        axs[i_prime].set_title(
            f"Dataset={args.dataset} | lp={args.metrics.lpq_norm.p_norm}_norm ({group_wise_mode}) | {args.num_of_items} items",
            fontsize=20,
        )
        axs[i_prime].legend()


def plot_rankings(ax, rankings, dataset=None, args=None):
    positions = [rank + 1 for rank in range(len(rankings))]
    g_ids = [item.group_id for item in rankings.items]
    thetas = [item.theta for item in rankings.items]

    # plot bar chart of thetas where x-axis is group id and y-axis is theta
    # and color the bar based on group-id
    group_to_color = {
        "g-0": "red",
        "g-1": "blue",
        "g-2": "green",
        "g-3": "yellow",
        "g-4": "orange",
        "g-5": "black",
    }
    colors = [group_to_color[g_id] for g_id in g_ids]
    ax.bar(positions, thetas, color=colors)
    # add legend
    for g_id in group_to_color.keys():
        if group_to_color[g_id] in colors:
            ax.plot([], [], color=group_to_color[g_id], label=f"Group : {g_id}")

    # legend should come inside the plot and scale to the plot size
    ax.legend(loc="upper right", fontsize=40)

    ax.set_ylabel("True Scores", fontsize=40)
    ax.set_xlabel("Items", fontsize=40)

    # increase the font size
    ax.tick_params(axis="x", labelsize=40)
    ax.tick_params(axis="y", labelsize=40)

    if dataset is not None:
        ax.set_title(f"Dataset: {dataset}", fontsize=40)


def plot_joint_rankings(axs, rankings, dataset_list):
    # same as plot rankings, but plot multiple rankings on the same plot
    for i, dataset in enumerate(dataset_list):
        plot_rankings(axs[i], rankings[dataset], dataset)


def plot_results(
    x_id,
    y_id,
    ax,
    results,
    dim="kendall_tau",
    args=None,
    offset=False,
    y_lim=None,
    show_legend=True,
):
    # Plot two line plots with error band
    # where x-axis is the sample size options
    # y axis is the result value accessed by results[sample_size]

    # get the mean and std of the results
    colors = {
        "color-aware-btp": "red",
        "color-blind-btp": "blue",
        "baseline": "green",
    }
    label_map = {
        "color-aware-btp": "Fair-BTP",
        "color-blind-btp": "BTP",
        "baseline": "AR",
    }
    for group_wise_mode in [
        "color-blind-btp",
        "color-aware-btp",
        "baseline",
    ]:
        if group_wise_mode not in results:
            continue

        mean_results = np.array(
            [np.mean(val[dim]) for val in results[group_wise_mode]["result"].values()]
        )
        std_results = np.array(
            [np.std(val[dim]) for val in results[group_wise_mode]["result"].values()]
        )
        sample_sizes = np.array(
            [
                np.mean(
                    results[group_wise_mode]["result"][num_of_rounds]["sample_size"]
                )
                for num_of_rounds in list(results[group_wise_mode]["result"].keys())
            ]
        )
        # sort the results by sample size
        sorted_indices = np.argsort(sample_sizes)
        sample_sizes = sample_sizes[sorted_indices]
        mean_results = mean_results[sorted_indices]
        std_results = std_results[sorted_indices]

        # plot the mean and 1 stddev
        ax.plot(
            sample_sizes,
            mean_results,
            label=label_map[group_wise_mode],
            color=colors[group_wise_mode],
            marker="o",
        )
        ax.fill_between(
            sample_sizes,
            mean_results - std_results,
            mean_results + std_results,
            color=colors[group_wise_mode],
            alpha=0.2,
        )

        metric_name = (
            "kendall_tau"
            if dim == "kendall_tau"
            else f"p={args.metrics.lpq_norm.p_norm}, q={args.metrics.lpq_norm.q_norm}-norm"
        )
        ax.set_xscale("log")

        ax.set_title(
            f"{args.dataset} | {metric_name} | {args.num_of_items} items",
            fontsize=20,
        )

        if y_lim is not None:
            ax.set_ylim([-0.01, y_lim])

        # ax.set_ylim([0.0, 0.07])
        ax.tick_params(axis="y", labelsize=20)
        ax.tick_params(axis="x", labelsize=20)
        if show_legend:
            ax.legend(fontsize=20, loc="upper right")


def log_rankings(original_ranking, predicted_ranking, args):
    # print original ranking of set_of_items by sorting them in descending order of theta
    args.logger.info("\n\nOriginal ranking:")
    for item in original_ranking.items:
        args.logger.info("Item %s: %f", item.identifier, item.theta)

    # log the sorted item_ids along with their p values
    args.logger.info("\n\nPredicted ranking:")
    for item_id in predicted_ranking:
        args.logger.info("Item %s", item_id)


def log_prot_attrs(prot_attrs, logger, args):
    # log a table of protected attributes representation percentage
    # per prefix range in prot_attrs

    logger.info("Prefix range Percentage")
    representation = {}
    for prefix_length in range(1, args.num_of_items + 1):
        representation[prefix_length] = {}
        for group in range(args.num_of_groups):
            representation[prefix_length][group] = (
                Counter(prot_attrs[:prefix_length])[group]
                / len(prot_attrs[:prefix_length]),
            )

    # log the representation table
    for prefix_length in range(1, args.num_of_items + 1):
        logger.info(
            "%d %s",
            prefix_length,
            " ".join(
                [
                    f"{group}: {representation[prefix_length][group]}"
                    for group in range(args.num_of_groups)
                ]
            ),
        )


def create_random_prot_attrs(args):
    # create protected attributes randomly for each item in num_of_items
    _set_seed(args.seed)

    prot_attrs = []
    for group_id in range(args.num_of_groups):
        prot_attrs.extend(
            [f"g-{group_id}"]
            * math.ceil(
                args.dataset_config[args.dataset].pvals[group_id] * args.num_of_items
            )
        )
    prot_attrs = prot_attrs[: args.num_of_items]
    # shuffle the prot_attrs
    random.shuffle(prot_attrs)
    return prot_attrs
