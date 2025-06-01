import argparse
from collections import defaultdict
import copy
import os
import pickle
from concurrent.futures import ProcessPoolExecutor

import matplotlib.pyplot as plt
from tqdm import tqdm

from active_fair_ranking.algorithms.metrics import kendal_tau_ranking_groupwise
from active_fair_ranking.algorithms.utils import (
    plot_groupwise_lineplot,
    plot_joint_rankings,
    plot_rankings,
    plot_results,
)


def compute_kendal_tau(predicted_ranking, set_of_items, args, logger):
    return kendal_tau_ranking_groupwise(
        set_of_items,
        predicted_ranking,
        args,
        p_norm=args.metrics.lpq_norm.p_norm,
        q_norm=args.metrics.lpq_norm.q_norm,
        logger=logger,
    )


def calc_metrics(results, args, logger):
    for args.dataset in results.keys():
        set_of_items = results[args.dataset]["set_of_items"]
        group_ids = set_of_items.get_group_ids()

        for groupwise_mode in ["color-aware", "color-blind"]:
            if groupwise_mode not in results[args.dataset]:
                continue
            for num_of_rounds in tqdm(
                results[args.dataset][groupwise_mode].keys(),
                total=len(results[args.dataset][groupwise_mode].keys()),
            ):
                # call kendal_tau_ranking_groupwise via multiprocessing
                with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
                    # Create a list of tasks
                    tasks = [
                        executor.submit(
                            compute_kendal_tau,
                            predicted_ranking,
                            set_of_items,
                            args,
                            logger,
                        )
                        for predicted_ranking in results[args.dataset][groupwise_mode][
                            num_of_rounds
                        ]["predicted_ranking"]
                    ]

                    for task in tasks:
                        (
                            kendal_tau_result_groupwise,
                            kendal_tau_result,
                            lpq_norm,
                            lpq_norm_groupwise,
                        ) = task.result()

                        results[args.dataset][groupwise_mode][num_of_rounds][
                            "kendall_tau"
                        ].append(kendal_tau_result)
                        results[args.dataset][groupwise_mode][num_of_rounds][
                            "lpq_norm"
                        ].append(lpq_norm)
                        for g_id in group_ids:
                            results[args.dataset][groupwise_mode][num_of_rounds][
                                f"kendall_tau_{g_id}"
                            ].append(kendal_tau_result_groupwise[g_id])
                            results[args.dataset][groupwise_mode][num_of_rounds][
                                f"lpq_norm_{g_id}"
                            ].append(lpq_norm_groupwise[g_id])

    return results


def plot(results, args, config):
    num_of_datasets = len(results)
    # Initialize figures for plotting
    fig, axs = plt.subplots(
        nrows=len(args.metrics.dimensions),
        ncols=2,
        # figsize=(50, 10),
        figsize=(30, 10),
        squeeze=False,
    )
    fig_groupwise, axs_groupwise = plt.subplots(
        nrows=1,
        ncols=2,  # we add one column to plot the thetas
        figsize=(20, 5),
        squeeze=False,
    )
    fig.tight_layout(pad=7.0)
    fig_groupwise.tight_layout(pad=6.0)

    dataset_id = 0
    for args.dataset in results.keys():
        if "35" in args.dataset or "25" in args.dataset or "sex" in args.dataset:
            continue

        for col_id, dim in enumerate(args.metrics.dimensions):
            plot_results(
                dataset_id,
                col_id,
                axs[col_id, dataset_id],
                copy.deepcopy(results[args.dataset]),
                dim=dim,
                args=args,
                y_lim=None,
            )

        plot_groupwise_lineplot(
            copy.deepcopy(results[args.dataset]),
            dim="lpq_norm",
            args=args,
            axs=axs_groupwise[:, dataset_id],
            # show_legend=True if dataset_id == len(results.keys()) - 1 else False,
            show_legend=True,
        )

        dataset_id += 1

    plt.subplots_adjust(hspace=1)
    fig.savefig(f"{config.plot_dir}/beat_the_pivot_exp_{args.num_of_items}.pdf")
    fig_groupwise.savefig(
        f"{config.plot_dir}/group_wise_lp_norm_{args.num_of_items}.pdf"
    )


def inflate_dataset_list(dataset_list):
    # replace "compas" in inflated_dataset_list with "compas_race" and "compas_sex"
    if "compas" in dataset_list:
        dataset_list.remove("compas")
        dataset_list.append("compas_sex")
        dataset_list.append("compas_race")
    if "german" in dataset_list:
        dataset_list.remove("german")
        dataset_list.append("german_age")
        # dataset_list.append("german_age25")
        # dataset_list.append("german_age35")
    return dataset_list


def process_filtered_results(results, exp_id, all_results, dataset_list, args, logger):

    # filter results based on provided datasets
    filtered_results = {}
    for dataset_name in results.keys():
        # print(dataset_name)
        # if "age25" in dataset_name or "age35" in dataset_name:
        #     continue

        if any([dataset_name.startswith(d) for d in dataset_list]):
            filtered_results[dataset_name] = results[dataset_name]

    filtered_results = calc_metrics(filtered_results, args, logger)

    for dataset, _ in filtered_results.items():

        if "baseline" in exp_id:
            if "color-aware" in filtered_results[dataset]:
                raise Exception("Should not be here")

            all_results[dataset]["baseline"]["result"] = filtered_results[dataset][
                "color-blind"
            ]
            all_results[dataset]["baseline"]["set_of_items"] = copy.deepcopy(
                filtered_results[dataset]["set_of_items"]
            )
        else:
            # this is btp experiment
            all_results[dataset]["color-aware-btp"]["result"] = filtered_results[
                dataset
            ]["color-aware"]
            all_results[dataset]["color-aware-btp"]["set_of_items"] = copy.deepcopy(
                filtered_results[dataset]["set_of_items"]
            )

            all_results[dataset]["color-blind-btp"]["result"] = filtered_results[
                dataset
            ]["color-blind"]
            all_results[dataset]["color-blind-btp"]["set_of_items"] = copy.deepcopy(
                filtered_results[dataset]["set_of_items"]
            )

    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp-ids",
        nargs="+",
        type=str,
        required=True,
        help="List of experiment IDs for which to load results",
    )
    # list of datasets to plot
    parser.add_argument(
        "--datasets",
        nargs="+",
        type=str,
        default=["compas", "german", "geo"],
    )
    # add flag to plot the set of items of the datasets
    parser.add_argument(
        "--plot-dataset",
        action="store_true",
        help="Plot the set of items of the datasets",
    )
    # add flag to run lpq ablations
    parser.add_argument(
        "--ablation-lpq",
        action="store_true",
        help="Run ablation over p, q norms",
    )
    parser.add_argument(
        "--ablation-num-items",
        action="store_true",
        help="Run ablation over num of items in ranking",
    )
    parser.add_argument(
        "--length-of-rankings",
        nargs="+",
        type=int,
        default=[25, 50, 100],
    )
    config = parser.parse_args()
    dataset_list = config.datasets
    inflated_dataset_list = copy.deepcopy(inflate_dataset_list(dataset_list))

    # create a new plotting folder with the joint name of exp_ids
    plot_dir = f"./checkpoints/{'_'.join(config.exp_ids)}"
    config.plot_dir = plot_dir
    os.makedirs(plot_dir, exist_ok=True)

    if getattr(config, "plot_dataset", False):
        # plot the set of items of the datasets

        # create plot subfolder if doesn't exists
        plot_subfolder = f"{plot_dir}/plots"
        os.makedirs(plot_subfolder, exist_ok=True)

        # create a joint row of plots for all the datasets
        joint_fig, joint_axs = plt.subplots(
            nrows=1, ncols=len(inflated_dataset_list), figsize=(50, 15), squeeze=False
        )

        dataset_fig = {}
        for dataset in inflated_dataset_list:
            set_of_items = pickle.load(
                open(
                    f"./checkpoints/{config.exp_ids[0]}/set_of_items_{dataset}.pkl",
                    "rb",
                )
            )
            fig, axs = plt.subplots(nrows=1, ncols=1, figsize=(20, 15), squeeze=False)
            axis = plot_rankings(
                axs[0, 0], copy.deepcopy(set_of_items), dataset=dataset
            )
            dataset_fig[dataset] = copy.deepcopy(set_of_items)
            plt.savefig(f"{plot_dir}/plots/{dataset}.pdf")
            # Print % of groups in the ranking
            group_ids = set_of_items.get_group_ids()
            group_counts = []
            for g_id in group_ids:
                group_counts.append(len(set_of_items.get_items_with_group_id(g_id)))
            group_counts = [count / len(set_of_items) for count in group_counts]
            print(f"Group counts for {dataset}: {group_counts}")

        # save the joint figure
        plot_joint_rankings(joint_axs[0], dataset_fig, inflated_dataset_list)
        joint_fig.savefig(f"{plot_dir}/plots/joint_set_of_items.pdf")

    all_results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    results = {}
    for exp_id in config.exp_ids:
        try:
            results[exp_id], args, logger = pickle.load(
                open(f"./checkpoints/{exp_id}/beat_the_pivot_exp_50.pkl", "rb")
            )
        except FileNotFoundError:
            results[exp_id], args, logger = pickle.load(
                open(f"./checkpoints/{exp_id}/baseline_50.pkl", "rb")
            )

        # Calculate metrics for filtered results
        all_results = process_filtered_results(
            results[exp_id], exp_id, all_results, dataset_list, args, logger
        )

    plot(all_results, args, config)

    # 3. ablation over p, q plots
    if getattr(config, "ablation_lpq", False):

        print("\nRunning ablation over p, q norms")
        ABLATION_DIR = f"{plot_dir}/ablations"
        os.makedirs(ABLATION_DIR, exist_ok=True)
        P_NORM_LIST = [1, 2, 10]
        Q_NORM_LIST = [1, 2, 10]

        # create a new figure for each dataset
        fig, ax = {}, {}
        for dataset in all_results.keys():
            fig[dataset], ax[dataset] = plt.subplots(
                len(P_NORM_LIST), len(Q_NORM_LIST), figsize=(20, 15), squeeze=False
            )
            fig[dataset].tight_layout(pad=5.0, h_pad=5.0, w_pad=5.0)

        # run ablation over p, q norms
        for p_id, p_norm in enumerate(P_NORM_LIST):
            for q_id, q_norm in enumerate(Q_NORM_LIST):
                print(f"Running ablation for p={p_norm}, q={q_norm}")
                args.metrics.lpq_norm.p_norm = p_norm
                args.metrics.lpq_norm.q_norm = q_norm

                # get all results for each exp_id
                all_results = defaultdict(
                    lambda: defaultdict(lambda: defaultdict(list))
                )
                for exp_id in config.exp_ids:
                    all_results = process_filtered_results(
                        results[exp_id], exp_id, all_results, dataset_list, args, logger
                    )

                y_lim = None
                if "compas" in dataset:
                    y_lim = 0.04
                elif "german" in dataset:
                    y_lim = 0.1

                dataset_id = 0
                for dataset in all_results.keys():
                    args.dataset = dataset
                    plot_results(
                        p_id,
                        q_id,
                        ax[dataset][p_id, q_id],
                        copy.deepcopy(all_results[dataset]),
                        dim="lpq_norm",
                        args=args,
                        y_lim=y_lim,
                        show_legend=(p_id == 0 and q_id == len(Q_NORM_LIST) - 1),
                    )

        plt.subplots_adjust(hspace=0.5)

        for dataset in all_results.keys():
            print(
                f"Saving ablation plot for {dataset} at {ABLATION_DIR}/ablation_lpq_{dataset}_{args.num_of_items}.pdf"
            )
            fig[dataset].savefig(
                f"{ABLATION_DIR}/ablation_pq_{dataset}_{args.num_of_items}.pdf"
            )
