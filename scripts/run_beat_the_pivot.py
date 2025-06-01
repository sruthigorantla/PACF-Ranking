import copy
import datetime
import logging
import math
import os
import pickle
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy

import omegaconf
from tqdm import tqdm

from active_fair_ranking.algorithms.beat_the_pivot import beat_the_pivot
from active_fair_ranking.algorithms.data import SetofItems, create_ranking
from active_fair_ranking.algorithms.merge_rankings import merge_rankings
from active_fair_ranking.algorithms.utils import (
    _set_seed,
    create_random_prot_attrs,
    setup_logging,
)


def read_config():
    config_file = "config.yaml"
    config = omegaconf.OmegaConf.load(config_file)

    # setup the logger
    logger = setup_logging(config)

    # write post init checks here

    # assert epsilon and delta are > 0
    for dataset in config.datasets:
        assert config.dataset_config[dataset].epsilon > 0, "epsilon must be > 0"
        assert config.dataset_config[dataset].delta > 0, "delta must be > 0"
    assert (
        config.beat_the_pivot.subset_size <= config.num_of_items
    ), f"subset size {config.subset_size} must be <= number of items {config.num_of_items}"

    return config, logger


def single_run_beat_the_pivot(
    set_of_items: SetofItems,
    args: omegaconf.omegaconf.DictConfig,
    logger: logging.Logger = None,
):
    # args.final_sample_size will keep a count of the exact
    # number of times oracle will be called.
    args.final_sample_size = 0

    if args.color_aware:
        group_ids = set_of_items.get_group_ids()

        # get group-wise set_of_items
        args.num_of_groups = len(group_ids)

        orig_num_of_rounds = args.num_of_rounds
        set_of_items_group_wise = {}
        predicted_ranking_group_wise = {}

        # num_of_rounds_per_group = defaultdict(lambda: 0)
        # multiply num_of_rounds by num of items in the group
        for g_id in group_ids:
            # get the set of items for this group
            set_of_items_group_wise[g_id] = set_of_items.get_items_with_group_id(g_id)
            # group_proportion = len(set_of_items_group_wise[g_id]) / len(set_of_items)
            # num_of_rounds_per_group[g_id] = math.ceil(group_proportion * args.num_of_rounds)

        final_sample_size_across_groups = 0
        oracle_complexity_across_groups = 0
        for g_id in group_ids:
            # call beat the pivot algorithm
            # update num_of_items in args
            # args.num_of_rounds = num_of_rounds_per_group[g_id]
            args.num_of_items = len(set_of_items_group_wise[g_id])

            (
                predicted_ranking,
                final_sample_size,
                oracle_complexity,
            ) = beat_the_pivot(
                set_of_items=set_of_items_group_wise[g_id],
                epsilon=args.dataset_config[args.dataset].epsilon,
                delta=args.dataset_config[args.dataset].delta,
                args=args,
                logger=logger,
            )
            assert (final_sample_size / oracle_complexity) == args.num_of_rounds
            assert oracle_complexity == int(
                2
                * math.ceil(len(predicted_ranking) - 1)
                / (args.beat_the_pivot.subset_size - 1)
            )
            args.final_sample_size = 0  # reset this for each group's beat the pivot
            final_sample_size_across_groups += final_sample_size
            oracle_complexity_across_groups += oracle_complexity

            predicted_ranking_group_wise[g_id] = predicted_ranking

            logger.info(
                f"\nOriginal Ranking for Group {g_id} is {set_of_items.get_item_ids_with_group_id(g_id)}"
            )
            logger.info(
                f"Predicted Ranking for Group {g_id} is {predicted_ranking}\n\n"
            )

        args.final_sample_size = final_sample_size_across_groups
        args.num_of_rounds = orig_num_of_rounds

        # call merging algorithm to merge the group-wise rankings
        predicted_ranking, merge_oracle_complexity = merge_rankings(
            set_of_items,
            set_of_items_group_wise,
            predicted_ranking_group_wise,
            group_ids=group_ids,
            args=args,
            merge_mode=args.group_wise.merge_mode,
            logger=logger,
        )

        logger.info(
            f"\nMerge sub-routine called the oracle: {merge_oracle_complexity} times"
        )
        assert (
            args.final_sample_size - final_sample_size_across_groups
        ) == merge_oracle_complexity * args.num_of_rounds
        final_sample_size = args.final_sample_size

    else:
        (
            predicted_ranking,
            final_sample_size,
            oracle_complexity,
        ) = beat_the_pivot(
            set_of_items=set_of_items,
            epsilon=args.dataset_config[args.dataset].epsilon,
            delta=args.dataset_config[args.dataset].delta,
            args=args,
            logger=logger,
        )

        # oracle_complexity stores the number of time the play() function was called
        assert (final_sample_size / oracle_complexity) == args.num_of_rounds
        assert oracle_complexity == int(
            2
            * math.ceil(len(predicted_ranking) - 1)
            / (args.beat_the_pivot.subset_size - 1)
        )

    logger.info(f"Predicted Ranking: {predicted_ranking}")
    logger.info(f"Final sample size: {final_sample_size}")

    # updating the num_of_items in case it was changed inside the beat_the_pivot function
    args.num_of_items = len(predicted_ranking)
    return predicted_ranking, final_sample_size


def single_run_wrapper(args):
    # Wrapper function to call single_run_beat_the_pivot
    _set_seed(args["args"].seed)
    return single_run_beat_the_pivot(args["set_of_items"], args["args"], args["logger"])


def multi_run_beat_the_pivot(
    set_of_items: SetofItems,
    args: omegaconf.omegaconf.DictConfig,
    logger: logging.Logger = None,
):
    results = defaultdict()
    for sample_size_seed in tqdm(
        args.sample_size_seeds, total=len(args.sample_size_seeds)
    ):
        results[sample_size_seed] = defaultdict(list)
        args.num_of_rounds = sample_size_seed
        experiment_args = []

        for seed in range(1, args.num_of_exps + 1):
            exp_args = deepcopy(args)
            exp_args.seed = seed
            experiment_args.append(
                {"set_of_items": set_of_items, "args": exp_args, "logger": logger}
            )

        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            for predicted_ranking, final_sample_size in executor.map(
                single_run_wrapper, experiment_args
            ):
                results[sample_size_seed]["predicted_ranking"].append(predicted_ranking)
                results[sample_size_seed]["sample_size"].append(final_sample_size)

    return results


def main(
    args: omegaconf.dictconfig.DictConfig,
    logger: logging.Logger = None,
    experiment_folder: str = None,
    dataset_name: str = None,
):
    args.num_of_groups = getattr(
        args.dataset_config[args.dataset], "num_of_groups", None
    )
    # Create the ranking
    set_of_items = create_ranking(
        num_of_items=args.num_of_items,
        num_of_groups=args.num_of_groups,
        dataset=args.dataset,
        prot_attrs=(
            None
            if args.dataset in ["compas", "german"]
            else create_random_prot_attrs(args)
        ),
        args=args,
        logger=logger,
    )

    # print group proportions in top 25, 50 and 100 positions
    # for k in [25, 50, 100]:
    #     print(f"\nTop {k} items:")
    #     set_of_items_top_k = set_of_items.get_items_by_ids(
    #         [f"item-{str(i+1)}" for i in range(k)]
    #     )
    #     group_ids = set_of_items_top_k.get_group_ids()
    #     for gid in group_ids:
    #         items_with_gid = set_of_items_top_k.get_item_ids_with_group_id(gid)
    #         print(
    #             f"Group {gid} has {len(items_with_gid)}: {len(items_with_gid)/k} proportion"
    #         )
    # return None

    if experiment_folder is not None:
        # save the set_of_items in the experiment folder
        with open(f"{experiment_folder}/set_of_items_{dataset_name}.pkl", "wb") as f:
            pickle.dump(set_of_items, f)

    results = {}
    args.sample_size_seeds = [2**val for val in range(args["powers_of_two"])]

    for args.color_aware in [False, True]:
        results["color-aware" if args.color_aware else "color-blind"] = (
            multi_run_beat_the_pivot(set_of_items, args, logger)
        )

    results["set_of_items"] = set_of_items

    return results


if __name__ == "__main__":
    """
    sample command: python run_beat_the_pivot.py
    """

    # Load omegaconf config from a YAML file
    args, logger = read_config()
    logger.info("Running multirun experiment")

    # create a folder inside args.checkpoint.path with the latest date
    today_date_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    experiment_folder = f"{args.checkpoint.path}/{today_date_time}"
    os.makedirs(experiment_folder, exist_ok=True)

    # save the config file in the experiment folder
    omegaconf.OmegaConf.save(args, f"{experiment_folder}/config.yaml")

    results = {}
    for args.dataset in args.datasets:
        if hasattr(args.dataset_config[args.dataset], "protected_group"):
            # assert that protected_group is a ListConf of OmegaConf
            assert isinstance(
                args.dataset_config[args.dataset].protected_group,
                omegaconf.listconfig.ListConfig,
            ), "protected_group must be a list"

            protected_groups = list(args.dataset_config[args.dataset].protected_group)

            # run the experiment for each protected_group of this dataset
            for prot_group in protected_groups:
                # set the current protected group for the dataset in args
                args.dataset_config[args.dataset].protected_group = prot_group
                print(f"\n\nRunning experiment for {args.dataset} with {prot_group}")
                # run the experiment
                results[f"{args.dataset}_{prot_group}"] = main(
                    args,
                    logger,
                    experiment_folder,
                    dataset_name=f"{args.dataset}_{prot_group}",
                )
        else:
            results[args.dataset] = main(
                args,
                logger,
                experiment_folder,
                dataset_name=f"{args.dataset}",
            )

    # pickle results and args
    with open(
        f"{experiment_folder}/beat_the_pivot_exp_{args.num_of_items}.pkl", "wb"
    ) as f:
        pickle.dump([results, args, logger], f)
