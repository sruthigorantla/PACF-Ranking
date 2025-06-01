import logging
from typing import List

import numpy as np
import omegaconf
import pandas as pd

from active_fair_ranking.algorithms.utils import softmax


class Item:
    # Class of item where the item has an identifier and a group ID.
    # The group ID is a list of IDs starting from 1 to num_of_groups
    def __init__(
        self, identifier: str, group_id: int, theta: float, theta_bias: float = 1.0
    ) -> None:
        self.identifier = identifier
        self.group_id = group_id
        self.theta = self.score = theta
        self.theta_bias = theta_bias

    # define printing of item
    def __repr__(self) -> str:
        return f"Item ID {self.identifier}: group ID = {self.group_id}, theta = {self.theta}, theta_bias = {self.theta_bias}"


class SetofItems:
    # Class of stores a list of items

    def __init__(self, items: List[Item]) -> None:
        self.items = items

    def __len__(self) -> int:
        # function that returns the number of items
        return len(self.items)

    def __repr__(self) -> str:
        return self.print_items

    # property that prints the items
    @property
    def print_items(self) -> str:
        # function that returns a string of all items with their components to logger
        return "\n".join([item.__repr__() for item in self.items])

    def get_item_ids(self) -> List[str]:
        # function that returns a list of item ids
        return [item.identifier for item in self.items]

    def get_group_ids(self) -> List[int]:
        # function that returns the unique group ids
        return sorted(list(set([item.group_id for item in self.items])))

    def get_items_with_group_id(self, group_id) -> "SetofItems":
        # function that returns a subset of items with the given group ID
        return SetofItems([item for item in self.items if item.group_id == group_id])

    def get_item_ids_with_group_id(self, group_id) -> List[str]:
        # function that returns a subset of items with the given group ID
        return [item.identifier for item in self.items if item.group_id == group_id]

    def get_items_by_ids(self, ids, do_sort=True) -> "SetofItems":
        # function that returns a subset of items with the item ids

        # if any id in ids is not in the list of item ids, raise an error
        if any(id not in self.get_item_ids() for id in ids):
            raise ValueError(
                "One or more of the given ids is not in the list of item ids"
            )

        if do_sort == True:
            return SetofItems(
                [item for item in self.items if item.identifier in sorted(list(ids))]
            )
        else:
            return SetofItems(
                [item for item in self.items if item.identifier in list(ids)]
            )

    def get_item_by_id(self, id) -> Item:
        # function that returns the item with the given id

        # if id is not in the list of item ids, raise an error
        if id not in self.get_item_ids():
            raise ValueError("The given id is not in the list of item ids")

        return self.get_items_by_ids([id]).items[0]

    def get_item_index_by_id(self, id) -> int:
        # function that returns the index of the item with the given id

        # if id is not in the list of item ids, raise an error
        if id not in self.get_item_ids():
            raise ValueError("The given id is not in the list of item ids")

        return self.get_item_ids().index(id)

    def get_thetas(self) -> List[float]:
        # function that returns a list of thetas of the items
        return [item.theta for item in self.items]

    def get_probabilities(self) -> List[float]:
        # function that returns a list of probabilities of the items
        # using the softmax of thetas

        # assert that get_probability is called only for two items
        assert len(self.items) == 2, "get_probability can be called only for two items"
        return softmax([item.theta * item.theta_bias for item in self.items])

    @staticmethod
    def create_set_of_items(
        thetas, ids=None, prot_attrs=None, args=None, theta_biases=None
    ):
        # assert that thetas are in decreasing order
        assert all(
            thetas[i] >= thetas[i + 1] for i in range(len(thetas) - 1)
        ), "Thetas are not in decreasing order"

        if ids is None:
            # if ids is None, create a list of ids
            ids = [f"item-{str(i+1)}" for i in range(len(thetas))]
        assert prot_attrs is not None, "prot_attrs cannot be None"

        # Create the items, include ids and prot_attrs if not None
        items = []
        for theta, id, group_id, theta_bias in zip(
            thetas, ids, prot_attrs, theta_biases
        ):
            # item with random identifier string and random group ID from 1 to num_of_groups
            # theta is a random float between 0 and 1
            items.append(
                Item(
                    identifier=id, group_id=group_id, theta=theta, theta_bias=theta_bias
                )
            )

        # Store the items in a SetofItems object
        return SetofItems(items)


def create_ranking(
    num_of_items: int,
    num_of_groups: int,
    dataset: str = None,
    prot_attrs: List[str] = None,
    args: omegaconf.dictconfig.DictConfig = None,
    logger: logging.Logger = None,
):
    """
    Function that creates a ranking of items with thetas and group IDs
    :param num_of_items: number of items in the ranking
    :param num_of_groups: number of groups in the ranking
    :param dataset: dataset name
    :param prot_attrs: list of group IDs for each item
    :param args: arguments
    :param logger: logger
    :return: SetofItems object
    """
    if prot_attrs is not None:
        assert (
            len(prot_attrs) == num_of_items
        ), f"Length of prot_attrs ({len(prot_attrs)}) is not equal to num_of_items ({num_of_items})"
        logger.info(
            f"Creating dataset with {num_of_items} items with given group IDs: {prot_attrs}"
        )

    # create a list of theta of size num_of_items
    thetas = []
    if dataset == "geo":
        # for geometric ranking, theta is 1 for the first item and theta * discount for the next item
        prev_theta = None
        for i in range(num_of_items):
            thetas.append(
                1 if i == 0 else prev_theta * args.dataset_config.geo.discount
            )
            prev_theta = thetas[-1]
        logger.info(f"Creating dataset with {num_of_items} items with geometric thetas")
        ids = [f"item-{val+1}" for val in range(num_of_items)]

        theta_biases = [
            args.dataset_config.geo.theta_bias[int(prot_attrs[i].split("-")[-1])]
            for i in range(num_of_items)
        ]

    elif dataset == "har":
        # for harmonic ranking, theta is 1 / (i + 1) for the i-th item
        for i in range(num_of_items):
            thetas.append(1 / (i + 1))
        logger.info(f"Creating dataset with {num_of_items} items with harmonic thetas")
        ids = [f"item-{val+1}" for val in range(num_of_items)]
        theta_biases = [
            args.dataset_config.har.theta_bias[int(prot_attrs[i].split("-")[-1])]
            for i in range(num_of_items)
        ]

    elif dataset == "arith":
        # for arithmetic ranking, theta is 1 for the first item and theta - discount for the next item
        thetas.append(1)
        for i in range(num_of_items - 1):
            thetas.append(thetas[-1] - args.dataset_config.arith.discount)
        logger.info(
            f"Creating dataset with {num_of_items} items with arithmetic thetas"
        )
        ids = [f"item-{val+1}" for val in range(num_of_items)]
        theta_biases = [
            args.dataset_config.arith.theta_bias[int(prot_attrs[i].split("-")[-1])]
            for i in range(num_of_items)
        ]
    elif dataset == "steps":
        # for steps ranking, theta is 1 for the first window_size items and theta - discount for the next window_size items

        thetas.extend([1] * args.dataset_config.steps.window_size)
        # reduce thetas by discount factor for every window size
        for i in range(
            0,
            num_of_items - args.dataset_config.steps.window_size,
            args.dataset_config.steps.window_size,
        ):
            thetas.extend(
                [thetas[-1] - args.dataset_config.steps.discount]
                * args.dataset_config.steps.window_size
            )
        thetas = thetas[:num_of_items]
        logger.info(f"Creating dataset with {num_of_items} items with steps thetas")
        ids = [f"item-{val+1}" for val in range(num_of_items)]
        theta_biases = [
            args.dataset_config.steps.theta_bias[int(prot_attrs[i].split("-")[-1])]
            for i in range(num_of_items)
        ]

    elif dataset == "compas":
        protected_group = args.dataset_config.compas.protected_group
        assert protected_group in [
            "race",
            "sex",
        ], "Invalid protected group"

        judgment = "Recidivism_rawscore"
        header = [
            "priors_count",
            "Violence_rawscore",
            "Recidivism_rawscore",
            protected_group,
        ]

        data = pd.read_csv(
            args.dataset_config[args.dataset]["path"][protected_group],
            names=header,
            header=0,
        )
        p = []
        num_of_groups = len(pd.unique(data[protected_group]))

        for i in range(num_of_groups):
            proportion = float(
                len(data.query(protected_group + "==" + str(i))) / len(data)
            )
            p.append(proportion)

        data.rename(
            columns={protected_group: "prot_attr"},
            inplace=True,
        )

        data["query_id"] = np.ones(len(data), dtype=int)
        new_header = np.append(header, "query_id")

        data[judgment] = data[judgment].apply(lambda val: 1 - val)
        data = (data.sort_values(by=[judgment], ascending=False)).reset_index(drop=True)
        data["doc_id"] = data.index + 1
        new_header = np.append(new_header, "doc_id")

        # prepare the data
        ids = data["doc_id"].tolist()
        prot_attrs = [f"g-{g_id}" for g_id in data["prot_attr"].tolist()]
        thetas = data[judgment].tolist()

        # Take the first num_of_items rows
        ids = [f"item-{int(id)}" for id in ids[:num_of_items]]
        prot_attrs = prot_attrs[:num_of_items]
        thetas = thetas[:num_of_items]

        # assign theta biases based on group-id
        theta_biases = [
            args.dataset_config.compas.theta_bias[int(prot_attrs[i].split("-")[-1])]
            for i in range(num_of_items)
        ]

    elif dataset == "german":
        protected_group = args.dataset_config.german.protected_group
        assert protected_group in [
            "age25",
            "age35",
            "age",
            "age_gender",
        ], "Invalid protected group"

        judgment = "score"
        header = ["DurationMonth", "CreditAmount", "score", protected_group]
        origFile = args.dataset_config[args.dataset]["path"][protected_group]
        data = pd.read_csv(origFile, names=header, header=0)
        p = []
        num_of_groups = len(pd.unique(data[protected_group]))

        for i in range(num_of_groups):
            proportion = float(
                len(data.query(protected_group + "==" + str(i))) / len(data)
            )
            p.append(proportion)

        data.rename(
            columns={protected_group: "prot_attr"},
            inplace=True,
        )

        data["query_id"] = np.ones(len(data), dtype=int)
        new_header = np.append(header, "query_id")

        data[judgment] = data[judgment].apply(lambda val: val)
        data = (data.sort_values(by=[judgment], ascending=False)).reset_index(drop=True)
        data["doc_id"] = data.index + 1
        new_header = np.append(new_header, "doc_id")

        # prepare the data
        ids = data["doc_id"].tolist()
        prot_attrs = [f"g-{g_id}" for g_id in data["prot_attr"].tolist()]
        thetas = data[judgment].tolist()

        # Take the first num_of_items rows
        ids = [f"item-{int(id)}" for id in ids[:num_of_items]]
        prot_attrs = prot_attrs[:num_of_items]
        thetas = thetas[:num_of_items]

        if num_of_groups == 2:
            # assign theta biases based on group-id
            theta_biases = [
                args.dataset_config.german.theta_bias[int(prot_attrs[i].split("-")[-1])]
                for i in range(num_of_items)
            ]
        else:
            theta_biases = [1.0] * num_of_items
    else:
        raise ValueError("Invalid data mode")

    set_of_items = SetofItems.create_set_of_items(
        thetas=thetas,
        ids=ids,
        prot_attrs=prot_attrs,
        args=args,
        theta_biases=theta_biases,
    )

    # assert that there are atleast 2 items per group
    assert all(
        len(set_of_items.get_items_with_group_id(group_id)) >= 2
        for group_id in set_of_items.get_group_ids()
    ), "There are less than 2 items per group"

    logger.info("Items:")
    logger.info(set_of_items.print_items)

    return set_of_items
