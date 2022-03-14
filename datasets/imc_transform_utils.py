from __future__ import annotations

import os

from typing import Union
from utils import memory, get_execute_function, file_path
import numpy as np
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import pickle
from analyses.imputation_score import Prediction


from datasets.imc import get_smu_file, get_merged_areas_per_split

s = get_smu_file("train", 0, read_only=True)
quantiles_for_normalization = s["imc"]["transformed_mean"].uns["scaling_factors"][...]
print(quantiles_for_normalization)
s.backing.close()

##
from enum import IntEnum
from typing import Callable

Space = IntEnum("Space", "raw_sum raw_mean asinh_sum asinh_mean scaled_mean", start=0)

from_to = [[None for _ in range(len(Space))] for _ in range(len(Space))]


def f_set(space0: Space, space1: Space, transformation: Callable):
    from_to[space0.value][space1.value] = transformation


def f_get(space0: Space, space1: Space):
    return from_to[space0.value][space1.value]


f_set(Space.raw_sum, Space.asinh_sum, lambda x, a: np.arcsinh(x))
f_set(Space.raw_mean, Space.asinh_mean, lambda x, a: np.arcsinh(x))
f_set(Space.asinh_mean, Space.scaled_mean, lambda x, a: x / quantiles_for_normalization)
f_set(Space.raw_sum, Space.raw_mean, lambda x, a: x / a)

f_set(Space.asinh_sum, Space.raw_sum, lambda x, a: np.sinh(x))
f_set(Space.asinh_mean, Space.raw_mean, lambda x, a: np.sinh(x))
f_set(Space.scaled_mean, Space.asinh_mean, lambda x, a: x * quantiles_for_normalization)
f_set(Space.raw_mean, Space.raw_sum, lambda x, a: x * a)

from typing import List


def find_path(from_node: int, to_node: int):
    def dfs(from_node: int, to_node: int, visited: List[int], path: List[int]):
        if from_node == to_node:
            return True
        for j, to in enumerate(from_to[from_node]):
            if to is not None:
                if not visited[j]:
                    visited[j] = True
                    b = dfs(from_node=j, to_node=to_node, visited=visited, path=path)
                    if b:
                        path.append(j)
                        return True
        return False

    visited = [False for _ in range(len(Space))]
    path = []
    dfs(from_node=from_node, to_node=to_node, visited=visited, path=path)
    path.append(from_node)
    return list(reversed(path))


test_path = find_path(from_node=Space.scaled_mean.value, to_node=Space.asinh_sum.value)
assert test_path == [4, 3, 1, 0, 2], test_path


def transform(
    x: np.ndarray, from_space: Space, to_space: Space, split: str
) -> np.ndarray:
    # old code
    # a = areas[split]
    # end old code
    # new code
    a = get_merged_areas_per_split()[split]
    a = np.tile(a.reshape(-1, 1), (1, x.shape[1]))
    # end new code
    assert len(a) == len(x)
    path = find_path(from_node=from_space.value, to_node=to_space.value)
    for i in range(len(path) - 1):
        start, end = path[i], path[i + 1]
        s, e = Space(start), Space(end)
        f = f_get(s, e)
        print(f"applying transformation from {s.name} to {e.name}")
        x = f(x, a)
    return x


import math
from scipy.stats import t as t_dist


class IMCPrediction(Prediction):
    def __init__(
        self,
        original: np.ndarray,
        corrupted_entries: np.ndarray,
        predictions_from_perturbed: np.ndarray,
        space: Union[Space, int],
        name: str,
        split: str,
    ):
        if type(space) == int:
            space = Space(space)
        self.space = space
        super().__init__(
            original=original,
            corrupted_entries=corrupted_entries,
            predictions_from_perturbed=predictions_from_perturbed,
            name=f"{name}, {self.space.name}",
        )
        self.split = split

    def transform_to(self, space: Space) -> IMCPrediction:
        original_transformed = transform(
            x=self.original,
            from_space=self.space,
            to_space=space,
            split=self.split,
        )
        predictions_from_perturbed_transformed = transform(
            x=self.predictions_from_perturbed,
            from_space=self.space,
            to_space=space,
            split=self.split,
        )
        p = IMCPrediction(
            original=original_transformed,
            corrupted_entries=self.corrupted_entries,
            predictions_from_perturbed=predictions_from_perturbed_transformed,
            space=space,
            name=self.name,
            split=self.split,
        )
        return p

    def plot_summary(self):
        if self.space != Space.scaled_mean:
            p = self.transform_to(Space.scaled_mean)
            p.plot_summary()
        else:
            super().plot_summary()


##
def compare_predictions(p0: IMCPrediction, p1: IMCPrediction, target_space: Space):
    q0 = p0.transform_to(target_space)
    q1 = p1.transform_to(target_space)
    m0, s0 = q0.compute_and_plot_scores()
    m1, s1 = q1.compute_and_plot_scores()
    Prediction.welch_t_test(s0, s1)


if __name__ == "__main__":
    from splits import validation

    x = np.random.rand(len(validation), 5)
    import torch
    from torch.distributions import Bernoulli

    dist = Bernoulli(probs=0.1)
    shape = x.shape
    state = torch.get_rng_state()
    torch.manual_seed(0)
    corrupted_entries = dist.sample(shape).bool().numpy()
    torch.set_rng_state(state)

    x_perturbed = x.copy()
    eps = np.random.rand(np.sum(corrupted_entries)) / 10
    x_perturbed[corrupted_entries] = x[corrupted_entries] + eps
    p = IMCPrediction(
        original=x,
        corrupted_entries=corrupted_entries,
        predictions_from_perturbed=x_perturbed,
        name="test",
        split="validation",
        space=Space.scaled_mean,
    )
    p.plot_reconstruction()
    p.plot_scores()
    p.plot_summary()
    print("ooo")
