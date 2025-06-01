import numpy as np


delta = 0.1
epsilon = 0.01
num_of_items = 1000
num_of_groups = 2
import matplotlib.pyplot as plt

prange = np.arange(1, 31)
qrange = np.arange(1, 31)
# frange = [0.05, 0.1, 0.2, 0.25]
frange = [0.5, 0.9, 0.995, 0.9999]

fig, axs = plt.subplots(
    2,
    2,
    figsize=(10, 10),
)
fig.tight_layout(pad=5.0)

for i in range(len(frange)):
    # n1 = np.int32(num_of_items * (frange[i]))
    # n2 = n3 = np.int32(num_of_items * ((1 - 2 * (frange[i])) / 3))
    # n4 = num_of_items - n1 - n2 - n3
    # num_of_items_per_group = np.array([n1, n2, n3, n4])
    # assert np.sum(num_of_items_per_group) == num_of_items
    n1 = np.int32(num_of_items * (frange[i]))
    n2 = num_of_items - n1
    num_of_items_per_group = np.array([n1, n2])
    val = []
    pvals = []
    qvals = []
    for p in prange:
        for q in qrange:
            phis = num_of_items_per_group / 3
            eps = np.zeros(num_of_groups)
            for h in range(num_of_groups):
                eps[h] = (
                    (epsilon)
                    * (num_of_items_per_group[h] ** (1 / q - 1 / p))
                    * ((phis[h] * num_of_groups) ** (-1 / q))
                )

            indices = np.argsort(eps)
            eps = eps[indices]
            n_h = num_of_items_per_group[indices]
            comp1 = 0
            comp2 = 0
            for h in range(num_of_groups):
                new_eps_h = eps[h] / (n_h[h] ** (1 / p))
                comp1 += (n_h[h] / new_eps_h) ** (2) * np.log2(
                    (2 * num_of_groups) / delta
                )
            for h in range(num_of_groups - 1):
                for j in range(1, num_of_groups):
                    comp1 += n_h[j] * (
                        (2 / eps[h]) ** (2) * np.log2((2 * num_of_groups) / delta)
                    )
            comp2 = (
                (num_of_items ** (1 + max(1 / p, 1 / q))) / epsilon
            ) ** 2 * np.log2((num_of_items) / delta)

            if comp1 - comp2 > 0:
                val.append("grey")
            else:
                val.append("lightblue")

            # val.append(comp1 - comp2)
            pvals.append(p)
            qvals.append(q)
    row_id = i // 2
    col_id = i % 2
    val = np.array(val)
    pvals = np.array(pvals)
    qvals = np.array(qvals)
    axs[row_id, col_id].scatter(pvals, qvals, c=val)
    axs[row_id, col_id].set_xlabel("p")
    axs[row_id, col_id].set_ylabel("q")
    axs[row_id, col_id].set_title(f"f={frange[i]}")
fig.set_tight_layout(True)
fig.suptitle(
    f"Sample complexity for delta={delta}, epsilon={epsilon}, n={num_of_items}\n group-aware wins (blue) | group-blind wins (grey) "
)
plt.savefig(f"sample_complexity.png")
