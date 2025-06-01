from numpy import *


from active_fair_ranking.algorithms.pairwise import pairwise
from active_fair_ranking.algorithms.ranking_algorithms import ARalg


def test_algorithms():

    n = 25
    kset = [val for val in range(1, n + 1)]

    delta = 0.1
    pmodel = pairwise(n)

    prev_theta = None
    thetas = []
    for i in range(n):
        thetas.append(1 if i == 0 else prev_theta * 0.875)
        prev_theta = thetas[-1]

    pmodel.generate_deterministic_BTL_custom(thetas)

    print("model complexity: ", pmodel.top1H())

    ar = ARalg(pmodel, kset)
    trackdata = ar.rank(delta, track=1000)
    print()
    print(trackdata)
    print("AR, # Comparisons:", ar.pairwise.ctr)
    print("..succeeded" if ar.evaluate_perfect_recovery() else "..failed")


test_algorithms()
