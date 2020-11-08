from wildfire import FireGrid
import numpy as np
import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import itertools
import copy

def approximate_utility(grid):
    return np.sum(grid.observation()[1] * grid.observation()[2])

def all_possible_actions(n):
    l = [False, True]
    possible_true_false_sequences = list(itertools.product(l, repeat=n**2))
    A = [np.asarray(l).reshape(n, n) for l in possible_true_false_sequences]
    return A

def sparse_sampling(A, grid, d, m=5, gamma=.95):
    if d <= 0:
        return (None, approximate_utility(grid))
    best = (None, float('-inf'))
    for a in A:
        u = 0.0
        for i in range(m):
            grid_cp = copy.deepcopy(grid)
            grid_cp.set_resources(a)
            r = grid_cp.transition()
            a_prime, u_prime = sparse_sampling(A, grid_cp, d-1)
            u += (r + gamma * u_prime) / m
        if u > best[1]:
            best = (a, u)
    return best

def simulate_sparse(n, m, num_sims=5, simulation_depth=10, d=2):
    print('m = {}'.format(m))
    performance = []
    A = all_possible_actions(n)
    for i in range(num_sims):
        grid = FireGrid(n)
        for t in range(simulation_depth):
            a, _ = sparse_sampling(A, grid, d, m=m)
            grid.set_resources(a)
            grid.transition()
        performance.append(grid.reward)
    return performance

ms = range(2, 10, 2)
performances = [np.mean(simulate_sparse(2, m)) for m in ms]
plt.plot(ms, performances)
plt.xlabel("m (number of samples)")
plt.ylabel("average reward")
plt.title("performance of sparse sampling")
plt.savefig("../figures/sparse_sample.png")
