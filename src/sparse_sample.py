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

def simulate_sparse(n, m, cpr=1, num_sims=10, simulation_depth=10, d=2):
    print('m = {}'.format(m))
    performance = []
    A = all_possible_actions(n)
    for i in range(num_sims):
        grid = FireGrid(n, cost_per_resource=cpr)
        for t in range(simulation_depth):
            a, _ = sparse_sampling(A, grid, d, m=m)
            grid.set_resources(a)
            grid.transition()
        performance.append(grid.reward)
    return performance

ms = range(2, 11, 2)
performances = [np.mean(simulate_sparse(2, m)) for m in ms]
cost_per_resource = np.linspace(.5, 1.5, num=5)
for cpr in cost_per_resource:
    rewards = [simulate_sparse(2, m, cpr=cpr) for m in ms] 
    plt.plot(ms, [np.mean(reward) for reward in rewards], label="{}".format(cpr))
plt.legend(title="cost per resource")
plt.ylabel("average reward")
plt.xlabel("m (number of samples)")
plt.title("Performance of sparse sampling")
plt.savefig("../figures/sparse_sample.png")
