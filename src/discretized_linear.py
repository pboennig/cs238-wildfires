from wildfire import FireGrid
import numpy as np
import itertools

def approximate_utility(grid):
    utility = 0.0
    for row in grid.observation()[0]:
        for region in row:
            utility -= region.fire * region.property
    return utility

def sparse_sampling(grid, d, m=10, gamma=.95):
    if d <= 0:
        return (None, approximate_utility(grid))
    best = (None, float('-inf'))
    for a in grid.A:
        u = 0.0
        for i in range(m):
            grid_cp = grid
            grid_cp.set_resources(a)
            r = grid_cp.transition()
            a_prime, u_prime = sparse_sampling(grid_cp, d-1)
            u += (r + gamma * u_prime) / m
        if u > best[1]:
            best = (a, u)
    return best

def simulate_sparse(n, num_sims=5, num_transitions=5, d=3):
    performance = []
    for i in range(num_sims):
        print(i)
        grid = FireGrid(n)
        for _ in range(num_transitions):
            a, _ = sparse_sampling(grid, d)
            grid.set_resources(a)
            grid.transition()
        performance.append(grid.reward)
    return performance

print(np.mean(simulate_sparse(2)))
