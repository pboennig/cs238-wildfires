import numpy as np
from wildfire import FireGrid, all_possible_actions
import matplotlib.pyplot as plt
import copy

"""
Follow a random policy until depth = 0. Return discounted utility.
"""
def random_rollout(grid, depth, gamma=.95, pp=.5):
    if depth <= 0:
        return 0.0
    else:
        random_flip = np.random.rand(grid.n, grid.n) < pp 
        grid.set_resources(random_flip)
        r = grid.transition()
        return r + gamma * random_rollout(grid, depth - 1, gamma=gamma, pp=pp)

"""
Estimate utility as the product of property and fire.
"""
def approximate_utility(grid):
    return np.sum(grid.property * grid.fire)


"""
One step lookahead using random rollout. Return the action with the highest average
utility ovver m samples.
"""
def lookahead(A, grid, d, m=5, gamma=.95):
    if d <= 0:
        return (None, approximate_utility(grid))
    best = (None, float('-inf'))
    for a in A:
        u = 0.0
        for i in range(m):
            grid_cp = copy.deepcopy(grid)
            grid_cp.set_resources(a)
            r = grid_cp.transition()
            u_prime = random_rollout(grid_cp, d-1, gamma=gamma) 
            u += (r + gamma * u_prime) / m
        if u > best[1]:
            best = (a, u)
    return best[0]

def simulate_lookahead(n, m, cpr=1, num_sims=20, simulation_depth=10, d=5):
    print('m = {}'.format(m))
    performance = []
    A = all_possible_actions(n)
    for i in range(num_sims):
        grid = FireGrid(n, cost_per_resource=cpr)
        for t in range(simulation_depth):
            a = lookahead(A, grid, d, m=m)
            grid.set_resources(a)
            grid.transition()
        performance.append(grid.reward)
    return performance

ms = range(2, 11, 2)
cost_per_resource = np.linspace(.5, 1.5, num=5)
for cpr in cost_per_resource:
    rewards = [simulate_lookahead(2, m, cpr=cpr) for m in ms] 
    plt.plot(ms, [np.mean(reward) for reward in rewards], label="{}".format(cpr))
plt.legend(title="cost per resource")
plt.ylabel("average reward")
plt.xlabel("m (number of samples)")
plt.title("Performance of lookahead")
plt.savefig("../figures/lookahead.png")
