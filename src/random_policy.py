import numpy as np
from wildfire import FireGrid
import matplotlib.pyplot as plt


def sim_random_policy(place_prob, cost_per_resource, grid_size=20, num_sims=50, simulation_depth=10):
    rewards = []
    for iter in range(num_sims):
        grid = FireGrid(grid_size)
        grid.cost_per_resource = cost_per_resource
        for sim in range(simulation_depth):
            random_flip = np.random.rand(grid_size, grid_size) < place_prob
            grid.set_resources(random_flip)
            grid.transition()
        rewards.append(grid.reward)
    return rewards

place_probs = np.linspace(.05, .95, num=10)
cost_per_resource = np.linspace(.5, 1.5, num=5)
for cpr in cost_per_resource:
    rewards = [sim_random_policy(pp,cpr) for pp in place_probs] 
    plt.plot(place_probs, [np.mean(reward) for reward in rewards], label=f'{cpr}')
plt.legend(title="cost per resource")
plt.xlabel("probability of placing resource")
plt.ylabel("average reward")
plt.title("Performance of random policy")
plt.savefig("../figures/random_policy.png")
