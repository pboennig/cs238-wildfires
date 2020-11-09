import numpy as np
import scipy.signal
import itertools

np.seterr(all='raise')

"""
A helper function to generate all possible actions on
a n x n grid. Note that is O(2^{n^2}), so large values of
n will lead to the process hanging.
"""
def all_possible_actions(n):
    l = [False, True]
    possible_true_false_sequences = list(itertools.product(l, repeat=n**2))
    A = [np.asarray(l).reshape(n, n) for l in possible_true_false_sequences]
    return A

class FireGrid:
    """
    Randomly generate a n x n grid of StateRegion cells.
    All cells have no resources to start, and the other attributes
    are randomly generated. 
    """
    def __init__(self, n, cost_per_resource=1):
        self.n = n
        self.fire = np.zeros((n, n), dtype=bool)
        self.wind = np.random.rand(n, n)
        self.fuel = np.random.rand(n, n)
        self.dryness = np.random.rand(n, n)
        self.property = 100 * np.random.rand(n, n)
        self.resource_assignment = np.zeros((n, n), dtype=bool)
        self.reward = 0
        self.cost_per_resource = cost_per_resource 
        self.kernel = np.array([[0, 1, 0,], [1, 0, 1], [0, 1, 0]]) # only direct neighbors influence fire risk


    """
    Transition function that changes the state of the grid based on existing fire conditions,
    the dryness of the cell, and our resources. Also updates our reward to include the cost
    of burning fires and using resources. Returns the reward from this particular step while updating 
    the current object.
    """
    def transition(self):
        new_reward = 0.0 
        percentage_burned = .2 * np.random.rand(self.n, self.n) * self.fire
        self.fuel = np.minimum(self.fuel - percentage_burned, np.zeros((self.n, self.n)))
        self.fire = self.fire * (self.fuel != 0) # if out of fuel, no fire
        new_reward -= np.sum(self.property * percentage_burned)
        self.property -= self.property * percentage_burned
        neighbors_on_fire = scipy.signal.convolve(self.fire, self.kernel, mode='same')
        threshold = (neighbors_on_fire + 1) / 5 *self.dryness * self.fuel + .1 * np.sqrt(self.wind)
        threshold -= .5 * self.resource_assignment
        self.fire = np.maximum(self.fire, np.random.rand(self.n, self.n) < threshold)
        self.wind = np.clip(self.wind + .1 * np.random.randn(self.n, self.n), .01, .99)
        new_reward -= np.sum(self.resource_assignment) * self.cost_per_resource 
        self.reward += new_reward
        return new_reward


    """
    Set the resource assignment using a grid of Booleans.
    """
    def set_resources(self, arrangement):
        self.resource_assignment = arrangement
