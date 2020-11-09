import numpy as np
import scipy.signal
import itertools

np.seterr(all='raise')

def all_possible_actions(n):
    l = [False, True]
    possible_true_false_sequences = list(itertools.product(l, repeat=n**2))
    A = [np.asarray(l).reshape(n, n) for l in possible_true_false_sequences]
    return A


class StateRegion:
    def __init__(self, fire, dryness, fuel, wind, property):
        self.fire = fire # a Boolean representing if the area is currently on fire
        self.dryness = dryness # a percentage representing how dry the region is
        self.fuel = fuel # a vale from 0 to 100 representing how much flammable material there is in this region 
        self.wind = wind # a percentage representing the windiness
        self.property = property # a number from 0 to 100 representing how valuable the property on the land is
    
    def __str__(self):
        return "Fire: {}, Dryness: {}, Fuel: {}, Property: {}, Resources: {}".format(self.fire, self.dryness, self.fuel, self.property, self.resources)
    def __repr__(self):
        return "(Fire: {}, Dryness: {}, Fuel: {}, Property: {}, Resources: {})".format(self.fire, self.dryness, self.fuel, self.property, self.resources)

class ObservationRegion:
    def __init__(self, state_region, fuel_sigma=.05, dryness_sigma=.05):
        self.fire = state_region.fire
        self.wind = state_region.wind # can perfectly measure wind
        self.fuel = state_region.fuel + np.random.normal(scale=fuel_sigma)
        self.dryness = state_region.dryness + np.random.normal(scale=dryness_sigma)
        self.property = state_region.property

    def __repr__(self):
        return "(Fire: {}, Dryness: {}, Fuel: {}, Property: {})".format(self.fire, self.dryness, self.fuel, self.property)

class FireGrid:
    """
    Randomly generate a n x n grid of StateRegion cells.
    All cells have no resources to start, and the other attributes
    are randomly generated. We seed fires randomly, using the fire_prob
    parameter to set the probability of a fire starting/
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
        self.kernel = np.array([[0, 1, 0,], [1, 0, 1], [0, 1, 0]]) # only direct neighbors


    """
    Helper function to calculate the number of neighboring cells on fire.
    """ 
    def neighbors_on_fire(self, i, j):
        num_neighbors = 0
        if i - 1 >= 0:
            num_neighbors += self.S[i-1][j].fire
        if i + 1 < self.n:
            num_neighbors += self.S[i+1][j].fire
        if j - 1 >= 0:
            num_neighbors += self.S[i][j-1].fire
        if j + 1 < self.n:
            num_neighbors += self.S[i][j+1].fire
        return num_neighbors

    """
    Transition function that changes the state of the grid based on existing fire conditions,
    the dryness of the cell, and our resources. Also updates our reward to include the cost
    of burning fires and using resources.
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


    def observation(self):
        return (self.fire, self.property + 3 * np.random.randn(self.n, self.n), self.resource_assignment)


    def set_resources(self, arrangement):
        self.resource_assignment = arrangement
