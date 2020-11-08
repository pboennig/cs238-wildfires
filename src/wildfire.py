import numpy as np
import itertools

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
    def __init__(self, n):
        S = []
        self.resource_assignment = np.zeros((n, n), dtype=bool)
        self.n = n
        self.reward = 0
        self.resource_cost = 0
        self.cost_per_resource = .8
        self.A = self.all_possible_actions()
        for i in range(n):
            r = []
            for j in range(n):
                fire = False 
                dryness = np.random.random_sample()
                property = 100 * np.random.random_sample()
                fuel = np.random.random_sample()
                wind = np.random.random_sample()
                r.append(StateRegion(fire, dryness, fuel, wind, property))
            S.append(r)

        self.S = S

    """
    Helper functions to display the fires, fuel, and resources.
    """
    def show_fires(self):
        symbol = {True: 'X', False: '_'}
        for row in self.S:
            print(' '.join([symbol[cell.fire] for cell in row]))

    def show_fuel_status(self):
        for row in self.S:
            print(' '.join([str(round(cell.fuel, 2)) for cell in row]))

    def show_wind(self):
        for row in self.S:
            print(' '.join([str(round(cell.wind, 2)) for cell in row]))

    def all_possible_actions(self):
        l = [False, True]
        possible_true_false_sequences = list(itertools.product(l, repeat=self.n**2))
        A = [np.asarray(l).reshape(self.n, self.n) for l in possible_true_false_sequences]
        return A

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
        S_prime = self.S
        new_reward = 0.0 
        for i in range(self.n):
            for j in range(self.n):
                if self.S[i][j].fire:
                    # for now, cannot put out fires but only contain/prevent them
                    percentage_burned = .2 * np.random.random_sample()
                    if S_prime[i][j].fuel < percentage_burned:
                        # we're out of fuel
                        S_prime[i][j].fuel = 0
                        S_prime[i][j].fire = False
                    else:
                        S_prime[i][j].fuel -= percentage_burned
                    property_lost = S_prime[i][j].property * percentage_burned
                    new_reward -= property_lost 
                    S_prime[i][j].property -= property_lost 
                else:
                    threshold = (self.neighbors_on_fire(i, j) + 1) / 5 * self.S[i][j].dryness * self.S[i][j].fuel + .1 * np.sqrt(self.S[i][j].wind)
                    threshold -= .5 * np.random.random_sample() * self.resource_assignment[i, j] # if using resources, can reduce threshold by .5
                    S_prime[i][j].fire = np.random.random_sample() < threshold
                S_prime[i][j].wind = min(self.S[i][j].wind + .1 * np.random.random_sample() - .05, .99)
        self.S = S_prime
        new_reward -= self.resource_cost
        self.reward += new_reward
        return new_reward


    def observation(self):
        O = []
        for i in range(self.n):
            r = []
            for j in range(self.n):
                r.append(ObservationRegion(self.S[i][j]))
            O.append(r)
        return (O, self.resource_assignment)


    def set_resources(self, arrangement):
        self.resource_assignment = arrangement
        self.resource_cost = self.cost_per_resource * np.sum(self.resource_assignment) # chosen by evaluating random policy
