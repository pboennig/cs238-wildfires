import numpy as np

class StateRegion:
    def __init__(self, fire, dryness, fuel, property, resources):
        self.fire = fire # a Boolean representing if the area is currently on fire
        self.dryness = dryness # a percentage representing how dry the region is
        self.fuel = fuel # a vale from 0 to 100 representing how much flammable material there is in this region 
        self.property = property # a number from 0 to 100 representing how valuable the property on the land is
        self.resources = resources # a Boolean, True if we've allocated resources
    
    def __str__(self):
        return "Fire: {}, Dryness: {}, Fuel: {}, Property: {}, Resources: {}".format(self.fire, self.dryness, self.fuel, self.property, self.resources)
    def __repr__(self):
        return "(Fire: {}, Dryness: {}, Fuel: {}, Property: {}, Resources: {})".format(self.fire, self.dryness, self.fuel, self.property, self.resources)


class FireGrid:
    """
    Randomly generate a n x n grid of StateRegion cells.
    All cells have no resources to start, and the other attributes
    are randomly generated. We seed fires randomly, using the fire_prob
    parameter to set the probability of a fire starting/
    """
    def __init__(self, n, fire_prob=.3):
        S = []
        self.n = n
        self.reward = 0
        self.resource_cost = 0
        for i in range(n):
            r = []
            for j in range(n):
                fire = np.random.random_sample() < fire_prob
                dryness = np.random.random_sample()
                property = 100 * np.random.random_sample()
                fuel = np.random.random_sample()
                resources = False
                r.append(StateRegion(fire, dryness, fuel, property, resources))
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

    def show_resources(self):
        symbol = {True: '0', False: '_'}
        for row in self.S:
            print(' '.join([symbol[cell.resources] for cell in row]))

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
                    self.reward -= property_lost 
                    S_prime[i][j].property -= property_lost 
                else:
                    threshold = (self.neighbors_on_fire(i, j) + 1) / 5 * self.S[i][j].dryness * self.S[i][j].fuel
                    threshold -= .5 * np.random.random_sample() * self.S[i][j].resources # if using resources, can reduce threshold by .5
                    S_prime[i][j].fire = np.random.random_sample() < threshold
        self.S = S_prime
        self.reward -= self.resource_cost

    """
    Actions to remove or add resources to a specific cell.
    """
    def place_resource(self, i, j):
        if not self.S[i][j].resources:
            self.S[i][j].resources = True
            self.resource_cost += 10
        self.transition()

    def remove_resource(self, i, j):
        if self.S[i][j].resources:
            self.S[i][j].resources = False
            self.resource_cost -= 10
        self.transition()
