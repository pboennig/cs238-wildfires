from wildfire import FireGrid

n = int(input("Size of grid: "))
num_iters = int(input("Number of transitions: "))
grid = FireGrid(n)
for _ in range(num_iters):
    grid.show_fires()
    print("Reward so far: {}".format(grid.reward))
    print(" ")
    grid.transition()

grid.show_fires()
print("Reward so far: {}".format(grid.reward))


