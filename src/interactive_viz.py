from wildfire import FireGrid

n = int(input("Size of grid: "))
num_iters = int(input("Number of transitions: "))
grid = FireGrid(n)
for _ in range(num_iters):
    grid.show_fires()
    grid.show_wind()
    grid.transition()

grid.show_fires()
grid.show_wind()
print(grid.observation())
print(grid.S)

