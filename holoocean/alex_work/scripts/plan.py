import itertools
from matplotlib import pyplot as plt
from itertools import product
import numpy as np
import scipy.interpolate as interp

# Input space
xx = np.loadtxt("xx.txt")
y = np.loadtxt("y.txt")
xx = xx
y = y

grid_res = 0.25

interp2d = interp.RBFInterpolator(xx, y, kernel="multiquadric", epsilon=10.0)
x1 = np.arange(round(xx[:,0].min()), round(xx[:,0].max()), grid_res)[::-1]
x2 = np.arange(round(xx[:,1].min()), round(xx[:,1].max()+5.0), grid_res)[::-1]
x1x2 = np.array(list(product(x1, x2)))
z_dense = interp2d(x1x2)
# print(x1x2.shape, x1.shape, x2.shape, z_dense.shape)

targets = np.array([
    [-1.0, -0.5],
    [-2.49, 8.9],
    [-17.5, 13.5],
])
path = np.array([
    [8.0, -1.0],
    [-28.0, -1.0],
    [-28.0, 7.0],
    [8.0, 7.0],
    [8.0, 15.0],
    [-28.0, 15.0],
])

fig = plt.figure(layout='constrained')

h = plt.contourf(x1, x2, z_dense.reshape((x1.shape[0], x2.shape[0])).T)
plt.scatter(targets[:,0], targets[:,1], s=80, c='r', marker=(5, 1), label="targets")
plt.plot(path[:,0], path[:,1], c='cyan', label="path")

plt.axis('scaled')
plt.colorbar()
# plt.tight_layout()

plt.legend()
# plt.show()

cities = ["start"]
coords = [path[-1]]
for i in range(len(targets)):
    cities.append(f"target_{i}")
    coords.append(targets[i])
cities.append("end")
coords.append(path[0])

distances = {}
for i, city in enumerate(cities):
    for j, dst in enumerate(cities[i+1:]):
        distances[(city, dst)] = np.linalg.norm(coords[i]-coords[i+1+j])

# Function to calculate the total cost of a route
def calculate_cost(route):
    total_cost = 0
    n = len(route)
    for i in range(n-1):
        current_city = route[i]
        next_city = route[(i + 1) % n]  # Wrap around to the start of the route
        # Look up the distance in both directions
        if (current_city, next_city) in distances:
            total_cost += distances[(current_city, next_city)]
        else:
            total_cost += distances[(next_city, current_city)]
    return total_cost

# Generate all permutations of the cities
all_permutations = itertools.permutations(cities[1:-1])

# Initialize variables to track the minimum cost and corresponding route
min_cost = float('inf')
optimal_route = None

# Iterate over all permutations and calculate costs
for perm in all_permutations:
    perm = [cities[0]]+list(perm)+[cities[-1]]
    cost = calculate_cost(perm)
    if cost < min_cost:
        min_cost = cost
        optimal_route = perm

# Print the optimal route and its cost
print(f"Optimal Route: {optimal_route}")
print(f"Total Cost: {min_cost}")

cities = np.array(cities)
n_path = []
xs = []
ys = []
for city in optimal_route:
    idx = np.where(cities==city)[0][0]
    print(city, idx)
    # print(cities)
    xs.append(coords[idx][0])
    ys.append(coords[idx][1])

fig = plt.figure(layout='constrained')

h = plt.contourf(x1, x2, z_dense.reshape((x1.shape[0], x2.shape[0])).T)
plt.scatter(targets[:,0], targets[:,1], s=80, c='r', marker=(5, 1), label="targets")
plt.plot(xs, ys, c='cyan', label="path")

plt.axis('scaled')
plt.colorbar()
# plt.tight_layout()

plt.legend()
plt.show()