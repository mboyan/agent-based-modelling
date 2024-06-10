import numpy as np


# directions and corresponding (unit)vectors on a grid
directions = {0: [0, 0],
              1: [0, 1], 2: [0, -1],
              3: [1, 0], 4: [-1, 0],
              5: [np.sqrt(2) / 2, np.sqrt(2) / 2], 6: [-np.sqrt(2) / 2, -np.sqrt(2) / 2],
              7: [-np.sqrt(2) / 2, np.sqrt(2) / 2], 8: [np.sqrt(2) / 2, -np.sqrt(2) / 2]}

indices = {0: [0, 0],
           1: [0, 1], 2: [0, -1],
           3: [1, 0], 4: [-1, 0],
           5: [1, 1], 6: [-1, -1],
           7: [-1, 1], 8: [1, -1]}

# make a dataset which indicates whether a neighbour exists on one of its 9 neighbouring gridpoints,
# and the last column indicates which direction the agent moves in as a response
# the agents in general walk away from their neighbours
datapoints = 300
data = np.zeros((datapoints, 10), dtype=np.int)
data[:, 1:9] = np.random.randint(0, 2, size=(datapoints, 8))
data[:, 0] = 1

for i in range(datapoints):
    # get the vector 'away' from its neighbours
    vector = np.array([-sum(x) for x in zip(*[directions[x] for x in np.where(data[i])[0]])])

    response = 0
    if np.linalg.norm(vector) > 0:
        uvector = vector / np.linalg.norm(vector)
        diffs = [np.linalg.norm(np.array(v) - uvector) for _, v in directions.items()]
        response = diffs.index(min(diffs))

    data[i, -1] = response

# add some random noise
flips = np.random.choice([0, 1], size=(datapoints, 8), p=[0.95, 0.05])
data[:, 1:-1] ^= flips

np.savetxt("direction_choices.csv", data, delimiter=";")
