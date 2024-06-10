from mesa import Agent
from generate_data import indices
import numpy as np


class Player(Agent):
    """
    A Player (agent), that walks around the grid. Depending on its neighbours it decides its response.
    In the default dataset the agent walks away from its neighbours.
    """
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)

    def step(self):
        """
        Check neighborhood for neighbors, then depending on the neighbors decide what to do (from the
        multinomial logit statsmodel).
        """
        neighbors = np.zeros(9)
        for i in indices.keys():
            x = self.pos[0] + indices[i][0]
            y = self.pos[1] + indices[i][1]
            if len(self.model.grid.get_neighbors((x, y), True, include_center=True, radius=0)) > 0:
                neighbors[i] = 1

        direction = np.argmax(self.model.stats_model.predict(neighbors))
        new_x = self.pos[0] + indices[direction][0]
        new_y = self.pos[1] + indices[direction][1]

        # we do not want to step on top of a neighbour
        if len(self.model.grid.get_neighbors((new_x, new_y), True, include_center=True, radius=0)) == 0:
            self.model.grid.move_agent(self, (new_x, new_y))
