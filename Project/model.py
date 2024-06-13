import numpy as np
import random
from mesa import Model
from mesa.space import MultiGrid, PropertyLayer
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
from agent import Tree, Fungus, Organism

"""
TODO
- implement global planting strategy (given a harvesting strategy)
  pick 2 planting, fix harvesting
    1. simply plant exactly where a tree cut down (remove -> plant (delay?))
    2. ANR: probabilistic reproduction based on density/fertility (pollen dispersion)
    3. top 3-5 fertile sites; plant tree there (human planting strategy)
- stochastic removal of fungus - either implement or remove from report
- initialize trees with different volumes
- function for addition of dead wood: p = V_t / (1 + V_t) (V_t is total volume in neighbourhood)
"""


class Forest(Model):
    '''
    TODO:
    Things to keep track of:
    X total volume of existing trees
    - total volume of harvested trees
    X number of trees
    - number of planted trees
    X number of fungi
    X number of infected trees
    '''
    def __init__(self, width, height, n_init_trees, n_init_fungi, harvest_params, max_substrate=3, max_soil_fertility=3):

        super().__init__()

        self.height = width
        self.width = height
        self.harvest_params = harvest_params

        # Initialize harvested volume
        self.harvest_volume = 0

        # Create schedule
        self.schedule_Tree = RandomActivation(self)
        self.schedule_Fungus = RandomActivation(self)
        
        self.grid = MultiGrid(self.width, self.height, torus=True)
        
        # Add initial substrate
        self.grid.add_property_layer(PropertyLayer('substrate', self.width, self.height, 1))
        self.grid.properties['substrate'].data = np.random.uniform(0, max_substrate, (self.width, self.height))

        # Add initial soil fertility
        self.grid.add_property_layer(PropertyLayer('soil_fertility', self.width, self.height, 1))
        self.grid.properties['soil_fertility'].data = np.random.uniform(0, max_soil_fertility, (self.width, self.height))

        self.datacollector = DataCollector(
             {"Trees": lambda m: self.schedule_Tree.get_agent_count(),
              "Fungi": lambda m: self.schedule_Fungus.get_agent_count(),
              "Living Trees Total Volume": lambda m: sum([agent.volume for agent in self.schedule_Tree.agents]),
              "Infected Trees": lambda m: sum([agent.infected for agent in self.schedule_Tree.agents]),
              "Mean Substrate": lambda m: np.mean(self.grid.properties['substrate'].data),
              "Mean Soil Fertility": lambda m: np.mean(self.grid.properties['soil_fertility'].data),
              "Harvested volume": lambda m: sum([agent.volume for agent in self.schedule_Tree.agents]),
              "Harvested volume": lambda m: m.harvest_volume})
        
        # Initialise populations
        self.init_population(n_init_trees, Tree, (5, 30), 4)
        self.init_population(n_init_fungi, Fungus, (1, 3), 1)

        self.running = True
        self.datacollector.collect(self)
    

    def init_population(self, n_agents, agent_type, init_size_range, dispersal_coeff=1):
        """
        Method that initializes the population of trees and fungi.
        Args:
            n_agents (int): number of agents to add
            agent_type (Organism): class of the agent to add
            init_size_range (tuple or list): range of agent sizes [min, max] -
                volume for trees and energy for fungi
        """

        # Get lattice coordinates
        coords = np.indices((self.width, self.height)).reshape(2, -1).T

        # Determine whether multiple agents can be placed on same site
        if agent_type == Tree:
            replace = False
        else:
            replace = True
        
        # Random coords sample
        coords_select = coords[np.random.choice(len(coords), n_agents, replace=replace)]

        # Add agents to the grid
        for coord in coords_select:
            self.new_agent(agent_type, coord, np.random.randint(init_size_range[0], init_size_range[1] + 1), dispersal_coeff)

    
    def new_agent(self, agent_type, pos, init_size=1, disp=1):
        """
        Method that enables us to add agents of a given type.
        """
        
        # Create a new agent of the given type
        new_agent = agent_type(self.next_id(), self, pos, init_size, disp)

        # Add agent to schedule
        getattr(self, f'schedule_{agent_type.__name__}').add(new_agent)
    

    def remove_agent(self, agent):
        """
        Method that enables us to remove passed agents.
        """
        
        # Remove agent from grid
        self.grid.remove_agent(agent)

        # Remove agent from schedule
        getattr(self, f'schedule_{agent.__class__.__name__}').remove(agent)
    

    def calc_dist(self, pos1, pos2):
        """
        Method that calculates the Euclidean distance between two points.
        """
        return np.sqrt((pos1[..., 0] - pos2[..., 0])**2 + (pos1[..., 1] - pos2[..., 1])**2)


    def add_substrate(self):
        """
        Stochastically adds substrate (woody debris)
        based on the distance to all trees in the lattice.
        On average, 2.5*1e-4 of the tree biomass is added per time step.
        """

        coords = np.transpose(np.indices((self.width, self.height)), (1, 2, 0))

        # Assign probabilities to all lattice sites
        lattice_probs = np.zeros((self.width, self.height))
        for tree in self.schedule_Tree.agents:
            lattice_tree_dist = self.calc_dist(np.array(tree.pos), coords)
            lattice_probs += np.exp(-lattice_tree_dist / (tree.volume ** (2/3)))
        
        # Normalize probabilities
        lattice_probs /= np.sum(lattice_probs)

        # Distribute substrate
        for tree in self.schedule_Tree.agents:
            # Portion of substrate to add to each lattice site
            # Assumes an average tree volume of 100
            n_portions = int(np.floor(0.75 * tree.volume))

            # Lattice sites to add substrate to
            coords_idx_select = np.random.choice(np.arange(self.width * self.height), n_portions, replace=True, p=lattice_probs.flatten())
            coords_select = coords.reshape(-1, 2)[coords_idx_select]

            for coord in coords_select:
                self.grid.properties['substrate'].data[tuple(coord)] += 1


    def step(self):
        """
        Method that calls the step method for each of the trees, and then for each of the fungi.
        """
        self.add_substrate()

        self.schedule_Tree.step()
        self.schedule_Fungus.step()

        # Save statistics
        self.datacollector.collect(self)


    def run_model(self, n_steps=100):
        """
        Method that runs the model for a given number of steps.
        """
        for i in range(n_steps):
            self.step()
            