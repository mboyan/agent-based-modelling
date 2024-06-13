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
    - total volume of existing trees
    - total volume of harvested trees
    - number of trees
    - number of planted trees
    - number of fungi
    - number of infected trees
    '''
    def __init__(self, width, height, n_init_trees, n_init_fungi, max_substrate=3, max_soil_fertility=3):

        super().__init__()

        self.height = width
        self.width = height

        # Create schedule
        self.schedule_Tree = RandomActivation(self)
        self.schedule_Fungus = RandomActivation(self)
        
        self.grid = MultiGrid(self.width, self.height, torus=True)
        
        # Add initial substrate
        self.grid.add_property_layer(PropertyLayer('substrate', self.width, self.height, 1))
        self.grid.properties['substrate'].data = np.random.randint(0, max_substrate, (self.width, self.height))

        # Add initial soil fertility
        self.grid.add_property_layer(PropertyLayer('soil_fertility', self.width, self.height, 1))
        self.grid.properties['soil_fertility'].data = np.random.randint(0, max_soil_fertility, (self.width, self.height))

        self.datacollector = DataCollector(
             {"Trees": lambda m: self.schedule_Tree.get_agent_count(),
              "Fungi": lambda m: self.schedule_Fungus.get_agent_count()})
        
        # Initialise populations
        self.init_population(n_init_trees, Tree, (5, 30))
        self.init_population(n_init_fungi, Fungus, (1, 3))

        self.running = True
        self.datacollector.collect(self)


    def init_population(self, n_agents, agent_type, init_size_range):
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
            self.new_agent(agent_type, coord, np.random.randint(init_size_range[0], init_size_range[1] + 1))

    
    def new_agent(self, agent_type, pos, init_size=1):
        """
        Method that enables us to add agents of a given type.
        """
        
        # Create a new agent of the given type
        new_agent = agent_type(self.next_id(), self, pos, init_size)

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
    

    def step(self):
        """
        Method that calls the step method for each of the trees, and then for each of the fungi.
        """
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
            