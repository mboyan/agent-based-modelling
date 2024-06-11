import numpy as np
import random
from mesa import Model
from mesa.space import MultiGrid, PropertyLayer
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
from agent import Tree, Fungus, Organism

class Forest(Model):
    def __init__(self, width, height, n_init_trees, n_init_fungi, max_substrate=3, max_soil_fertility=3):

        super().__init__()

        self.height = width
        self.width = height

        # Create schedule
        self.schedule_Tree = RandomActivation(self)
        self.schedule_Fungus = RandomActivation(self)
        
        self.grid = MultiGrid(self.width, self.height, torus=True)
        
        # Add initial substrate
        substrate = np.random.randint(max_substrate)
        self.grid.add_property_layer(PropertyLayer('substrate', self.width, self.height, substrate))

        # Add initial soil fertility
        soil_fertility = np.random.randint(max_soil_fertility)
        self.grid.add_property_layer(PropertyLayer('soil_fertility', self.width, self.height, soil_fertility))

        self.datacollector = DataCollector(
             {"Trees": lambda m: self.schedule_Tree.get_agent_count(),
              "Fungi": lambda m: self.schedule_Fungus.get_agent_count()})
        
        # Initialise populations
        self.init_population(n_init_trees, Tree)
        self.init_population(n_init_fungi, Fungus)

        self.running = True
        self.datacollector.collect(self)


    def init_population(self, n_agents, agent_type):
        """
        Method that initializes the population of trees and fungi.
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
            self.new_agent(agent_type, coord)

        # for i in range(n_agents):
        #     x = random.randrange(self.width)
        #     y = random.randrange(self.height)
            
        #     self.new_agent(agent_type, (x, y))

    
    def new_agent(self, agent_type, pos):
        """
        Method that enables us to add agents of a given type.
        """
        # self.n_agents += 1
        
        # Create a new agent of the given type
        new_agent = agent_type(self.next_id(), self, pos)
        
        # Place the agent on the grid
        self.grid.place_agent(new_agent, pos)
        
        # And add the agent to the model so we can track it
        # self.agents.append(new_agent)

        # Add agent to schedule
        getattr(self, f'schedule_{agent_type.__name__}').add(new_agent)
    

    def remove_agent(self, agent):
        """
        Method that enables us to remove passed agents.
        """
        # self.n_agents -= 1
        
        # Remove agent from grid
        self.grid.remove_agent(agent)
        
        # Remove agent from model
        # self.agents.remove(agent)

        # Remove agent from schedule
        getattr(self, f'schedule_{agent.__class__.__name__}').remove(agent)
    

    def step(self):
        """
        Method that calls the step method for each of the trees, and then for each of the fungi.
        """
        # print(self.grid.properties['substrate'].data[0,0])
        # for agent in list(self.agents):
        #     agent.step()
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
            