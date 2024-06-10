import numpy as np
from mesa import Model, Agent
from mesa.space import MultiGrid


class Forest(Model):
    def __init__(self, width, height, n_init_trees, n_init_fungi, max_substrate=3, max_soil_fertility=3):
        self.height = width
        self.width = height
        
        self.grid = MultiGrid(self.width, self.height, torus=True)

        # Add initial substrate
        substrate = np.random.randint(max_substrate)
        self.grid.add_property_layer('substrate', self.width, self.height, substrate)

        # Add initial soil fertility
        soil_fertility = np.random.randint(max_soil_fertility)
        self.grid.add_property_layer('soil_fertility', self.width, self.height, soil_fertility)
        
        self.n_agents = n_init_trees + n_init_fungi
        self.agents = []


    def init_population(self, n_agents, type):
        """
        Method that initializes the population of trees and fungi.
        """
        for i in range(n_agents):
            x = self.random.randrange(self.width)
            y = self.random.randrange(self.height)
            
            if type == 'tree':
                self.new_agent(Tree, (x, y))
            elif type == 'fungus':
                self.new_agent(Fungus, (x, y))

    
    def new_agent(self, agent_type, pos):
        """
        Method that enables us to add agents of a given type.
        """
        self.n_agents += 1
        
        # Create a new agent of the given type
        new_agent = agent_type(self.n_agents, self, pos)
        
        # Place the agent on the grid
        self.grid.place_agent(new_agent, pos)
        
        # And add the agent to the model so we can track it
        self.agents.append(new_agent)
    

    def remove_agent(self, agent):
        """
        Method that enables us to remove passed agents.
        """
        self.n_agents -= 1
        
        # Remove agent from grid
        self.grid.remove_agent(agent)
        
        # Remove agent from model
        self.agents.remove(agent)
    

    def step(self):
        """
        Method that steps every agent. 
        
        Prevents applying step on new agents by creating a local list.
        """
        for agent in list(self.agents):
            agent.step()


class Organism(Agent):
    """
    General class for all organisms in the model.
    """
    
    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)

        self.pos = pos


class Tree(Organism):
    """
    A tree agent.
    """

    def __init__(self, unique_id, model, pos, start_volume):
        super().__init__(unique_id, model, pos)

        self.volume = start_volume
        self.dispersal_coeff = 4
        self.infected = False


    def grow(self):
        """
        Grow the tree.
        """

        # Growth rate proportional to soil fertility
        growth_rate = self.model.grid.get_property('soil_fertility', self.pos[0], self.pos[1]) * 0.1

        self.volume += growth_rate

    
    def shed_leaves(self):
        """
        Shed leaves.
        """
        # Scan all substrate on lattice
        for x in range(self.model.width):
            for y in range(self.model.height):
                if self.model.grid.get_property('substrate', x, y) > 0:
                    
                    dist = np.sqrt((x - self.pos[0])**2 + (y - self.pos[1])**2)

                    # Inoculate substrate
                    if np.random.random() < np.exp(-dist/self.dispersal_coeff):
                        self.model.new_agent(Fungus, (x, y))
    
    
    def step(self):
        """
        Tree development step.
        """
        self.grow()

        if self.infected:
            self.shed_leaves()


class Fungus(Organism):
    """
    A fungus agent.
    """
    
    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model, pos)

        # Start with 1 energy
        self.energy = 1
    

    def consume(self):
        """
        Consume substrate.
        """
        x, y = self.pos
        
        substrate = self.model.grid.get_property('substrate', x, y)
        self.model.grid.set_property('substrate', x, y, substrate - 1)

        self.energy += 1
        self.dispersal_coeff = 1
    

    def reproduce(self):
        """
        Reproduce if enough energy.
        """
        if self.energy > 4:
            # Sporulate
            self.energy -= 4

            # Scan all substrate on lattice
            for x in range(self.model.width):
                for y in range(self.model.height):
                    if self.model.grid.get_property('substrate', x, y) > 0:
                        
                        dist = np.sqrt((x - self.pos[0])**2 + (y - self.pos[1])**2)

                        # Inoculate substrate
                        if np.random.random() < np.exp(-dist/self.dispersal_coeff):
                            self.model.new_agent(Fungus, (x, y))
                        
                        # Inoculate tree
                        for agent in self.model.grid.get_cell_list_contents([(x, y)]):
                            if isinstance(agent, Tree):
                                if not agent.infected:
                                    if np.random.random() < np.exp(-dist/self.dispersal_coeff):
                                        agent.infected = True
                                        agent.volume -= 1
                                        break
        
    def die(self):
        """
        Die if no energy.
        """
        if self.energy <= 0:
            self.model.remove_agent(self)