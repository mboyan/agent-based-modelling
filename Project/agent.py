import numpy as np
from mesa import Agent

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

    def __init__(self, unique_id, model, pos, start_volume=1):
        super().__init__(unique_id, model, pos)

        self.volume = start_volume
        self.dispersal_coeff = 4
        self.infected = False


    def grow(self):
        """
        Grow the tree.
        """

        # Growth rate proportional to soil fertility
        growth_rate = self.model.grid.properties['soil_fertility'].data[self.pos[0], self.pos[1]] * 0.1

        self.volume += growth_rate

    
    def shed_leaves(self):
        """
        Shed leaves.
        """
        # Scan all substrate on lattice
        for x in range(self.model.width):
            for y in range(self.model.height):
                if self.model.grid.properties['substrate'].data[x, y] > 0:
                    
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
        
        substrate = self.model.grid.properties['substrate'].data[x, y]
        self.model.grid.properties['substrate'].set_cell((x, y), substrate - 1)

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
                    if self.model.grid.properties['substrate'].data[x, y] > 0:
                        
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
    

    def step(self):
        """
        Fungus development step.
        """
        self.consume()
        self.reproduce()
        self.die()