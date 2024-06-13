import numpy as np
from mesa import Agent
import random

class Organism(Agent):
    """
    General class for all organisms in the model.
    """
    
    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)
        
        self.pos = pos
        self.model.grid.place_agent(self, pos)


class Tree(Organism):
    """
    A tree agent.
    TODO
    - leaffall constant?
    - embed shed_leaves from agent.py + update to not scan entire lattice
    - add/define tree growth + consume fertility (from entire Moore neighborhood (+?))
    - define harvesting in die function
    - different initialization volumes
    """

    def __init__(self, unique_id, model, pos, init_volume=1, base_growth_rate=1):
        super().__init__(unique_id, model, pos)

        self.agent_type = 'Tree'

        self.volume = init_volume
        self.dispersal_coeff = 4
        self.infected = False
        self.base_growth_rate = base_growth_rate


        self.harvest_params = [20,0.5,0.6]


    def grow(self):
        """
        Grow the tree.
        TODO:
        - find correct way to represent root system / inlcusion of neighbour fertility
        """

        # Get fertility of current cell and its neighbours
        fertility_center = self.model.grid.properties['soil_fertility'].data[self.pos[0], self.pos[1]]
        neighbourhood = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
        fertility_nbrs = [self.model.grid.properties['soil_fertility'].data[x, y] for x, y in neighbourhood]
        fertility = 0.5 * fertility_center + 0.5 * np.mean(fertility_nbrs)

        # Get neighbours occupied by trees
        nbr_agents = self.model.grid.get_neighbors(tuple(self.pos), moore=True, include_center=False)
        nbrs_with_trees = [1 for agent in nbr_agents if isinstance(agent, Tree)]
        nbrs_with_trees = sum(nbrs_with_trees)

        # Get total volume of neighbour trees
        volume_nbrs = [agent.volume for agent in nbr_agents if isinstance(agent, Tree)]
        volume_nbrs = sum(volume_nbrs)

        # Growth term (can inclue)
        # THIS IS SOOO COMPUTATIONALLY HEAVY oops

        base_vol_increase = 0.0392/4 * self.volume
        vol_fertility = base_vol_increase * 1.05 if (self.volume < 10 and fertility > 1) else base_vol_increase * 1
        vol_add = vol_fertility * 0.9 if volume_nbrs/self.volume < 0.5 else vol_fertility * 1
        self.volume += vol_add


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
    
    def harvest(self):
        """
        How a tree 'harvests itself':
        
            If the volume is above a threshold and if x percent of the surrounding 8 trees are still present
            -> can be harvested with probability p
        """
        harvest_vol_threshold, harvest_percent_threshold, harvest_probability = self.harvest_params

        # Check volume threshold
        if self.volume > harvest_vol_threshold:
            # Check percentage of surrounding trees threshold
            neighbouring_agents = self.model.grid.get_neighbors(tuple(self.pos), moore=True, include_center=False)
            count_trees = sum(1 for agent in neighbouring_agents if isinstance(agent, Tree))
            if count_trees / 8 > harvest_percent_threshold:
                # Include a random probability
                if random.random() < harvest_probability:
                    self.model.grid.remove_agent(self)  # Remove the tree
                    return True
        return False  # Tree is not harvested


    

    
    def step(self):
        """
        Tree development step.
        """
        self.grow()

        if self.infected:
            self.shed_leaves()
        
        self.harvest()



class Fungus(Organism):
    """
    A fungus agent.
    TODO
    - define fertility: rates, quantities, units
    - remove endophyticism check
    - sporulate & spore_infect = reproduce in agent.py -> combine
    """
    
    def __init__(self, unique_id, model, pos, init_energy=1):
        super().__init__(unique_id, model, pos)

        # Start with 1 energy
        self.energy = init_energy
        self.agent_type = 'Fungus'
    

    def consume(self):
        """
        Consume substrate.
        TODO:
        - substrate cannot go below 0
        - multiple fungi? account for simulateous eating
        """
        x, y = self.pos
        
        substrate = self.model.grid.properties['substrate'].data[x, y]
        self.model.grid.properties['substrate'].set_cell((x, y), substrate - 1)

        
        self.energy += 1
        self.dispersal_coeff = 1
    

    def reproduce(self):
        """
        Reproduce if enough energy.
        TODO
        - don't inoculate the same substrate twice
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
                        if np.random.random() < np.exp(-dist/self.dispersal_coeff): # and dist > 0
                            self.model.new_agent(Fungus, (x, y))
                        
                        # Inoculate tree
                        for agent in self.model.grid.get_cell_list_contents([(x, y)]):
                            if isinstance(agent, Tree):
                                if not agent.infected:
                                    
                                    # Decay status
                                    subs_site = self.model.grid.properties['substrate'].data[x, y]
                                    fert_site = self.model.grid.properties['soil_fertility'].data[x, y]
                                    decay = subs_site / (subs_site + fert_site) if subs_site + fert_site else 0

                                    if np.random.random() < decay * np.exp(-dist/self.dispersal_coeff):
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
        self.reproduce()
        self.consume()
        self.die()