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
        print(fertility)

        # Get neighbours occupied by trees
        nbr_agents = self.model.grid.get_neighbors(tuple(self.pos), moore=True, include_center=False)
        nbrs_with_trees = [1 for agent in nbr_agents if isinstance(agent, Tree)]
        nbrs_with_trees = sum(nbrs_with_trees)

        # Get total volume of neighbour trees
        volume_nbrs = [agent.volume for agent in nbr_agents if isinstance(agent, Tree)]
        volume_nbrs = sum(volume_nbrs)

        # Competition with neighbours
        competition = self.volume / (volume_nbrs + self.volume)

        # Growth term
        volume_add = (self.base_growth_rate / self.volume + self.volume * fertility) * competition

        self.volume += volume_add

    
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
    TODO
    - sporulate & spore_infect = reproduce in agent.py -> combine
    """
    
    def __init__(self, unique_id, model, pos, init_energy=1):
        super().__init__(unique_id, model, pos)

        self.agent_type = 'Fungus'
        self.energy = init_energy
        # self.disp = disp
    

    def consume(self):
        """
        Consume substrate.
        """
        x, y = self.pos
        
        substrate = self.model.grid.properties['substrate'].data[x, y]
        fertility = self.model.grid.properties['soil_fertility'].data[x, y]
        # eat wood + deposit fertility
        if substrate > 0:
            self.model.grid.properties['substrate'].set_cell((x, y), substrate - 1)
            self.model.grid.properties['soil_fertility'].set_cell((x,y), fertility + 1)
            self.energy += 1
        # consume reserve if no wood
        else:
            self.energy -= 1
        
        # self.dispersal_coeff = 1 ?
    
    def infect_wood(self, cell):
        '''
        Try to infect woody debris.
        '''
        x,y = cell
        dist = np.sqrt((x - self.pos[0])**2 + (y - self.pos[1])**2)
        
        subs_site = self.model.grid.properties['substrate'].data[x, y]
        # fert_site = self.model.grid.properties['soil_fertility'].data[x, y]
        # decay = subs_site / (subs_site + fert_site) if subs_site !=0 else 1
        
        # count fungi in cell
        contents = self.model.grid.get_cell_list_contents(cell)
        fun_count = len([agent for agent in contents if type(agent)==Fungus])
        
        # inoculation probability
        prob = 1/(fun_count+1) * np.exp(-dist/self.dispersal_coeff)
        
        # try to infect wood at different location
        if np.random.random() < prob and cell != self.pos: 
            self.model.new_agent(Fungus, (x, y))
        
        
    def infect_tree(self,tree):
        '''
        Try to infect tree.
        '''
        x,y = tree.pos
        dist = np.sqrt((x - self.pos[0])**2 + (y - self.pos[1])**2)
        prob = np.exp(-dist/self.dispersal_coeff)

        # infect tree with calculated probability
        if np.random.random() < prob:
            tree.infected = True

    def sporulate(self):
        """
        Reproduce if enough energy.
        TODO
        - don't inoculate the same substrate twice
        """
        # sporulate
        self.energy -= 4

        # substrate infection      
        substrate_layer = self.model.grid.properties['substrate']

        def has_substrate(cell_value):
            return cell_value > 0

        # select cells that have substrate
        woody_cells = substrate_layer.select_cells(has_substrate)
        for cell in woody_cells:
            self.infect_wood(cell)
        
        # tree infection
        trees = self.model.getall(Tree)
        for tree in trees:
            if not tree.infected:
                self.infect_tree(tree)
                

        
    def die(self):
        """
        Die if no energy or killed through environment
        """
        self.model.remove_agent(self)
    

    def step(self):
        """
        Fungus self-development step.
        """
        self.consume()
        
        if self.energy > 4:
            self.sporulate()
        elif self.energy < 1:
            self.die()
            