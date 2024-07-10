import numpy as np
from mesa import Agent
import random


def check_non_empty(cell_value):
        """
        Check for available property in a given cell.
        
        Args:
        ----------
        cell_value: float
            Value of the cell.
        
        Returns:
        ----------
        bool
            True if the cell value is greater than 0, False otherwise.
        """
        return cell_value > 0


class Organism(Agent):
    """
    General class for all organisms in the forest model.
    
    Attributes
    ---------------------------------------------------
    unique_id : int
        Unique id for agent.
    model : Model
        Forest model agent is placed in.
    pos : tuple
        Position of agent on forest grid.
    disp : int
        Dispersal coefficient of agent.
        
    Methods
    ---------------------------------------------------
    calc_dist()
        Calculates the distance between two positions 
        on forest grid.
    """

    def __init__(self, unique_id, model, pos, disp):
        super().__init__(unique_id, model)

        self.disp = disp
        self.model.grid.place_agent(self, pos)
        self.pos = pos

    def calc_dist(self, pos1, pos2):
        """
        Calculate the distance between two positions on forest grid.
        """
        x1, y1 = pos1
        x2, y2 = pos2

        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


class Tree(Organism):
    """
    Represents an individual tree agent in the forest.
    
    Attributes
    ---------------------------------------------------
    unique_id : int
        Unique id for agent.
    model : Model
        Forest model tree agent is placed in.
    pos : tuple
        Position of tree on forest grid.
    disp : int
        Dispersal coefficient for leaf fall.
    init_volume : float
        Volume a tree is initialized with.
        
    Methods
    ---------------------------------------------------
    grow ()
        Calculates tree growth based on current volume
        and environmental factors.
    shed_leaves()
        Called in autumn if a tree is infected.
        Drops leaves on grid cells and attempts to 
        infect woody substrate on forest floor.
    harvest()
        Remove tree agent if all three harvest 
        thresholds are satisfied:
        - volume
        - number of neighbors
        - undefined probability
    stochastic_removal()
        Stochastically remove tree agents based
        on their average lifespan (120 years).
    step()
        Executes a single step of the tree's behavior.

    """

    def __init__(self, unique_id, model, pos, disp, init_volume=1):
        super().__init__(unique_id, model, pos, disp)

        self.agent_type = 'Tree'
        self.volume = init_volume
        self.dispersal_coeff = 4
        self.infected = False
        self.v_max = 350
        self.leaffall = 4

        # Mark tree site in forest model
        self.model.tree_sites[tuple(self.pos)] = True

    def grow(self):
        """
        Calculates tree growth based on current volume and environmental factors.
        """

        v_current = self.volume
        r = self.model.calc_r(self.pos, self.v_max, True, self.volume)
        v_update = v_current * np.exp(r*(1-v_current/self.v_max)**4)

        self.volume = v_update

    def shed_leaves(self):
        """
        Called in autumn if a tree is infected.
        Drops leaves on grid cells and attempts to 
        infect woody substrate on forest floor.
        """
        # Check for substrate
        substrate_cells = self.model.grid.properties['substrate'].select_cells(check_non_empty)

        for cell in substrate_cells:
            dist = self.calc_dist(cell, self.pos)

            # Inoculate substrate
            if np.random.random() < np.exp(-dist / self.dispersal_coeff):
                self.model.new_agent(Fungus, cell)


    def harvest(self):
        """
        Remove tree agent if all three harvest thresholds are satisfied:
        - volume
        - number of neighbors
        - undefined probability
        """
        harvest_vol_threshold, harvest_nbrs_threshold, harvest_probability = self.model.harvest_params

        # Check volume threshold
        if self.volume > harvest_vol_threshold:

            # Check percentage of surrounding trees threshold
            neighbouring_agents = self.model.grid.get_neighbors(tuple(self.pos), moore=True, include_center=False)
            count_trees = sum(agent.agent_type == "Tree" for agent in neighbouring_agents)

            if count_trees >= harvest_nbrs_threshold:
                # Include an udefined environmental probability
                if random.random() < harvest_probability:
                    # Remove tree if all three thresholds are satisfied
                    self.model.harvest_volume += self.volume
                    self.model.remove_agent(self) 

    def stochastic_removal(self):
        ''' 
        Stochastically remove tree agents based on their average lifespan (120 years).
        '''

        rate_parameter = 1 / 480
        p_die = 1 - np.exp(-rate_parameter)
        if np.random.random() < p_die:
            # Add substrate to the soil of dead tree
            coord = self.pos
            self.model.grid.properties['substrate'].data[tuple(coord)] += self.volume*0.1

            # Remove tree agent
            self.model.remove_agent(self)
            

    def step(self):
        """
        Executes a single step of the tree's behavior.
        """
        self.grow()

        # Shed leaves in fall
        if self.infected and self.model.schedule.time % self.leaffall == 0:
            self.shed_leaves()

        # Check if tree can be harvested
        if self.harvest():
            return
        
        # Stochastically die off
        self.stochastic_removal()


class Fungus(Organism):
    """
    Represents an individual fungus agent in the forest.
    
    Attributes
    ---------------------------------------------------
    unique_id : int
        Unique id for agent.
    model : Model
        Forest model fungus agent is placed in.
    pos : tuple
        Position of fungus on forest grid.
    disp : int
        Dispersal coefficient for sporulation on forest floor.
    init_energy : int
        Energy a fungus agent is initialized with.
        
    Methods
    ---------------------------------------------------
    consume()
        Consumes substrate and deposit fertility on 
        lattice cell the fungus occupies.
        Consumes its own energy if no substrate is present.
    infect_wood()
        Attempt to infect woody debris based on dispersal
        coefficient and number of fungi present.
    infect_tree()
        Attempt to infect tree based on dispersal coefficient.
    sporulate()
        Reproduce to other forest grid cells if woody
        debris is present.
        Attempt to reproduce into trees.
    die()
        Remove fungus agent from forest.
    stochastic_removal()
        Stochastically kill off fungus agent based on average 
        lifespan (5 years).
    step()
        Executes a single step of the fungus' behavior.
    """

    def __init__(self, unique_id, model, pos, disp, init_energy=1):
        super().__init__(unique_id, model, pos, disp)

        self.agent_type = 'Fungus'
        self.energy = init_energy

    def consume(self):
        """
        Consumes substrate and deposit fertility on lattice cell the fungus occupies.
        Consumes its own energy if no substrate is present.
        """
        x, y = self.pos
        substrate = self.model.grid.properties['substrate'].data[x, y]
        fertility = self.model.grid.properties['soil_fertility'].data[x, y]

        # Eat wood + deposit fertility
        if substrate > 0:
            self.model.grid.properties['substrate'].set_cell((x, y), substrate - 1)
            self.model.grid.properties['soil_fertility'].set_cell((x, y), fertility + 1)
            self.energy += 1
        # Consume reserve if no wood present
        else:
            self.energy -= 1

    def infect_wood(self, cell):
        '''
        Attempt to infect woody debris based on dispersal coefficient and number of fungi present.
        
        Args:
        ------------
        cell : tuple
            Forest grid cell to infect (x,y).
        '''
        dist = self.calc_dist(cell, self.pos)

        # Count fungi in cell to scale probability with space competition
        contents = self.model.grid.get_cell_list_contents(cell)
        fun_count = sum(1 for agent in contents if isinstance(agent, Fungus))

        # Inoculation probability
        prob = 1 / (fun_count + 1) * np.exp(-dist / self.disp)

        # Probabilistic infection
        if np.random.random() < prob:
            self.model.new_agent(Fungus, cell)

    def infect_tree(self, tree):
        '''
        Attempt to infect tree based on dispersal coefficient.
        
        Args:
        -----------
        tree : Agent
            Tree in forest to infect.   
        '''
        dist = self.calc_dist(tree.pos, self.pos)
        prob = np.exp(-dist / self.disp)

        # Probabilistic infection
        if np.random.random() < prob:
            tree.infected = True

    def sporulate(self):
        """
        Reproduce to other forest grid cells if woody debris is present.
        Attempt to reproduce into trees.
        """
        self.energy -= 4

        # Substrate infection
        substrate_layer = self.model.grid.properties['substrate']

        # Select cells that have substrate
        def has_substrate(cell_value):
            return cell_value > 0

        woody_cells = substrate_layer.select_cells(has_substrate)
        for cell in woody_cells:
            # Only infect at different location
            if cell[0] != self.pos[0] and cell[1] != self.pos[1]:
                self.infect_wood(cell)

        # Tree infection
        trees = self.model.getall("Tree")
        for tree in trees:
            if not tree.infected:
                self.infect_tree(tree)

    def die(self):
        """
        Remove fungus agent from forest.
        """
        self.model.remove_agent(self)


    def stochastic_removal(self):
        ''' 
        Stochastically kill off fungus agent based on average lifespan (5 years).
        '''
        if np.random.random() < 0.1:
           self.model.remove_agent(self)

    def step(self):
        """
       Executes a single step of the fungus' behavior.
        """
        # Eat substrate
        self.consume()

        # Reproduce once a year or die if no energy left
        if self.energy > 4:
            self.sporulate()
        elif self.energy < 1:
            self.die()
            return

        # Stochastically die off
        self.stochastic_removal()
        
        