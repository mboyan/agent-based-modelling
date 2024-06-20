import numpy as np
from mesa import Agent
import random


def check_non_empty(cell_value):
        """
        Check for available property in a given cell.
        Args:
            cell_value: Value of the cell.
        """
        return cell_value > 0

class Organism(Agent):
    """
    General class for shared attributes in all organisms of the model.
    """

    def __init__(self, unique_id, model, pos, disp):
        super().__init__(unique_id, model)

        self.disp = disp
        self.pos = pos
        self.model.grid.place_agent(self, pos)

    def calc_dist(self, pos1, pos2):
        """
        Calculate the distance between two positions.
        """
        x1, y1 = pos1
        x2, y2 = pos2

        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


class Tree(Organism):
    """
    Tree agent:
    A tree can become infected with viaphytic fungus and spread infection through leaf drop.
    Each timestep, the tree grows and is possibly harvested.
    """

    def __init__(self, unique_id, model, pos, disp, init_volume=1):
        super().__init__(unique_id, model, pos, disp)

        self.agent_type = 'Tree'
        self.volume = init_volume
        self.dispersal_coeff = 4
        self.infected = False
        self.v_max = 350
        self.leaffall = 4
        self.inf_loss = 0.05


        # Mark tree site
        self.model.tree_sites[tuple(self.pos)] = True

    def grow(self):
        """
        Calculate tree growth for next timestep. 
        """

        v_current = self.volume
        r = self.model.calc_r(self.pos, self.v_max, True, self.volume)
        v_update = v_current * np.exp(r*(1-v_current/self.v_max)**4)

        self.volume = v_update


    def shed_leaves(self):
        """     
        Infected trees stochastically drop leaves on the grid
        and inoculate substrate on the forest floor.
        """
        substrate_cells = self.model.grid.properties['substrate'].select_cells(check_non_empty)

        for cell in substrate_cells:
            dist = self.calc_dist(cell, self.pos)

            # Inoculate substrate
            if np.random.random() < np.exp(-dist / self.dispersal_coeff):
                self.model.new_agent(Fungus, cell)


    def harvest(self):
        """
        Trees are stochastically harvested once they reach 
        a threshold volume and number of neighbors.
        """
        harvest_vol_threshold, harvest_nbrs_threshold, harvest_probability = self.model.harvest_params

        # volume threshold
        if self.volume > harvest_vol_threshold:
            # calculate number of neighbouring trees
            neighbouring_agents = self.model.grid.get_neighbors(tuple(self.pos), moore=True, include_center=False)
            count_trees = sum(agent.agent_type == "Tree" for agent in neighbouring_agents)

            # remove tree stochastically
            if count_trees >= harvest_nbrs_threshold:
                if random.random() < harvest_probability:
                    self.model.harvest_volume += self.volume
                    self.model.remove_agent(self)  
    

    def stochastic_removal(self):
        ''' Stochastic removal of trees: assuming they live on average of 120 years (= 4 * 120 = 480 timesteps)
        '''

        rate_parameter = 1 / 480
        p_die = 1 - np.exp(-rate_parameter)
        if np.random.random() < p_die:
            # Add substrate to the soil of dead tree
            coord = self.pos
            self.model.grid.properties['substrate'].data[tuple(coord)] += self.volume*0.1

            # Remove tree
            self.model.remove_agent(self)
            

    def step(self):
        """
        Step function for tree.
        """
        if self.infected:
            # leaffall once a year
            if self.model.schedule.time % self.leaffall == 0:
                self.shed_leaves()
            # stochastic loss of infection
            if random.random() < self.inf_loss:
                self.infected = False

        if self.harvest():
            return
        
        self.stochastic_removal()



class Fungus(Organism):
    """
    Fungus agent:
    Fungi consume substrate and convert it into soil fertility.
    They sporulate once a year and can infect trees and substrate in the process.
    """

    def __init__(self, unique_id, model, pos, disp, init_energy=1):
        super().__init__(unique_id, model, pos, disp)

        self.agent_type = 'Fungus'
        self.energy = init_energy

    def consume(self):
        """
        Consume substrate and convert into soil fertility.
        """
        x, y = self.pos
        substrate = self.model.grid.properties['substrate'].data[x, y]
        fertility = self.model.grid.properties['soil_fertility'].data[x, y]

        if substrate > 0:
            # eat wood
            self.model.grid.properties['substrate'].set_cell((x, y), substrate - 1)
            self.energy += 1
            # deposit fertility
            self.model.grid.properties['soil_fertility'].set_cell((x, y), fertility + 1)
        # consume reserve if no wood
        else:
            self.energy -= 1

    def infect_wood(self, cell):
        '''
        Try to infect woody debris.
        '''
        dist = self.calc_dist(cell, self.pos)

        # count fungi in cell to scale probability with space competition
        contents = self.model.grid.get_cell_list_contents(cell)
        fun_count = sum(1 for agent in contents if isinstance(agent, Fungus))

        # inoculation probability
        prob = 1 / (fun_count + 1) * np.exp(-dist / self.disp)

        # probabilistic infection
        if np.random.random() < prob:
            self.model.new_agent(Fungus, cell)


    def infect_tree(self, tree):
        '''
        Try to infect tree.
        '''
        dist = self.calc_dist(tree.pos, self.pos)
        prob = np.exp(-dist / self.disp)

        # probabilistic infection
        if np.random.random() < prob:
            tree.infected = True


    def sporulate(self):
        """
        Reproduce if enough energy.
        """
        self.energy -= 4

        # substrate infection
        substrate_layer = self.model.grid.properties['substrate']

        # filter cells for substrate
        def has_substrate(cell_value):
            return cell_value > 0

        woody_cells = substrate_layer.select_cells(has_substrate)
        for cell in woody_cells:
            # only infect at different location
            if cell[0] != self.pos[0] and cell[1] != self.pos[1]:
                self.infect_wood(cell)

        # tree infection
        trees = self.model.getall("Tree")
        for tree in trees:
            if not tree.infected:
                self.infect_tree(tree)


    def die(self):
        """
        Die if no energy or killed through environment
        """
        self.model.remove_agent(self)


    def stochastic_removal(self):
        ''' 
        Stochastic removal of fungi: assuming they live on average for 5 years (=20 timesteps = 20*3 months)
         and the probability of them dying during 5 years is 0.9 = 1−(prob not dying)^20
        -> then per timestep the probability of stochastic removal is approx 0.1
        '''
        if np.random.random() < 0.1:
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
            return

        self.stochastic_removal()
        
        