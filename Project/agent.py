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
    General class for all organisms in the model.
    """

    def __init__(self, unique_id, model, pos, disp):
        super().__init__(unique_id, model)

        self.disp = disp
        self.model.grid.place_agent(self, pos)
        self.pos = pos

    def calc_dist(self, pos1, pos2):
        """
        Calculate the distance between two positions.
        """
        x1, y1 = pos1
        x2, y2 = pos2

        return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


class Tree(Organism):
    """
    A tree agent.
    """

    def __init__(self, unique_id, model, pos, disp, init_volume=1):
        super().__init__(unique_id, model, pos, disp)

        self.agent_type = 'Tree'
        self.volume = init_volume
        self.dispersal_coeff = 4
        self.infected = False
        self.v_max = 350
        self.leaffall = 4

        # Mark tree site
        self.model.tree_sites[tuple(self.pos)] = True

    def grow(self):
        """
        Grow the tree.
        """

        v_current = self.volume
        r = self.model.calc_r(self.pos, self.v_max, True, self.volume)
        v_update = v_current * np.exp(r*(1-v_current/self.v_max)**4)

        self.volume = v_update

    def shed_leaves(self):
        """
        Shed leaves.
        TODO
        - don't scan entire lattice
        """
        substrate_cells = self.model.grid.properties['substrate'].select_cells(check_non_empty)

        for cell in substrate_cells:
            dist = self.calc_dist(cell, self.pos)

            # Inoculate substrate
            if np.random.random() < np.exp(-dist / self.dispersal_coeff):
                self.model.new_agent(Fungus, cell)
        # Scan all substrate on lattice
        # for x in range(self.model.width):
        #     for y in range(self.model.height):
        #         if self.model.grid.properties['substrate'].data[x, y] > 0:

        #             dist = self.calc_dist((x, y), self.pos)

        #             # Inoculate substrate
        #             if np.random.random() < np.exp(-dist / self.dispersal_coeff):
        #                 self.model.new_agent(Fungus, (x, y))

    def harvest(self):
        """
        A tree 'harvests itself' if:
        If the volume is above a threshold and if x percent of the surrounding 8 trees are still present
            -> can be harvested with probability p
            
        TODO
        - finish up + remove returns
        """
        harvest_vol_threshold, harvest_nbrs_threshold, harvest_probability = self.model.harvest_params

        # Check volume threshold
        if self.volume > harvest_vol_threshold:

            # Check percentage of surrounding trees threshold
            neighbouring_agents = self.model.grid.get_neighbors(tuple(self.pos), moore=True, include_center=False)
            count_trees = sum(agent.agent_type == "Tree" for agent in neighbouring_agents)

            if count_trees >= harvest_nbrs_threshold:
                # Include a random probability
                if random.random() < harvest_probability:
                    self.model.harvest_volume += self.volume
                    self.model.remove_agent(self)  # Remove the tree
                    return True
        return False  # Tree is not harvested

    def stochastic_removal(self):
        ''' Stochastic removal of trees: assuming they live on average of 120 years (= 4 * 120 = 480 timesteps)
        '''

        rate_parameter = 1 / 480
        p_die = 1 - np.exp(-rate_parameter)
        if np.random.random() < p_die:
            # Add substrate to the soil of dead tree
            coord = self.pos
            self.model.grid.properties['substrate'].data[tuple(coord)] += self.volume#1

            # Remove tree
            self.model.remove_agent(self)
            

    def step(self):
        """
        Tree development step.
        """
        self.grow()

        if self.infected and self.model.schedule.time % self.leaffall == 0:
            self.shed_leaves()

        if self.harvest():
            return
        
        self.stochastic_removal()


class Fungus(Organism):
    """
    A fungus agent.
    """

    def __init__(self, unique_id, model, pos, disp, init_energy=1):
        super().__init__(unique_id, model, pos, disp)

        self.agent_type = 'Fungus'
        self.energy = init_energy

    def consume(self):
        """
        Consume substrate.
        """
        x, y = self.pos
        substrate = self.model.grid.properties['substrate'].data[x, y]
        fertility = self.model.grid.properties['soil_fertility'].data[x, y]

        # eat wood + deposit fertility
        if substrate > 0:
            # self.model.grid.properties['substrate'].set_cell((x, y), max(0, substrate - 1))
            self.model.grid.properties['substrate'].set_cell((x, y), substrate - 1)
            self.model.grid.properties['soil_fertility'].set_cell((x, y), fertility + 1)
            self.energy += 1
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
            # self.model.new_agent(Fungus, (x, y))
            self.model.new_agent(Fungus, cell)

    def infect_tree(self, tree):
        '''
        Try to infect tree.
        '''
        dist = self.calc_dist(tree.pos, self.pos)
        prob = np.exp(-dist / self.disp)

        # print(prob)

        # probabilistic infection
        if np.random.random() < prob:
            # print("Tree infected!")
            tree.infected = True

    def sporulate(self):
        """
        Reproduce if enough energy.
        """
        self.energy -= 4

        # substrate infection
        substrate_layer = self.model.grid.properties['substrate']

        # select cells that have substrate
        def has_substrate(cell_value):
            return cell_value > 0

        woody_cells = substrate_layer.select_cells(has_substrate)
        for cell in woody_cells:
            # only infect at different location
            if cell[0] != self.pos[0] and cell[1] != self.pos[1]:
                self.infect_wood(cell)

        # tree infection
        trees = self.model.getall("Tree")
        # print(trees)
        for tree in trees:
            if not tree.infected:
                self.infect_tree(tree)

    def die(self):
        """
        Die if no energy or killed through environment
        """
        self.model.remove_agent(self)


    def stochastic_removal(self):
        ''' Stochastic removal of fungi: assuming they live on average for 5 years (=20 timesteps = 20*3 months)
         and the probability of them dying during 5 years is 0.9 = 1−(prob not dying)^20
        -> then per timestep the probability of stochastic removal is approx 0.1'''
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
        
        