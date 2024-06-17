import numpy as np
from mesa import Agent
import random


class Organism(Agent):
    """
    General class for all organisms in the model.
    """

    def __init__(self, unique_id, model, pos, disp):
        super().__init__(unique_id, model)

        self.disp = disp
        self.model.grid.place_agent(self, pos)
        self.pos = pos


class Tree(Organism):
    """
    A tree agent.
    """

    def __init__(self, unique_id, model, pos, disp, init_volume=1, base_growth_rate=1.05):
        super().__init__(unique_id, model, pos, disp)

        self.agent_type = 'Tree'

        self.volume = init_volume
        self.dispersal_coeff = 4
        self.infected = False
        self.v_max = 100
        self.base_growth_rate = base_growth_rate # Must be at least 1.05 to avoid negative r_effective
        self.leaffall = 4

    def grow(self):
        """
        Grow the tree.
        """

        v_current = self.volume
        r = self.model.calc_r(self.pos, self.base_growth_rate, self.v_max, self.volume)
        v_update = v_current * np.exp(r * np.log(self.v_max/self.volume))

        self.volume = v_update

        if r < 0:
            print(f"Warning! Negative growth rate! r={r}")

    def shed_leaves(self):
        """
        Shed leaves.
        """
        # Scan all substrate on lattice
        for x in range(self.model.width):
            for y in range(self.model.height):
                if self.model.grid.properties['substrate'].data[x, y] > 0:

                    dist = np.sqrt((x - self.pos[0]) ** 2 + (y - self.pos[1]) ** 2)

                    # Inoculate substrate
                    if np.random.random() < np.exp(-dist / self.dispersal_coeff):
                        self.model.new_agent(Fungus, (x, y))

    def harvest(self):
        """
        A tree 'harvests itself' if:
        If the volume is above a threshold and if x percent of the surrounding 8 trees are still present
            -> can be harvested with probability p
        """
        harvest_vol_threshold, harvest_percent_threshold, harvest_probability = self.model.harvest_params

        # Check volume threshold
        if self.volume > harvest_vol_threshold:

            # Check percentage of surrounding trees threshold
            neighbouring_agents = self.model.grid.get_neighbors(tuple(self.pos), moore=True, include_center=False)
            count_trees = len([agent for agent in neighbouring_agents if agent.agent_type == "Tree"])

            if count_trees / 8 > harvest_percent_threshold:
                # Include a random probability
                if random.random() < harvest_probability:
                    self.model.harvest_volume += self.volume
                    self.model.remove_agent(self)  # Remove the tree
                    return True
        return False  # Tree is not harvested

    def step(self):
        """
        Tree development step.
        """
        self.grow()

        if self.infected and self.model.schedule.time % self.leaffall == 0:
            self.shed_leaves()

        self.harvest()


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
        x, y = cell
        dist = np.sqrt((x - self.pos[0]) ** 2 + (y - self.pos[1]) ** 2)

        # count fungi in cell to scale probability with space competition
        contents = self.model.grid.get_cell_list_contents(cell)
        fun_count = len([agent for agent in contents if type(agent) == Fungus])

        # inoculation probability
        prob = 1 / (fun_count + 1) * np.exp(-dist / self.disp)

        # probabilistic infection
        if np.random.random() < prob:
            # self.model.new_agent(Fungus, (x, y))
            self.model.new_agent(Fungus, (x, y))

    def infect_tree(self, tree):
        '''
        Try to infect tree.
        '''
        x, y = tree.pos
        dist = np.sqrt((x - self.pos[0]) ** 2 + (y - self.pos[1]) ** 2)
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

    def step(self):
        """
        Fungus self-development step.
        """
        self.consume()

        if self.energy > 4:
            self.sporulate()
        elif self.energy < 1:
            self.die()
