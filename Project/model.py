import numpy as np
import random
from mesa import Model
from mesa.space import MultiGrid, PropertyLayer
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
from agent import Tree, Fungus, Organism


class Forest(Model):

    def __init__(self, 
                 width, 
                 height, 
                 n_init_trees, 
                 n_init_fungi, 
                 harvest_params, 
                 fert_comp_ratio,
                 max_substrate=3, 
                 max_soil_fertility=3,
                 top_n_sites=5):

        super().__init__()

        self.height = width
        self.width = height
        self.v_max_global = 100 
        #self.r0_global = 1.05 # guess this isn't needed anymore
        self.harvest_params = harvest_params
        self.fert_comp_ratio = fert_comp_ratio

        # Top n sites to plant a tree based on fertility and competition
        # TO DO: Make this a percentage of lattice sites relative to the grid size
        self.top_n_sites = top_n_sites

        # Initialize harvested volume
        self.harvest_volume = 0

        # Create schedule
        self.schedule = RandomActivation(self)
        
        self.grid = MultiGrid(self.width, self.height, torus=True)

        # Add initial substrate
        self.grid.add_property_layer(PropertyLayer('substrate', self.width, self.height, 1))
        self.grid.properties['substrate'].data = np.random.randint(0, max_substrate, (self.width, self.height))

        # Add initial soil fertility
        self.grid.add_property_layer(PropertyLayer('soil_fertility', self.width, self.height, 1))
        self.grid.properties['soil_fertility'].data = np.random.uniform(0, max_soil_fertility,
                                                                        (self.width, self.height))

        self.datacollector = DataCollector(
             {"Trees": lambda m: len(self.getall("Tree")),
              "Fungi": lambda m: len(self.getall("Fungus")),
              "Living Trees Total Volume": lambda m: sum([agent.volume for agent in self.getall("Tree")]),
              "Infected Trees": lambda m: sum([agent.infected for agent in self.getall("Tree")]),
              "Mean Substrate": lambda m: np.mean(self.grid.properties['substrate'].data),
              "Substrate Variance": lambda m: np.var(self.grid.properties['substrate'].data),
              "Mean Soil Fertility": lambda m: np.mean(self.grid.properties['soil_fertility'].data),
              "Soil Fertility Variance": lambda m: np.var(self.grid.properties['soil_fertility'].data),
              "Harvested volume": lambda m: m.harvest_volume})
        
        # Initialise populations
        self.init_population(n_init_trees, Tree, (5, 30), 4)
        self.init_population(n_init_fungi, Fungus, (1, 3), 1)

        self.running = True
        self.datacollector.collect(self)


    def init_population(self, n_agents, agent_type, init_size_range, dispersal_coeff=1):
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
            self.new_agent(agent_type, coord, np.random.randint(init_size_range[0], init_size_range[1] + 1), dispersal_coeff)
            # params = [coord, np.random.randint(init_size_range[0], init_size_range[1] + 1), dispersal_coeff]


    def new_agent(self, agent_type, pos, init_size=1, disp=1):
        """
        Method that enables us to add agents of a given type.
        """

        # Create a new agent of the given type
        new_agent = agent_type(self.next_id(), self, pos, disp, init_size)

        # Add agent to schedule
        self.schedule.add(new_agent)
        
    
    def find_agent(self, agent):
        # This method should return the position of the agent or None if the agent is not found
        for x in range(self.width):
            for y in range(self.height):
                if agent in self.grid[x][y]:
                    return (x, y)
        return None
    
    
    def remove_agent(self, agent):
        """
        Method that enables us to remove passed agents.
        """

        # Remove agent from grid
        self.grid.remove_agent(agent)

        # Remove agent from schedule
        self.schedule.remove(agent)

    
    def calc_dist(self, pos1, pos2):
        """
        Method that calculates the Euclidean distance between two points.
        """
        return np.sqrt((pos1[..., 0] - pos2[..., 0]) ** 2 + (pos1[..., 1] - pos2[..., 1]) ** 2)


    def calc_fert(self, pos, v_self, v_max, Grow:bool):
        """
        Method that calculates the fertility of the soil at position pos and consumes if tree is growing
        """
        coord_nbrs = self.grid.get_neighborhood(tuple(pos), moore=True, include_center=False)

        # fert_self = min((1,self.grid.properties['soil_fertility'].data[tuple(pos)]))
        # fert_nbrs = [min(0.5, self.grid.properties['soil_fertility'].data[tuple(coord)]) for coord in coord_nbrs]
        # fert_nbrs_sum = sum(fert_nbrs)
        # f_c = v_self/v_max * (fert_self + fert_nbrs_sum)

        fert_self = min((v_self/v_max*2,self.grid.properties['soil_fertility'].data[tuple(pos)]))
        fert_nbrs = [min(v_self/v_max*1, self.grid.properties['soil_fertility'].data[tuple(coord)]) for coord in coord_nbrs]
        fert_nbrs_sum = sum(fert_nbrs)
        f_c = (fert_self + fert_nbrs_sum)

        if Grow:
            self.grid.properties['soil_fertility'].data[tuple(pos)] -= fert_self
            for idx,coord in enumerate(coord_nbrs):
                self.grid.properties['soil_fertility'].data[tuple(coord)] -= fert_nbrs[idx]

        return f_c


    def calc_comp(self, pos, v_self):
        """
        Method that calculates the competition of the position pos
        """
        nbr_agents = self.grid.get_neighbors(tuple(pos), moore=True, include_center=False)

        vol_self = v_self
        vol_nbrs = sum([agent.volume for agent in nbr_agents if isinstance(agent, Tree)])

        competition = vol_nbrs / (vol_self + vol_nbrs)
        return competition


    #def calc_r(self, pos, r0, v_max, grow=True, v_self=1):
    def calc_r(self, pos, v_max, grow=True, v_self=1):
        """
        Methods calculates the r_effective of the position pos
        Args:
            pos (tuple): position of the agent
            r0 (float): base growth rate
            v_max (float): maximum volume of a tree
            v_self (float): volume of the agent
        """

        beta = 0.1
        alpha = self.fert_comp_ratio * beta

        f_c = self.calc_fert(pos, v_self, v_max, grow)
        comp = self.calc_comp(pos, v_self)
        F = f_c / (f_c + 10)

        r = beta + alpha * F - beta * comp 
        return r


    def getall(self, typeof):
        if not any([agent.agent_type == typeof for agent in self.schedule.agents]):
            return ([])
        else:
            return [agent for agent in self.schedule.agents if agent.agent_type == typeof]

    def add_substrate(self):
        """
        Stochastically adds substrate (woody debris)
        based on the distance to all trees in the lattice.
        On average, 2.5*1e-4 of the tree biomass is added per time step.
        """

        coords = np.transpose(np.indices((self.width, self.height)), (1, 2, 0))

        # Assign probabilities to all lattice sites
        lattice_probs = np.zeros((self.width, self.height))
        for tree in self.getall("Tree"):
            lattice_tree_dist = self.calc_dist(np.array(tree.pos), coords)
            lattice_probs += np.exp(-lattice_tree_dist / (tree.volume ** (1 / 3)))

        # Normalize probabilities
        lattice_probs /= np.sum(lattice_probs)

        # Distribute substrate
        total_volume = sum([agent.volume for agent in self.getall("Tree")])
        n_portions = int(total_volume / 1.2e5 * 100)
        print(n_portions)

        # Lattice sites to add substrate to
        coords_idx_select = np.random.choice(np.arange(self.width * self.height), n_portions, replace=True,
                                                p=lattice_probs.flatten())
        coords_select = coords.reshape(-1, 2)[coords_idx_select]

        for coord in coords_select:
            self.grid.properties['substrate'].data[tuple(coord)] += 1


    def plant_trees(self):
        if self.schedule.steps % 4 == 0:  # Every 4 time steps i.e plantation every year
            self.plant_trees_top_r()


    def plant_trees_top_r(self):
        # Calculate r_effective for all positions
        
        all_positions = []
        for x in range(self.width):
            for y in range(self.height):
                cell_agents = self.grid.get_cell_list_contents([x,y])
                if not any(isinstance(agent, Tree) for agent in cell_agents):
                    all_positions.append((x,y))

        random.shuffle(all_positions)

        r_effective_values = [(pos, self.calc_r(pos, self.v_max_global, grow=False)) 
                              for pos in all_positions]

        # Sort positions by r_effective
        r_effective_values.sort(key=lambda x: x[1], reverse=True)

        # Plant trees in top n positions
        for pos, _ in r_effective_values[:self.top_n_sites]:
            self.new_agent(Tree, pos, init_size=1, disp=1)


    def step(self):
        """
        Method that calls the step method for trees and fungi in randomized order.
        """

        # Zero harvest volume
        self.harvest_volume = 0

        self.add_substrate()
        self.schedule.step()
        self.plant_trees()
        # Save statistics
        self.datacollector.collect(self)


    def run_model(self, n_steps=100):
        """
        Method that runs the model for a given number of steps.
        """
        for i in range(n_steps):
            self.step()
