import numpy as np
import random
import heapq
from mesa import Model
from mesa.space import MultiGrid, PropertyLayer
from mesa.datacollection import DataCollector
from mesa.time import RandomActivation
from agent import Tree, Fungus, Organism




class Forest(Model):
    """
    Represents a forest model with tree and fungus agents.
    The model manages agent progression, the forest environment 
    as well as initialization, timestepping and datacollection.
    
    Args
    ---------------------------------------------------
    width/height : ints
        Forest grid dimensions.
    n_init_trees : int
        Number of trees the forest is initialized with.
    n_init_fungi : int
        Number of fungi the forest is initialized with.
    harvest_params : list
        Default harvesting threshold parameters.
        Only used if no separate parameters are supplied.
    fert_comp_ratio_exponent : float
        Ratio between fertility and competition in tree growth calculations.
    max_substrate : int
        Max substrate any grid cell can collect.
    max_soil_fertility : int
        Max soil fertility any grid cell can reach.
    top_n_sites_percent : float
        Planting percentage parameter.
    harvest_volume : int 
        Threshold volume for tree harvesting.
    harvest_nbrs : float
        Threshold number of neighbors for tree harvesting.
    harvest_prob : float
        Undefined environmental harvesting probability.
    
    Additional attributes
    ---------------------------------------------------
    v_max_global : int
        Model wide maximum volume trees can reach.
    harvest_volume : float
        Volume of trees that has been harvested in each timestep
    schedule : RandomActivation
        Forest schedule with all agents.
    grid : MultiGrid
        Forest lattice.
    datacollector : DataCollector
        Object that keeps track of relevant data during simulation.
    tree_sites : array
        Keeps track of locations of trees on forest lattice.
    
    Methods
    ---------------------------------------------------
    init_population()
        Initializes the tree and fungus agent populations of the forest.
    new_agent()
        Adds new agent to forest grid and model schedule.
    find_agent()
        Given an agent, returns its position.
    remove_agent()
        Given an agent, removes it from the forest grid and model schedule.
    calc_dist()
        Calculates the Euclidean distance between two forest grid points.
    calc_fert()
        Calculates the fertility of the soil at input position and consumes if a tree is growing.
    calc_comp()
        Calculates the competition term at a given grid point.
    calc_r()
        Calculates the effective growth rate at input position.
    getall()
        Creates a list of agents of a given type.
    add_substrate()
        Stochastically adds substrate (woody debris) based on the distance to all trees in the lattice.
    plant_trees()
        Plants trees at unoccupied lattice sites based on growth potential ranking.
    step()
        Executes a single step of the model's behavior. This includes all agents, harvesting, 
        planting and datacollection.
    run_model()
        Runs the model for a given number of steps.
    """

    def __init__(self, 
                 width=20, 
                 height=20, 
                 n_init_trees=100, 
                 n_init_fungi=50, 
                 harvest_params=[150,4,0.5], 
                 fert_comp_ratio_exponent=-0.3,
                 max_substrate=3, 
                 max_soil_fertility=1,
                 top_n_sites_percent=0.05,
                 harvest_volume=None,
                 harvest_nbrs=None,
                 harvest_prob=None):

        super().__init__()

        # Overwrite harvest parameters if passed
        if harvest_volume is not None:
            harvest_params[0] = harvest_volume
        if harvest_nbrs is not None:
            harvest_params[1] = harvest_nbrs
        if harvest_prob is not None:
            harvest_params[2] = harvest_prob

        self.height = height
        self.width = width
        self.v_max_global = 350
        self.harvest_params = harvest_params
        self.fert_comp_ratio = 10**fert_comp_ratio_exponent

        # Top n sites to plant a tree based on fertility and competition
        self.top_n_sites = int(top_n_sites_percent * self.width * self.height)

        # Initialize harvested volume
        self.harvest_volume = 0

        # Create schedule
        self.schedule = RandomActivation(self)
        
        self.grid = MultiGrid(self.width, self.height, torus=True)

        # Add initial substrate
        self.grid.add_property_layer(PropertyLayer('substrate', self.width, self.height, 1))
        self.grid.properties['substrate'].data = np.random.randint(1, max_substrate, (self.width, self.height))

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
        
        # Initialize tree_sites array
        self.tree_sites = np.zeros((self.width, self.height), dtype=bool)

        # Initialise populations
        self.init_population(n_init_trees, Tree, (5, 270), 4)
        self.init_population(n_init_fungi, Fungus, (1, 3), 1)

        self.running = True
        self.datacollector.collect(self)


    def init_population(self, n_agents, agent_type, init_size_range, dispersal_coeff=1):
        """
        Method that initializes the population of trees and fungi.
        
        Args:
        --------
        n_agents : int
            Number of agents to add.
        agent_type : string
            Class of agent to add to the model.
        init_size_range : tuple
            range of agent sizes [min, max] -
            volume for trees and energy for fungi.
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


    def new_agent(self, agent_type, pos, init_size=1, disp=1):
        """
        Method that enables us to add agents of a given type.
        
        Args:
        --------
        agent_type : string
            Type of agent to be added to the model.
        pos : tuple
            Location at which agent is to be added.
        init_size : int
            Initial size/energy of agent.
        disp : float
            Dispersal coefficient of agent.
        """
        # Create a new agent of the given type and add to model schedule.
        new_agent = agent_type(self.next_id(), self, pos, disp, init_size)
        self.schedule.add(new_agent)

    
    def find_agent(self, agent):
        """
        Given an agent, returns its position.
        
        Args:
        --------
        agent : Agent
            Tree or fungus agent.
            
        Returns:
        --------
        pos : tuple
            Position of input agent, if it has one, otherwise None.
        """
        for x in range(self.width):
            for y in range(self.height):
                if agent in self.grid[x][y]:
                    return (x, y)
        return None
    
    
    def remove_agent(self, agent):
        """
        Given an agent, removes it from the forest grid and model schedule.
        
        Args:
        --------
        agent : Agent
            Tree or fungus agent.
        """
        # If tree, remove from tree_sites
        if isinstance(agent, Tree):
            self.tree_sites[tuple(agent.pos)] = False

        # Remove agent from grid and schedule
        self.grid.remove_agent(agent)
        self.schedule.remove(agent)
        
    
    def calc_dist(self, pos1, pos2):
        """
        Calculates the Euclidean distance between two forest grid points.
        
        Args:
        --------
        pos1/pos2 : tuples
            Two distinct grid points in the forest.
            
        Returns:
        --------
        dist : float
            Distance between two input grid points. 
        """
        return np.sqrt((pos1[..., 0] - pos2[..., 0]) ** 2 + (pos1[..., 1] - pos2[..., 1]) ** 2)


    def calc_fert(self, pos, v_self, v_max, Grow:bool):
        """
        Method that calculates the fertility of the soil at position pos and consumes if tree is growing
        
        Args:
        --------
        pos : tuple
            Position at which to do calculation.
        v_self : float
            Volume of current tree.
        v_max : float
            Maximum volume a tree can reach.
        Grow : bool
            If function is called when growing a tree, set to True, otherwise False
            
        Returns:
        --------
        f_c : float
            Fertility consumed by tree.
        """
        coord_nbrs = self.grid.get_neighborhood(tuple(pos), moore=True, include_center=False)

        # Calculate fertility to be consumed at tree's own position and it's Moore neighborhood
        fert_self = min((v_self/v_max*2,self.grid.properties['soil_fertility'].data[tuple(pos)]))
        fert_nbrs = [min(v_self/v_max*1, self.grid.properties['soil_fertility'].data[tuple(coord)]) for coord in coord_nbrs]
        # Calculate total fertility consumed.
        fert_nbrs_sum = sum(fert_nbrs)
        f_c = (fert_self + fert_nbrs_sum)

        # When used to grow a tree, remove fertility from lattice sites.
        if Grow:
            self.grid.properties['soil_fertility'].data[tuple(pos)] -= fert_self
            for idx,coord in enumerate(coord_nbrs):
                self.grid.properties['soil_fertility'].data[tuple(coord)] -= fert_nbrs[idx]

        return f_c


    def calc_comp(self, pos, v_self):
        """
        Calculates the competition term at a given grid point.
        
        Args:
        --------
        pos : tuple
            Location of grid point.
        v_self : float
            Volume of tree at location.
            
        Returns:
        --------
        competition : float
            Calculated competition term at grid point.
        """
        nbr_agents = self.grid.get_neighbors(tuple(pos), moore=True, include_center=False)

        # Calculate tree volume in Moore neighborhood
        vol_nbrs = sum([agent.volume for agent in nbr_agents if isinstance(agent, Tree)])

        # Calculate competition term
        return vol_nbrs / (v_self + vol_nbrs)


    def calc_r(self, pos, v_max, grow=True, v_self=1):
        """
        Calculates the effective growth rate at input position.
        
        Args:
        --------
        pos : tuple
            Position of the input agent.
        r0 : float
            Base growth rate of tree.
        v_max : float
            Maximum volume a tree can reach.
        v_self : float
            Current volume of tree agent.
            
        Returns:
        --------
        r : float
            Calculated effective growth rate of (potential) tree.
        """
        # Set fertility and competition term based on fert/comp ratio 
        beta = 0.1
        alpha = self.fert_comp_ratio * beta

        # Calculate consumed fertility and competition term
        f_c = self.calc_fert(pos, v_self, v_max, grow)
        comp = self.calc_comp(pos, v_self)
        F = f_c / (f_c + 10)

        # Return effective growth rate
        return beta + alpha * F - beta * comp 


    def getall(self, typeof):
        """
        Creates a list of agents of a given type.
        
        Args:
        --------
        typeof : string
            Agent type to be checked for
            
        Returns:
        --------
        agents : list
            List of agents of given type.
        """
        # Return empty list when no agents of a given type are present
        if not any([agent.agent_type == typeof for agent in self.schedule.agents]):
            return ([])
        # Create list of agents of a given type
        else:
            return [agent for agent in self.schedule.agents if agent.agent_type == typeof]

    def add_substrate(self):
        """
        Stochastically adds substrate (woody debris) based on the distance to all trees in the lattice.
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
        n_portions = int(total_volume / 1.2e5 * 75) # Based on paper

        # Lattice sites to add substrate to
        coords_idx_select = np.random.choice(np.arange(self.width * self.height), n_portions, replace=True,
                                                p=lattice_probs.flatten())
        coords_select = coords.reshape(-1, 2)[coords_idx_select]

        # Add substrate to the selected positions
        for coord in coords_select:
            self.grid.properties['substrate'].data[tuple(coord)] += 1


    def plant_trees(self):
        """
        Plants trees at unoccupied lattice sites based on growth potential ranking. 
        """
        # Get and shuffle empty positions
        empty_positions = np.where(self.tree_sites == False)
        empty_positions = list(zip(empty_positions[0], empty_positions[1]))
        random.shuffle(empty_positions)
        
        # Select a batch of candidates if applicable
        candidate_positions = empty_positions[:min(len(empty_positions), self.top_n_sites * 2)]  # Adjust multiplier based on expected density of trees
        
        # Calculate r_effective for candidates and use a heap to find top n
        heap = []
        for pos in candidate_positions:
            r_effective = self.calc_r(pos, self.v_max_global, grow=False)
            # Use negative r_effective because heapq is a min-heap, but we need max values
            heapq.heappush(heap, (-r_effective, pos))
            if len(heap) > self.top_n_sites:
                heapq.heappop(heap)
        
        # Plant trees in top n positions
        for _, pos in heap:
            self.new_agent(Tree, pos, init_size=1, disp=1)


    def step(self):
        """
        Executes a single step of the model's behavior. This includes all agents,
        harvesting, planting and datacollection.
        """

        # Reset harvest volume
        self.harvest_volume = 0

        # Add substrate to grid and perform a step for all agents
        self.add_substrate()
        self.schedule.step()
        
        # Plant new trees once a year in spring.
        if self.schedule.steps % 4 == 0:
            self.plant_trees()
            
        # Save statistics
        self.datacollector.collect(self)


    def run_model(self, n_steps=100):
        """
        Runs the model for a given number of steps.
        """
        for _ in range(n_steps):
            self.step()
