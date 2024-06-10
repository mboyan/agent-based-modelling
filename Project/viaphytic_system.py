from mesa import Model, Agent
from mesa.space import MultiGrid


class Forest(Model):
    def __init__(self, width, height):
        self.height = width
        self.width = height
        
        self.grid = MultiGrid(self.width, self.height, torus=True)
        
        self.n_agents = 0
        self.agents = []

    
    def new_agent(self, agent_type, pos):
        '''
        Method that enables us to add agents of a given type.
        '''
        self.n_agents += 1
        
        # Create a new agent of the given type
        new_agent = agent_type(self.n_agents, self, pos)
        
        # Place the agent on the grid
        self.grid.place_agent(new_agent, pos)
        
        # And add the agent to the model so we can track it
        self.agents.append(new_agent)
    

    def remove_agent(self, agent):
        '''
        Method that enables us to remove passed agents.
        '''
        self.n_agents -= 1
        
        # Remove agent from grid
        self.grid.remove_agent(agent)
        
        # Remove agent from model
        self.agents.remove(agent)
    
    
    def step(self):
        '''
        Method that steps every agent. 
        
        Prevents applying step on new agents by creating a local list.
        '''
        for agent in list(self.agents):
            agent.step()


class Organism(Agent):
    '''
    General class for all organisms in the model.
    '''
    
    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model)

        self.pos = pos


class Tree(Organism):
    '''
    A tree agent.
    '''

    def __init__(self, unique_id, model, pos, start_biomass):
        super().__init__(unique_id, model, pos)

        self.biomass = start_biomass
        self.infected = False


    def grow(self):
        '''
        Grow the tree.
        '''
        self.biomass += 1


class Fungus(Organism):
    '''
    A fungus agent.
    '''
    
    def __init__(self, unique_id, model, pos):
        super().__init__(unique_id, model, pos)
    
