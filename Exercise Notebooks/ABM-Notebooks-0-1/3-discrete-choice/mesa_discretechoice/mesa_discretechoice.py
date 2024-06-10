import numpy as np
from mesa import Model
from mesa.space import SingleGrid
from mesa.time import RandomActivation
import statsmodels.api as sm
from agent import Player
import random


class MesaDiscretechoice(Model):
    def __init__(self, height=30, width=30, players=180, dataset='direction_choices.csv'):
        super().__init__()

        self.height = height
        self.width = width

        # add a scheduler
        self.scheduler = RandomActivation(self)

        # add a grid
        self.grid = SingleGrid(self.width, self.height, torus=True)

        # fit the multinomial logit according to dataset
        # (default dataset is neighbor avoidance)
        data = np.genfromtxt(dataset, delimiter=';')
        mnLogit = sm.MNLogit(data[:, -1], data[:, :-1])
        self.stats_model = mnLogit.fit(disp=0)

        # random players
        for i in range(players):
            x = i % width
            y = i // height
            agent = Player(self.next_id(), self)
            self.scheduler.add(agent)
            self.grid.position_agent(agent, x=x, y=y)

    def step(self):
        """
        step through 10 agents and step them
        """
        for i in range(10):
            agent = self.scheduler._agents[random.choice(list(self.scheduler._agents.keys()))]
            agent.step()

    def run_model(self, step_count=200):
        """
        Method that runs the model for a specific amount of steps.
        """
        for i in range(step_count):
            self.step()
