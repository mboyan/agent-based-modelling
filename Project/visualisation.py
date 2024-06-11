import matplotlib.pyplot as plt
import numpy as np

from mesa.experimental import JupyterViz
from agent import Tree, Fungus, Organism

def agent_portrayal(agent):
    if isinstance(agent, Fungus):
        size = 10
        color = "tab:blue"
        alpha = agent.energy / 4
    if isinstance(agent, Tree):
        size = agent.volume
        color = "tab:green"
        alpha = 1
    return {"size": size, "color": color, "alpha": alpha}


def create_jupyter_viz(model, model_params, measures):
    """
    Function to create a JupyterViz object for the model.
    """

    page = JupyterViz(model, model_params, measures, name="FA Model", agent_portrayal=agent_portrayal)

    return page