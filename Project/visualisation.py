import matplotlib.pyplot as plt
import numpy as np
import agent as agt

from mesa.experimental import JupyterViz
from itertools import combinations

'''
Analysis stuff
- phase space plot
- bifurcation analysis
'''

def agent_portrayal(agent):
    size = 5
    color = "black"
    alpha = 1
    if agent.agent_type == "Fungus":
        size = 4
        color = "tab:blue"
        alpha = agent.energy / 4
    if agent.agent_type == "Tree":
        size = agent.volume
        if agent.infected:
            color = "tab:olive"
        else:
            color = "tab:green"
        alpha = 1
    return {"size": size, "color": color, "alpha": alpha}


def create_jupyter_viz(model, model_params, measures):
    """
    Function to create a JupyterViz object for the model.
    """

    page = JupyterViz(model, model_params, measures, name="FA Model", agent_portrayal=agent_portrayal)

    return page


def plot_property_layer(model, layer_name):
    """
    Function to plot a property layer of the model.
    """

    plt.imshow(model.grid.properties[layer_name].data, cmap='viridis')
    plt.title(layer_name)
    plt.colorbar()
    plt.show()


def plot_index(s, params, title=''):
    """
    Creates a plot for Sobol sensitivity analysis that shows the contributions
    of each parameter to the global sensitivity.

    Args:
        s (dict): nested dictionary {'output': {'S#': dict, 'S#_conf': dict}} that holds
            the values for a set of parameters of all outputs.
        params (list): the parameters taken from s
        title (str): title for the plot
    """

    # Order of Si
    orders = ['1', '2', 'T']
    order_names = ['First', 'Second', 'Total']

    for output in s.keys():
        for i, order in enumerate(orders):
            print('=========')
            if i == 'Second':
                p = len(params)
                params = list(combinations(params, 2))
                print('=========')
                print(s['output'])
                indices = s[output]['S' + order].reshape((p ** 2))
                indices = indices[~np.isnan(indices)]
                errors = s[output]['S' + order + '_conf'].reshape((p ** 2))
                errors = errors[~np.isnan(errors)]
                print(indices.shape)
                print(errors.shape)
            else:
                print(s[output])
                indices = s[output]['S' + order]
                errors = s[output]['S' + order + '_conf']
                plt.figure()

            print(indices)
            l = len(indices)

            plt.title(f'{order_names[i]} order sensitivity')
            plt.ylim([-0.2, len(indices) - 1 + 0.2])
            plt.yticks(range(l), params)
            plt.errorbar(indices, range(l), xerr=errors, linestyle='None', marker='o')
            plt.axvline(0, c='k')

            plt.show()