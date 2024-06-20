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


def plot_index(s, params):
    """
    Creates a plot for Sobol sensitivity analysis that shows the contributions
    of each parameter to the global sensitivity.

    Args:
        s (dict): nested dictionary {'output': {'S#': dict, 'S#_conf': dict}} that holds
            the values for a set of parameters of all outputs.
        params (list): the parameters taken from s
    """

    # Order of Si
    orders = ['1', '2', 'T']
    order_names = ['First', 'Second', 'Total']

    for output in s.keys():
        for i, order in enumerate(orders):
            if order_names[i] == 'Second':
                p = len(params)
                params_combo = list(combinations(params, 2))
                indices = s[output]['S' + order].reshape((p ** 2))
                indices = indices[~np.isnan(indices)]
                errors = s[output]['S' + order + '_conf'].reshape((p ** 2))
                errors = errors[~np.isnan(errors)]
            else:
                params_combo = params
                indices = s[output]['S' + order]
                errors = s[output]['S' + order + '_conf']
                plt.figure()

            l = len(indices)

            fig, ax = plt.subplots()
            fig.set_size_inches(6, 4)

            fig.suptitle(f'{output}: {order_names[i]} order sensitivity')
            ax.set_ylim([-0.2, len(indices) - 1 + 0.2])
            ax.set_yticks(range(l), params_combo)
            ax.errorbar(indices, range(l), xerr=errors, linestyle='None', marker='o')
            ax.axvline(0, c='k')

            fig.show()

def query_simulation_run(data, sim_id, outputs, problem):
    """
    Retrieve and plot the time series data for a specific simulation run.
    Args:
        data (pd.DataFrame): collected data from the model runs
        sim_id (int): the ID of the simulation run to plot
        outputs (list of str): list of output variable names to plot
        problem (dict): SALib problem dictionary
    """

    # Get the data for the specific run
    run_data = data[data['SimId'] == sim_id]
    run_data.reset_index(drop=True, inplace=True)

    # Print the parameter values for the run
    print(f'Parameter values for simulation {sim_id}:')
    print(run_data.iloc[0, :][['SimId'] + problem['names']])

    # Plot the time series data
    for output in outputs:
        fig, ax = plt.subplots()
        fig.suptitle(f'{output} for simulation {sim_id}')
        fig.set_size_inches(6, 4)
        ax.plot(run_data['Step'], run_data[output], label=output)
        ax.set_xlabel('Timestep')
        ax.set_ylabel(output)
        fig.show()