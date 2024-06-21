import matplotlib.pyplot as plt
import numpy as np
import agent as agt

from mesa.experimental import JupyterViz
from itertools import combinations
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

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

    # Plot Trees/Fungi phase space
    fig, ax = plt.subplots()
    fig.set_size_inches(6, 5)
    ax.plot(run_data['Trees'].values, run_data['Fungi'].values)
    ax.set_xlabel('Trees')
    ax.set_ylabel('Fungi')
    ax.set_title('Trees/Fungi phase space')
    fig.show()

    # Plot the time series data
    for output in outputs:
        fig, ax = plt.subplots()
        fig.suptitle(f'{output} for simulation {sim_id}')
        fig.set_size_inches(6, 4)
        ax.plot(run_data['Step'], run_data[output], label=output)
        ax.set_xlabel('Timestep')
        ax.set_ylabel(output)
        fig.show()


def plot_param_space(data, output):
    """
    Plot specific outputs in a sampled 2D parameter space.
    Args:
        data (pd.DataFrame): collected data from the model runs
        output (str): name of the output of interest
    """

    fig, ax = plt.subplots()
    fig.set_size_inches(6, 5)

    max_sims = data['SimId'].max() + 1

    harvest_params = []
    planting_params = []
    mean_outputs = []

    for sim_id in range(max_sims):
        data_subset = data[data['SimId'] == sim_id]
        mean_output = data_subset[output].mean()
        mean_outputs.append(mean_output)
        
        harvest_volume = data_subset['harvest_volume'].values[0]
        harvest_nbrs = data_subset['harvest_nbrs'].values[0]
        harvest_prob = data_subset['harvest_prob'].values[0]
        harvest_param = (1 - harvest_volume / 350) * (1 - harvest_nbrs / 8) * harvest_prob
        harvest_params.append(harvest_param)

        planting_param = data_subset['top_n_sites_percent'].values[0]
        planting_params.append(planting_param)

    # Set cmap range
    vmin = min(mean_outputs)
    vmax = max(mean_outputs)
    cmap = plt.cm.get_cmap('viridis')

    ax.scatter(harvest_params, planting_params, c=mean_outputs, cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xlabel('Harvest parameter')
    ax.set_ylabel('Planting parameter')
    ax.set_title(output)

    # Create a ScalarMappable object with the same colormap and normalization as your scatter points
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap)

    fig.colorbar(sm, ax=ax, label=output)
    fig.show()