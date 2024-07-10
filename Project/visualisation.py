import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib as mpl
import seaborn as sns
import plotly.graph_objs as go
import plotly.io as pio
pio.renderers.default = 'notebook'

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
                # plt.figure()

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


def plot_param_space(data, output, param1, param2, mean_over_last=100, ax=None):
    """
    Plot specific outputs in a sampled 2D parameter space.
    Args:
        data (pd.DataFrame): collected data from the model runs
        output (str): name of the output of interest
        param1 (str): name of the first parameter to plot
        param2 (str): name of the second parameter to plot
        mean_over_last (int): number of timesteps to average over for each replicate
        ax (matplotlib.axes.Axes): axes object to plot on
    """

    if ax is None:
        fig, ax = plt.subplots()
        fig.set_size_inches(6, 5)
    else:
        fig = ax.get_figure()

    max_sims = data['SimId'].max() + 1

    param_name_mapping = {
        'harvest_volume': '$V_H$',
        'harvest_nbrs': '$N_H$',
        'harvest_prob': '$P_H$',
        'top_n_sites_percent': '$P_\%$'
    }


    param_values_1 = []
    param_values_2 = []
    mean_outputs = []

    for sim_id in range(max_sims):
        data_subset = data[data['SimId'] == sim_id]
        n_values = data_subset.shape[0]
        mean_output = data_subset[output].values[n_values - mean_over_last:].mean()
        mean_outputs.append(mean_output)

        param_values_1_subset = data_subset[param1].values[0]
        param_values_1.append(param_values_1_subset)

        param_values_2_subset = data_subset[param2].values[0]
        param_values_2.append(param_values_2_subset)

    # Set cmap range
    vmin = min(mean_outputs)
    vmax = max(mean_outputs)
    cmap = mpl.colormaps['viridis'] #plt.cm.get_cmap('viridis')

    ax.scatter(param_values_1, param_values_2, c=mean_outputs, cmap=cmap, vmin=vmin, vmax=vmax, marker='.', s=5)

    ax.set_xlabel(param_name_mapping[param1])
    ax.set_ylabel(param_name_mapping[param2])
    ax.set_title(output, fontsize=11)

    # Create a ScalarMappable object with the same colormap and normalization as your scatter points
    norm = Normalize(vmin=vmin, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap)

    fig.colorbar(sm, ax=ax)
    if ax is None:
        fig.show()


def plot_param_space_array(data, params, outputs):
    """
    Create a series of 2D plots for each pair of input parameters.
    Args:
        data (pd.DataFrame): collected data from the model runs
        params (list of str): list of parameter names to plot
        outputs (list of str): list of output variable names to plot
    """

    param_combos = list(combinations(params, 2))

    fig, ax = plt.subplots(len(outputs), len(param_combos))
    fig.set_size_inches(2*len(param_combos), 2*len(outputs))

    for i, output in enumerate(outputs):
        for j, combo in enumerate(param_combos):
            plot_param_space(data, output, combo[0], combo[1], ax=ax[i, j])
    
    fig.tight_layout()
    fig.show()


def plot_param_range(data, param, output, param_range, mean_over_last=100):
    """
    Create a plot that shows the output for a specific parameter over a range of values.
    Args:
        data (pd.DataFrame): collected data from the model runs
        param (str): name of the parameter to plot
        outputs (str): name of the output of interest
        param_range (tuple): (min, max) range of values for the parameter
        mean_over_last (int): number of timesteps to average over for each replicate
    """

    param_name_mapping = {
        'harvest_volume': '$V_H$',
        'harvest_nbrs': '$N_H$',
        'harvest_prob': '$P_H$',
        'top_n_sites_percent': '$P_\%$'
    }

    fig, ax = plt.subplots()
    fig.set_size_inches(4, 3)
    
    data_subset = data[(data[param] >= param_range[0]) & (data[param] <= param_range[1])]
    param_values = data_subset[param].unique()
    mean_outputs = []

    for param_value in param_values:
        data_subset_param = data_subset[data_subset[param] == param_value]
        n_values = data_subset_param.shape[0]
        mean_output = data_subset_param[output].values[n_values - mean_over_last:].mean()
        mean_outputs.append(mean_output)

    ax.scatter(param_values, mean_outputs, marker='o')

    ax.set_xlabel(param_name_mapping[param])
    ax.set_ylabel(output)
    ax.set_title(f'Average {output} over parameter range', fontsize=11)
    fig.show()
    
    
def phase_space_2d(data, output_var='Trees', input_var='harvest_nbrs', input_value=None, start_step=50):

    if not input_value:
        input_value = sorted(data[input_var].unique())[0]

    subset = data[data[input_var] == input_value]
    subset = subset.sort_values(by=['Step', 'RunId'])

    # Compute mean and derivative over the 20 runs for each time step
    mean_output = subset.groupby('Step')[output_var].mean().values
    derivative_output = np.gradient(mean_output)

    # Consider only the data from the specified start step onwards
    mean_output = mean_output[start_step:]
    derivative_output = derivative_output[start_step:]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(mean_output, derivative_output, label=f'Derivative vs {output_var}', color='blue')
    plt.xlabel(output_var)
    plt.ylabel(f'Derivative of {output_var}')
    plt.title(f'{output_var} vs Derivative for {input_var} = {input_value}')
    plt.legend()
    plt.grid(True)
    plt.show()

def phase_space_3d(data, output_var='Trees', input_var='harvest_nbrs', start_step=50):
    # Initialize lists to store the results
    input_var_values = sorted(data[input_var].unique())
    output_means = []
    output_derivatives = []

    # Process each unique value of the input variable
    for input_value in input_var_values:
        subset = data[data[input_var] == input_value]
        # Ensure the data is sorted by Step and RunId
        subset = subset.sort_values(by=['Step', 'RunId'])
        # Compute mean output variable over the 20 runs for each time step
        mean_output = subset.groupby('Step')[output_var].mean().values
        # Compute the derivative of the mean output variable
        derivative_output = np.gradient(mean_output)
        # Store the results starting from the specified start step
        output_means.append(mean_output[start_step:])
        output_derivatives.append(derivative_output[start_step:])

    # Convert lists to arrays for plotting
    output_means = np.array(output_means)
    output_derivatives = np.array(output_derivatives)

    # Create 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plotting
    for i, input_value in enumerate(input_var_values):
        ax.plot(input_value * np.ones_like(output_means[i]), output_means[i], output_derivatives[i], lw=0.5)

    # Labels
    ax.set_xlabel(input_var)
    ax.set_ylabel(output_var)
    ax.set_zlabel(f'Derivative')

    plt.title(f'Phase space of Fungi Population for varying $V_H$')
    plt.show()