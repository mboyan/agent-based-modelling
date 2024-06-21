import mesa.batchrunner as batch
from SALib.sample import saltelli
from SALib.analyze import sobol
import pandas as pd
import numpy as np
import random
from IPython.display import clear_output


def run_batches(model, problem, outputs, n_max_timesteps, n_replicates, n_distinct_samples, seed=123):
    """
    Generate parameter samples with SALib and run batches of model replicates.
    Args:
        problem: SALib problem dictionary
        outputs: list of output variables to record
        n_max_timesteps: maximum number of timesteps to run the model for
        n_replicates: number of replicates per sample
        n_distinct_samples: number of distinct samples to generate
        seed: random seed
    """

    np.random.seed(seed)
    random.seed(seed)

    # Generate samples
    param_values = saltelli.sample(problem, n_distinct_samples)

    print(f'Running a total of {param_values.shape[0]} samples.')

    # Find indices of parameters that should be integers
    integer_param_names = ['n_init_trees', 'n_init_fungi', 'width', 'height', 'harvest_nbrs']
    integer_param_indices = [problem['names'].index(name) for name in integer_param_names if name in problem['names']]

    collected_data = []

    count = 0
    for vals in param_values: 
        # Change parameters that should be integers
        vals = list(vals)
        # Turn integer parameters into integers
        for index in integer_param_indices:
            vals[index] = int(vals[index])
        # Transform to dict with parameter names and their values
        variable_parameters = {}
        for name, val in zip(problem['names'], vals):
            variable_parameters[name] = val


        run_result = batch.batch_run(model, variable_parameters,
                                     iterations=n_replicates, max_steps=n_max_timesteps,
                                     data_collection_period=1)
        
        # Get relevant keys from the run result
        run_keys_relevant = [key for key in run_result[0].keys() if key in outputs or key in ['RunId', 'Step'] or key in problem['names']]
        run_result_relevant = []
        for step in range(len(run_result)):
            run_result_step = {key: run_result[step][key] for key in run_keys_relevant}
            run_result_step['SimId'] = count * n_replicates + run_result_step['RunId']
            run_result_relevant.append(run_result_step)

        # Store the results in the data frame
        collected_data.extend(run_result_relevant)

        clear_output()

        count += 1
        print(f'{count / len(param_values) * 100:.2f}% done')
    
    data = pd.DataFrame(collected_data)

    data.reset_index(drop=True, inplace=True)

    return data


def sobol_analyse(data, problem, outputs, mean_over_last):
    """
    Analyse the collected data with Sobol sensitivity analysis and plot the results.
    Args:
        data (pd.DataFrame): collected data from the model runs
        problem (dict): SALib problem dictionary
        outputs (list of str): list of output variable names to analyse
        mean_over_last (int): number of timesteps to average over for each replicate
    """

    max_steps = data['Step'].max() + 1
    max_sims = data['SimId'].max() + 1
    
    # Take the last mean_over_last timesteps for each simulation and average over them
    data_averages = pd.DataFrame(columns=data.columns)
    for sim_id in range(max_sims):
        data_average = data[data['SimId'] == sim_id][data['Step'] == 0]
        for output in outputs:
            Y_last = data[output][data['SimId'] == sim_id][data['Step'] > max_steps - mean_over_last]
            Y_mean = Y_last.mean()
            data_average[output] = [Y_mean]
        data_averages = pd.concat([data_averages, data_average], ignore_index=True)

    # Calculate the sensitivity indices
    sensitivity_indices = {}

    for output in outputs:
        Y = data_averages[output].values
        Si = sobol.analyze(problem, Y, print_to_console=False)
        sensitivity_indices[output] = Si
    
    return sensitivity_indices