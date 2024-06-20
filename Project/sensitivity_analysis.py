import mesa.batchrunner as batch
from SALib.sample import saltelli
from SALib.analyze import sobol
import pandas as pd
import numpy as np
from IPython.display import clear_output


def run_batches(model, problem, outputs, n_max_timesteps, n_replicates, n_distinct_samples):
    """
    Generate parameter samples with SALib and run batches of model replicates.
    Args:
        problem: SALib problem dictionary
        outputs: list of output variables to record
        n_max_timesteps: maximum number of timesteps to run the model for
        n_replicates: number of replicates per sample
        n_distinct_samples: number of distinct samples to generate
    """

    # Generate samples
    param_values = saltelli.sample(problem, n_distinct_samples)

    print(f'Running a total of {param_values.shape[0]} samples.')

    # Find indices of parameters that should be integers
    integer_param_names = ['n_init_trees', 'width', 'height', 'n_init_fungi']
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
        run_result_relevant = [{key: run_result[step][key] for key in run_keys_relevant} for step in range(len(run_result))]

        # Store the results in the data frame
        collected_data.extend(run_result_relevant)

        clear_output()

        count += 1
        print(f'{count / len(param_values) * 100:.2f}% done')
    
    data = pd.DataFrame(collected_data)

    data.reset_index(drop=True, inplace=True)

    return data