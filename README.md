# Investigating the Effects of Forestry Practices on the Dynamics of the Viaphytic Fungi-Trees Ecosystem
_Core course Agent-Based Modelling, MSc Computational Science (UvA/VU, 2024)_

This repository contains the code for the course project assignment in Agent-Based Modelling (ABM). The foundation for our framework is a model proposed by D. C. Thomas, R. Vandegrift and B. A. Roy (2020)[1], investigating the dynamics between viaphytic fungi and trees, which the fungi can use as propagation vectors. We expand this model by creating a mutualist feedback cycle in which both agent types deposit resources, which in turn affect the other agent type's behaviour.

A particular focus of our research is the exploration of forestry techniques consisting of harvesting and planting, which can optimise timber yield while preserving the tree and fungi populations.

## Contents

The experiments with the model are structured in several Jupyter notebooks:
- `runmodel.ipynb` contains general purpose procedures for running the model with predefined parameters and analysing model results saved as `.parquet` files;
- `SA_paramspace.ipynb` contains a global and local sensitivity analysis of several model parameters, visualisations of specific metrics in the sampled parameter space and additional simulations for determining the tree volume distribution;
- `planting_experiments.ipynb` contains the setup for more specific simulations at a more narrowed-down range of planting and harvesting parameters;
- `phase_space.ipynb` contains analysis procedures for analysing the temporal dynamics of model outputs at sampled input parameters.

The main Python modules imported in the notebooks partly follow Mesa conventions and include:
- `model.py` - model-related procedures;
- `agent.py` - agent-related procedures;
- `visualisation.py` - procedures for visualising results;
- `sensitivity_analysis.py` - procedures for global and local sensitivity analysis.

## Requirements

To use the model, please install the dependencies by running `pip install -r requirements.text`.

The packages used in the model include:
- NumPy
- SciPy
- Matplotlib
- Seaborn
- Mesa
- Pandas
- SALib
- pyarrow

## References
[1] Thomas, Daniel C., Roo Vandegrift, and Bitty A. Roy. "An agent-based model of the foraging ascomycete hypothesis." Fungal Ecology 47 (2020): 100963.
