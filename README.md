# HKU_DASE7111
It is DASE7111 course project 2 code source. 

This project aims to compare different methods for solving the Traveling Salesman Problem (TSP), including Reinforcement Learning (RL) models and classic heuristic algorithms. The comparison involves training an RL model, evaluating it alongside other methods on multiple instances of the problem, and visualizing the results.

## Requirements

Before running the scripts, ensure you have installed all necessary dependencies. You can install them using pip:

```bash
pip install -r requirements.txt
```

The `requirements.txt` should include at least:
- torch
- numpy
- matplotlib
- scipy
- pytorch-lightning
- rl4co
- concorde (optional)

Note: For using Concorde solver, make sure to install its Python wrapper separately as it's optional and might require additional setup steps.

## Scripts Overview

### 1. Training Script (`train_tsp_model.py`)

This script is used to train a TSP-solving model with configurable hyperparameters. To run this script, use:

```bash
python train_tsp_model.py --epochs <num_epochs> --num_loc <number_of_cities> --lr <learning_rate> --batch_size <size> --seed <random_seed>
```

Parameters are adjustable via command-line arguments.

### 2. Comparison Script (`compare_methods.py`)

After training the model, this script compares the trained model against several baseline methods on various TSP instances. It also generates visualizations to illustrate the performance differences.

To execute the comparison, use:

```bash
python compare_methods.py --version <model_version_name> --num_instances <number_of_test_instances> [--sampling] [--aug]
```

- Use `--sampling` to enable sampling during evaluation.
- Use `--aug` for applying 8-augmentation during evaluation.

If the `--version` argument is not provided, the script will automatically find and use the latest trained model version.

## Outputs

The comparison script generates several visual outputs in the specified output directory (`Desktop/DASE7111_TR_ENHANCED` by default):

- Instance-specific solution plots comparing all methods side-by-side for the first three test cases.
- A boxplot illustrating the distribution of tour lengths across methods, excluding the random method.
- Line plots showing the tour length for each instance and method, with one plot specifically excluding the random method.
- Statistical summaries indicating the average performance, standard deviation, number of wins, and percentage gap to the best known solution for each method.

For further details or customization, please refer to the comments within the code files.
