# Fair Pagerank

This folder contains the files associated with our proposed method for minimizing the cost of **Pagerank Fairness**.

## Content

	experiments.py

This file contains the our defined classes.

	script.py

This file contains execution examples.

	group_preferential_attachment.py

This file contains code for the generation of the random graphs used in our experiments.

## Usage

In order to perform an experiment, we must instatiate an `Experiment` object. The `Experiment` class implements the logic of our method. The arguments of its constructor are:

1. A string containing the name of the dataset.
2. A `Configuration` object.
3. A `Settings` object.

The `Configuration` class is a container class for the parameters of our algorithm. We instantiate it with our desired values for the experiment. The available parameters are:

- `gamma`, the pagerank jump probability
- `phi`, the target pagerank fairness
- `selector`, the function selecting the next node to be modified
- `intervention`, the form of node modification to apply

The `Settings` class is a container class for parameters enabling optional functionality. The available parameters are `save` and `verbose`.