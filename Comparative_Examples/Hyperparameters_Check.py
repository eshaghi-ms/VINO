#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
This Python script is designed to systematically evaluate the effect of a varying hyperparameter
on the performance of the VINO_Antiderivative_Arg model. The performance metric used is the
relative error (test_l2_set), and the results are visualized using a box plot.
"""
import argparse
import VINO_Antiderivative_Arg
from argparse import Namespace
import matplotlib.pyplot as plt
import os

# Define the parameter to vary and its values
parameters = {
    'batch_size': [5, 10, 20, 25, 50, 100],
    'learning_rate': [0.0001, 0.0002, 0.0005, 0.0007, 0.001, 0.002, 0.005, 0.007, 0.01],
    'gamma': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
    'modes': [8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 64],
    'width': [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256],
    'n_layer': [2, 3, 4, 5, 6, 7, 8, 9, 10]
}

parser = argparse.ArgumentParser(description="Experiment with VINO Antiderivative")
parser.add_argument(
    'parameter_name',
    type=str,
    choices=parameters.keys(),
    help=f"Parameter to vary, choose from {list(parameters.keys())}."
)
args = parser.parse_args()

parameter_name = args.parameter_name
parameter_values = parameters[parameter_name]

title = {
    'batch_size': 'Batch Size',
    'learning_rate': 'Learning Rate',
    'gamma': 'Decay Factor',
    'modes': 'Fourier Modes',
    'width': 'Fourier Layers Width',
    'n_layer': 'Number of Layers'
}

# Fixed default arguments
default_args = Namespace(
    nTrain=1000,
    nTest=100,
    scaling=100,
    s=256,
    batch_size=20,
    learning_rate=0.001,
    epochs=500,
    gamma=0.5,
    modes=16,
    width=64,
    n_layer=4,
)

# Initialize a dictionary to store test_l2_set for each parameter value
test_l2_sets = {}

# Loop over each value for the chosen parameter
for value in parameter_values:
    # Override the parameter value in default_args
    setattr(default_args, parameter_name, value)

    # Print progress for clarity
    print(f"Running experiment with {parameter_name} = {value}...")

    # Call the main function and collect test_l2_set
    # Ensure VINO_Antiderivative_Arg.main(args) returns test_l2_set
    test_l2_set = VINO_Antiderivative_Arg.main(default_args)
    test_l2_sets[value] = test_l2_set

# Prepare data for box plot
box_plot_data = [test_l2_sets[value] for value in parameter_values]

# Plot the box plot
fig_font = "DejaVu Serif"
plt.rcParams["font.family"] = fig_font
plt.figure(figsize=(12, 6))
plt.boxplot(
    box_plot_data,
    vert=False,
    patch_artist=True,
    showmeans=True,
)

plt.yticks(range(1, len(parameter_values) + 1), parameter_values)
plt.xlabel('Relative Error')
plt.ylabel(title[parameter_name])
plt.title(f'Relative Error Distribution Across Different {title[parameter_name]} Values')

# Customize grid values
plt.xscale('log')  # Logarithmic scale for grid
plt.xticks([1e-3, 1e-2, 1e-1], ['1e-3', '1e-2', '1e-1'])
plt.grid(axis='x', which='both', linestyle='--', linewidth=0.5)

# Save the figure
output_dir = "Comparative_Examples/plots"
os.makedirs(output_dir, exist_ok=True)
output_filename = os.path.join(output_dir, f"{parameter_name}_test_error_boxplot.png")
plt.savefig(output_filename, bbox_inches='tight', dpi=1200)
print(f"Figure saved as {output_filename}")

# Show the plot
plt.show()
