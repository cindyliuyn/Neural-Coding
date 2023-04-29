import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def read_data_from_csv(file_name):
    df = pd.read_csv('data/' + file_name, header=None)
    return df.to_numpy()
def plot_receptive_field(location, ax):
    small_radius= 0.1
    large_radius = 0.3
    position_x = location[0]
    position_y = location[1]

    # Add two circle to the axis object to represent the receptive field
    s_circle = plt.Circle((position_x, position_y), small_radius, facecolor='white', edgecolor='purple', alpha=0.7)
    ax.add_patch(s_circle)
    b_circle = plt.Circle((position_x, position_y), large_radius, facecolor='white', edgecolor='grey', alpha=0.5)
    ax.add_patch(b_circle)

    # Plot the point of the receptive field
    ax.scatter(position_x, position_y, color='blue', s=5)

def plot_location(location, ax):
    ax.scatter(location[0], location[1], color='black', s=5)


def plot_neuron_locations(locations, pair_indice, title_index):
    # Create a figure and axis object
    fig, fig_ax = plt.subplots(figsize=(6, 6))

    # Create a Polygon object from the vertices
    square = plt.Polygon([(0, 0), (1, 0), (1, 1), (0, 1)], facecolor='white', closed=True)
    fig_ax.add_patch(square)

    for cell_location in locations:
        plot_location(location=cell_location, ax=fig_ax)

    for cell_location in pair_indice:
        plot_receptive_field(location=cell_location, ax=fig_ax)

    # Set the limits of the plot to the range [0, 1] on both axes
    fig_ax.set_xlim([0, 1])
    fig_ax.set_ylim([0, 1])
    axis_values = np.round(np.arange(0, 1.1, 0.1), 1)
    fig_ax.set_xticks(axis_values)
    fig_ax.set_yticks(axis_values)
    fig_ax.set_xticklabels(axis_values)
    fig_ax.set_yticklabels(axis_values)

    # Display the plot
    plt.savefig(f'pid_results/neuron_locations_{title_index}.png')
    plt.show()

cell_locations = read_data_from_csv('cell_locations.csv')

point_a = cell_locations[0]
point_b = list(filter(lambda x: x[1] > 0.7 and x[0] < 0.3, cell_locations))[0]
point_c = list(filter(lambda x: x[1] > 0.7 and x[0] > 0.7, cell_locations))[0]
point_d = list(filter(lambda x: x[1] < 0.5 and x[0] < 0.5, cell_locations))[15]

plot_neuron_locations(cell_locations, [point_a, point_b], 1)
plot_neuron_locations(cell_locations, [point_c, point_d], 2)



