import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import os
import math

from idtxl.data import Data
from idtxl.bivariate_pid import BivariatePID
from sklearn.metrics import mutual_info_score


def read_data_from_csv(file_name):
    df = pd.read_csv('data/' + file_name, header=None)
    return df.to_numpy()

def preprocess_spike_counts():
    # Data shape: number of neurons * number of stimulus * number of time bins
    data = read_data_from_csv('spikes_count.csv')
    num_time_bins = 36
    num_of_stimulus = 400
    gc_spike_counts = np.empty((len(data), num_of_stimulus, num_time_bins))
    for i in range(len(data)):
        gc_spike_counts[i] = data[i].reshape(num_of_stimulus, num_time_bins)
    return gc_spike_counts


def preprocess_stimulus_location():
    data = read_data_from_csv('stimulus_location.csv')
    return data[0]

def preprocess_cell_location():
    return read_data_from_csv('cell_locations.csv')


def preprocess_pair_list():
    return read_data_from_csv('pair_list.csv')


def get_value_counts(x):
    return np.unique(x, return_counts=True)


def get_pid(stimulus_loc, neuron_a, neuron_b, estimator_type='SydneyPID'):
    unique_a, counts_a = get_value_counts(neuron_a)
    unique_b, counts_b = get_value_counts(neuron_b)

    unique_c, counts_c = get_value_counts(stimulus_loc)
    alph_s1 = max(unique_a) + 1
    alpha_s2 = max(unique_b) + 1
    alpha_t = len(unique_c) + 1

    data = Data(np.vstack((stimulus_loc, neuron_a, neuron_b)), 'ps', normalise=False)

    settings = {
        'pid_estimator': estimator_type,
        'alph_s1': alph_s1,
        'alph_s2': alpha_s2,
        'alph_t': alpha_t,
        'max_unsuc_swaps_row_parm': 60,
        'num_reps': 63,
        'max_iters': 1000
        # 'lags_pid': [2, 3]
    }
    pid_analysis = BivariatePID()
    results = pid_analysis.analyse_single_target(settings=settings,
                                                 data=data,
                                                 target=0,
                                                 sources=[1, 2])
    target_result = results._single_target[0]
    return target_result['unq_s1'], target_result['unq_s2'], target_result['shd_s1_s2'], target_result['syn_s1_s2']

# https://scipython.com/book/chapter-8-scipy/problems/p84/overlapping-circles/
def circle_overlap_area(x1, y1, r1, x2, y2, r2):
    d = math.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    if d >= r1 + r2:
        # the circles are disjoint
        return 0
    elif d <= abs(r1 - r2):
        # one circle is completely contained within the other
        return math.pi * min(r1, r2)**2
    else:
        # the circles intersect
        A1 = r1**2 * math.acos((d**2 + r1**2 - r2**2) / (2*d*r1))
        A2 = r2**2 * math.acos((d**2 + r2**2 - r1**2) / (2*d*r2))
        A3 = 0.5 * math.sqrt((-d+r1+r2)*(d+r1-r2)*(d-r1+r2)*(d+r1+r2))
        return A1 + A2 - A3

def save_to_csv(time_step, a_ind, b_ind, overlapping, u1, u2, rd, syn):
    param_values = list(locals().values())
    row_values = [str(v) for v in param_values]
    string = '\n' + ','.join(row_values)
    file_name = f'all_pid'
    csv_path = './pid_results/' + str(file_name) + '.csv'
    write_header = False
    if not os.path.isfile(csv_path):
        write_header = True
    with open(csv_path, 'a+') as fd:
        if write_header:
            fd.write('time_step,source_neuron_1,source_neuron_2,overlapping,uniq_s1,uniq_s2,redun_s1_s2,syn_s1_s2')
        fd.write(string)

def get_all_data(n_samples, spike_counts, estimator_type): # time_step,
    for i in range(n_samples):
        pair = pair_list[i]
        a = pair[0]
        b = pair[1]
        print(f'Processing sample {i}: source neurons {a} and {b}')

        # 1. Calculate the distance between the two neurons
        loc_a = cell_locations[a]
        loc_b = cell_locations[b]

        r = 0.3 # large circle of the receptive field
        overlap = circle_overlap_area(loc_a[0], loc_a[1], r, loc_b[0], loc_b[1], r)

        # 2. Get the spike counts for a pair
        for time_step in range(9, 26):  # timesteps during which stimulus is on
            spike_count_a = spike_counts[a, :, time_step].astype(int)
            spike_count_b = spike_counts[b, :, time_step].astype(int)

            # 3. Get pid results
            uniq_s1, uniq_s2, redun_s1_s2, syn_s1_s2 = get_pid(stimulus_location, spike_count_a, spike_count_b, estimator_type)
            save_to_csv(time_step, a, b, overlap, uniq_s1, uniq_s2, redun_s1_s2, syn_s1_s2)


def get_mi(stimulus_loc, spike_count):
    all_mi = []
    # for each time step, the mutual information between stimulus location and spike count
    for i in range(spike_count.shape[1]):
        source = spike_count[:, i]
        mi = mutual_info_score(source, stimulus_loc)
        all_mi.append(mi)
    return all_mi


def visualize_mi_results(mi):
    bin_duration = 25 # milliseconds
    plt.rcParams.update({'font.size': 15})
    mi_x_axis = [i * bin_duration - 200 for i in range(len(mi))]
    plt.scatter(mi_x_axis, mi)
    plt.ylabel('Mutual Information (bits)')
    plt.xlabel('Time (ms)')
    plt.savefig('pid_results/mi.png')
    plt.show()

# Preprocess data
cell_locations = preprocess_cell_location()
pair_list = preprocess_pair_list()
stimulus_location = preprocess_stimulus_location()
all_spike_counts = preprocess_spike_counts()

# Compute MI between the stimulus location and the neuron at the center of the grid
mi_results = get_mi(stimulus_location, all_spike_counts[0])
visualize_mi_results(mi_results)

# Compute PID for a pair of neurons and stimulus location
get_all_data(300, all_spike_counts, 'TartuPID')



