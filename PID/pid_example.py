import numpy as np
from matplotlib import pyplot as plt

from idtxl.data import Data
from idtxl.bivariate_pid import BivariatePID
import pandas as pd
import math
from sklearn.metrics import mutual_info_score
import seaborn as sns


def read_data_from_csv(file_name):
    df = pd.read_csv('data/' + file_name, header=None)
    return df.to_numpy()

def preprocess_spike_counts():
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

def preprocss_cell_location():
    return read_data_from_csv('cell_locations.csv')


def preprocss_pair_list():
    return read_data_from_csv('pair_list.csv')


def get_value_counts(x):
    return np.unique(x, return_counts=True)


def get_pid(stimulus_loc, neuron_a, neuron_b, estimater_type='SydneyPID'):
    unique_a, counts_a = get_value_counts(neuron_a)
    unique_b, counts_b = get_value_counts(neuron_b)

    unique_c, counts_c = get_value_counts(stimulus_loc)
    alph_s1 = max(unique_a) + 1
    alpha_s2 = max(unique_b) + 1
    alpha_t = len(unique_c) + 1

    data = Data(np.vstack((stimulus_loc, neuron_a, neuron_b)), 'ps', normalise=False)

    settings = {
        'pid_estimator': estimater_type,
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


def save_to_csv(time_step, a_ind, b_ind, source_distance, u1, u2, rd, syn):
    param_values = list(locals().values())
    row_values = [str(v) for v in param_values]
    string = '\n' + ','.join(row_values)
    file_name = f'pid_{time_step}'
    with open('./pid_results/' + str(file_name) + '.csv', 'a+') as fd:
        fd.write(string)


def get_distance(location_a, location_b):
    xVarTile = location_b[0] + np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1])
    yVarTile = location_b[1] + np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])
    r = np.min(np.sqrt((location_a[0] * np.ones([1, 9]) - xVarTile) ** 2
                       + (location_b[0] * np.ones([1, 9]) - yVarTile) ** 2))
    return r


def get_all_data(n_samples, spike_counts, time_step, estimater_type):
    for i in range(n_samples):
        pair = pair_list[i]
        a = pair[0]
        b = pair[1]
        print(f'Processing sample {i}: source neurons {a} and {b}')

        # 1. Calculate the distance between the two neurons
        loc_a = cell_locations[a]
        loc_b = cell_locations[b]

        distance = get_distance(loc_a, loc_b)
        print(f'Distance between the neurons: {distance}')

        # 2. Get the spike counts for a pair
        spike_count_a = spike_counts[a, :, time_step].astype(int)
        spike_count_b = spike_counts[b, :, time_step].astype(int)

        # 3. Get pid results
        uniq_s1, uniq_s2, redun_s1_s2, syn_s1_s2 = get_pid(stimulus_location, spike_count_a, spike_count_b, estimater_type)
        save_to_csv(time_step, a, b, distance, uniq_s1, uniq_s2, redun_s1_s2, syn_s1_s2)


cell_locations = preprocss_cell_location()
pair_list = preprocss_pair_list()
stimulus_location = preprocess_stimulus_location()
all_spike_counts = preprocess_spike_counts()


get_all_data(1, all_spike_counts, 14, 'TartuPID')

df_pid = pd.read_csv('pid_results/pid_12.csv')

df_pid.plot(x='distance', y='uniq_a', kind='scatter')
df_pid.plot(x='distance', y='uniq_b', kind='scatter')
df_pid.plot(x='distance', y='redun', kind='scatter')
df_pid.plot(x='distance', y='syn', kind='scatter')
plt.show()

def get_hist_plot(col_names, colors_, figsize):
    fig, ax = plt.subplots(figsize=figsize)
    for col_name, color_ in zip(col_names, colors_.copy()):
        label = col_name + ' $\mu$ = ' + str(round(df_pid[col_name].mean(), 2)) \
                + ' $\sigma$ = ' + str(round(df_pid[col_name].std(), 2))
        sns.distplot(df_pid[col_name], ax=ax, color=color_, label=label)
    ax.set_xlabel('Information (bits)')
    ax.grid(True)
    ax.set_title('')
    ax.legend()
    plt.tight_layout()
    plt.show()


get_hist_plot(['syn', 'uniq_a'], ['#FF5733', '#33B9FF'], (6, 6))
get_hist_plot(['redun'], ['#BAA16E'], (5, 5))

# def get_mi(stimulus_loc, spike_count):
#     all_mi = []
#     for i in range(spike_count.shape[1]):
#         source = spike_count[:, i]
#         mi = mutual_info_score(source, stimulus_loc)
#         all_mi.append(mi)
#     return all_mi
#
# mi_results = get_mi(stimulus_location, all_spike_counts[0])
#
# bin_duration = 25 # milliseconds
# plt.rcParams.update({'font.size': 15})
# mi_x_axis = [i * bin_duration - 200 for i in range(len(mi_results))]
# plt.scatter(mi_x_axis, mi_results)
# plt.ylabel('Mutual Information (bits)')
# plt.xlabel('Time (ms)')
# plt.show()
