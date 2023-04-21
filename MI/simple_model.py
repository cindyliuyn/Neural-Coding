# """
# Adapted from Nicholas M. Timme and Christopher Lapish. A tutorial for information theory in neuro-
# science, 2018. ISSN 23732822
#
# """
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# from sklearn.metrics.cluster import mutual_info_score
# from simulation.rs_neuron import simulate_rs
# from simulation.inverseNoise import inverse_noise
#
# # model parameters
# noise_scale = 0.1
# inter_stim_interval = 500
# stim_duration = 500
# num_stim_sets = 50
# tau = 0.1
# pulseValueSTD = 5
# pulseOnsetSTD = 5  # standard deviation for pulse onset value
# pulseSize = 500  # input current magnitude
#
# pulseSize = [pulseSize, 0] # on, off
# num_of_pulses = len(pulseSize)
#
# # Find the total length of the recording in milliseconds
# T = num_stim_sets * num_of_pulses * (stim_duration + inter_stim_interval) + inter_stim_interval
#
# # Find the total number of time bins
# n = round(T / tau)
#
# # Find the number of time bins for stimulus interval
# interval_between_stim = round(inter_stim_interval / tau)
# interval_within_stim = round(stim_duration / tau)
#
# # Generate input current and stimulus data for one set of input pulse values
# v_input = []
# stims = []
# for i in range(num_of_pulses):
#     v_input += [0] * interval_between_stim
#     v_input += [pulseSize[i]] * interval_within_stim
#     stims += [0] * interval_between_stim
#     pulse_identifier = i + 1
#     stims.append(pulse_identifier)
#     stims += [0] * (interval_within_stim - 1)
#
# # Repeat the same input current multiple times - simulate data from multiple trials
# v_input = v_input * num_stim_sets
# v_input += [0] * interval_between_stim
# while len(v_input) > n:
#     v_input.pop()
# while len(v_input) < n:
#     v_input.append(0)
#
# stims = stims * num_stim_sets
# stim = {
#         'times': [tau * i for i, x in enumerate(stims) if x != 0], # start times for the 'on' voltage
#         'types': [x for x in stims if x != 0],
#         'currentOriginal': v_input
# }
#
# # Find transition points
# transition_points = np.where(np.diff(v_input) != 0)[0]
#
# # Generate noise
# noise = np.round((pulseOnsetSTD / tau) * np.random.randn(transition_points.size))
#
# # Add noise to input signal at transition points
# v_input = np.array(v_input)
# for trans_idx, t_point in enumerate(transition_points):
#     if noise[trans_idx] > 0:
#         start_ind = int(t_point + 1)
#         end_ind = int(t_point + noise[trans_idx])
#         v_input[start_ind:end_ind] = v_input[t_point]
#     elif noise[trans_idx] < 0:
#         start_ind = int(t_point + noise[trans_idx] + 1)
#         v_input[start_ind:t_point] = v_input[(t_point+1)]
#
# # Add noise to the input current
# v_input = v_input + pulseValueSTD * np.random.randn(len(v_input))
# # Keep a copy of the stimulation current received by the neuron
# stim['currentRec'] = v_input
#
#
# noise_input = inverse_noise(n, 1, 1000/tau)
# noise_input = noise_input / np.std(noise_input)
# noise_input = noise_input * noise_scale
#
# spike_locs, _ = simulate_rs(v_input, n, tau, noise_input, True)
# spike_locs = np.array(spike_locs) * tau  # convert into milliseconds
#
# #
# window_start = 200
# window_end = 800
# time_boundaries = []
# for s_idx, stim_onset in enumerate(stim['times']):
#     time_boundaries.append([stim_onset - window_start, stim_onset + window_end])
#
# bin_size = 50  # milliseconds
#
# # convert the spike location to whether firing or not at every location
# spike_train = np.zeros(T)
# for loc in spike_locs:
#     spike_train[int(loc)] = 1
#
# total_count = []
# for boundary in time_boundaries:
#     spikes_within_bounds = spike_train[int(boundary[0]): int(boundary[1])]
#
#     # bin the spikes
#     count_per_trial = []
#     for i in range(0, len(spikes_within_bounds), bin_size):
#         spikes = spikes_within_bounds[i:i+bin_size]
#         spike_count = np.count_nonzero(spikes == 1)
#         count_per_trial.append(spike_count)
#
#     total_count.append(count_per_trial)
#
# # Calculate mutual information between stimulus and spikes
# total_count = np.array(total_count)
#
# # iterate through each time bin
# all_mi = []
# for j in range(len(total_count[0])):
#     spike_count_var = total_count[:, j]
#     stim_var = stim['types']
#     count_matrix = pd.crosstab(stim_var, spike_count_var)
#
#     # todo: try to calculate in bits
#     mi = mutual_info_score(stim_var, spike_count_var)
#     all_mi.append(mi)
#
# plt.rcParams.update({'font.size': 15})
# mi_x_axis = [i * 50 - window_start for i in range(len(all_mi))]
# plt.scatter(mi_x_axis, all_mi)
# plt.ylabel('Mutual Information (bits)')
# plt.xlabel('Time (ms)')
# plt.show()
#
#
#
