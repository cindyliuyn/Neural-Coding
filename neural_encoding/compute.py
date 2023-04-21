"""
Adapted from Coursera Computational Neuroscience - Quiz 2
"""

from __future__ import division
import numpy as np


def compute_sta(stim, rho, num_timesteps):
    """Compute the spike-triggered average from a stimulus and spike-train.
    
    Args:
        stim: stimulus time-series
        rho: spike-train time-series
        num_timesteps: how many timesteps to use in STA
        
    Returns:
        spike-triggered average for num_timesteps timesteps before spike"""
    
    sta = np.zeros((num_timesteps,))

    # This command finds the indices of all of the spikes that occur
    # after 300 ms into the recording.
    spike_times = rho[num_timesteps:].nonzero()[0] + num_timesteps

    # Fill in this value. Note that you should not count spikes that occur
    # before 300 ms into the recording.
    num_spikes = len(spike_times)
    print(num_spikes)

    # Interval Distribtion

    # Plot the mean of interspike and compare against exponential distribution
    # https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test
    
    # Compute the spike-triggered average of the spikes found.
    # To do this, compute the average of all of the vectors
    # starting 300 ms (exclusive) before a spike and ending at the time of
    # the event (inclusive). Each of these vectors defines a list of
    # samples that is contained within a window of 300 ms before each
    # spike. The average of these vectors should be completed in an
    # element-wise manner.
    # 
    # Your code goes here.
    for i in range(num_spikes):
        spike_index = spike_times[i]
        stim_windows = stim[spike_index - num_timesteps:spike_index]
        sta = sta + stim_windows  # Adding the stimulus on each time step
    return sta / num_spikes


def compute_interspike_interval(rho):
    spike_times = rho.nonzero()[0]
    return np.diff(spike_times)