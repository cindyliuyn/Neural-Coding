"""
Simulate neuron firing data for a regular spiking neuron

From izhikevich 2007, page 274
Izhikevich, E. (2007). Dynamical Systems In Neuroscience. MIT Press, 111.

"""
import numpy as np
import matplotlib.pyplot as plt


def simulate_rs(v_input, num_samples, tau, noise_input, plot=True):
    # parameters used for RS
    C = 100
    vr = -60
    vt = -40
    k = 0.7

    # neo cortical pyramidal neurons
    a = 0.03
    b = -2
    c = -50
    d = 100

    v_peak = 35  # threshold for spike
    v = vr * np.ones(num_samples)  # ones
    u = 0 * v  # initial values
    spike_location = []
    spike_train = []

    for i in range(num_samples - 1):  # forward Euler method
        v[i + 1] = v[i] + tau * (k * (v[i] - vr) * (v[i] - vt) - u[i] + v_input[i]) / C + noise_input[i]
        u[i + 1] = u[i] + tau * a * (b * (v[i] - vr) - u[i])
        if v[i + 1] >= v_peak:  # a spike is fired!
            v[i] = v_peak  # padding the spike amplitude
            v[i + 1] = c  # membrane voltage reset
            u[i + 1] = u[i + 1] + d  # recovery variable update
            spike_location.append(i)  # Store spike location
            spike_train.append(1)
        else:
            spike_train.append(0)

    # Plot
    if plot:
        timestamps = [tau * j for j in range(1, num_samples + 1)]
        plt.figure(figsize=(30, 10))
        plt.rcParams.update({'font.size': 25})
        plt.plot(timestamps, v)
        plt.xlabel('Time (ms)')
        plt.ylabel('Voltage')
        plt.show()

    return spike_location, spike_train

# T = 1000
# tau = 1  # time span and step (ms)
# n = round(T / tau)  # number of simulation steps
# v_in = np.concatenate([np.zeros(int(0.1 * n)), 70 * np.ones(int(0.9 * n))])
# simulate_rs(v_in, n, tau)
