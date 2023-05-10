"""
Generate noise input to be added to a regular spiking neuron

From Neuroscience-Information-Theory-Toolbox by Nicholas M. Timme
https://github.com/nmtimme/Neuroscience-Information-Theory-Toolbox/blob/master/Simulations/inverseFNoise.m
"""

import numpy as np


def inverse_noise(n, alpha, Fs):
    min_f = 10  # can also pass in as parameter
    const_low = 1  # can also pass in as parameter
    max_f = Fs / 2
    # Calculate the filter coefficient
    b = 10**(np.log10(1) + alpha*np.log10(min_f))

    # Calculate the frequencies
    if n % 2 == 0:
        bins = np.arange(1, n/2 + 1)
        freqs = (Fs/n) * np.concatenate(([0], bins, np.flip(bins[:-1])))
    else:
        bins = np.arange(1, np.floor(n/2) + 1)
        freqs = (Fs/n) * np.concatenate(([0], bins, np.flip(bins)))

    # Make the filter
    filter = b * freqs ** (-alpha)
    filter[freqs < 10**(np.log10(min_f) - const_low)] = 0
    filter[(freqs >= 10**(np.log10(min_f) - const_low)) & (freqs < min_f)] = 1
    filter[freqs > max_f] = 0

    # Generate the noise
    noise = np.random.randn(n)

    # Apply the 1/f^alpha filter in the frequency domain
    noise = np.fft.fft(noise)
    noise = filter * noise
    noise = np.fft.ifft(noise)

    # Get the real part of the filtered noise
    return np.real(noise)


# tau = 0.1
# n = 45000
# nzInput = inverse_noise(n, 1, 1000/tau)
#%%
