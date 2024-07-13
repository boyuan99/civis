import numpy as np
import h5py
import pandas as pd
from scipy.signal import savgol_filter, find_peaks

def compute_deltaF_over_F(fluorescence, baseline_indices=None):
    """
    Compute delta F / F from a fluorescence signal using specified baseline indices.

    Parameters:
    - fluorescence: numpy array of the signal from which to estimate noise.
    - baseline_indices: indices of the signal considered to be baseline/noise.

    Returns:
    - deltaF_F: estimated delta F / F of the signal
    .
    """
    if baseline_indices is None:
        F0 = np.mean(fluorescence)
    else:
        F0 = np.mean(fluorescence[baseline_indices[0]:baseline_indices[1]])

    deltaF_F = (fluorescence - F0) / F0

    return deltaF_F


def normalize_signal_per_neuron(signal):
    # Initialize an empty array for the normalized signal
    normalized_signal = np.zeros(signal.shape)

    # Iterate over each neuron
    for neuron_idx in range(signal.shape[0]):
        neuron_signal = signal[neuron_idx, :]
        min_val = np.min(neuron_signal)
        max_val = np.max(neuron_signal)
        normalized_signal[neuron_idx, :] = (neuron_signal - min_val) / (max_val - min_val)

    return normalized_signal


def normalize_signal(signal):
    min_val = np.min(signal)
    max_val = np.max(signal)
    normalized_signal = (signal - min_val) / (max_val - min_val)
    return normalized_signal


def shift_signal(fluorescence):
    shifted = np.zeros_like(fluorescence)
    for i, signal in enumerate(fluorescence):
        shifted[i] = signal - np.mean(signal)

    return shifted


def load_data(filename):
    """
    Load all data
    :param filename: .mat file containing config, spatial, Cn, Coor, ids, etc...
    :return: essential variables for plotting
    """
    with h5py.File(filename, 'r') as file:
        data = file['data']
        Cn = np.transpose(data['Cn'][()])
        ids = data['ids'][()] - 1
        Coor_cell_array = data['Coor']
        Coor = []
        C_raw = np.transpose(data['C_raw'][()])
        C = np.transpose(data['C'][()])
        centroids = np.transpose(data['centroids'][()])
        virmenPath = data['virmenPath'][()].tobytes().decode('utf-16le')

        for i in range(Coor_cell_array.shape[1]):
            ref = Coor_cell_array[0, i]  # Get the reference
            coor_data = file[ref]  # Use the reference to access the data
            coor_matrix = np.array(coor_data)  # Convert to a NumPy array

            Coor.append(np.transpose(coor_matrix))

        # C = np.zeros_like(C_raw)
        # for i, C_pick in enumerate(C_raw):
        #     C_base = savgol_filter(C_pick, window_length=2000, polyorder=2, mode='interp')
        #     C[i] = C_pick - C_base

    return C, C_raw, Cn, ids, Coor, centroids, virmenPath

