import pandas as pd
import numpy as np
import h5py
import torch
from scipy.signal import find_peaks, savgol_filter


class CalciumTank:
    def __init__(self,
                 neuron_path,
                 ci_rate=20,  # calcium imaging rate
                 session_duration=30 * 60,  # in seconds
                 velocity_height=0.7,
                 velocity_distance=100,
                 threshold=[100.0, -100.0],
                 window_length=51,
                 polyorder=3, ):
        self.neuron_path = neuron_path
        self.ci_rate = ci_rate
        self.session_duration = session_duration
        self.t = np.arange(0, self.session_duration, 1 / self.ci_rate)
        self.t_raw = np.arange(0, len(self.t), 1)
        self.C, self.C_raw, self.Cn, self.ids, self.Coor, self.centroids, self.virmenPath = self.load_data(neuron_path)
        self.C_raw = self.shift_signal(self.compute_deltaF_over_F(self.C_raw))
        self.ca_all = np.mean(self.C_raw, axis=0)
        self.neuron_num = self.C_raw.shape[0]
        self.velocity_height = velocity_height  # typically 1 for 364, 0.7 for 366
        self.velocity_distance = velocity_distance

        self.threshold = threshold
        if self.threshold[0] < self.threshold[1]:
            raise ValueError("threshold[0] should be greater than threshold[1]")

        self.virmen_trials, self.virmen_data = self.read_and_process_data(self.virmenPath, threshold=self.threshold)
        self.trials_end_indices = self.calculate_virmen_trials_end_indices()
        self.lick_raw, self.lick_raw_mask, self.lick = self.find_lick_data()
        self.pstcr_raw, self.pstcr = self.find_position_movement_rate()  # position change rate
        self.velocity = self.find_velocity()
        self.window_length = window_length  # Window length: number of coefficients (odd number)
        self.polyorder = polyorder  # Polynomial order
        self.smoothed_pstcr = savgol_filter(self.pstcr, self.window_length, self.polyorder)
        self.smoothed_velocity = savgol_filter(self.velocity, self.window_length, self.polyorder)
        self.dr, self.dr_raw = self.find_rotation()

        # for analysis usage:
        self.velocity_peak_indices = find_peaks(self.smoothed_velocity, height=8, distance=self.velocity_distance)[0]
        self.lick_edge_indices = np.where(np.diff(self.lick) > 0)[0]
        self.movement_onset_indices = self.find_movement_onset(self.smoothed_velocity, self.velocity_peak_indices)

    @staticmethod
    def compute_deltaF_over_F(fluorescence, baseline_indices=None):
        if baseline_indices is None:
            F0 = np.mean(fluorescence)
        else:
            F0 = np.mean(fluorescence[baseline_indices[0]:baseline_indices[1]])

        deltaF_F = (fluorescence - F0) / F0
        return deltaF_F

    @staticmethod
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

    @staticmethod
    def normalize_signal(signal):
        min_val = np.min(signal)
        max_val = np.max(signal)
        normalized_signal = (signal - min_val) / (max_val - min_val)
        return normalized_signal

    @staticmethod
    def shift_signal(fluorescence):
        shifted = np.zeros_like(fluorescence)
        for i, signal in enumerate(fluorescence):
            shifted[i] = signal - np.mean(signal)

        return shifted

    def load_data(self, filename):
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

        return C, C_raw, Cn, ids, Coor, centroids, virmenPath

    @staticmethod
    def read_and_process_data(file_path, threshold=None):
        raise NotImplementedError("Subclasses are required to read and process the virmen data.")

    def calculate_virmen_trials_end_indices(self):
        raise NotImplementedError("Subclasses are required to get the end indices for the virmen data.")

    def find_lick_data(self):
        lick_all = np.array(self.virmen_data['lick'])
        # Convert lick data into binary
        lick_all[lick_all != 0] = 1
        lick_all_mask = []
        flag = False
        for i in range(len(lick_all) - 10):
            lick_all_mask_data = 0

            if (lick_all[i] != 0):
                flag = True
            if (flag):
                lick_all_mask_data = max(lick_all[i:i + 10])
                if (lick_all_mask_data == 0):
                    flag = False
            lick_all_mask.append(lick_all_mask_data)

        lick_all_mask.extend([0 for i in range(10)])
        lick_all_mask = np.array(lick_all_mask)

        # Initialize an empty mask for valid lick periods
        valid_lick_mask = np.zeros(len(lick_all_mask), dtype=bool)

        # Loop through each trial end to mark valid lick periods
        for end_idx in self.trials_end_indices:
            # Find the next occurrence of a lick after the trial end
            try:
                start_idx = np.where(lick_all_mask[end_idx:] == 1)[0][0] + end_idx
                # From start_idx, find when the licking stops
                stop_idxs = np.where(lick_all_mask[start_idx:] == 0)[0]
                if len(stop_idxs) > 0:
                    valid_end_idx = stop_idxs[0] + start_idx
                    valid_lick_mask[start_idx:valid_end_idx] = True
                else:
                    # In case there's no stop, mark until the end
                    valid_lick_mask[start_idx:] = True
            except IndexError:
                # No licking after this trial end, move to the next
                continue

        lick_all = lick_all[:self.session_duration * self.ci_rate]
        lick_all_mask = lick_all_mask[:self.session_duration * self.ci_rate]
        valid_lick_mask = valid_lick_mask[:self.session_duration * self.ci_rate]

        return lick_all, lick_all_mask, valid_lick_mask

    def find_position_movement_rate(self, reset_threshold=1.5, method='outlier'):
        """
        Finds the position movement rate, in centimeterers per 50ms.
        Notice that this is not the really running velocity on the ball,
         just the position movement rate calculated from the positions in the maze
        :param method: 'fixed' or 'outlier'
        :param reset_threshold: set the movement rate to 0 if exceeded the threshold
        :return: velocity movement rate in centimeters per 50 ms
        """
        data_array = np.array(self.virmen_data)
        dx = np.diff(data_array[:, 0])
        dy = np.diff(data_array[:, 1])
        velocity_raw = np.sqrt(dx ** 2 + dy ** 2)
        velocity = None
        if method == 'fixed':
            velocity = np.where(velocity_raw > reset_threshold, 0, velocity_raw)
        elif method == 'outlier':
            data = self.normalize_signal(velocity_raw)
            mean = np.mean(data)
            std_dev = np.std(data)
            z_scores = [(point - mean) / std_dev for point in data]
            outliers = np.where(np.abs(z_scores) > reset_threshold)
            velocity = velocity_raw.copy()
            velocity[outliers] = 0

        velocity_raw = velocity_raw[: self.session_duration * self.ci_rate]
        velocity = velocity[: self.session_duration * self.ci_rate]

        return velocity_raw, velocity

    def find_velocity(self, threshold=3):
        dx = np.array(self.virmen_data['dx'])
        dy = np.array(self.virmen_data['dy'])
        velocity = np.sqrt(dx ** 2 + dy ** 2)
        outliers = self.find_outliers_indices(self.normalize_signal(velocity), threshold=threshold)
        velocity[outliers] = 0
        velocity = velocity[: self.session_duration * self.ci_rate]

        return velocity

    def find_rotation(self, threshold=0.4):
        face_angle = np.array(self.virmen_data['face_angle'])
        dr_raw = np.diff(face_angle)
        dr_raw = np.insert(dr_raw, 0, 0)
        outliers = self.find_outliers_indices(self.normalize_signal(dr_raw), threshold=threshold)
        dr = dr_raw.copy()
        dr[outliers] = 0
        dr = dr[: self.session_duration * self.ci_rate]

        return dr, dr_raw

    @staticmethod
    def find_outliers_indices(data, threshold=3.0):
        mean = np.mean(data)
        std_dev = np.std(data)
        z_scores = (data - mean) / std_dev
        outliers = np.where(np.abs(z_scores) > threshold)[0]
        return outliers

    def find_peaks_in_traces(self, notebook=False, use_tqdm=True):
        """
        Finds peaks in each calcium trace in C_raw.
        Applies baseline correction using Savitzky-Golay filter before finding peaks.

        Returns:
        - peak_indices_filtered_all: List of arrays, each containing peak indices for the filtered signal of corresponding trace.
        """
        if notebook:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm

        peak_indices_all = []
        peak_indices_filtered_all = []

        iterator = tqdm(self.C_raw, desc="Finds peaks in each calcium trace") if use_tqdm else self.C_raw

        for i, C_pick in enumerate(iterator):
            peak_height = np.average(C_pick) + 3 * np.std(C_pick)

            peak_calciums, _ = find_peaks(C_pick, height=peak_height, distance=100, prominence=0.2)

            C_base = savgol_filter(C_pick, window_length=2000, polyorder=2, mode='interp')
            peak_calciums_filtered, _ = find_peaks(C_base, height=np.average(C_base) + 3 * np.std(C_base), distance=200)

            C_filtered = C_pick - C_base
            peak_calciums_filtered, _ = find_peaks(C_filtered, height=np.average(C_filtered) + 3 * np.std(C_filtered),
                                                   distance=200)

            peak_indices_all.append(peak_calciums)
            peak_indices_filtered_all.append(peak_calciums_filtered)

        return peak_indices_filtered_all

    def compute_correlation_ca_instance(self, instance):
        ca_instance_correlation_array = np.zeros(self.neuron_num)
        for neuron_pick in range(self.neuron_num):
            ca_instance_correlation_array[neuron_pick] = np.corrcoef(instance, self.C_raw[neuron_pick])[0, 1]

        return ca_instance_correlation_array

    def average_across_indices(self, indices, cut_interval=50):
        """
        Average the calcium signal across the specified indices.
        :param indices: indices to average across
        :param cut_interval: interval to cut around the indices
        :return: averaged calcium signal, normalized signal, sorted signal, mean signal, sorted indices
        """
        C_avg = np.zeros([self.neuron_num, cut_interval * 2])
        time_points = len(self.t)

        for neu in range(self.neuron_num):
            C_tmp = np.zeros([len(indices), cut_interval * 2])
            for i, idx in enumerate(indices):
                start_idx = max(idx - cut_interval, 0)
                end_idx = min(idx + cut_interval, time_points)

                # Check if the window exceeds the array bounds
                if start_idx == 0 or end_idx == time_points:
                    # If the window exceeds, keep the row in C_tmp as zeros
                    continue
                else:
                    # Otherwise, assign the values from C_raw to the appropriate row in C_tmp
                    C_tmp[i, (cut_interval - (idx - start_idx)):(cut_interval + (end_idx - idx))] = self.C_raw[neu,
                                                                                                    start_idx:end_idx]

            C_avg[neu] = np.mean(self.shift_signal(C_tmp), axis=0)

        peak_values = np.max(C_avg, axis=1)
        C_avg_normalized = self.normalize_signal_per_neuron(C_avg)
        peak_times = np.argmax(C_avg_normalized, axis=1)
        sorted_indices = np.argsort(peak_times)
        C_avg_normalized_sorted = C_avg_normalized[sorted_indices]
        C_avg_mean = np.mean(C_avg_normalized_sorted, axis=0)

        return C_avg, C_avg_normalized, C_avg_normalized_sorted, C_avg_mean, sorted_indices

    def find_movement_onset(self, velocity, velocity_peak_indices, threshold_ratio=0.1):
        """
        Find the onset of movement based on the velocity peaks and a threshold ratio.
        :param velocity: velocity data, can be position change rate or real velocity
        :param velocity_peak_indices: indices of the velocity peaks
        :param threshold_ratio: ratio of the peak value to be used as the threshold
        :return: list of onset indices
        """
        onsets = []
        for i, peak_index in enumerate(velocity_peak_indices):
            peak_value = velocity[peak_index]
            threshold = peak_value * threshold_ratio

            # Search backwards from the peak to find when the velocity first rises above the threshold
            pre_peak_values = velocity[:peak_index]
            below_threshold_indices = np.where(pre_peak_values < threshold)[0]

            if below_threshold_indices.size > 0:
                onset_index = below_threshold_indices[-1] + 1  # Corrected to point to the first index above threshold

                # If the onset index is before the last peak (misidentification), find local minimum between peaks
                if i > 0 and onset_index <= velocity_peak_indices[i - 1]:
                    segment = velocity[velocity_peak_indices[i - 1]:peak_index]
                    local_min_index = np.argmin(segment) + velocity_peak_indices[i - 1]
                    onset_index = local_min_index
            else:
                # If no below-threshold value is found, default to start of the data
                onset_index = 0

            onsets.append(onset_index)

        return onsets

    @staticmethod
    def ensure_tensor(data, device):
        """
        Ensure that the data is a tensor on the specified device.
        :param data: input data
        :param device: device to move the data to
        :return: tensor on the specified device
        """
        if isinstance(data, np.ndarray):
            return torch.from_numpy(data).to(device)
        elif isinstance(data, torch.Tensor):
            return data.to(device)
        else:
            raise ValueError("Unsupported data type. Expected numpy.ndarray or torch.Tensor")

    def compute_correlation_statistics_batched(self, trace, num_trials=2000, max_shift=6000, device=None,
                                               use_tqdm=True, notebook=False):
        """
        Compute the correlation statistics for the specified trace. This function uses batch processing to speed up the computation.
        Randomly shifts the calcium traces and computes the correlation coefficients and shifts for each trial.
        :param trace: trace to compute the correlation statistics for (e.g., lick, position change rate, velocity)
        :param num_trials: number of shifts to run
        :param max_shift: maximum shift to apply to the calcium traces
        :param device: device to run the computation on
        :param use_tqdm: flag to use tqdm for progress bar
        :param notebook: flag to use tqdm.notebook for Jupyter notebook
        :return: correlation coefficients, shifts, precomputed data
        """
        if notebook:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Check if inputs are tensors; if not, convert them
        C_raw_tensor = self.ensure_tensor(self.C_raw, device)
        trace_tensor = self.ensure_tensor(trace, device)

        # Compute the correlation coefficients and shifts using batch processing
        coefficients_all, shifts_all = self.randomly_shift_crop_and_correlate_batched(C_raw_tensor, trace_tensor,
                                                                                      max_shift,
                                                                                      num_trials, device, use_tqdm,
                                                                                      notebook)

        coefficients_all_numpy = coefficients_all.cpu().numpy()
        shifts_all_numpy = shifts_all.cpu().numpy()

        precomputed_data = {
            'histograms': [],
            'edges': [],
            'details': []
        }

        # Choose iterator based on use_tqdm flag
        iterator = tqdm(range(coefficients_all_numpy.shape[0]),
                        desc="Computing histograms and details") if use_tqdm else range(coefficients_all_numpy.shape[0])
        for i in iterator:
            coefficients = coefficients_all_numpy[i]
            shifts = shifts_all_numpy[i]

            hist, edges = np.histogram(coefficients, bins=30)
            precomputed_data['histograms'].append(hist.tolist())
            precomputed_data['edges'].append(edges.tolist())

            bin_details = [[] for _ in range(len(hist))]
            for coef, shift in zip(coefficients, shifts):
                bin_index = np.digitize(coef, edges) - 1
                bin_index = min(bin_index, len(hist) - 1)
                bin_details[bin_index].append(str(shift))

            bin_details_str = [", ".join(details) for details in bin_details]
            precomputed_data['details'].append(bin_details_str)

        return coefficients_all_numpy, shifts_all_numpy, precomputed_data

    def randomly_shift_crop_and_correlate_batched(self, C_raw_tensor, trace_tensor, max_shift, num_trials, device,
                                                  use_tqdm, notebook):
        """
        Randomly shift the calcium traces, crop the signals, and compute the correlation coefficients for each trial.
        :param C_raw_tensor: tensor of calcium traces
        :param trace_tensor: tensor of the trace to correlate with
        :param max_shift: maximum shift to apply to the calcium traces
        :param num_trials: number of trials to run
        :param device: device to run the computation on (e.g., 'cpu' or 'cuda')
        :param use_tqdm: flag to use tqdm for progress bar
        :param notebook: flag to use tqdm.notebook for Jupyter notebook
        :return: correlation coefficients, shifts
        """
        if notebook:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm
        num_rows = C_raw_tensor.size(0)
        signal_length = C_raw_tensor.size(1)

        # Generate a single set of random shifts for all trials
        shifts = torch.randint(-max_shift, max_shift + 1, (num_trials,), device=device)

        # Initialize tensors to hold the correlation coefficients and shifts for all rows and trials
        correlation_coefficients = torch.zeros((num_rows, num_trials), device=device)
        # Choose iterator based on use_tqdm flag
        iterator = tqdm(shifts, desc="Computing correlations") if use_tqdm else shifts
        for trial, shift_amount in enumerate(iterator):
            shifted_signals = torch.roll(C_raw_tensor, shifts=int(shift_amount.item()), dims=1)
            valid_range = slice(max_shift, signal_length - max_shift)
            cropped_signals1 = shifted_signals[:, valid_range]
            cropped_signals2 = trace_tensor[valid_range].expand_as(cropped_signals1)
            cropped_signals1 = cropped_signals1.float()
            cropped_signals2 = cropped_signals2.float()

            correlations = self.pearson_correlation_batched(cropped_signals1, cropped_signals2)
            correlation_coefficients[:, trial] = correlations

        shift_amounts = shifts.expand(num_rows, num_trials)
        return correlation_coefficients, shift_amounts

    @staticmethod
    def pearson_correlation_batched(x, y):
        vx = x - torch.mean(x, dim=1, keepdim=True)
        vy = y - torch.mean(y, dim=1, keepdim=True)
        cost = torch.sum(vx * vy, dim=1) / (
                torch.sqrt(torch.sum(vx ** 2, dim=1)) * torch.sqrt(torch.sum(vy ** 2, dim=1)))
        return cost

    @staticmethod
    def compute_outliers(coefficients_all_numpy, threshold=4):
        """
        Computes the outliers based on the deviation from the mean beyond a specified threshold.

        :param numpy.ndarray coefficients_all_numpy: Array of coefficients from which to compute outliers.
        :param float threshold: The threshold for determining outliers, in terms of standard deviations from the mean.
        :return: A tuple containing two lists: `outliers_all`, which includes the mean of outliers or 100 if no outliers were found for a row, and `outliers_all_raw`, which includes arrays of outliers for each row.
        :rtype: (numpy.ndarray, list)
        """
        outliers_all = []
        outliers_all_raw = []
        num_rows = coefficients_all_numpy.shape[0]
        interval_lines = np.zeros((num_rows, 2))

        for i in range(num_rows):
            coefficients = coefficients_all_numpy[i]
            mean = np.mean(coefficients)
            std_dev = np.std(coefficients)
            outliers = coefficients[
                (coefficients < mean - threshold * std_dev) | (coefficients > mean + threshold * std_dev)]
            outliers_all_raw.append(outliers)
            interval_lines[i] = [mean - threshold * std_dev, mean + threshold * std_dev]

            if len(outliers) == 0:
                outliers_all.append(100)
            else:
                outliers_all.append(np.mean(outliers))

        outliers_all = np.array(outliers_all)
        return outliers_all_raw, interval_lines

    def compute_correlation_statistics_pairwise_batched_optimized(self, num_trials=2000, max_shift=6000, device=None,
                                                                  use_tqdm=True, notebook=False):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        C_raw_tensor = torch.from_numpy(self.C_raw).to(device)

        # Process in batched manner without explicit Python loops if possible
        coefficients_all_tensor, shifts_all_tensor = self.randomly_shift_crop_and_correlate_batched_optimized(
            C_raw_tensor, max_shift, num_trials, device, use_tqdm, notebook
        )

        coefficients_all = coefficients_all_tensor.cpu().numpy()
        shifts_all = shifts_all_tensor.cpu().numpy()

        return coefficients_all, shifts_all

    def randomly_shift_crop_and_correlate_batched_optimized(self, C_raw_tensor, max_shift, num_trials, device, use_tqdm,
                                                            notebook):
        if notebook:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm

        num_rows = C_raw_tensor.size(0)
        signal_length = C_raw_tensor.size(1)

        # Generate random shifts
        shifts = torch.randint(-max_shift, max_shift + 1, (num_trials,), device=device)

        # Initialize tensors to hold the correlation coefficients for all rows and trials
        correlation_coefficients = torch.zeros((num_rows, num_rows, num_trials), device=device)

        # Choose iterator based on use_tqdm flag
        iterator = tqdm(shifts, desc="Computing correlations") if use_tqdm else shifts

        for trial, shift_amount in enumerate(iterator):
            shifted_C_raw_tensor = torch.roll(C_raw_tensor, shifts=int(shift_amount.item()), dims=1)
            valid_range = slice(max_shift, signal_length - max_shift)
            cropped_C_raw_tensor = shifted_C_raw_tensor[:, valid_range]

            # Compute correlations for each row with every other row
            for i in range(num_rows):
                trace_tensor = C_raw_tensor[i, valid_range].unsqueeze(0).expand(num_rows, -1)
                cropped_trace_tensor = trace_tensor.float()
                correlations = self.pearson_correlation_batched(cropped_C_raw_tensor.float(), cropped_trace_tensor)
                correlation_coefficients[:, i, trial] = correlations

        # The shifts tensor will be the same for each pair, so we replicate it across the dimensions to match the output shape
        shifts_all = shifts.expand(num_rows, num_rows, -1)

        return correlation_coefficients, shifts_all

    @staticmethod
    def compute_outliers_pairwise(coefficients_all_numpy, threshold=4, use_tqdm=True, notebook=False):
        """
        Computes the pairwise correlation coefficients outliers based on the deviation from the mean beyond a specified threshold.

        :param numpy.ndarray coefficients_all_numpy: Array of coefficients from which to compute outliers.
        :param float threshold: The threshold for determining outliers, in terms of standard deviations from the mean.
        :return: A tuple containing two lists: `outliers_all`, which includes the mean of outliers or 100 if no outliers were found for a row, and `outliers_all_raw`, which includes arrays of outliers for each row.
        """
        if notebook:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm

        num_rows = coefficients_all_numpy.shape[0]
        outliers_all = np.zeros((num_rows, num_rows))
        outliers_all_raw = [[[] for _ in range(num_rows)] for _ in range(num_rows)]

        iterator = tqdm(range(num_rows), desc="Computing outliers") if use_tqdm else range(num_rows)

        for i in iterator:
            for j in range(i, num_rows):
                coefficients = coefficients_all_numpy[i, j]
                mean = np.mean(coefficients)
                std_dev = np.std(coefficients)
                outliers = coefficients[
                    (coefficients < mean - threshold * std_dev) | (coefficients > mean + threshold * std_dev)]
                outliers_all_raw[i][j] = outliers
                outliers_all_raw[j][i] = outliers

                if len(outliers) != 0:
                    outliers_all[i, j] = 1
                    outliers_all[j, i] = 1

        return outliers_all, outliers_all_raw

    def compute_pairwise_correlation_matrix_bs(self, outliers_pairwise_raw):
        """
        Compute the bootstrapping pairwise correlation matrix.
        :return: bootstrapping pairwise correlation matrix
        """
        real_cor_all = np.zeros((self.neuron_num, self.neuron_num))
        for i in range(self.neuron_num):
            for j in range(i, self.neuron_num):
                if len(outliers_pairwise_raw[i][j]) > 0:
                    real_cor = outliers_pairwise_raw[i][j][np.argmax(abs(outliers_pairwise_raw[i][j]))]
                    real_cor_all[i, j] = real_cor
                    real_cor_all[j, i] = real_cor

        return real_cor_all

    @staticmethod
    def output_bokeh_plot(plot, save_path, title, notebook, overwrite):
        import os
        from bokeh.io import output_notebook, output_file, reset_output, save, show, curdoc

        if save_path is not None:
            reset_output()
            if os.path.exists(save_path) and not overwrite:
                print("File already exists and overwrite is set to False.")
            else:
                output_file(save_path, title=title)
                curdoc().clear()
                curdoc().add_root(plot)
                save(curdoc())
                print("File saved or overwritten.")
                curdoc().clear()

        if notebook:
            reset_output()
            output_notebook()
            show(plot)

    def plot_t_ca_all(self, notebook=False, save_path=None, overwrite=False):
        """
        Plot the average calcium trace for all neurons.
        :param notebook: Flag to indicate if the plot is for a Jupyter notebook.
        :param save_path: Path to save the plot as an HTML file.
        :return:
        """
        from bokeh.plotting import figure

        p = figure(width=800, height=400, active_scroll="wheel_zoom", title="Average Calcium Trace (Δf/f)")
        p.line(self.t, self.ca_all, line_width=2)

        self.output_bokeh_plot(p, save_path=save_path, title=str(p.title.text), notebook=notebook, overwrite=overwrite)

        return p

    def plot_lick_and_velocity(self, title="Lick And Velocity", notebook=False, save_path=None, overwrite=False):
        """
        Plot the lick data and velocity data.
        :param notebook: Flag to indicate if the plot is for a Jupyter notebook.
        :param save_path:  Path to save the plot as an HTML file.
        :param overwrite: Flag to indicate whether to overwrite the existing plot
        :return: None
        """
        from bokeh.plotting import figure
        from bokeh.models import Span, HoverTool
        from bokeh.layouts import column

        p = figure(width=800, height=400, y_range=[0, 1.2], active_drag='pan', active_scroll='wheel_zoom', title='Lick')
        p.line(self.t, self.lick_raw, line_color='navy', legend_label='raw', line_width=1)
        p.line(self.t, self.lick_raw_mask, line_color='palevioletred', legend_label='mask', line_width=2)
        p.line(self.t, self.lick, line_color='gold', legend_label='filtered', line_width=2)

        for idx in self.trials_end_indices:
            vline = Span(location=idx / self.ci_rate, dimension='height', line_color='green', line_width=2)
            p.add_layout(vline)

        p.legend.click_policy = "hide"

        hover = HoverTool()
        hover.tooltips = [
            ("Index", "$index"),
            ("(x, y)", "($x, $y)"),
        ]

        p.add_tools(hover)

        ################################################################

        p_v = figure(width=800, height=400, x_range=p.x_range,
                     active_drag='pan', active_scroll='wheel_zoom', title="Velocity")
        p_v.line(self.t, self.pstcr_raw, line_color='navy', legend_label='dp_raw', line_width=2)
        p_v.line(self.t, self.pstcr, line_color='red', legend_label='dp_filtered', line_width=2)
        p_v.line(self.t, self.velocity, line_color='purple', legend_label='velocity', line_width=2)

        hover_v = HoverTool()
        hover_v.tooltips = [
            ("Index", "$index"),
            ("(x, y)", "($x, $y)"),
        ]

        p_v.add_tools(hover_v)
        p_v.legend.click_policy = "hide"

        for idx in self.trials_end_indices:
            vline = Span(location=idx / self.ci_rate, dimension='height', line_color='green', line_width=2)
            p_v.add_layout(vline)

        layout = column(p, p_v)

        self.output_bokeh_plot(layout, save_path=save_path, title=title, notebook=notebook, overwrite=overwrite)

        return layout

    def plot_ca_around_indices(self, map_data, indices, cut_interval=50, title="None", notebook=False, save_path=None,
                               overwrite=False):
        """
        Plot the calcium temporal traces around the specified indices.
        :param map_data  A 2D NumPy array containing the calcium traces.
        :param indices: Indices to plot around.
        :param cut_interval: Interval to cut around the indices.
        :param notebook: Flag to indicate if the plot is for a Jupyter notebook.
        :param save_path: Path to save the plot as an HTML file.
        :param overwrite: Flag to indicate whether to overwrite the existing file
        :return: None
        """

        from bokeh.plotting import figure
        from bokeh.models import ColumnDataSource, LinearColorMapper, ColorBar, LabelSet, CustomJSTickFormatter
        from bokeh.layouts import column

        x_value = (np.array(range(2 * cut_interval)) - cut_interval) / self.ci_rate

        mapper = LinearColorMapper(palette="Viridis256", low=0, high=1)

        p = figure(width=800, height=800, title=title,
                   x_axis_label='Time', y_axis_label='Neuron Number', active_scroll='wheel_zoom',
                   x_range=(-cut_interval, cut_interval), y_range=(0, map_data.shape[0]))

        p.image(image=[np.flipud(map_data)], x=-cut_interval, y=0, dw=map_data.shape[1], dh=map_data.shape[0],
                color_mapper=mapper)

        color_bar = ColorBar(color_mapper=mapper, label_standoff=12, location=(0, 0))
        p.add_layout(color_bar, 'right')

        source = ColumnDataSource(data=dict(
            y=list(range(map_data.shape[0])),
            neuron_index=indices[::-1]
        ))

        labels = LabelSet(x=-cut_interval, y='y', text='neuron_index', level='glyph',
                          x_offset=-20, y_offset=5, source=source, text_font_size="8pt", text_color="black")
        p.add_layout(labels)

        p.xaxis.formatter = CustomJSTickFormatter(code="""
            return (tick / 20).toFixed(2);
        """)

        v = figure(width=800, height=250, active_scroll="wheel_zoom",
                   toolbar_location=None,
                   x_range=(min(x_value), max(x_value)))
        v.line(x_value,
               np.mean(map_data, axis=0), line_width=2,
               color='red')

        layout = column(p, v)

        self.output_bokeh_plot(layout, save_path=save_path, title=title, notebook=notebook, overwrite=overwrite)

        return layout

    def create_outlier_visualization(self, precomputed_data, coefficients_all_numpy, threshold=4, title=None,
                                     notebook=False, save_path=None, overwrite=False):
        """
        Creates an interactive visualization to display outliers and mean value for a set of coefficients.

        :param precomputed_data: A dictionary containing histograms, edges, and details for visualization.
        :param coefficients_all_numpy: A NumPy array of coefficients to compute outliers and mean values.
        :param threshold: The threshold for determining outliers, defaults to 4.
        :param notebook: Flag to indicate if the visualization is for a Jupyter notebook.
        :param save_path: Path to save the visualization as an HTML file.
        """

        from bokeh.plotting import figure
        from bokeh.models import ColumnDataSource, Span, Slider, CustomJS, Div, TapTool
        from bokeh.layouts import column, row

        def find_outliers(data, threshold=4.0):
            """Find the lower and upper bounds of outliers."""
            mean = np.mean(data)
            std_dev = np.std(data)
            lower_bound = mean - threshold * std_dev
            upper_bound = mean + threshold * std_dev
            return lower_bound, upper_bound


        # Initialize the plot data source
        initial_histogram = precomputed_data['histograms'][0]
        initial_edges = precomputed_data['edges'][0]
        initial_details = precomputed_data['details'][0]
        source = ColumnDataSource(data=dict(
            top=initial_histogram,
            bottom=[0] * len(initial_histogram),
            left=initial_edges[:-1],
            right=initial_edges[1:],
            details=initial_details
        ))

        # Compute positions for outlier spans and mean value
        positions_arrays = [[], [], []]
        for coefficients in coefficients_all_numpy:
            lower, upper = find_outliers(coefficients, threshold=threshold)
            mean_val = np.mean(coefficients)
            positions_arrays[0].append(lower)
            positions_arrays[1].append(upper)
            positions_arrays[2].append(mean_val)

        # Create the figure
        p = figure(width=700, height=400, title=title, active_scroll="wheel_zoom")
        p.quad(top='top', bottom='bottom', left='left', right='right', source=source,
               fill_color="skyblue", line_color="white", alpha=0.5)

        # Add spans for lower bound, upper bound, and mean
        spans = [Span(location=positions_arrays[0][0], dimension='height', line_color='red', line_width=2,
                      line_dash='dashed'),
                 Span(location=positions_arrays[1][0], dimension='height', line_color='green', line_width=2,
                      line_dash='dashed'),
                 Span(location=positions_arrays[2][0], dimension='height', line_color='navy', line_width=2)]
        for span in spans:
            p.add_layout(span)

        # Additional plot components
        details_div = Div(width=300, height=p.height, sizing_mode="fixed", text="Shift amounts will appear here")
        row_slider = Slider(start=0, end=len(coefficients_all_numpy) - 1, value=0, step=1, width=600,
                            title="Select Row")

        # Configure callbacks for interactivity
        callback_code = """
            var row = row_slider.value;
            var histograms = precomputed_data['histograms'][row];
            var edges = precomputed_data['edges'][row];
            var details = precomputed_data['details'][row];

            source.data['top'] = histograms;
            source.data['bottom'] = new Array(histograms.length).fill(0);
            source.data['left'] = edges.slice(0, -1);
            source.data['right'] = edges.slice(1);
            source.data['details'] = details;

            source.change.emit();

            for (let i = 0; i < spans.length; i++) {
                spans[i].location = positions_arrays[i][row];
            }
        """
        callback = CustomJS(args=dict(source=source, details_div=details_div, precomputed_data=precomputed_data,
                                      row_slider=row_slider, spans=spans, positions_arrays=positions_arrays),
                            code=callback_code)

        tap_callback = CustomJS(args=dict(source=source, details_div=details_div), code="""
            const indices = cb_obj.indices;
            const details = source.data['details'][indices[0]];
            details_div.text = "Shift amounts: " + details;
        """)

        # Add tools and configure plot layout
        p.add_tools(TapTool())
        source.selected.js_on_change('indices', tap_callback)
        row_slider.js_on_change('value', callback)
        layout = column(row_slider, row(p, details_div))

        self.output_bokeh_plot(layout, save_path=save_path, title=title, notebook=notebook, overwrite=overwrite)

        return layout

    def generate_colors(self, num_categories):
        """
        Generates a list of colors for the pie chart. Uses a predefined palette and extends it if necessary.
        """
        from bokeh.palettes import Category20c
        import itertools
        palette = Category20c[20]  # Using a larger palette from Bokeh
        if num_categories <= len(palette):
            return palette[:num_categories]
        else:
            # Extend the palette by repeating it
            return list(itertools.islice(itertools.cycle(palette), num_categories))

    def categorize_neurons(self, picked_neurons_velocity, picked_neurons_lick, picked_neurons_movement):
        """
        Categorize neurons based on the picked neurons for velocity, lick, and movement.
        :param picked_neurons_velocity:  Set of neuron indices picked for velocity
        :param picked_neurons_lick:  Set of neuron indices picked for lick
        :param picked_neurons_movement:  Set of neuron indices picked for movement
        :return: Dictionary containing neuron categories as keys and neuron indices as values
        """
        all_neurons = set(range(self.neuron_num))

        # Ensure that the inputs are sets
        if not isinstance(picked_neurons_velocity, set):
            picked_neurons_velocity = set(picked_neurons_velocity)

        if not isinstance(picked_neurons_lick, set):
            picked_neurons_lick = set(picked_neurons_lick)

        if not isinstance(picked_neurons_movement, set):
            picked_neurons_movement = set(picked_neurons_movement)

        # Exclusive categories
        velocity_only = picked_neurons_velocity - picked_neurons_lick - picked_neurons_movement
        lick_only = picked_neurons_lick - picked_neurons_velocity - picked_neurons_movement
        movement_only = picked_neurons_movement - picked_neurons_velocity - picked_neurons_lick

        # Intersection categories
        velocity_and_lick = picked_neurons_velocity & picked_neurons_lick - picked_neurons_movement
        velocity_and_movement = picked_neurons_velocity & picked_neurons_movement - picked_neurons_lick
        lick_and_movement = picked_neurons_lick & picked_neurons_movement - picked_neurons_velocity
        all_three = picked_neurons_velocity & picked_neurons_lick & picked_neurons_movement

        # Neurons not in any category
        none = all_neurons - (
                velocity_only | lick_only | movement_only | velocity_and_lick | velocity_and_movement | lick_and_movement | all_three)

        # Prepare the categories with labels and neuron indices
        categories = {
            'Velocity Only': velocity_only,
            'Lick Only': lick_only,
            'Movement Only': movement_only,
            'Velocity & Lick': velocity_and_lick,
            'Velocity & Movement': velocity_and_movement,
            'Lick & Movement': lick_and_movement,
            'All Three': all_three,
            'None': none
        }

        return categories

    def create_neuron_categories_pie_chart(self, neuron_categories, title="Neuron Categories", notebook=False,
                                           save_path=None, overwrite=False):
        """
        Create a pie chart to visualize the distribution of neurons across different categories.
        :param neuron_categories: dictionary containing neuron categories as keys and neuron indices as values (e.g., {'Category 1': [0, 1, 2], 'Category 2': [3, 4, 5]})
        :param title: title of the pie chart
        :param notebook: flag to indicate if the visualization is for a Jupyter notebook
        :param save_path: path to save the visualization as an HTML file
        :return: None

        """
        from bokeh.plotting import figure, show
        from bokeh.models import ColumnDataSource
        from bokeh.io import output_notebook, output_file, save

        counts = {category: len(neurons) for category, neurons in neuron_categories.items()}

        # Prepare data for the pie chart
        data = pd.Series(counts).reset_index(name='value').rename(columns={'index': 'category'})
        data['angle'] = data['value'] / data['value'].sum() * 2 * np.pi

        # Generate colors for the categories
        num_categories = len(counts)
        colors = self.generate_colors(num_categories)
        data['color'] = colors[:num_categories]

        data['end_angle'] = np.cumsum(data['angle'])
        data['start_angle'] = np.cumsum(data['angle']) - data['angle']

        p = figure(width=600, height=400, title=title, tooltips="@category: @value", x_range=(-0.5, 1.0))

        p.wedge(x=0, y=1, radius=0.4,
                start_angle='start_angle', end_angle='end_angle',
                line_color="white", fill_color='color', legend_field='category', source=ColumnDataSource(data))

        p.axis.axis_label = None
        p.axis.visible = False
        p.grid.grid_line_color = None
        p.legend.location = "top_right"

        self.output_bokeh_plot(p, save_path=save_path, title=title, notebook=notebook, overwrite=overwrite)

        return p

    def create_heatmap_with_trace(self, sorted_indices, trace, cutoff_index=10000, title=None,
                                  save_path=None, notebook=False, overwrite=False):
        """
         Create a heatmap
         :param sorted_indices: sorted indices (could be based on correlation coefficient)
         :param trace: trace to plot
         :param cutoff_index: Only plot part of the calcium heatmap
         :param title: title of the plot
         :param save_path: path to save the plot
         :param notebook: whether use notebook to visualize the heatmap
         :param overwrite: whether to overwrite the existing plot

         :return: the heatmap plot

        """

        from bokeh.plotting import figure
        from bokeh.models import LinearColorMapper, ColorBar, CustomJSTickFormatter
        from bokeh.layouts import column

        map_data = self.C_raw[sorted_indices, :cutoff_index]

        mapper = LinearColorMapper(palette="Viridis256", low=-0.2, high=0.2)

        p = figure(width=800, height=800, title=title,
                   x_axis_label='Time', y_axis_label='Neuron Number', active_scroll='wheel_zoom',
                   x_range=(0, 5000), y_range=(0, map_data.shape[0]))

        p.image(image=[map_data], x=0, y=0, dw=map_data.shape[1], dh=map_data.shape[0], color_mapper=mapper)

        p.xaxis.formatter = CustomJSTickFormatter(code="""
            return (tick / 20).toFixed(2);
        """)

        color_bar = ColorBar(color_mapper=mapper, label_standoff=12, location=(0, 0))
        p.add_layout(color_bar, 'right')

        v = figure(width=800, height=200, x_range=p.x_range, active_scroll="wheel_zoom")
        v.line(np.arange(0, len(self.t), 1), trace, line_width=2, color='red')

        v.xaxis.formatter = CustomJSTickFormatter(code="""
            return (tick / 20).toFixed(2);
        """)

        layout = column(p, v)

        self.output_bokeh_plot(layout, save_path=save_path, title=title, notebook=notebook, overwrite=overwrite)

        return layout

    def plot_calcium_trace_with_lick_and_velocity(self, title="Average Calcium Trace with Lick and Speed",
                                                  save_path=None, notebook=False, overwrite=False):

        from bokeh.plotting import figure
        from bokeh.models import Span

        p = figure(width=800, height=400, active_scroll="wheel_zoom", title=title)
        p.line(self.t, self.normalize_signal(self.ca_all), line_width=2, legend_label='Ca', color='blue')
        p.line(self.t, self.lick, line_width=2, legend_label='Lick', color='gold')
        p.line(self.t, self.normalize_signal(self.velocity), line_width=2, legend_label='Velocity', color='red')
        p.line(self.t, self.normalize_signal(self.smoothed_velocity), line_width=2, legend_label='Smoothed Velocity',
               color='green')

        for idx in self.velocity_peak_indices:
            vline = Span(location=idx / self.ci_rate, dimension='height', line_color='green', line_width=2)
            p.add_layout(vline, 'below')
        for idx in self.movement_onset_indices:
            vline = Span(location=idx / self.ci_rate, dimension='height', line_color='brown', line_width=2)
            p.add_layout(vline, 'below')

        p.legend.click_policy = 'hide'
        self.output_bokeh_plot(p, save_path=save_path, title=title, notebook=notebook, overwrite=overwrite)

        return p

    def plot_calcium_trace_around_indices(self, cut_interval=50, save_path=None, title="Average Calcium Trace around Indices",
                                          notebook=False, overwrite=False):

        from bokeh.plotting import figure
        from bokeh.models import CustomJSTickFormatter

        [_, _, _, C_avg_mean_velocity_peak, _] = self.average_across_indices(self.velocity_peak_indices,
                                                                             cut_interval=cut_interval)
        [_, _, _, C_avg_mean_lick_edge, _] = self.average_across_indices(self.lick_edge_indices,
                                                                         cut_interval=cut_interval)
        [_, _, _, C_avg_mean_movement_onset, _] = self.average_across_indices(self.movement_onset_indices,
                                                                              cut_interval=cut_interval)

        p = figure(width=800, height=400, active_scroll="wheel_zoom", title="Average Calcium Trace around Indices")
        p.line((np.array(range(2 * cut_interval)) - cut_interval)/self.ci_rate, C_avg_mean_velocity_peak, line_width=2,
               legend_label='Velocity Peak', color='red')
        p.line((np.array(range(2 * cut_interval)) - cut_interval)/self.ci_rate, C_avg_mean_lick_edge, line_width=2,
               legend_label='Lick', color='navy')
        p.line((np.array(range(2 * cut_interval)) - cut_interval)/self.ci_rate, C_avg_mean_movement_onset, line_width=2,
               legend_label='Movement Onset', color='purple')

        p.legend.click_policy = 'hide'

        p.xaxis.formatter = CustomJSTickFormatter(code="""
            return (tick / 20).toFixed(2);
        """)

        self.output_bokeh_plot(p, save_path=save_path, title=title, notebook=notebook, overwrite=overwrite)

        return p


if __name__ == "__main__":
    print("Starting CivisServer")
