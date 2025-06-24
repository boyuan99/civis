import pandas as pd
import numpy as np
import h5py
import json
import os
from civis.src.VirmenTank import VirmenTank
from scipy.signal import savgol_filter


class CITank(VirmenTank):
    def __init__(self,
                 session_name,
                 ci_path=None,
                 virmen_path=None,
                 gcamp_path=None,
                 tdt_org_path=None,
                 tdt_adjusted_path=None,
                 maze_type=None,
                 ci_rate=20,
                 vm_rate=20,
                 session_duration=30 * 60):

        self.session_name = session_name
        self.config = self.load_config()

        ci_path = ci_path or os.path.join(self.config['ProcessedFilePath'], session_name, f"{session_name}_v7.mat")
        virmen_path = virmen_path or os.path.join(self.config['VirmenFilePath'], f"{session_name}.txt")
        gcamp_path = gcamp_path or os.path.join(self.config['ProcessedFilePath'], session_name,
                                                f"{session_name}_tiff_projections", f"{session_name}_max.tif")
        tdt_org_path = tdt_org_path or os.path.join(self.config['ProcessedFilePath'], session_name,
                                                f"{session_name}_alignment_check", f"{session_name}_tdt_original_16bit.tif")
        tdt_adjusted_path = tdt_adjusted_path or os.path.join(self.config['ProcessedFilePath'], session_name,
                                                f"{session_name}_tdt_adjustment", f"{session_name}_tdt_adjusted_16bit.tif")


        super().__init__(
            session_name=session_name,
            virmen_path=virmen_path,
            maze_type=maze_type,
            vm_rate=vm_rate,
            session_duration=session_duration)

        self.ci_path = ci_path
        self.gcamp_path = gcamp_path
        self.tdt_adjusted_path = tdt_adjusted_path
        self.tdt_org_path = tdt_org_path
        self.ci_rate = ci_rate
        self.session_duration = session_duration

        (self.C, self.C_raw, self.Cn, self.ids, self.Coor, self.centroids,
         self.C_denoised, self.C_deconvolved, self.C_baseline, self.C_reraw, self.A) = self._load_data(ci_path)

        self.neuron_num = self.C_raw.shape[0]
        self.C_raw_deltaF_over_F = self._compute_deltaF_over_F()
        self.C_zsc = self._z_score_normalize_all()
        self.ca_all = self.normalize_signal(self.shift_signal_single(np.mean(self.C_zsc, axis=0)))
        self.peak_indices = self._find_peaks_in_traces()
        self.rising_edges_starts = self._find_rising_edges_starts()
        

    @staticmethod
    def _load_data(filename):
        """
        Load all data
        :param filename: .mat file containing config, spatial, Cn, Coor, ids, etc...
        :return: essential variables for plotting
        """
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        config_path = os.path.join(project_root, 'config.json')

        # Load the config
        with open(config_path, 'r') as config_file:
            config = json.load(config_file)

        print(f"Opening: {filename}...")
        with h5py.File(filename, 'r') as file:
            data = file['data']
            Cn = np.transpose(data['Cn'][()])
            ids = data['ids'][()] - 1
            Coor_cell_array = data['Coor']
            Coor = []
            C = np.transpose(data['C'][()])
            C_raw = np.transpose(data['C_raw'][()])
            C_denoised = np.transpose(data['C_denoised'][()])
            C_deconvolved = np.transpose(data['C_deconvolved'][()])
            C_baseline = np.transpose(data['C_baseline'][()])
            C_reraw = np.transpose(data['C_reraw'][()])
            centroids = np.transpose(data['centroids'][()])
            A = data['A'][()]

            for i in range(Coor_cell_array.shape[1]):
                ref = Coor_cell_array[0, i]  # Get the reference
                coor_data = file[ref]  # Use the reference to access the data
                coor_matrix = np.array(coor_data)  # Convert to a NumPy array

                Coor.append(np.transpose(coor_matrix))

        return C, C_raw, Cn, ids, Coor, centroids, C_denoised, C_deconvolved, C_baseline, C_reraw, A

    @staticmethod
    def calculate_delta_f_over_f(signal, baseline_percentile=10):
        """计算ΔF/F"""
        F0 = np.percentile(signal, baseline_percentile)
        delta_f_over_f = (signal - F0) / F0

        return delta_f_over_f, F0

    def _compute_deltaF_over_F(self):

        signals = np.zeros_like(self.C_raw)
        for i in range(self.neuron_num):
            delta_f_over_f, _ = self.calculate_delta_f_over_f(self.C_raw[i])
            signals[i] = delta_f_over_f

        return signals
    
    @staticmethod
    def z_score_normalize(signal):
        """Z-score normalization"""
        mean_val = np.mean(signal)
        std_val = np.std(signal)
        if std_val == 0:
            return np.zeros_like(signal)
        return (signal - mean_val) / std_val
    
    def _z_score_normalize_all(self):
        signals = np.zeros_like(self.C_raw)
        for i in range(self.neuron_num):
            signals[i] = self.z_score_normalize(self.C_raw[i])

        return signals

    @staticmethod
    def find_outliers_indices(data, threshold=3.0):
        mean = np.mean(data)
        std_dev = np.std(data)
        z_scores = (data - mean) / std_dev
        outliers = np.where(np.abs(z_scores) > threshold)[0]
        return outliers

    def _find_peaks_in_traces(self, prominence=0.5, min_width=3, wlen=None):
        """
        Improved peak detection function, using prominence and width as main parameters
        
        Parameters:
        prominence : float or None
            Prominence factor relative to signal noise level, if None then calculated for each neuron
        min_width : int
            Minimum peak width (samples)
        wlen : int or None
            Window length for calculating prominence
            
        Returns:
        list of arrays
            List of peak indices for each neuron
        """
        from scipy.signal import find_peaks

        peak_indices_all = []
        
        for i in range(len(self.C_denoised)):
            signal = self.C_denoised[i]
            
            # adaptive calculation of prominence threshold
            if prominence is None:
                # use median absolute deviation (MAD) as noise estimator, more robust
                noise_level = np.median(np.abs(signal - np.median(signal))) * 1.4826
                min_prominence = 2.0 * noise_level  # 2 times MAD as default threshold
            else:
                # use user-specified prominence factor
                noise_level = np.std(signal)
                min_prominence = prominence * noise_level
            
            # find peaks, mainly rely on prominence and width
            peaks, _ = find_peaks(signal, 
                                 prominence=min_prominence,
                                 width=min_width,
                                 wlen=wlen)
            
            peak_indices_all.append(peaks)
        
        return peak_indices_all
    
    def _find_rising_edges_starts(self):
        """
        Find the start indices of rising edges in the calcium traces.
        """
        rising_edges_starts = []
        for i in range(self.neuron_num):
            c1 = np.where(self.C_deconvolved[i] > 0.0001, 1, 0)
            c2 = np.diff(c1, append=c1[-1])
            c3 = np.where(c2 == 1, 1, 0)

            rising_edges_starts.append(np.where(c3 == 1)[0])

        return rising_edges_starts

    def compute_correlation_ca_instance(self, instance):
        """
        Calculate the Pearson correlation coefficient between a given instance and each calcium trace in the dataset.

        :param instance: A numpy array representing a single instance (e.g., velocity, lick).
        :type instance: np.ndarray
        :returns: A numpy array of Pearson correlation coefficients between the given instance and each neuron's calcium trace.
        :rtype: np.ndarray

        The returned array has one correlation coefficient per neuron.
        """
        ca_instance_correlation_array = np.zeros(self.neuron_num)
        for neuron_pick in range(self.neuron_num):
            ca_instance_correlation_array[neuron_pick] = np.corrcoef(instance, self.C_raw[neuron_pick])[0, 1]

        return ca_instance_correlation_array

    def compute_cross_correlation_ca_instance(self, instance, mode='same'):
        """
        Compute the cross-correlation between a given instance and each calcium trace across all neurons.

        :param instance: A numpy array representing a single instance (e.g., velociity, lick).
        :type instance: np.ndarray
        :param mode: Specifies the size of the output: 'full', 'valid', or 'same'. Default is 'same'.
        :type mode: str
        :returns: A tuple containing:
            - correlations: An array where each row represents the cross-correlation of the instance with a neuron's calcium trace.
            - lags: A numpy array of lags used in the cross-correlation, adjusted by the acquisition rate.
        :rtype: tuple
        """

        import scipy
        lags = scipy.signal.correlation_lags(len(self.C_raw[0]), len(instance), mode=mode) / self.vm_rate

        correlations = np.array([scipy.signal.correlate(d, instance, mode=mode) for d in self.C_raw])

        for i in range(self.neuron_num):
            multiplier = np.corrcoef(self.C_raw[i], instance)[0, 1] / correlations[i][np.where(lags == 0)]
            correlations[i] = correlations[i] * multiplier

        return correlations, lags

    def average_across_indices(self, indices, signal=None, cut_interval=50):
        """
        Average the calcium signal across the specified indices.
        :param indices: indices to average across
        :param signal: optional input signal to use instead of self.C_raw
        :param cut_interval: interval to cut around the indices
        :return: averaged calcium signal, normalized signal, sorted signal, mean signal, sorted indices
        """
        # Use provided signal if available, otherwise use self.C_raw
        input_signal = signal if signal is not None else self.C_zsc
        if len(input_signal.shape) == 1:
            input_signal = np.expand_dims(input_signal, axis=0)

        # Get number of neurons from the input signal
        neuron_num = len(input_signal) if signal is not None else self.neuron_num

        C_avg = np.zeros([neuron_num, cut_interval * 2])
        time_points = len(self.t)

        for neu in range(neuron_num):
            C_tmp = np.zeros([len(indices), cut_interval * 2])
            for i, idx in enumerate(indices):
                start_idx = max(idx - cut_interval, 0)
                end_idx = min(idx + cut_interval, time_points)

                # Check if the window exceeds the array bounds
                if start_idx == 0 or end_idx == time_points:
                    # If the window exceeds, keep the row in C_tmp as zeros
                    continue
                else:
                    # Use input_signal instead of self.C_raw
                    C_tmp[i, (cut_interval - (idx - start_idx)):(cut_interval + (end_idx - idx))] = input_signal[neu,
                                                                                                    start_idx:end_idx]

            C_avg[neu] = np.mean(self.shift_signal(C_tmp), axis=0)

        peak_values = np.max(C_avg, axis=1)
        C_avg_normalized = self.normalize_signal_per_row(C_avg)
        peak_times = np.argmax(C_avg_normalized, axis=1)
        sorted_indices = np.argsort(peak_times)
        C_avg_normalized_sorted = C_avg_normalized[sorted_indices]
        C_avg_mean = np.mean(C_avg, axis=0)

        return C_avg, C_avg_normalized, C_avg_normalized_sorted, C_avg_mean, sorted_indices

    @staticmethod
    def ensure_tensor(data, device):
        """
        Ensure that the data is a tensor on the specified device.
        :param data: input data
        :param device: device to move the data to
        :return: tensor on the specified device
        """
        import torch

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
        import torch

        if notebook:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Check if inputs are tensors; if not, convert them
        C_raw_tensor = self.ensure_tensor(self.C_zsc, device)
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
        import torch

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
        import torch

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
        import torch

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
        import torch

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
    def find_neurons_with_peaks_near_indices(peak_indices, dependent_indices, window=10, choice='before'):
        """
        find neurons with peaks near indicated indices

        :param peak_indices: indices of calcium traces peaks
        :param dependent_indices: indices of (velocity peak, movement onset, lick edge, etc...)
        :param window: window size
        :param choice: 'before', 'after', or 'around'

        :return: dictionary of found neurons
        :example: neurons_with_peaks_before_lick = ci.find_neurons_with_peaks_near_indices(peak_indices, ci.lick_edge_indices, window=10, choice='before')
        """

        neurons_with_peaks_near_trial = {}

        for neuron_index, peaks in enumerate(peak_indices):
            for trial in dependent_indices:
                if choice.lower() == 'before':
                    near_peaks = peaks[(np.abs(peaks - trial) <= window) & (peaks < trial)]
                elif choice.lower() == 'after':
                    near_peaks = peaks[(np.abs(peaks - trial) <= window) & (peaks > trial)]
                elif choice.lower() == 'around':
                    near_peaks = peaks[(np.abs(peaks - trial) <= window)]
                else:
                    raise ValueError("Invalid choice")

                if near_peaks.size > 0:
                    if neuron_index not in neurons_with_peaks_near_trial:
                        neurons_with_peaks_near_trial[neuron_index] = []
                    neurons_with_peaks_near_trial[neuron_index].extend(near_peaks.tolist())

        return neurons_with_peaks_near_trial

    def get_spike_statistics(self, peak_indices):
        """
        :param peak_indices: the indices when a neuron activated, use ci.find_peaks_in_traces()

        :return spike_stats: time series of spike counts in each time point
        """
        spike_stats = np.zeros_like(self.t)
        for i in peak_indices:
            spike_stats[i] += 1

        return spike_stats

    def plot_t_ca_all(self, notebook=False, save_path=None, overwrite=False, font_size=None):
        """
        Plot the average calcium signal over time.
        """
        from bokeh.plotting import figure

        p = figure(width=800, height=300, active_scroll='wheel_zoom', title="Average Calcium Signal")
        p.line(self.t, self.ca_all, color='blue', alpha=0.7, legend_label="ca_all")

        p.legend.click_policy = 'hide'

        self.output_bokeh_plot(p, save_path=save_path, title=str(p.title.text), notebook=notebook, overwrite=overwrite, font_size=font_size)

    def plot_ca_around_indices(self, map_data, indices, cut_interval=50, title="None", notebook=False, save_path=None,
                               overwrite=False, font_size=None):
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

        self.output_bokeh_plot(layout, save_path=save_path, title=title, notebook=notebook, overwrite=overwrite, font_size=font_size)

        return layout

    def create_outlier_visualization(self, precomputed_data, coefficients_all_numpy, threshold=4, title=None,
                                     notebook=False, save_path=None, overwrite=False, font_size=None):
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

        self.output_bokeh_plot(layout, save_path=save_path, title=title, notebook=notebook, overwrite=overwrite, font_size=font_size)

        return layout

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

    def categorize_neurons_four_classes(self, picked_neurons_velocity, picked_neurons_lick):
        all_neurons = set(range(self.neuron_num))

        # Ensure that the inputs are sets

        if not isinstance(picked_neurons_velocity, set):
            picked_neurons_velocity = set(picked_neurons_velocity)

        if not isinstance(picked_neurons_lick, set):
            picked_neurons_lick = set(picked_neurons_lick)

        velocity_only = picked_neurons_velocity - picked_neurons_lick
        lick_only = picked_neurons_lick - picked_neurons_velocity

        # Intersection categories
        velocity_and_lick = picked_neurons_velocity & picked_neurons_lick

        # Neurons not in any category
        none = all_neurons - (
                velocity_only | lick_only | velocity_and_lick)

        # Prepare the categories with labels and neuron indices
        categories = {
            'Velocity Only': velocity_only,
            'Lick Only': lick_only,
            'Velocity & Lick': velocity_and_lick,
            'None': none
        }

        return categories

    @staticmethod
    def find_category_by_index(dictionary, target_value):
        """
        Find neuron category based on its index

        Args:
            dictionary: neuron categories dictionary
            target_value: neuron index

        Returns:
            neuron category, or "Not Found"

        Example:
             key = find_category_by_index(neuron_categories, 0)
        """
        for key, value in dictionary.items():
            value = list(value)
            for val in value:
                if np.abs(val - target_value) < 0.01:
                    return key
        return "Not Found"

    def create_neuron_categories_pie_chart(self, neuron_categories, title="Neuron Categories", notebook=False,
                                           save_path=None, overwrite=False, font_size=None):
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

        self.output_bokeh_plot(p, save_path=save_path, title=title, notebook=notebook, overwrite=overwrite, font_size=font_size)

        return p

    def create_heatmap_with_trace(self, sorted_indices, trace, cutoff_index=10000, title=None,
                                  save_path=None, notebook=False, overwrite=False, font_size=None):
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

        map_data = self.normalize_signal_per_row(self.C_zsc[sorted_indices, :cutoff_index])
        
        data_min = np.min(map_data)
        data_max = np.max(map_data)
        
        low_value = np.min(map_data)*0.6
        high_value = np.max(map_data)*0.6
        
        mapper = LinearColorMapper(palette="Viridis256", low=low_value, high=high_value)

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

        self.output_bokeh_plot(layout, save_path=save_path, title=title, notebook=notebook, overwrite=overwrite, font_size=font_size)

        return layout

    def plot_calcium_trace_with_lick_and_velocity(self, title="Average Calcium Trace with Lick and Speed",
                                                  save_path=None, notebook=False, overwrite=False, font_size=None):

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
        self.output_bokeh_plot(p, save_path=save_path, title=title, notebook=notebook, overwrite=overwrite, font_size=font_size)

        return p

    def plot_calcium_trace_around_indices(self, cut_interval=50, save_path=None,
                                          title="Average Calcium Trace around Indices",
                                          notebook=False, overwrite=False, font_size=None):

        from bokeh.plotting import figure
        from bokeh.models import CustomJSTickFormatter

        [_, _, _, C_avg_mean_velocity_peak, _] = self.average_across_indices(self.velocity_peak_indices,
                                                                             cut_interval=cut_interval)
        [_, _, _, C_avg_mean_lick_edge, _] = self.average_across_indices(self.lick_edge_indices,
                                                                         cut_interval=cut_interval)
        [_, _, _, C_avg_mean_movement_onset, _] = self.average_across_indices(self.movement_onset_indices,
                                                                              cut_interval=cut_interval)
        [_, _, _, C_avg_mean_movement_offset, _] = self.average_across_indices(self.movement_offset_indices,
                                                                              cut_interval=cut_interval)

        p = figure(width=800, height=400, active_scroll="wheel_zoom", title="Average Calcium Trace around Indices")
        
        # Convert time points from samples to seconds
        time_points = (np.array(range(2 * cut_interval)) - cut_interval) / self.ci_rate
        
        p.line(time_points, C_avg_mean_velocity_peak, line_width=2,
               legend_label='Velocity Peak', color='red')
        p.line(time_points, C_avg_mean_lick_edge, line_width=2,
               legend_label='Lick', color='navy')
        p.line(time_points, C_avg_mean_movement_onset, line_width=2,
               legend_label='Movement Onset', color='brown')
        p.line(time_points, C_avg_mean_movement_offset, line_width=2,
               legend_label='Movement Offset', color='purple')

        # Configure legend to be outside the plot
        p.legend.click_policy = 'hide'

        # p.xaxis.formatter = CustomJSTickFormatter(code="""
        #     return (tick / 20).toFixed(2);
        # """)

        self.output_bokeh_plot(p, save_path=save_path, title=title, notebook=notebook, overwrite=overwrite, font_size=font_size)

        return p

    def generate_raster_plot(self, save_path=None, title="Raster Plot", notebook=False, overwrite=False, font_size=None):

        from bokeh.models import ColumnDataSource
        from bokeh.plotting import figure
        from bokeh.layouts import column

        num_neurons = self.neuron_num
        spike_times = self.peak_indices

        # Prepare data for Bokeh
        data = {'x_starts': [], 'y_starts': [], 'x_ends': [], 'y_ends': []}

        for neuron_idx, spikes in enumerate(spike_times):
            for spike_time in spikes:
                data['x_starts'].append(spike_time / self.ci_rate)
                data['y_starts'].append(neuron_idx + 1)
                data['x_ends'].append(spike_time / self.ci_rate)
                data['y_ends'].append(neuron_idx + 1 + 0.7)  # Adjust the height of spikes with 0.4

        source = ColumnDataSource(data)

        # Create a Bokeh plot
        p = figure(width=800, height=1000, title="Raster Plot", x_axis_label='Time (s)', y_axis_label='Neuron',
                   active_scroll='wheel_zoom')

        # Use segment glyphs to represent spikes
        p.segment(x0='x_starts', y0='y_starts', x1='x_ends', y1='y_ends', source=source, color="black", line_width=2)

        # Set y-axis ticks and labels
        p.yaxis.ticker = np.arange(1, num_neurons + 1)
        p.yaxis.major_label_overrides = {i + 1: f"Neuron {i + 1}" for i in range(num_neurons)}

        v = figure(width=800, height=200, x_range=p.x_range, active_scroll='wheel_zoom')
        v.line(self.t, self.normalize_signal(self.velocity), color='SteelBlue', legend_label='velocity')
        v.line(self.t, self.lick, color='SandyBrown', legend_label='lick')
        v.line(self.t, self.normalize_signal_with_sign(self.acceleration), color='Green', legend_label='acceleration')

        v.legend.click_policy = 'hide'

        layout = column(p, v)

        self.output_bokeh_plot(layout, save_path=save_path, title=title, notebook=notebook, overwrite=overwrite, font_size=font_size)

        return layout

    def create_cross_correlation_slider_plot(self, instance, lags, correlations,
                                             save_path=None, title="Raster Plot", notebook=False, overwrite=False, font_size=None):

        from bokeh.models import ColumnDataSource, Slider, CustomJS
        from bokeh.plotting import figure
        from bokeh.layouts import gridplot, column

        source = ColumnDataSource(data={
            't': self.t,
            'calcium': self.C_raw[0],
            'instance': instance,
            'lags': lags,
            'correlation': correlations[0]
        })

        # Prepare the initial plot
        p1 = figure(width=800, height=300, title=title, x_axis_label='Time',
                    y_axis_label='Amplitude')
        p1.line('t', 'instance', source=source, legend_label='Instance', color='blue')
        p1.line('t', 'calcium', source=source, legend_label='Calcium', color='red')

        p2 = figure(width=800, height=300, x_axis_label='Lag',
                    y_axis_label='Correlation')
        p2.line('lags', 'correlation', source=source, legend_label='Cross-correlation', line_color='green')

        # Create a slider
        slider = Slider(start=0, end=len(self.C_raw) - 1, value=0, step=1, width=800,
                        title="Select Calcium Signal Index")

        # Define a callback to update the data based on the selected index
        callback = CustomJS(args=dict(source=source,
                                      correlations=[list(row) for row in correlations],
                                      C_raw=[list(row) for row in self.C_raw]),
                            code=
                            """
                                const index = cb_obj.value;
                                source.data.calcium = C_raw[index];
                                source.data.correlation = correlations[index];
                                source.change.emit();
                            """
                            )

        slider.js_on_change('value', callback)

        grid = gridplot([[p1], [p2]])
        layout = column(slider, grid)

        self.output_bokeh_plot(layout, save_path=save_path, title=title, notebook=notebook, overwrite=overwrite, font_size=font_size)

        return layout

    def show_single_signal(self, index,
                           save_path=None, title=f"example signal", notebook=False, overwrite=False, font_size=None):
        from bokeh.plotting import figure

        p = figure(width=800, height=300, title=title)

        p.line(self.t, self.C_raw[index], color="blue", alpha=0.7, legend_label="Calcium Trace")

        self.output_bokeh_plot(p, save_path=save_path, title=title, notebook=notebook, overwrite=overwrite, font_size=font_size)

        return p

    def show_multiple_signal(self, signal_gap=2, scale_gap=2,
                             save_path=None, title=f"Multiple Example signals", notebook=False, overwrite=False, font_size=None):
        from bokeh.plotting import figure
        from bokeh.palettes import Category10

        p = figure(width=800, height=600, title="Multiple Signals Examples")

        colors = Category10[10]

        for i in range(10):
            p.line(self.t, self.C_raw[i * signal_gap] + i * scale_gap, color=colors[i], alpha=0.7)

        self.output_bokeh_plot(p, save_path=save_path, title=title, notebook=notebook, overwrite=overwrite, font_size=font_size)

        return p

    @staticmethod
    def extract_statistical_features(region, mask):
        """
        Extracts comprehensive statistical features from a masked region.

        Args:
            region (ndarray): The image region
            mask (ndarray): Binary mask for the region

        Returns:
            list: Statistical features of the region
        """
        from scipy import stats
        from skimage.measure import shannon_entropy

        non_zero_pixels = region[mask > 0]

        if len(non_zero_pixels) == 0:
            return [0] * 12

        features = []

        # Basic statistics
        features.extend([
            np.mean(non_zero_pixels),
            np.median(non_zero_pixels),
            np.std(non_zero_pixels),
            stats.skew(non_zero_pixels),
            stats.kurtosis(non_zero_pixels)
        ])

        # Percentile-based features
        percentiles = np.percentile(non_zero_pixels, [10, 25, 75, 90])
        features.extend([
            percentiles[0],
            percentiles[1],
            percentiles[2],
            percentiles[3],
            percentiles[2] - percentiles[1]
        ])

        # Additional features
        features.extend([
            shannon_entropy(non_zero_pixels),
            np.sum(non_zero_pixels) / len(non_zero_pixels)
        ])

        return features

    def extract_and_classify_regions(self, target_image_path):
        """
        Extracts masked regions and classifies them using statistical features.
        Returns both the clustering results and separated masks.

        Args:
            target_image_path (str): Path to the source image

        Returns:
            tuple: (extracted_regions, labels, feature_matrix, feature_importance, cluster_masks)
                - cluster_masks is a dictionary containing the masks for each cluster
        """
        from tifffile import imread
        import cv2
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA

        target_image = imread(target_image_path)
        if len(target_image.shape) == 3:
            target_image = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)

        target_height, target_width = target_image.shape
        original_masks = []
        extracted_regions = []
        features = []

        for idx in range(self.A.shape[0]):
            mask = self.A[idx]
            mask_resized = cv2.resize(mask, (target_width, target_height),
                                      interpolation=cv2.INTER_LINEAR)
            binary_mask = (mask_resized != 0).astype(np.uint8)

            masked_region = target_image * binary_mask
            region_features = self.extract_statistical_features(masked_region, binary_mask)

            original_masks.append(mask)
            extracted_regions.append(masked_region)
            features.append(region_features)

        feature_matrix = np.array(features)
        original_masks = np.array(original_masks)

        scaler = StandardScaler()
        normalized_features = scaler.fit_transform(feature_matrix)

        pca = PCA()
        pca.fit(normalized_features)

        kmeans = KMeans(n_clusters=2, random_state=42)
        labels = kmeans.fit_predict(normalized_features)

        feature_importance = np.abs(pca.components_[0])

        masks_cluster_1 = original_masks[labels == 0]
        masks_cluster_2 = original_masks[labels == 1]

        cluster_masks = {
            'cluster1': masks_cluster_1,
            'cluster2': masks_cluster_2
        }

        return extracted_regions, labels, feature_matrix, feature_importance, cluster_masks

    @staticmethod
    def visualize_clustering_results(extracted_regions, labels, feature_matrix, feature_importance, cluster_masks,
                                     output_dir=None, show_plots=True):
        """
        Creates comprehensive visualizations of the clustering results.

        Args:
            extracted_regions (list): List of extracted region arrays
            labels (array): Cluster assignments
            feature_matrix (array): Matrix of features
            feature_importance (array): Feature importance scores
            cluster_masks (dict): Dictionary containing masks for each cluster
            output_dir (str, optional): Directory to save visualizations. If None, plots won't be saved.
            show_plots (bool): Whether to display the plots. Defaults to True.
        """
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(20, 15))

        # Sample regions from each cluster
        for cluster_idx in range(2):
            cluster_indices = np.where(labels == cluster_idx)[0]
            if len(cluster_indices) > 0:
                sample_indices = np.random.choice(cluster_indices,
                                                  size=min(3, len(cluster_indices)),
                                                  replace=False)

                plt.subplot(3, 2, cluster_idx + 1)
                plt.imshow(extracted_regions[sample_indices[0]], cmap='gray')
                for idx in sample_indices[1:]:
                    plt.imshow(extracted_regions[idx], cmap='gray', alpha=0.5)

                plt.title(f'Cluster {cluster_idx + 1} Examples\n({len(cluster_indices)} regions)')
                plt.axis('off')

        # Feature importance plot
        feature_names = [
            'Mean', 'Median', 'Std Dev', 'Skewness', 'Kurtosis',
            '10th Perc', '25th Perc', '75th Perc', '90th Perc', 'IQR',
            'Entropy', 'Avg Signal'
        ]

        plt.subplot(3, 2, 3)
        feature_importance_normalized = feature_importance / np.max(feature_importance)
        sns.barplot(x=feature_names, y=feature_importance_normalized)
        plt.xticks(rotation=45, ha='right')
        plt.title('Feature Importance')

        # Cluster distribution plot
        plt.subplot(3, 2, 4)
        for cluster_idx in range(2):
            cluster_features = feature_matrix[labels == cluster_idx]
            plt.hist(cluster_features[:, 0], bins=20, alpha=0.5,
                     label=f'Cluster {cluster_idx + 1}')
        plt.title('Distribution of Mean Intensity by Cluster')
        plt.legend()

        # Mask examples from each cluster
        for cluster_idx, (cluster_name, masks) in enumerate(cluster_masks.items()):
            plt.subplot(3, 2, 5 + cluster_idx)
            if len(masks) > 0:
                composite = np.zeros_like(masks[0])
                for i in range(min(3, len(masks))):
                    composite += masks[i]
                plt.imshow(composite, cmap='gray')
                plt.title(f'{cluster_name} Mask Examples\n({len(masks)} masks)')
            plt.axis('off')

        plt.tight_layout()

        # Save plot if output directory is specified
        if output_dir is not None:
            os.makedirs(output_dir, exist_ok=True)
            plt.savefig(os.path.join(output_dir, "clustering_analysis.png"))

        # Show plot if requested
        if show_plots:
            plt.show()
        else:
            plt.close()

        # Print summary statistics
        print("\nCluster Summary:")
        for cluster_idx in range(2):
            cluster_size = np.sum(labels == cluster_idx)
            cluster_features = feature_matrix[labels == cluster_idx]
            print(f"\nCluster {cluster_idx + 1} Statistics:")
            print(f"Size: {cluster_size}")
            print(f"Mean intensity: {np.mean(cluster_features[:, 0]):.2f}")
            print(f"Mean std dev: {np.mean(cluster_features[:, 2]):.2f}")
            print(f"Mean entropy: {np.mean(cluster_features[:, -2]):.2f}")
            print(f"Number of masks: {len(cluster_masks[f'cluster{cluster_idx + 1}'])}")

    def analyze_regions(self, target_image_path, output_dir=None, show_plots=True):
        """
        Convenience method to perform region extraction, classification, and visualization in one step.

        Args:
            target_image_path (str): Path to the source image
            output_dir (str, optional): Directory to save visualizations. If None, plots won't be saved.
            show_plots (bool): Whether to display the plots. Defaults to True.

        Returns:
            tuple: (cluster_masks, labels) for further analysis if needed
        """
        regions, labels, features, importance, masks = self.extract_and_classify_regions(target_image_path)
        self.visualize_clustering_results(
            regions,
            labels,
            features,
            importance,
            masks,
            output_dir=output_dir,
            show_plots=show_plots
        )
        return masks, labels
    

    def analyze_signal_around_spikes(self, d1_indices, d2_indices, signal, signal_name="Signal", time_window=4.0, height=0.5, 
                                    save_path=None, title=None, notebook=False, overwrite=False):
        """
        Analyze any signal before and after spikes in D1 and D2 neurons.
        
        Parameters:
        -----------
        d1_indices : array-like
            Indices of D1 neurons
        d2_indices : array-like
            Indices of D2 neurons
        signal : numpy.ndarray
            Signal array to analyze (e.g., self.smoothed_velocity, self.acceleration)
        signal_name : str
            Name of signal for plot labels (e.g., "Velocity", "Acceleration")
        time_window : float
            Time window in seconds before and after spike to analyze
        height : float
            Height threshold for peak detection
        save_path : str, optional
            Path to save the plot
        title : str, optional
            Title for the plot (if None, will be generated based on signal_name)
        notebook : bool
            Whether to display in notebook
        overwrite : bool
            Whether to overwrite existing file
        
        Returns:
        --------
        dict : Dictionary containing analysis results and plot objects
        """
        from bokeh.plotting import figure
        from bokeh.layouts import gridplot
        from bokeh.models import ColumnDataSource, Span
        from scipy.signal import find_peaks
        
        # Set default title if not provided
        if title is None:
            title = f"{signal_name} around D1 and D2 Neuron Spikes"
        
        # Convert time window to samples
        samples_window = int(time_window * self.ci_rate)
        
        # Find peaks for D1 and D2 neurons
        d1_peak_indices = []
        for i in d1_indices:
            x, _ = find_peaks(self.C_denoised[i], height=height)
            d1_peak_indices.append(x)
        
        d2_peak_indices = []
        for i in d2_indices:
            x, _ = find_peaks(self.C_denoised[i], height=height)
            d2_peak_indices.append(x)
        
        # Initialize arrays to store signal profiles around spikes
        d1_signals = []
        d2_signals = []
        
        # Extract signal profiles for D1 neurons
        for neuron_spikes in d1_peak_indices:
            for spike_idx in neuron_spikes:
                # Check if we have enough data before and after the spike
                if spike_idx - samples_window >= 0 and spike_idx + samples_window < len(signal):
                    signal_segment = signal[spike_idx - samples_window:spike_idx + samples_window + 1]
                    d1_signals.append(signal_segment)
        
        # Extract signal profiles for D2 neurons
        for neuron_spikes in d2_peak_indices:
            for spike_idx in neuron_spikes:
                # Check if we have enough data before and after the spike
                if spike_idx - samples_window >= 0 and spike_idx + samples_window < len(signal):
                    signal_segment = signal[spike_idx - samples_window:spike_idx + samples_window + 1]
                    d2_signals.append(signal_segment)
        
        # Convert to numpy arrays
        d1_signals = np.array(d1_signals)
        d2_signals = np.array(d2_signals)
        
        # Calculate average signal profiles
        d1_avg_signal = np.mean(d1_signals, axis=0)
        d2_avg_signal = np.mean(d2_signals, axis=0)
        
        # Calculate standard error of the mean
        d1_sem_signal = np.std(d1_signals, axis=0) / np.sqrt(len(d1_signals))
        d2_sem_signal = np.std(d2_signals, axis=0) / np.sqrt(len(d2_signals))
        
        # Create time axis in seconds relative to spike
        time_axis = np.linspace(-time_window, time_window, len(d1_avg_signal))
        
        # Create plots using Bokeh
        p1 = figure(width=900, height=400, 
                title=f"{signal_name} around D1 Neuron Spikes (n={len(d1_signals)})",
                x_axis_label="Time relative to spike (s)",
                y_axis_label=signal_name)
        
        # Add D1 line with confidence band
        p1.line(time_axis, d1_avg_signal, line_width=3, color='red', legend_label="D1 Average")
        p1.patch(
            np.concatenate([time_axis, time_axis[::-1]]),
            np.concatenate([
                d1_avg_signal + d1_sem_signal,
                (d1_avg_signal - d1_sem_signal)[::-1]
            ]),
            color='red', alpha=0.2
        )
        
        # Add vertical line at spike time
        p1.line([0, 0], [np.min(d1_avg_signal)-1, np.max(d1_avg_signal) * 1.2], line_width=2, color='black', line_dash='dashed')
        
        # Add horizontal line at y=0 for acceleration signals
        if signal_name.lower() == "acceleration":
            zero_line = Span(location=0, dimension='width', line_color='black', line_width=2, line_dash='dotted')
            p1.add_layout(zero_line)
        
        # Remove grid
        p1.xgrid.grid_line_color = None
        p1.ygrid.grid_line_color = None
        
        p2 = figure(width=900, height=400, 
                title=f"{signal_name} around D2 Neuron Spikes (n={len(d2_signals)})",
                x_axis_label="Time relative to spike (s)",
                y_axis_label=signal_name,
                x_range=p1.x_range)
        
        # Add D2 line with confidence band
        p2.line(time_axis, d2_avg_signal, line_width=3, color='blue', legend_label="D2 Average")
        p2.patch(
            np.concatenate([time_axis, time_axis[::-1]]),
            np.concatenate([
                d2_avg_signal + d2_sem_signal,
                (d2_avg_signal - d2_sem_signal)[::-1]
            ]),
            color='blue', alpha=0.2
        )
        
        # Add vertical line at spike time
        p2.line([0, 0], [np.min(d2_avg_signal)-1, np.max(d2_avg_signal) * 1.2], line_width=2, color='black', line_dash='dashed')
        
        # Add horizontal line at y=0 for acceleration signals
        if signal_name.lower() == "acceleration":
            zero_line = Span(location=0, dimension='width', line_color='black', line_width=2, line_dash='dotted')
            p2.add_layout(zero_line)
        
        # Remove grid
        p2.xgrid.grid_line_color = None
        p2.ygrid.grid_line_color = None
        
        # Create a comparison plot
        p3 = figure(width=900, height=500, 
                title=f"Comparison of {signal_name} around D1 vs D2 Neuron Spikes",
                x_axis_label="Time relative to spike (s)",
                y_axis_label=signal_name)
        
        # Add both lines to the comparison plot
        p3.line(time_axis, d1_avg_signal, line_width=3, color='red', legend_label="D1 Average")
        p3.line(time_axis, d2_avg_signal, line_width=3, color='blue', legend_label="D2 Average")
        
        # Add vertical line at spike time
        p3.line([0, 0], [np.min(d1_avg_signal) -1, np.max(d1_avg_signal) * 1.2], 
            line_width=2, color='black', line_dash='dashed')
        
        # Add horizontal line at y=0 for acceleration signals
        if signal_name.lower() == "acceleration":
            zero_line = Span(location=0, dimension='width', line_color='black', line_width=2, line_dash='dotted')
            p3.add_layout(zero_line)
        
        # Remove grid
        p3.xgrid.grid_line_color = None
        p3.ygrid.grid_line_color = None
        
        # Customize all plots
        for p in [p1, p2, p3]:
            p.legend.location = "top_right"
            p.legend.click_policy = "hide"
        
        # Create the layout
        layout = gridplot([[p1], [p2], [p3]])
        
        # Output the plot
        self.output_bokeh_plot(layout, save_path=save_path, title=title, notebook=notebook, overwrite=overwrite)
        
        # Store results
        results = {
            'time_axis': time_axis,
            'd1_avg_signal': d1_avg_signal,
            'd2_avg_signal': d2_avg_signal,
            'd1_sem_signal': d1_sem_signal,
            'd2_sem_signal': d2_sem_signal,
            'd1_sample_count': len(d1_signals),
            'd2_sample_count': len(d2_signals),
            'signal_name': signal_name,
            'plot': layout
        }
        
        return results

    # Going to remove this function
    def analyze_d1_d2_spikes_signal(self, json_masks, signal, signal_name="Signal", time_window=4, height=0.0, 
                               save_path=None, title=None, notebook=False, overwrite=False):
        """
        Wrapper function to analyze and plot any signal around D1 and D2 neuron spikes.
        
        Parameters:
        -----------
        json_masks : dict
            Dictionary containing D1 and D2 neuron indices from create_mask_from_json
        signal : numpy.ndarray
            Signal array to analyze (e.g., self.smoothed_velocity, self.acceleration)
        signal_name : str
            Name of signal for plot labels (e.g., "Velocity", "Acceleration")
        time_window : float
            Time window in seconds before and after spike to analyze
        height : float
            Height threshold for peak detection
        save_path : str, optional
            Path to save the figure
        title : str, optional
            Title for the plot (if None, will be generated based on signal_name)
        notebook : bool
            Whether to display in notebook
        overwrite : bool
            Whether to overwrite existing file
        
        Returns:
        --------
        dict : Results dictionary
        """
        # Get D1 and D2 indices
        d1_indices = json_masks["d1_indices"]
        d2_indices = json_masks["d2_indices"][:len(d1_indices)]
        
        # Call the analysis function
        results = self.analyze_signal_around_spikes(
            d1_indices=d1_indices,
            d2_indices=d2_indices,
            signal=signal,
            signal_name=signal_name,
            time_window=time_window,
            height=height,
            save_path=save_path,
            title=title,
            notebook=notebook,
            overwrite=overwrite
        )
        
        return results


    def plot_d1_d2_calcium_around_events(self, event_type='movement_onset', d1_signal=None, d2_signal=None, 
                                     cut_interval=50, save_path=None, title=None, notebook=False, overwrite=False, font_size=None):
        """
        Plot average calcium traces for D1 and D2 neuron populations, along with velocity, around specific events.
        Uses dual y-axes to separately scale calcium and velocity signals, with optimized scaling.
        
        Parameters:
        -----------
        event_type : str
            Type of event to plot around. Options: 'movement_onset', 'velocity_peak', 'movement_offset'
        d1_signal : numpy.ndarray, optional
            Signal from D1 neurons. If None, will use self.C_raw
        d2_signal : numpy.ndarray, optional
            Signal from D2 neurons. If None, will use self.C_raw
        cut_interval : int
            Interval to cut around the indices (in samples)
        save_path : str, optional
            Path to save the plot as an HTML file
        title : str, optional
            Title for the plot. If None, will generate based on event_type
        notebook : bool
            Flag to indicate if the plot is for a Jupyter notebook
        overwrite : bool
            Flag to indicate whether to overwrite existing file
        
        Returns:
        --------
        bokeh.plotting.figure
            The created plot
        """
        from bokeh.plotting import figure
        from bokeh.models import LinearAxis, Range1d, Legend
        
        # Select indices based on event type
        if event_type == 'movement_onset':
            indices = self.movement_onset_indices
            default_title = "Average Calcium Trace around Movement Onsets"
        elif event_type == 'velocity_peak':
            indices = self.velocity_peak_indices
            default_title = "Average Calcium Trace around Velocity Peaks"
        elif event_type == 'movement_offset':
            indices = self.movement_offset_indices
            default_title = "Average Calcium Trace around Movement Offsets"
        else:
            raise ValueError("event_type must be one of: 'movement_onset', 'velocity_peak', 'movement_offset'")
        
        # Use provided title or default
        plot_title = title or default_title
        
        # Calculate average signals around the event
        [_, _, _, C_avg_mean_d1, _] = self.average_across_indices(indices, signal=d1_signal, cut_interval=cut_interval)
        [_, _, _, C_avg_mean_d2, _] = self.average_across_indices(indices, signal=d2_signal, cut_interval=cut_interval)
        [_, _, _, velocity_mean, _] = self.average_across_indices(indices, signal=self.smoothed_velocity, cut_interval=cut_interval)
        
        # Create time axis in seconds
        time_axis = (np.array(range(2 * cut_interval)) - cut_interval) / self.ci_rate
        
        # Calculate optimal y-axis ranges for calcium signals with padding
        ca_min = min(C_avg_mean_d1.min(), C_avg_mean_d2.min())
        ca_max = max(C_avg_mean_d1.max(), C_avg_mean_d2.max())
        
        # Add padding to ensure the signals are well-positioned in the plot (10% padding)
        ca_range = ca_max - ca_min
        ca_min = ca_min - (ca_range * 0.1)
        ca_max = ca_max + (ca_range * 0.1)
        
        # Create plot with optimized y-axis for calcium signals
        p = figure(width=800, height=400, active_scroll="wheel_zoom", title=plot_title,
                   x_axis_label='Time (s)', y_axis_label='Calcium Signal (ΔF/F)',
                   y_range=Range1d(ca_min, ca_max))
        
        # Add secondary y-axis for velocity with its own optimized range
        vel_range = velocity_mean.max() - velocity_mean.min()
        vel_min = velocity_mean.min() - (vel_range * 0.1)
        vel_max = velocity_mean.max() + (vel_range * 0.1)
        
        p.extra_y_ranges = {"velocity_axis": Range1d(start=vel_min, end=vel_max)}
        p.add_layout(LinearAxis(y_range_name="velocity_axis", axis_label="Velocity"), 'right')
        
        # Add calcium traces and velocity to plot (without adding to legend)
        d1_line = p.line(time_axis, C_avg_mean_d1, line_width=3, color='navy', alpha=0.6)
        d2_line = p.line(time_axis, C_avg_mean_d2, line_width=3, color='red', alpha=0.6)
        vel_line = p.line(time_axis, velocity_mean, line_width=3, color='gray', 
                         y_range_name="velocity_axis", alpha=0.6)
        
        # Create a legend outside the plot
        legend = Legend(
            items=[
                ("D1", [d1_line]),
                ("D2", [d2_line]),
                ("Velocity", [vel_line])
            ],
            location="center",
            orientation="horizontal",
            click_policy="hide"
        )
        
        # Add the legend to the top of the plot (outside the plot area)
        p.add_layout(legend, 'above')
        
        # Style the plot
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
        
        # Add vertical line at time=0 (the event)
        p.line([0, 0], [ca_min, ca_max], line_width=1, line_dash='dashed', color='black', alpha=0.7)
        
        # Output the plot
        self.output_bokeh_plot(p, save_path=save_path, title=plot_title, notebook=notebook, overwrite=overwrite, font_size=font_size)
        
        return p

    def create_d1_d2_spike_visualizations(self, d1_peaks, d2_peaks, save_path=None, title="D1D2SpikePlots", notebook=False, overwrite=False, font_size=None):
        """
        Create comprehensive visualizations of D1 and D2 neural spike data.
        
        Parameters:
        -----------
        d1_peaks : list of arrays
            List of peak indices for each D1 neuron
        d2_peaks : list of arrays
            List of peak indices for each D2 neuron
        save_path : str, optional
            Path to save the visualization
        title : str
            Title for the visualization
        notebook : bool
            Flag to indicate if the visualization is for a Jupyter notebook
        overwrite : bool
            Flag to indicate whether to overwrite existing file
        
        Returns:
        --------
        bokeh.layouts.layout
            The created visualization layout
        """
        from bokeh.plotting import figure
        from bokeh.layouts import column, row
        from bokeh.models import ColumnDataSource, HoverTool, Legend, LegendItem
        
        def convert_to_seconds(indices):
            """Convert sample indices to seconds"""
            return np.array(indices) / self.ci_rate
        
        # Create figure for firing rate histogram
        p1 = figure(width=400, height=300, title='Distribution of Spike Counts')
        
        # Remove grid
        p1.grid.grid_line_color = None
        
        # Calculate spike counts per neuron
        d1_counts = [len(peaks) for peaks in d1_peaks]
        d2_counts = [len(peaks) for peaks in d2_peaks]
        
        # Create histograms with shared bins
        # Find the overall min and max for both datasets
        min_count = min(min(d1_counts), min(d2_counts))
        max_count = max(max(d1_counts), max(d2_counts))
        shared_edges = np.linspace(min_count, max_count, 21)  # 20 bins
        
        # Create histograms using shared edges
        hist1, _ = np.histogram(d1_counts, bins=shared_edges)
        hist2, _ = np.histogram(d2_counts, bins=shared_edges)
        
        # Plot histograms (without specifying legend_label in the glyph)
        d1_hist_plot = p1.quad(top=hist1, bottom=0, left=shared_edges[:-1], right=shared_edges[1:],
                fill_color='navy', alpha=0.5, line_color='navy')
        d2_hist_plot = p1.quad(top=hist2, bottom=0, left=shared_edges[:-1], right=shared_edges[1:],
                fill_color='crimson', alpha=0.5, line_color='crimson')
        
        # Create legend items
        legend_items1 = [
            LegendItem(label="D1", renderers=[d1_hist_plot]),
            LegendItem(label="D2", renderers=[d2_hist_plot])
        ]
        
        # Create legend and place it outside the plot
        legend1 = Legend(items=legend_items1, location="center")
        p1.add_layout(legend1, 'right')
        
        # Customize first histogram
        p1.xaxis.axis_label = 'Number of Spikes per Neuron'
        p1.yaxis.axis_label = 'Count'
        
        # Create figure for inter-spike interval histogram
        p2 = figure(width=400, height=300, title='Distribution of Inter-spike Intervals')
        
        # Remove grid
        p2.grid.grid_line_color = None
        
        # Calculate inter-spike intervals in seconds
        d1_isis = []
        d2_isis = []
        
        for peaks in d1_peaks:
            if len(peaks) > 1:
                isis = np.diff(np.sort(peaks)) / self.ci_rate  # Convert to seconds
                d1_isis.extend(isis)
                
        for peaks in d2_peaks:
            if len(peaks) > 1:
                isis = np.diff(np.sort(peaks)) / self.ci_rate  # Convert to seconds
                d2_isis.extend(isis)
        
        # Create histograms for ISIs with shared bins
        # Find the overall min and max for ISIs
        min_isi = min(min(d1_isis), min(d2_isis)) if d1_isis and d2_isis else 0
        max_isi = max(max(d1_isis), max(d2_isis)) if d1_isis and d2_isis else 1
        shared_isi_edges = np.linspace(min_isi, max_isi, 31)  # 30 bins
        
        # Create histograms using shared edges
        hist3, _ = np.histogram(d1_isis, bins=shared_isi_edges)
        hist4, _ = np.histogram(d2_isis, bins=shared_isi_edges)
        
        # Plot ISI histograms (without specifying legend_label in the glyph)
        d1_isi_plot = p2.quad(top=hist3, bottom=0, left=shared_isi_edges[:-1], right=shared_isi_edges[1:],
                fill_color='navy', alpha=0.5, line_color='navy')
        d2_isi_plot = p2.quad(top=hist4, bottom=0, left=shared_isi_edges[:-1], right=shared_isi_edges[1:],
                fill_color='crimson', alpha=0.5, line_color='crimson')
        
        # Create legend items
        legend_items2 = [
            LegendItem(label="D1", renderers=[d1_isi_plot]),
            LegendItem(label="D2", renderers=[d2_isi_plot])
        ]
        
        # Create legend and place it outside the plot
        legend2 = Legend(items=legend_items2, location="center")
        p2.add_layout(legend2, 'right')
        
        # Customize second histogram
        p2.xaxis.axis_label = 'Inter-spike Interval (seconds)'
        p2.yaxis.axis_label = 'Count'
        
        # Create raster plot
        p3 = figure(width=800, height=400, title='Neural Activity Patterns')
        
        # Remove grid
        p3.grid.grid_line_color = None
        
        # Create data for D1 neurons with time in seconds
        d1_data = []
        for i, peaks in enumerate(d1_peaks):
            times = convert_to_seconds(peaks)
            y_values = [i for _ in peaks]
            d1_data.extend(list(zip(times, y_values, [i+1]*len(peaks))))
        
        # Create data for D2 neurons with time in seconds
        d2_data = []
        max_d1 = len(d1_peaks)
        for i, peaks in enumerate(d2_peaks):
            times = convert_to_seconds(peaks)
            y_values = [i + max_d1 + 2 for _ in peaks]
            d2_data.extend(list(zip(times, y_values, [i+1]*len(peaks))))
        
        # Create ColumnDataSource for both types
        d1_source = ColumnDataSource(data=dict(
            x=[d[0] for d in d1_data],
            y=[d[1] for d in d1_data],
            neuron=[d[2] for d in d1_data]
        ))
        
        d2_source = ColumnDataSource(data=dict(
            x=[d[0] for d in d2_data],
            y=[d[1] for d in d2_data],
            neuron=[d[2] for d in d2_data]
        ))
        
        # Add scatter plots (without specifying legend_label in the glyph)
        d1_scatter = p3.scatter('x', 'y', source=d1_source, color='navy', alpha=0.6, size=3)
        d2_scatter = p3.scatter('x', 'y', source=d2_source, color='crimson', alpha=0.6, size=3)
        
        # Create legend items
        legend_items3 = [
            LegendItem(label="D1 Neurons", renderers=[d1_scatter]),
            LegendItem(label="D2 Neurons", renderers=[d2_scatter])
        ]
        
        # Create legend and place it outside the plot
        legend3 = Legend(items=legend_items3, location="center")
        p3.add_layout(legend3, 'right')
        legend3.click_policy = "hide"
        
        # Customize raster plot
        p3.xaxis.axis_label = 'Time (seconds)'
        p3.yaxis.axis_label = 'Neuron ID'
        
        # Add hover tool
        hover = HoverTool(tooltips=[
            ('Time', '@x{0.00} s'),
            ('Neuron', '@neuron')
        ])
        p3.add_tools(hover)

        p4 = figure(width=800, height=200, x_range=p3.x_range, 
                    title='Velocity')
        
        # Remove grid
        p4.grid.grid_line_color = None

        velocity_line = p4.line(self.t, self.smoothed_velocity, color="black", line_width=2)
        
        # Create legend items for velocity plot
        legend_items4 = [
            LegendItem(label="Velocity", renderers=[velocity_line])
        ]
        
        # Create legend and place it outside the plot
        legend4 = Legend(items=legend_items4, location="center")
        p4.add_layout(legend4, 'right')
        
        # Calculate and print summary statistics
        d1_mean_count = np.mean(d1_counts)
        d2_mean_count = np.mean(d2_counts)
        d1_mean_isi = np.mean(d1_isis) if d1_isis else 0
        d2_mean_isi = np.mean(d2_isis) if d2_isis else 0
        
        print(f"\nSummary Statistics:") 
        print(f"D1 Neurons:")
        print(f"- Average spikes per neuron: {d1_mean_count:.2f}")
        print(f"- Average inter-spike interval: {d1_mean_isi:.2f} seconds")
        print(f"D2 Neurons:")
        print(f"- Average spikes per neuron: {d2_mean_count:.2f}")
        print(f"- Average inter-spike interval: {d2_mean_isi:.2f} seconds")
        
        # Combine the plots into a layout
        layout = column(
            row(p1, p2),
            p3,
            p4
        )
        
        # Output the plot
        self.output_bokeh_plot(layout, save_path=save_path, title=title, notebook=notebook, overwrite=overwrite, font_size=font_size)
        
        return layout

    def find_synchronous_events(self, time_window=5, min_neurons=3, min_event_interval=None, 
                              peak_indices=None, return_details=True):
        """
        Find time points where many neurons are active together (synchronous events).
        
        Parameters:
        -----------
        time_window : int
            Time window size in samples to consider neurons as synchronous
        min_neurons : int  
            Minimum number of neurons that must be active to consider it a synchronous event
        min_event_interval : int, optional
            Minimum interval between events in samples. If None, defaults to time_window * 2
            This prevents detection of events that are too close in time
        peak_indices : list, optional
            List of peak indices for each neuron. If None, uses self.peak_indices
        return_details : bool
            Whether to return detailed information about each event
            
        Returns:
        --------
        dict
            Dictionary containing:
            - 'event_times': array of event time points (in samples)
            - 'event_strengths': array of number of neurons participating in each event
            - 'event_details': list of detailed info for each event (if return_details=True)
            - 'total_events': total number of synchronous events found
            - 'parameters': parameters used for detection
        """
        # Use provided peak_indices or default to self.peak_indices
        if peak_indices is None:
            if not hasattr(self, 'peak_indices') or self.peak_indices is None:
                raise ValueError("peak_indices not available. Run _find_peaks_in_traces() first.")
            peak_indices = self.peak_indices
        
        # Set default minimum event interval if not provided
        if min_event_interval is None:
            min_event_interval = time_window * 2
        
        # Create a time series of spike counts
        max_time = len(self.t) if hasattr(self, 't') else max(max(peaks) for peaks in peak_indices if len(peaks) > 0)
        spike_count_series = np.zeros(max_time, dtype=int)
        
        # Count spikes at each time point
        for neuron_idx, peaks in enumerate(peak_indices):
            for peak_time in peaks:
                if peak_time < max_time:
                    spike_count_series[peak_time] += 1
        
        # Find synchronous events using sliding window
        synchronous_events = []
        event_times = []
        event_strengths = []
        
        # Slide window across the time series
        for t in range(max_time - time_window + 1):
            window_spike_count = np.sum(spike_count_series[t:t + time_window])
            
            if window_spike_count >= min_neurons:
                # Find the exact time point with maximum activity in this window
                window_spikes = spike_count_series[t:t + time_window]
                peak_time_in_window = np.argmax(window_spikes)
                actual_event_time = t + peak_time_in_window
                actual_spike_count = window_spikes[peak_time_in_window]
                
                # Check if this is a new event (not too close to previous ones)
                if not event_times or (actual_event_time - event_times[-1]) >= min_event_interval:
                    event_times.append(actual_event_time)
                    event_strengths.append(actual_spike_count)
                    
                    if return_details:
                        # Find which neurons participated in this event
                        participating_neurons = []
                        for neuron_idx, peaks in enumerate(peak_indices):
                            # Check if this neuron has a peak within the time window around the event
                            nearby_peaks = peaks[(peaks >= actual_event_time - time_window//2) & 
                                               (peaks <= actual_event_time + time_window//2)]
                            if len(nearby_peaks) > 0:
                                participating_neurons.append({
                                    'neuron_id': neuron_idx,
                                    'peak_times': nearby_peaks.tolist()
                                })
                        
                        synchronous_events.append({
                            'event_time': actual_event_time,
                            'event_time_seconds': actual_event_time / self.ci_rate if hasattr(self, 'ci_rate') else actual_event_time,
                            'strength': actual_spike_count,
                            'participating_neurons': participating_neurons,
                            'time_window_used': time_window
                        })
        
        results = {
            'event_times': np.array(event_times),
            'event_strengths': np.array(event_strengths),
            'total_events': len(event_times),
            'parameters': {
                'time_window': time_window,
                'min_neurons': min_neurons,
                'min_event_interval': min_event_interval,
                'total_neurons_analyzed': len(peak_indices)
            }
        }
        
        if return_details:
            results['event_details'] = synchronous_events
            
        return results

    def plot_synchronous_events(self, synchronous_events=None, time_window=5, min_neurons=3,
                              min_event_interval=None, show_top_events=10, signal_to_plot='C_zsc', 
                              save_path=None, title=None, notebook=False, 
                              overwrite=False, font_size=None):
        """
        Visualize synchronous neural activity events.
        
        Parameters:
        -----------
        synchronous_events : dict, optional
            Results from find_synchronous_events(). If None, will compute automatically
        time_window : int
            Time window for event detection (used if synchronous_events is None)
        min_neurons : int
            Minimum neurons for event detection (used if synchronous_events is None)
        min_event_interval : int, optional
            Minimum interval between events in samples (used if synchronous_events is None)
        show_top_events : int
            Number of top events to highlight in the plot
        signal_to_plot : str
            Which signal to use for background ('C_zsc', 'C_raw', 'C_denoised')
        save_path : str, optional
            Path to save the plot
        title : str, optional
            Title for the plot
        notebook : bool
            Whether to display in notebook
        overwrite : bool
            Whether to overwrite existing file
        font_size : str, optional
            Font size for plot elements
            
        Returns:
        --------
        bokeh.layouts.layout
            The created visualization layout
        """
        from bokeh.plotting import figure
        from bokeh.layouts import column, row
        from bokeh.models import ColumnDataSource, Span, HoverTool, ColorBar, LinearColorMapper
        from bokeh.palettes import Reds9
        
        # Get synchronous events if not provided
        if synchronous_events is None:
            synchronous_events = self.find_synchronous_events(
                time_window=time_window, 
                min_neurons=min_neurons,
                min_event_interval=min_event_interval,
                return_details=True
            )
        
        if synchronous_events['total_events'] == 0:
            print("No synchronous events found with the given parameters.")
            return None
        
        # Set default title
        if title is None:
            title = f"Synchronous Neural Activity Events (n={synchronous_events['total_events']})"
        
        # Get the signal to plot
        if signal_to_plot == 'C_zsc' and hasattr(self, 'C_zsc'):
            signal = self.C_zsc
        elif signal_to_plot == 'C_denoised' and hasattr(self, 'C_denoised'):
            signal = self.C_denoised
        elif hasattr(self, 'C_raw'):
            signal = self.C_raw
        else:
            raise ValueError("No suitable signal found for plotting")
        
        # Create time axis
        time_axis = self.t if hasattr(self, 't') else np.arange(signal.shape[1]) / self.ci_rate
        
        # 1. Overview plot showing all events
        p1 = figure(width=1000, height=300, 
                   title="Population Activity and Synchronous Events",
                   x_axis_label="Time (s)", 
                   y_axis_label="Population Activity")
        
        # Plot population average
        population_avg = np.mean(signal, axis=0)
        p1.line(time_axis, population_avg, line_width=2, color='blue', alpha=0.7, legend_label="Population Average")
        
        # Add event markers
        event_times_sec = synchronous_events['event_times'] / self.ci_rate
        event_strengths = synchronous_events['event_strengths']
        
        # Create color mapping for event strength
        max_strength = np.max(event_strengths)
        min_strength = np.min(event_strengths)
        
        # Normalize strengths for color mapping
        normalized_strengths = (event_strengths - min_strength) / (max_strength - min_strength)
        colors = [Reds9[min(8, int(strength * 8))] for strength in normalized_strengths]
        
        # Add event scatter plot
        event_source = ColumnDataSource(data=dict(
            x=event_times_sec,
            y=[population_avg[int(t)] for t in synchronous_events['event_times']],
            strength=event_strengths,
            colors=colors
        ))
        
        scatter = p1.scatter('x', 'y', source=event_source, size=10, color='colors', alpha=0.8)
        
        # Add hover tool
        hover = HoverTool(renderers=[scatter], tooltips=[
            ('Time', '@x{0.00} s'),
            ('Strength', '@strength neurons'),
            ('Population Activity', '@y{0.000}')
        ])
        p1.add_tools(hover)
        
        p1.legend.location = "top_left"
        p1.legend.click_policy = "hide"
        
        # 2. Velocity plot with event markers (NEW)
        pv = figure(width=1000, height=200, 
                   title="Velocity and Synchronous Events",
                   x_axis_label="Time (s)", 
                   y_axis_label="Velocity",
                   x_range=p1.x_range)  # Link x-axis with population plot
        
        # Plot velocity if available
        if hasattr(self, 'smoothed_velocity') and self.smoothed_velocity is not None:
            pv.line(time_axis, self.smoothed_velocity, line_width=2, color='green', alpha=0.7, legend_label="Velocity")
        elif hasattr(self, 'velocity') and self.velocity is not None:
            pv.line(time_axis, self.velocity, line_width=2, color='green', alpha=0.7, legend_label="Velocity")
        else:
            # Create dummy velocity line if no velocity data
            pv.line(time_axis, np.zeros_like(time_axis), line_width=2, color='green', alpha=0.7, legend_label="No Velocity Data")
        
        # Add vertical lines for synchronous events
        for event_time in event_times_sec:
            event_line = Span(location=event_time, dimension='height', 
                            line_color='red', line_width=1, line_dash='dashed', line_alpha=0.5)
            pv.add_layout(event_line)
        
        pv.legend.location = "top_left"
        pv.legend.click_policy = "hide"
        
        # 3. Complete neural raster plot with event markers (NEW)
        p_raster = figure(width=1000, height=500,
                         title="Complete Neural Raster Plot with Synchronous Events",
                         x_axis_label="Time (s)",
                         y_axis_label="Neuron ID",
                         x_range=p1.x_range)  # Link x-axis with population plot
        
        # Create raster data for all neurons across entire time period
        raster_data = {'x': [], 'y': [], 'colors': []}
        
        # Create a set of event times for quick lookup
        event_times_set = set(synchronous_events['event_times'])
        
        # Plot spikes for all neurons
        for neuron_idx, peaks in enumerate(self.peak_indices):
            for peak in peaks:
                peak_time_sec = peak / self.ci_rate
                raster_data['x'].append(peak_time_sec)
                raster_data['y'].append(neuron_idx)
                
                # Color code spikes that are close to synchronous events
                is_event_spike = False
                for event_time in synchronous_events['event_times']:
                    if abs(peak - event_time) <= time_window:
                        is_event_spike = True
                        break
                
                if is_event_spike:
                    raster_data['colors'].append('red')
                else:
                    raster_data['colors'].append('black')
        
        # Add raster plot
        if raster_data['x']:  # Only plot if we have data
            raster_source = ColumnDataSource(data=raster_data)
            p_raster.scatter('x', 'y', source=raster_source, size=2, color='colors', alpha=0.6)
        
        # Add vertical lines for synchronous events
        for event_time in event_times_sec:
            event_line = Span(location=event_time, dimension='height', 
                            line_color='red', line_width=2, line_dash='dashed', line_alpha=0.7)
            p_raster.add_layout(event_line)
        
        # 4. Event strength histogram (moved to bottom)
        p2 = figure(width=500, height=300, 
                   title="Distribution of Event Strengths",
                   x_axis_label="Number of Participating Neurons",
                   y_axis_label="Count")
        
        hist, edges = np.histogram(event_strengths, bins=20)
        p2.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], 
               fill_color='orange', alpha=0.7, line_color='orange')
        
        # 5. Event timeline (moved to bottom)
        p4 = figure(width=500, height=300,
                   title="Event Timeline", 
                   x_axis_label="Time (s)",
                   y_axis_label="Event Strength")
        
        p4.scatter(event_times_sec, event_strengths, size=8, color='purple', alpha=0.7)
        p4.line(event_times_sec, event_strengths, line_width=1, color='purple', alpha=0.5)
        
        # Add summary statistics
        mean_strength = np.mean(event_strengths)
        p4.line([event_times_sec[0], event_times_sec[-1]], [mean_strength, mean_strength], 
               line_color='red', line_dash='dashed', line_width=2, legend_label=f"Mean: {mean_strength:.1f}")
        
        p4.legend.location = "top_right"
        
        # Combine plots in the new layout
        layout = column(
            p1,           # Population activity overview
            pv,           # Velocity plot
            p_raster,     # Complete neural raster plot
            row(p2, p4)   # Event statistics at the bottom
        )
        
        # Print summary
        print(f"\nSynchronous Events Summary:")
        print(f"Total events found: {synchronous_events['total_events']}")
        print(f"Time window used: {synchronous_events['parameters']['time_window']} samples")
        print(f"Minimum neurons required: {synchronous_events['parameters']['min_neurons']}")
        print(f"Minimum event interval: {synchronous_events['parameters']['min_event_interval']} samples ({synchronous_events['parameters']['min_event_interval']/self.ci_rate:.2f}s)")
        print(f"Average event strength: {np.mean(event_strengths):.1f} neurons")
        print(f"Maximum event strength: {np.max(event_strengths)} neurons")
        print(f"Event rate: {synchronous_events['total_events'] / (time_axis[-1] / 60):.2f} events/minute")
        
        # Output the plot
        self.output_bokeh_plot(layout, save_path=save_path, title=title, 
                             notebook=notebook, overwrite=overwrite, font_size=font_size)
        
        return layout

    def analyze_synchronous_events_statistics(self, synchronous_events=None, time_window=5, min_neurons=3, min_event_interval=None):
        """
        Analyze statistical properties of synchronous events.
        
        Parameters:
        -----------
        synchronous_events : dict, optional
            Results from find_synchronous_events(). If None, will compute automatically
        time_window : int
            Time window for event detection (used if synchronous_events is None)
        min_neurons : int
            Minimum neurons for event detection (used if synchronous_events is None)
        min_event_interval : int, optional
            Minimum interval between events in samples (used if synchronous_events is None)
            
        Returns:
        --------
        dict
            Dictionary containing statistical analysis results
        """
        # Get synchronous events if not provided
        if synchronous_events is None:
            synchronous_events = self.find_synchronous_events(
                time_window=time_window, 
                min_neurons=min_neurons,
                min_event_interval=min_event_interval,
                return_details=True
            )
        
        if synchronous_events['total_events'] == 0:
            return {'error': 'No synchronous events found'}
        
        event_times = synchronous_events['event_times']
        event_strengths = synchronous_events['event_strengths']
        
        # Calculate inter-event intervals
        if len(event_times) > 1:
            inter_event_intervals = np.diff(event_times) / self.ci_rate  # Convert to seconds
        else:
            inter_event_intervals = np.array([])
        
        # Calculate statistics
        stats = {
            'total_events': synchronous_events['total_events'],
            'event_rate_per_minute': synchronous_events['total_events'] / (len(self.t) / self.ci_rate / 60),
            'strength_statistics': {
                'mean': np.mean(event_strengths),
                'median': np.median(event_strengths),
                'std': np.std(event_strengths),
                'min': np.min(event_strengths), 
                'max': np.max(event_strengths)
            },
            'inter_event_intervals': {
                'mean_seconds': np.mean(inter_event_intervals) if len(inter_event_intervals) > 0 else 0,
                'median_seconds': np.median(inter_event_intervals) if len(inter_event_intervals) > 0 else 0,
                'std_seconds': np.std(inter_event_intervals) if len(inter_event_intervals) > 0 else 0
            },
            'temporal_distribution': {
                'first_event_time_seconds': event_times[0] / self.ci_rate,
                'last_event_time_seconds': event_times[-1] / self.ci_rate,
                'total_duration_seconds': len(self.t) / self.ci_rate
            }
        }
        
        # Analyze participating neurons
        if 'event_details' in synchronous_events:
            all_participating_neurons = set()
            neuron_participation_count = {}
            
            for event in synchronous_events['event_details']:
                for neuron_info in event['participating_neurons']:
                    neuron_id = neuron_info['neuron_id']
                    all_participating_neurons.add(neuron_id)
                    neuron_participation_count[neuron_id] = neuron_participation_count.get(neuron_id, 0) + 1
            
            stats['neuron_participation'] = {
                'total_participating_neurons': len(all_participating_neurons),
                'participation_percentage': len(all_participating_neurons) / len(self.peak_indices) * 100,
                'most_active_neuron_id': max(neuron_participation_count, key=neuron_participation_count.get),
                'most_active_neuron_events': max(neuron_participation_count.values()),
                'average_events_per_participating_neuron': np.mean(list(neuron_participation_count.values()))
            }
        
        return stats


if __name__ == "__main__":
    print("Starting CivisServer")
