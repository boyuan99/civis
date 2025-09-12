import pandas as pd
import numpy as np
import os
import json
from scipy.signal import find_peaks, savgol_filter

# Default parameters for movement detection
DEFAULT_ONSET_PARAMS = {
    'window_size': 10,
    'baseline_window': 20,
    'threshold': 2.0,
    'min_distance': 100,
    'steady_state_threshold': 0.25,
    'baseline_max': 3
}

DEFAULT_PEAK_PARAMS = {
    'search_window': 200,
    'min_peak_height': 10.0
}

DEFAULT_OFFSET_PARAMS = {
    'search_window': 200,
    'offset_threshold': 0.1,
    'window_size': 10
}

class VirmenTank:
    def __init__(self,
                 session_name,
                 virmen_path=None,
                 maze_type=None,
                 vm_rate=20,
                 session_duration=30 * 60,
                 onset_params=DEFAULT_ONSET_PARAMS,
                 peak_params=DEFAULT_PEAK_PARAMS,
                 offset_params=DEFAULT_OFFSET_PARAMS):

        self.session_name = session_name
        self.config = self.load_config()
        if ".txt" in session_name:
            virmen_path = os.path.join(self.config['VirmenFilePath'],
                                       session_name) if virmen_path is None else virmen_path
        else:
            virmen_path = os.path.join(self.config['VirmenFilePath'],
                                       f"{session_name}.txt") if virmen_path is None else virmen_path

        self.extend_data = None
        self.virmen_path = virmen_path
        self.vm_rate = vm_rate
        self.session_duration = session_duration
        self.maze_type = self.determine_maze_type(self.virmen_path) if maze_type is None else maze_type
        self.t = np.arange(0, self.session_duration, 1 / self.vm_rate)

        self.virmen_trials, self.virmen_data, self.trials_start_indices = self.read_and_process_data(self.virmen_path,
                                                                                                     maze_type=maze_type)
        self.trials_end_indices, self.trials_end_indices_all = self.calculate_virmen_trials_end_indices(self.maze_type)
        self.trial_num = len(self.trials_end_indices)
        self.trial_num_all = len(self.trials_start_indices)
        self.lick_raw, self.lick_raw_mask, self.lick = self.find_lick_data()
        self.pstcr_raw, self.pstcr = self.find_position_movement_rate()  # position change rate
        self.velocity = self.find_velocity()
        self.smoothed_pstcr = self.butter_lowpass_filter(self.pstcr, 0.5, self.vm_rate)
        self.smoothed_pstcr[self.smoothed_pstcr <= 0] = 0
        self.smoothed_velocity = self.butter_lowpass_filter(self.velocity, 0.5, self.vm_rate)
        self.smoothed_velocity[self.smoothed_velocity <= 0] = 0
        self.dr, self.dr_raw = self.find_rotation()
        self.smoothed_dr = self.butter_lowpass_filter(self.dr, 1, self.vm_rate)
        self.acceleration = self.find_acceleration()

        self.movement_onset_indices = self.detect_movement_onsets(onset_params)
        self.velocity_peak_indices = self.detect_velocity_peaks(peak_params)
        self.movement_offset_indices = self.detect_movement_offsets(offset_params)
        self.lick_edge_indices = np.where(np.diff(self.lick) > 0)[0]

    def load_config(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        civis_dir = os.path.dirname(current_dir)
        config_path = os.path.join(civis_dir, 'config.json')

        try:
            with open(config_path, 'r') as config_file:
                return json.load(config_file)
        except FileNotFoundError:
            print(f"Config file not found at {config_path}. Using default configuration.")
            return {}

    def read_and_process_data(self, file_path, length=None, maze_type=None):

        if maze_type is None:
            maze_type = self.determine_maze_type(file_path)

        determined_maze_type = self.determine_maze_type(file_path)
        if maze_type != determined_maze_type:
            raise ValueError(
                f"Provided maze_type '{maze_type}' does not match the determined maze type '{determined_maze_type}'. "
                f"Please check your input or the maze configuration.")

        data = pd.read_csv(file_path, sep=r'\s+|,', engine='python', header=None)

        if length is not None:
            data = data.iloc[:length, :]

        potential_names = ['x', 'y', 'face_angle', 'dx', 'dy', 'lick', 'time_stamp', 'maze_type']
        data.columns = potential_names[:data.shape[1]]

        # Identifying trials
        trials = []
        starts = []
        start = 0

        if maze_type.lower() == 'straight25':
            for i in range(len(data)):
                if data.iloc[i]['y'] >= 25 or data.iloc[i]['y'] <= -25:
                    trials.append(data[start:i + 1].to_dict(orient='list'))
                    starts.append(start)
                    start = i + 1

        elif maze_type.lower() == 'straight50':
            for i in range(len(data)):
                if data.iloc[i]['y'] >= 50 or data.iloc[i]['y'] <= -50:
                    trials.append(data[start:i + 1].to_dict(orient='list'))
                    starts.append(start)
                    start = i + 1

        elif maze_type.lower() == 'straight70':
            for i in range(len(data)):
                if data.iloc[i]['y'] >= 70 or data.iloc[i]['y'] <= -70:
                    trials.append(data[start:i + 1].to_dict(orient='list'))
                    starts.append(start)
                    start = i + 1

        elif maze_type.lower() == 'straight70v3':
            for i in range(len(data)):
                if data.iloc[i]['y'] >= 70 or data.iloc[i]['y'] <= -70:
                    trials.append(data[start:i + 1].to_dict(orient='list'))
                    starts.append(start)
                    start = i + 1
            
            # Add extend_data for straight70v3 maze type to track falls
            self.extend_data = MazeV3Tank(trials, data)

        elif maze_type.lower() in ['turnv0', 'turnv1']:
            for i in range(len(data)):
                if abs(data.iloc[i]['y']) + abs(data.iloc[i]['x']) >= 175:
                    trials.append(data[start:i + 1].to_dict(orient='list'))
                    starts.append(start)
                    start = i + 1

            if maze_type.lower() == 'turnv1':
                self.extend_data = MazeV1Tank(trials, data)

        return [trials, data, starts]

    @classmethod
    def determine_maze_type(cls, virmen_path):
        data = pd.read_csv(virmen_path, sep=r'\s+|,', engine='python', header=None)
        potential_names = ['x', 'y', 'face_angle', 'dx', 'dy', 'lick', 'time_stamp', 'maze_type']
        data.columns = potential_names[:data.shape[1]]

        if 'maze_type' in data.columns:
            if len(data['maze_type'].iloc[0]) < 3:
                arr = data['x']
                mask = np.abs(arr) < 0.001
                if np.any(np.convolve(mask, np.ones(40), mode='valid') == 40):
                    return 'turnv1'
                else:
                    return 'turnv0'
            else:
                return data['maze_type'].iloc[0]

        elif np.any(data['y'] > 60) or np.any(data['y'] < -60):
            return "straight70"
        elif np.any(data['y'] > 30) or np.any(data['y'] < -30):
            return "straight50"
        return "short25"

    def calculate_virmen_trials_end_indices(self, maze_type=None):
        """
        Calculate the end indices for virmen trials based on the specified threshold.

        :return: List of indices where 'y' value exceeds the threshold.
        """
        if maze_type.lower() == 'straight25':
            indices_all = np.array(self.virmen_data.index[self.virmen_data['y'].abs() > 25].tolist())
            indices = indices_all[np.where(indices_all < self.vm_rate * self.session_duration)]

        elif maze_type.lower() == 'straight50':
            indices_all = np.array(self.virmen_data.index[self.virmen_data['y'].abs() > 50].tolist())
            indices = indices_all[np.where(indices_all < self.vm_rate * self.session_duration)]

        elif maze_type.lower() == 'straight70':
            indices_all = np.array(self.virmen_data.index[self.virmen_data['y'].abs() > 70].tolist())
            indices = indices_all[np.where(indices_all < self.vm_rate * self.session_duration)]

        elif maze_type.lower() == 'straight70v3':
            indices_all = np.array(self.virmen_data.index[self.virmen_data['y'].abs() > 70].tolist())
            indices = indices_all[np.where(indices_all < self.vm_rate * self.session_duration)]

        elif maze_type.lower() in ['turnv0', 'turnv1']:
            indices_all = np.array(self.virmen_data.index[abs(self.virmen_data['y']) +
                                                          abs(self.virmen_data['x']) >= 175].tolist())
            indices = indices_all[np.where(indices_all < self.vm_rate * self.session_duration)]

        else:
            indices = []
            indices_all = []

        return indices, indices_all

    def compute_trial_bounds(self):
        """
        Compute trial boundaries using existing start and end indices
        """
        trial_bounds = []
        for start, end in zip(self.trials_start_indices, self.trials_end_indices):
            trial_bounds.append([start, end])
        return trial_bounds

    @staticmethod
    def find_trial_for_indices(trial_bounds, indices):
        """
        Find trials for indices

        Example:
        1.
            trial_index = find_trial_for_indices(trial_bounds, indices)
        2.
            trial_indices = []
            for indices in peak_indices:
                trial_index = find_trial_for_indices(trial_bounds, indices)
                trial_indices.append(trial_index)
        """

        # Convert trial_bounds into a suitable format for np.searchsorted
        starts = np.array([bound[0] for bound in trial_bounds])
        ends = np.array([bound[1] for bound in trial_bounds])

        index_trial_map = {}
        for index in indices:
            pos = np.searchsorted(ends, index)
            if pos < len(starts) and starts[pos] <= index <= ends[pos]:
                index_trial_map[index] = pos

        return index_trial_map

    @staticmethod
    def normalize_signal(signal):
        min_val = np.min(signal)
        max_val = np.max(signal)
        normalized_signal = (signal - min_val) / (max_val - min_val)
        return normalized_signal

    @staticmethod
    def normalize_signal_with_sign(signal):

        max_val = np.max(np.abs(signal))
        normalized_signal = signal / max_val
        return normalized_signal

    @staticmethod
    def normalize_signal_per_row(signal):
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
    def shift_signal(fluorescence):
        shifted = np.zeros_like(fluorescence)
        for i, signal in enumerate(fluorescence):
            shifted[i] = signal - np.mean(signal)

        return shifted

    @staticmethod
    def shift_signal_single(signal):
        signal = signal - np.mean(signal)

        return signal

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

        lick_all = lick_all[:self.session_duration * self.vm_rate]
        lick_all_mask = lick_all_mask[:self.session_duration * self.vm_rate]
        valid_lick_mask = valid_lick_mask[:self.session_duration * self.vm_rate]

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
        dx = np.array(dx, dtype=np.float64)
        dy = np.array(dy, dtype=np.float64)
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

        # velocity_raw = velocity_raw[: self.session_duration * self.vm_rate]
        velocity = velocity[: self.session_duration * self.vm_rate]

        return velocity_raw, velocity

    def find_velocity(self, threshold=3):
        dx = np.array(self.virmen_data['dx'])
        dy = np.array(self.virmen_data['dy'])
        velocity = np.sqrt(dx ** 2 + dy ** 2)
        outliers = self.find_outliers_indices(self.normalize_signal(velocity), threshold=threshold)
        velocity[outliers] = 0
        velocity = velocity[: self.session_duration * self.vm_rate]

        return velocity

    def find_rotation(self, threshold=0.4):
        face_angle = np.array(self.virmen_data['face_angle'])
        dr_raw = np.diff(face_angle)
        dr_raw = np.insert(dr_raw, 0, 0)
        outliers = self.find_outliers_indices(self.normalize_signal(dr_raw), threshold=threshold)
        dr = dr_raw.copy()
        dr[outliers] = 0
        dr = dr[: self.session_duration * self.vm_rate]

        return dr, dr_raw

    def find_acceleration(self):
        dt = 1 / self.vm_rate
        dv = np.diff(self.smoothed_velocity)

        acceleration = dv / dt
        acceleration = np.pad(acceleration, (0, len(self.t) - len(dv)), 'constant', constant_values=(0,))

        return acceleration

    @staticmethod
    def find_outliers_indices(data, threshold=3.0):
        mean = np.mean(data)
        std_dev = np.std(data)
        z_scores = (data - mean) / std_dev
        outliers = np.where(np.abs(z_scores) > threshold)[0]
        return outliers

    @staticmethod
    def butter_lowpass_filter(data, cutoff, fs, order=4):
        from scipy.signal import butter, filtfilt

        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        [b, a] = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    def detect_movement_onsets(self, params=None):
        """
        Detect movement onsets in smoothed velocity data.
        
        Parameters:
        -----------
        params : dict, optional
            Dictionary of detection parameters including:
            - window_size: Size of forward-looking window
            - baseline_window: Size of baseline window
            - threshold: Minimum velocity change threshold
            - min_distance: Minimum samples between onsets
            - steady_state_threshold: Maximum allowed baseline variance
            - baseline_max: Maximum allowed baseline value
            
        Returns:
        --------
        onsets : ndarray
            Indices where movement onsets were detected
        """
        if params is None:
            params = {
                'window_size': 20,           # Size of forward-looking window
                'baseline_window': 20,       # Size of baseline window
                'threshold': 2.0,            # Minimum velocity change threshold
                'min_distance': 150,         # Minimum samples between onsets
                'steady_state_threshold': 0.25, # Maximum allowed baseline variance
                'baseline_max': 3.0          # Maximum allowed baseline value
            }
        
        data = np.array(self.smoothed_velocity)
        onsets = []
        
        window_size = params['window_size']
        baseline_window = params['baseline_window']
        threshold = params['threshold']
        min_distance = params['min_distance']
        steady_thresh = params['steady_state_threshold']
        baseline_max = params['baseline_max']
        
        for i in range(baseline_window, len(data) - window_size):
            baseline = data[i - baseline_window:i]
            baseline_mean = np.mean(baseline)
            baseline_std = np.std(baseline)
            
            forward = data[i:i + window_size]
            forward_mean = np.mean(forward)
            
            velocity_change = forward_mean - baseline_mean
            is_baseline_steady = baseline_std < steady_thresh
            is_significant_change = velocity_change > threshold
            is_far_from_last = len(onsets) == 0 or (i - onsets[-1]) > min_distance
            is_baseline_low = baseline_mean < baseline_max
            
            if (is_baseline_steady and is_significant_change and 
                is_far_from_last and is_baseline_low):
                next_window = data[i + window_size:i + window_size * 2]
                if len(next_window) > 0:
                    next_mean = np.mean(next_window)
                    if next_mean > forward_mean:
                        onsets.append(i)
        
        return np.array(onsets)

    def detect_velocity_peaks(self, params=None):
        """
        Detect velocity peaks after each movement onset using self.movement_onset_indices.
        
        Parameters:
        -----------
        params : dict, optional
            Dictionary of detection parameters including:
            - search_window: How far to look for peak after onset
            - min_peak_height: Minimum velocity for peak detection
        
        Returns:
        --------
        peaks : ndarray
            Indices of detected velocity peaks
        """
        if params is None:
            params = {
                'search_window': 200,  # How far to look for peak after onset
                'min_peak_height': 10.0  # Minimum velocity for peak detection
            }
        
        data = np.array(self.smoothed_velocity)
        peaks = []
        
        for onset in self.movement_onset_indices:
            end_idx = min(onset + params['search_window'], len(data))
            search_window = data[onset:end_idx]
            
            if len(search_window) > 0:
                peak_idx = onset + np.argmax(search_window)
                if data[peak_idx] >= params['min_peak_height']:
                    peaks.append(peak_idx)
        
        return np.array(peaks)
    

    def detect_movement_offsets(self, params=None):
        """
        Detect movement offsets after velocity peaks using self.velocity_peak_indices.
        
        Parameters:
        -----------
        params : dict, optional
            Dictionary of detection parameters including:
            - search_window: How far to look for offset after peak
            - offset_threshold: Velocity threshold for offset detection
            - window_size: Window size for smoothing
        
        Returns:
        --------
        offsets : ndarray
            Indices of movement offsets
        """
        if params is None:
            params = {
                'search_window': 200,    # How far to look for offset after peak
                'offset_threshold': 1.0,  # Velocity threshold for offset detection
                'window_size': 10        # Window size for smoothing
            }
        
        data = np.array(self.smoothed_velocity)
        offsets = []
        
        for peak in self.velocity_peak_indices:
            # Define search window after peak
            end_idx = min(peak + params['search_window'], len(data))
            search_window = data[peak:end_idx]
            
            # Look for point where velocity drops below threshold
            for i in range(len(search_window) - params['window_size']):
                window_mean = np.mean(search_window[i:i + params['window_size']])
                if window_mean < params['offset_threshold']:
                    offsets.append(peak + i)
                    break
                    
        return np.array(offsets)
        


    @classmethod
    def output_bokeh_plot(cls, plot, save_path=None, title=None, notebook=False, overwrite=False, font_size="12pt"):
        import os
        from bokeh.plotting import figure
        from bokeh.layouts import LayoutDOM
        from bokeh.io import output_notebook, output_file, reset_output, export_svg, export_png, save, show, curdoc

        def set_figures_backend(obj, backend):
            """Recursively set the output_backend of all figure objects"""
            if isinstance(obj, figure):
                obj.output_backend = backend
            elif isinstance(obj, LayoutDOM) and hasattr(obj, 'children'):
                for child in obj.children:
                    set_figures_backend(child, backend)
        
        def set_font_sizes(obj, font_size="24pt"):
            """Recursively set font sizes for all figure objects"""
            if isinstance(obj, figure):
                # Set title font size
                if obj.title:
                    obj.title.text_font_size = font_size
                
                # Set axis label font sizes
                obj.xaxis.axis_label_text_font_size = font_size
                obj.yaxis.axis_label_text_font_size = font_size
                
                # Set axis tick label font sizes
                obj.xaxis.major_label_text_font_size = font_size
                obj.yaxis.major_label_text_font_size = font_size
                
                # Set legend font size
                if obj.legend:
                    obj.legend.label_text_font_size = font_size
                    
            elif isinstance(obj, LayoutDOM) and hasattr(obj, 'children'):
                for child in obj.children:
                    set_font_sizes(child, font_size)

        if save_path is not None:
            reset_output()
            if os.path.exists(save_path) and not overwrite:
                print("File already exists and overwrite is set to False.")
            else:
                # Create directory structure if it doesn't exist
                directory = os.path.dirname(save_path)
                if directory and not os.path.exists(directory):
                    os.makedirs(directory, exist_ok=True)
                    print(f"Created directory: {directory}")
                
                # Set font sizes for all output formats if font_size is specified
                if font_size is not None:
                    set_font_sizes(plot, font_size)
                
                file_extension = save_path.split(".")[-1].lower()
                
                if file_extension == 'html':
                    output_file(save_path, title=title)
                    curdoc().clear()
                    curdoc().add_root(plot)
                    save(curdoc())
                    print("File saved as html.")
                    curdoc().clear()
                elif file_extension == 'svg':
                    # set all figures to SVG backend
                    set_figures_backend(plot, 'svg')
                    
                    # export SVG
                    export_svg(plot, filename=save_path)
                    print("Plot successfully saved as svg.")
                    
                    # set back to canvas backend
                    set_figures_backend(plot, 'canvas')
                elif file_extension == 'png':
                    # set all figures to canvas backend for PNG export
                    set_figures_backend(plot, 'canvas')
                    
                    # export PNG
                    export_png(plot, filename=save_path)
                    print("Plot successfully saved as png.")
                else:
                    raise ValueError("Invalid file type. Supported formats are: html, svg, png")

        if notebook:
            reset_output()
            output_notebook()
            show(plot)

    @staticmethod
    def generate_colors(num_categories):
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

    def plot_lick_and_velocity(self, title="Lick And Velocity", notebook=False, save_path=None, overwrite=False, font_size=None):
        """
        Plot the lick data and velocity data with vertical lines for events.
        """
        from bokeh.plotting import figure
        from bokeh.models import HoverTool
        from bokeh.layouts import column

        # Prepare vertical lines data
        def create_vertical_lines(indices, y_range):
            """Helper function to create vertical line coordinates"""
            xs = [[idx/self.vm_rate, idx/self.vm_rate] for idx in indices]
            ys = [[y_range[0], y_range[1]] for _ in indices]
            return xs, ys

        # Create lick plot
        lick_y_range = [0, 1.2]
        p = figure(width=800, height=400, y_range=lick_y_range, 
                  active_drag='pan', active_scroll='wheel_zoom', title='Lick')
        
        # Add main data lines
        p.line(self.t, self.lick_raw, line_color='navy', legend_label='raw', line_width=1)
        p.line(self.t, self.lick_raw_mask, line_color='palevioletred', legend_label='mask', line_width=2)
        p.line(self.t, self.lick, line_color='gold', legend_label='filtered', line_width=2)

        # Add vertical lines for events
        trials_xs, trials_ys = create_vertical_lines(self.trials_end_indices, lick_y_range)
        velocity_xs, velocity_ys = create_vertical_lines(self.velocity_peak_indices, lick_y_range)
        movement_onset_xs, movement_onset_ys = create_vertical_lines(self.movement_onset_indices, lick_y_range)
        movement_offset_xs, movement_offset_ys = create_vertical_lines(self.movement_offset_indices, lick_y_range)

        p.multi_line(trials_xs, trials_ys, line_color='green', line_width=2, 
                    legend_label='Trial Ends', level='underlay')
        p.multi_line(velocity_xs, velocity_ys, line_color='blue', line_width=2, 
                    legend_label='Velocity Peaks', level='underlay')
        p.multi_line(movement_onset_xs, movement_onset_ys, line_color='brown', line_width=2, 
                    legend_label='Movement Onsets', level='underlay')
        p.multi_line(movement_offset_xs, movement_offset_ys, line_color='purple', line_width=2, 
                    legend_label='Movement Offsets', level='underlay')

        p.legend.click_policy = "hide"
        p.add_tools(HoverTool(tooltips=[("Index", "$index"), ("(x, y)", "($x, $y)")]))

        # Create velocity plot
        p_v = figure(width=800, height=400, x_range=p.x_range,
                    active_drag='pan', active_scroll='wheel_zoom', title="Velocity")
        
        # Add main velocity data lines
        p_v.line(self.t, self.smoothed_velocity, line_color='navy', 
                legend_label='smoothed_velocity', line_width=2)
        p_v.line(self.t, self.pstcr, line_color='red', 
                legend_label='position change rate', line_width=2)
        p_v.line(self.t, self.velocity, line_color='purple', 
                legend_label='velocity', line_width=2)

        # Add vertical lines for events (using the plot's y range)
        velocity_y_range = [min(self.velocity), max(self.velocity)]
        trials_xs, trials_ys = create_vertical_lines(self.trials_end_indices, velocity_y_range)
        velocity_xs, velocity_ys = create_vertical_lines(self.velocity_peak_indices, velocity_y_range)
        movement_onset_xs, movement_onset_ys = create_vertical_lines(self.movement_onset_indices, velocity_y_range)
        movement_offset_xs, movement_offset_ys = create_vertical_lines(self.movement_offset_indices, velocity_y_range)

        p_v.multi_line(trials_xs, trials_ys, line_color='green', line_width=2, 
                      legend_label='Trial Ends', level='underlay')
        p_v.multi_line(velocity_xs, velocity_ys, line_color='blue', line_width=2, 
                      legend_label='Velocity Peaks', level='underlay')
        p_v.multi_line(movement_onset_xs, movement_onset_ys, line_color='brown', line_width=2, 
                      legend_label='Movement Onsets', level='underlay')
        p_v.multi_line(movement_offset_xs, movement_offset_ys, line_color='purple', line_width=2, 
                      legend_label='Movement Offsets', level='underlay')

        p_v.legend.click_policy = "hide"
        p_v.add_tools(HoverTool(tooltips=[("Index", "$index"), ("(x, y)", "($x, $y)")]))

        # Create layout and output
        layout = column(p, p_v)
        self.output_bokeh_plot(layout, save_path=save_path, title=title, 
                             notebook=notebook, overwrite=overwrite, font_size=font_size)

        return layout

    def plot_summary_trajectory(self, title="Summary Trajectory", notebook=False, save_path=None, overwrite=False, font_size=None):
        """
        Plot summary trajectory showing all trials overlaid for the current maze type.
        
        Parameters:
        -----------
        title : str, optional
            Title for the plot
        notebook : bool, optional
            Whether to display the plot in a notebook
        save_path : str, optional
            Path to save the plot
        overwrite : bool, optional
            Whether to overwrite existing files
        font_size : str, optional
            Font size for plot elements
            
        Returns:
        --------
        bokeh plot
            The generated trajectory visualization
        """
        from bokeh.plotting import figure
        from bokeh.models import ColumnDataSource, HoverTool
        from bokeh.palettes import Category10
        import numpy as np
        
        # Set figure dimensions based on maze type
        if self.maze_type.lower() in ['turnv0', 'turnv1']:
            # Turn mazes are wider and more square
            width, height = 600, 600
        elif self.maze_type.lower() in ['straight70v3']:
            # Straight70v3 is narrower and taller
            width, height = 400, 800
        else:
            # Other straight mazes are narrow and tall
            width, height = 350, 800
        
        # Create figure
        p = figure(title=f"{title} - {self.maze_type.upper()}", width=width, height=height,
                  active_drag='pan', active_scroll='wheel_zoom')
        
        # Plot individual trials
        colors = Category10[10]
        for i, trial in enumerate(self.virmen_trials):
            if len(trial['x']) > 0:  # Only plot non-empty trials
                color = colors[i % len(colors)]
                alpha = 0.6 if len(self.virmen_trials) > 10 else 0.8
                
                # Create trial source
                trial_source = ColumnDataSource(data={
                    'x': trial['x'],
                    'y': trial['y'],
                    'trial': [i] * len(trial['x'])
                })
                
                p.line('x', 'y', source=trial_source, line_color=color, 
                      line_width=1, alpha=alpha, legend_label=f'Trial {i+1}')
        
        # Add overall trajectory (all data points)
        all_source = ColumnDataSource(data={
            'x': self.virmen_data['x'],
            'y': self.virmen_data['y'],
            'index': list(range(len(self.virmen_data)))
        })
        
        # Plot maze boundaries based on maze type
        self._add_maze_boundaries(p)
        
        # Add trial end points
        if len(self.trials_end_indices) > 0:
            end_x = [self.virmen_data['x'].iloc[idx] for idx in self.trials_end_indices if idx < len(self.virmen_data)]
            end_y = [self.virmen_data['y'].iloc[idx] for idx in self.trials_end_indices if idx < len(self.virmen_data)]
            
            end_source = ColumnDataSource(data={
                'x': end_x,
                'y': end_y,
                'trial_num': list(range(len(end_x)))
            })
            
            p.scatter('x', 'y', source=end_source, color='red', size=8,
                    legend_label='Trial Ends', alpha=0.8)
        
        # Add hover tool
        hover = HoverTool(tooltips=[
            ("Position", "(@x, @y)"),
            ("Trial", "@trial"),
            ("Index", "@index")
        ])
        p.add_tools(hover)
        
        # Configure plot appearance
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"
        p.grid.grid_line_color = "lightgray"
        p.axis.axis_line_color = "black"
        p.axis.major_tick_line_color = "black"
        p.axis.minor_tick_line_color = "gray"
        p.xaxis.axis_label = "X Position (cm)"
        p.yaxis.axis_label = "Y Position (cm)"
        
        # Make legend more readable for many trials
        if len(self.virmen_trials) > 10:
            p.legend.visible = False
        
        # Output the plot
        self.output_bokeh_plot(p, save_path=save_path, title=title,
                             notebook=notebook, overwrite=overwrite, font_size=font_size)
        
        return p
    
    def _add_maze_boundaries(self, plot):
        """
        Add maze boundary lines based on maze type.
        
        Parameters:
        -----------
        plot : bokeh figure
            The plot to add boundaries to
        """
        # Get data range for boundary lines
        x_data = self.virmen_data['x']
        y_data = self.virmen_data['y']
        x_range = [min(x_data), max(x_data)]
        y_range = [min(y_data), max(y_data)]
        
        if self.maze_type.lower() == 'straight25':
            # Add horizontal boundary lines at ±25
            plot.line(x_range, [25, 25], line_color='black', line_width=2, 
                     line_dash='dashed', legend_label='Maze Boundary')
            plot.line(x_range, [-25, -25], line_color='black', line_width=2, 
                     line_dash='dashed')
                     
        elif self.maze_type.lower() == 'straight50':
            # Add horizontal boundary lines at ±50
            plot.line(x_range, [50, 50], line_color='black', line_width=2, 
                     line_dash='dashed', legend_label='Maze Boundary')
            plot.line(x_range, [-50, -50], line_color='black', line_width=2, 
                     line_dash='dashed')
                     
        elif self.maze_type.lower() in ['straight70', 'straight70v3']:
            # Add horizontal boundary lines at ±70
            plot.line(x_range, [70, 70], line_color='black', line_width=2, 
                     line_dash='dashed', legend_label='Maze Boundary')
            plot.line(x_range, [-70, -70], line_color='black', line_width=2, 
                     line_dash='dashed')
                     
        elif self.maze_type.lower() in ['turnv0', 'turnv1']:
            # Add boundary lines for turn maze (|x| + |y| = 175)
            # Create boundary points
            boundary_points = []
            for angle in np.linspace(0, 2*np.pi, 100):
                x = 175 * np.cos(angle)
                y = 175 * np.sin(angle)
                if abs(x) + abs(y) <= 175:
                    boundary_points.append((x, y))
            
            if boundary_points:
                boundary_x, boundary_y = zip(*boundary_points)
                plot.line(boundary_x, boundary_y, line_color='black', line_width=2,
                         line_dash='dashed', legend_label='Maze Boundary')


class MazeV3Tank():
    def __init__(self, trials, data):
        self.virmen_trials = trials
        self.virmen_data = data
        self.fall_indices = self.detect_falls()
        self.fall_count = len(self.fall_indices)
        
        # Discriminate between left and right falls
        self.left_fall_indices, self.right_fall_indices = self.classify_fall_directions()
        self.left_fall_count = len(self.left_fall_indices)
        self.right_fall_count = len(self.right_fall_indices)
        
        # Enhanced trial tracking
        self.enhanced_trials = self.detect_enhanced_trials()
        self.enhanced_trial_start_indices = [trial['start_idx'] for trial in self.enhanced_trials]
        self.enhanced_trial_end_indices = [trial['end_idx'] for trial in self.enhanced_trials]
        self.trial_outcomes = [trial['outcome'] for trial in self.enhanced_trials]
        self.success_count = sum(1 for outcome in self.trial_outcomes if outcome == 'success')
        self.failure_count = sum(1 for outcome in self.trial_outcomes if outcome == 'failure')
        self.left_failure_count = sum(1 for outcome in self.trial_outcomes if outcome == 'left_failure')
        self.right_failure_count = sum(1 for outcome in self.trial_outcomes if outcome == 'right_failure')
        self.total_enhanced_trials = len(self.enhanced_trials)
        
    def detect_falls(self, x_boundary=20, y_success_threshold=70, jump_threshold=10, 
                     freeze_threshold=0.5, max_lookback=1000):
        """
        Lookback fall start detection method
        
        Detection logic:
        1. First find fall end moments (large position jump + pre-jump out-of-bounds and unsuccessful)
        2. From end moment, look back to find when position started freezing
        3. Return fall start moment indices
        
        Parameters:
        -----------
        x_boundary : float
            X-direction boundary threshold, default 20cm
        y_success_threshold : float  
            Y-direction success threshold, default 70cm
        jump_threshold : float
            Position jump threshold, default 10cm
        freeze_threshold : float
            Position freeze detection threshold, default 0.5cm
        max_lookback : int
            Maximum lookback frames, default 100 frames
            
        Returns:
        --------
        fall_indices : list
            List of fall start moment indices
        """
        x_positions = np.array(self.virmen_data['x'])
        y_positions = np.array(self.virmen_data['y'])
        
        # Calculate frame-by-frame position changes
        dx = np.diff(x_positions)
        dy = np.diff(y_positions)
        position_changes = np.sqrt(dx**2 + dy**2)
        position_changes = np.append([0], position_changes)  # Pad to match length
        
        # Step 1: Find fall end moments
        fall_end_indices = []
        
        for i in range(1, len(x_positions)):
            # Detect large position jumps
            if position_changes[i] > jump_threshold:
                previous_x = x_positions[i-1]
                previous_y = y_positions[i-1]
                
                # Confirm this is a fall, not trial success
                was_outside_x = abs(previous_x) > x_boundary
                was_not_successful = abs(previous_y) < y_success_threshold
                
                if was_outside_x and was_not_successful:
                    fall_end_indices.append(i-1)
        
        # Step 2: Look back from each fall end moment to find start moment
        fall_start_indices = []
        
        for end_idx in fall_end_indices:
            # Look back to find when position started freezing
            freeze_start_idx = end_idx
            
            for lookback in range(1, min(max_lookback, end_idx)):
                check_idx = end_idx - lookback
                
                # Check position change at this frame
                if position_changes[check_idx] > freeze_threshold:
                    # Position still changing, hasn't started freezing yet
                    freeze_start_idx = check_idx + 1
                    break
            
            # Avoid adding duplicate nearby fall start moments
            if (len(fall_start_indices) == 0 or 
                freeze_start_idx - fall_start_indices[-1] > 20):
                fall_start_indices.append(freeze_start_idx)
        
        return fall_start_indices
    
    def classify_fall_directions(self):
        """
        Classify fall events into left and right falls based on x-position at fall moment.
        
        Returns:
        --------
        tuple
            (left_fall_indices, right_fall_indices) - Lists of fall indices separated by direction
        """
        x_positions = np.array(self.virmen_data['x'])
        
        left_fall_indices = []
        right_fall_indices = []
        
        for fall_idx in self.fall_indices:
            if fall_idx < len(x_positions):
                x_at_fall = x_positions[fall_idx]
                if x_at_fall < 0:
                    left_fall_indices.append(fall_idx)
                else:
                    right_fall_indices.append(fall_idx)
        
        return left_fall_indices, right_fall_indices
    
    def detect_enhanced_trials(self, x_boundary=20, y_success_threshold=70, jump_threshold=10, 
                               freeze_threshold=0.5, max_lookback=1000):
        """
        Detect enhanced trials that separate based on both success and failure endpoints.
        
        This method creates a comprehensive trial structure that includes:
        - Success trials: trials ending when animal reaches success threshold (±70cm in y)
        - Failure trials: trials ending when animal falls (position jump after being out of bounds)
        
        Parameters:
        -----------
        x_boundary : float
            X-direction boundary threshold, default 20cm
        y_success_threshold : float  
            Y-direction success threshold, default 70cm
        jump_threshold : float
            Position jump threshold, default 10cm
        freeze_threshold : float
            Position freeze detection threshold, default 0.5cm
        max_lookback : int
            Maximum lookback frames, default 1000 frames
            
        Returns:
        --------
        enhanced_trials : list
            List of dictionaries containing trial information with keys:
            - start_idx: Trial start index
            - end_idx: Trial end index
            - outcome: 'success' or 'failure'
            - duration: Trial duration in frames
            - trial_data: Dictionary containing x, y, and other data for this trial
        """
        x_positions = np.array(self.virmen_data['x'])
        y_positions = np.array(self.virmen_data['y'])
        
        # Find success endpoints (reaching ±70cm in y direction)
        success_indices = []
        for i in range(len(y_positions)):
            if abs(y_positions[i]) >= y_success_threshold:
                success_indices.append(i)
        
        # Find failure endpoints using fall detection logic
        fall_end_indices = []
        
        # Calculate frame-by-frame position changes
        dx = np.diff(x_positions)
        dy = np.diff(y_positions)
        position_changes = np.sqrt(dx**2 + dy**2)
        position_changes = np.append([0], position_changes)  # Pad to match length
        
        # Find fall end moments (position jumps after being out of bounds)
        for i in range(1, len(x_positions)):
            if position_changes[i] > jump_threshold:
                previous_x = x_positions[i-1]
                previous_y = y_positions[i-1]
                
                # Confirm this is a fall (out of x bounds and below success threshold)
                was_outside_x = abs(previous_x) > x_boundary
                was_not_successful = abs(previous_y) < y_success_threshold
                
                if was_outside_x and was_not_successful:
                    fall_end_indices.append(i-1)  # Record the moment before jump
        
        # Combine and sort all endpoints, classifying failures by direction
        all_endpoints = []
        for idx in success_indices:
            all_endpoints.append((idx, 'success'))
        for idx in fall_end_indices:
            # Classify failure direction based on x-position at fall moment
            if idx < len(x_positions):
                x_at_fall = x_positions[idx]
                outcome = 'left_failure' if x_at_fall < 0 else 'right_failure'
            else:
                outcome = 'failure'  # fallback if index is out of bounds
            all_endpoints.append((idx, outcome))
        
        # Sort by index to get chronological order
        all_endpoints.sort(key=lambda x: x[0])
        
        # Create enhanced trials
        enhanced_trials = []
        current_start = 0
        
        for end_idx, outcome in all_endpoints:
            if end_idx > current_start:  # Ensure we have a valid trial
                # Extract trial data
                trial_data = {}
                for column in self.virmen_data.columns:
                    trial_data[column] = list(self.virmen_data[column].iloc[current_start:end_idx+1])
                
                trial_info = {
                    'start_idx': current_start,
                    'end_idx': end_idx,
                    'outcome': outcome,
                    'duration': end_idx - current_start + 1,
                    'trial_data': trial_data
                }
                enhanced_trials.append(trial_info)
                
                # Next trial starts after current endpoint
                current_start = end_idx + 1
        
        # Handle remaining data if any (incomplete trial at the end)
        if current_start < len(self.virmen_data) - 1:
            trial_data = {}
            for column in self.virmen_data.columns:
                trial_data[column] = list(self.virmen_data[column].iloc[current_start:])
                
            # Determine outcome based on final position
            final_y = y_positions[-1]
            final_outcome = 'success' if abs(final_y) >= y_success_threshold else 'incomplete'
            
            trial_info = {
                'start_idx': current_start,
                'end_idx': len(self.virmen_data) - 1,
                'outcome': final_outcome,
                'duration': len(self.virmen_data) - current_start,
                'trial_data': trial_data
            }
            enhanced_trials.append(trial_info)
        
        return enhanced_trials

    def get_fall_summary(self):
        """
        Returns a summary of the falls detected in the session, including directional fall statistics.
        
        Returns:
        --------
        dict
            A dictionary containing fall statistics including left/right falls
        """
        return {
            "total_falls": self.fall_count,
            "left_falls": self.left_fall_count,
            "right_falls": self.right_fall_count,
            "fall_indices": self.fall_indices,
            "left_fall_indices": self.left_fall_indices,
            "right_fall_indices": self.right_fall_indices,
            "left_fall_rate": self.left_fall_count / self.fall_count if self.fall_count > 0 else 0,
            "right_fall_rate": self.right_fall_count / self.fall_count if self.fall_count > 0 else 0,
            "average_trial_length": np.mean([len(trial['x']) for trial in self.virmen_trials]) if self.virmen_trials else 0,
            "total_trials": len(self.virmen_trials),
            "enhanced_trial_summary": {
                "total_enhanced_trials": self.total_enhanced_trials,
                "success_count": self.success_count,
                "left_failure_count": self.left_failure_count,
                "right_failure_count": self.right_failure_count,
                "success_rate": self.success_count / self.total_enhanced_trials if self.total_enhanced_trials > 0 else 0,
                "left_failure_rate": self.left_failure_count / self.total_enhanced_trials if self.total_enhanced_trials > 0 else 0,
                "right_failure_rate": self.right_failure_count / self.total_enhanced_trials if self.total_enhanced_trials > 0 else 0
            }
        }
    
    def plot_falls(self, save_path=None, title="Animal Falls in Maze", notebook=False, overwrite=False, font_size=None):
        """
        Creates a visualization of animal trajectory with fall locations highlighted.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the plot
        title : str, optional
            Title for the plot
        notebook : bool, optional
            Whether to display the plot in a notebook
        overwrite : bool, optional
            Whether to overwrite existing files
            
        Returns:
        --------
        bokeh plot
            The generated visualization
        """
        from bokeh.plotting import figure
        from bokeh.models import ColumnDataSource, HoverTool
        
        # Create position data source
        source = ColumnDataSource(data={
            'x': self.virmen_data['x'],
            'y': self.virmen_data['y'],
            'index': list(range(len(self.virmen_data)))
        })
        
        # Create fall data source
        fall_x = [self.virmen_data['x'].iloc[idx] for idx in self.fall_indices]
        fall_y = [self.virmen_data['y'].iloc[idx] for idx in self.fall_indices]
        fall_source = ColumnDataSource(data={
            'x': fall_x,
            'y': fall_y,
            'index': self.fall_indices
        })
        
        # Create figure
        p = figure(title=title, width=800, height=600, 
                  active_drag='pan', active_scroll='wheel_zoom')
        
        # Plot trajectory
        p.line('x', 'y', source=source, line_color='navy', line_width=1, alpha=0.6)
        
        # Highlight falls
        p.scatter('x', 'y', source=fall_source, color='red', size=10,
                alpha=0.8, legend_label=f'Falls ({self.fall_count})')
        
        # Add hover tool
        hover = HoverTool(tooltips=[
            ("Index", "@index"),
            ("Position", "(@x, @y)")
        ])
        p.add_tools(hover)
        
        # Configure plot
        p.legend.location = "top_left"
        p.legend.click_policy = "hide"
        p.grid.grid_line_color = "lightgray"
        p.axis.axis_line_color = "black"
        p.axis.major_tick_line_color = "black"
        p.axis.minor_tick_line_color = "gray"
        
        # Output
        VirmenTank.output_bokeh_plot(p, save_path=save_path, title=title, 
                                   notebook=notebook, overwrite=overwrite, font_size=font_size)
        
        return p


class MazeV1Tank():
    def __init__(self, trials, data):
        self.virmen_trials = trials
        self.virmen_data = data
        self.correct_array = self.analyze_trials_correctness()
        self.maze_type_array = self.get_maze_type_array()
        self.confusion_matrix = self.generate_confusion_matrix()

    @staticmethod
    def determine_trial_correctness(trial_data):
        """
        Determines whether a trial is correct based on the turning direction and maze_type.

        :param trial_data: Dictionary containing the trial data
        :return: Boolean indicating whether the trial is correct
        """
        # Extract relevant data
        x_values = trial_data['x']
        maze_type = trial_data['maze_type'][7]

        actual_turn = 'left' if x_values[-1] < 0 else 'right'
        correct_turn = 'left' if maze_type == 0 else 'right'

        # Check if the actual turn matches the correct turn
        return actual_turn == correct_turn

    def analyze_trials_correctness(self):
        """
        Analyzes all trials and returns a list of booleans indicating correctness.

        :return: List of booleans, True for correct trials, False for incorrect ones
        """
        correctness = []
        for trial in self.virmen_trials:
            is_correct = self.determine_trial_correctness(trial)
            correctness.append(is_correct)
        return correctness

    def get_maze_type_array(self):

        maze_type_array = []
        for i in range(len(self.virmen_trials)):
            maze_type_array.append(self.virmen_trials[i]['maze_type'][-1])

        return maze_type_array

    def generate_confusion_matrix(self, print_info=False):
        """
        Generates a confusion matrix based on maze types and trial correctness.

        :param maze_types: List of maze types (0 for left, 1 for right)
        :param correctness: List of booleans indicating trial correctness
        :return: 2x2 numpy array representing the confusion matrix
        """
        confusion_matrix = np.zeros((2, 2), dtype=int)

        for maze_type, is_correct in zip(self.maze_type_array, self.correct_array):
            if maze_type == 0:  # Left turn
                if is_correct:
                    confusion_matrix[0, 0] += 1  # Correct Left Turn
                else:
                    confusion_matrix[0, 1] += 1  # Incorrect Right Turn
            else:  # Right turn
                if is_correct:
                    confusion_matrix[1, 1] += 1  # Correct Right Turn
                else:
                    confusion_matrix[1, 0] += 1  # False Positive

        if print_info:
            cm = confusion_matrix
            print("Confusion Matrix:")
            print(cm)
            print("\nInterpretation:")
            print(f"Correct Left Turns: {cm[0, 0]}")
            print(f"Incorrect Right Turns: {cm[0, 1]}")
            print(f"Incorrect Left Turns: {cm[1, 0]}")
            print(f"Correct Right Turns: {cm[1, 1]}")

            # Calculate and print additional metrics
            total_trials = np.sum(cm)
            accuracy = (cm[0, 0] + cm[1, 1]) / total_trials
            print(f"\nAccuracy: {accuracy:.2%}")

            left_precision = cm[0, 0] / (cm[0, 0] + cm[1, 0]) if (cm[0, 0] + cm[1, 0]) > 0 else 0
            right_precision = cm[1, 1] / (cm[1, 1] + cm[0, 1]) if (cm[1, 1] + cm[0, 1]) > 0 else 0
            print(f"Left Turn Precision: {left_precision:.2%}")
            print(f"Right Turn Precision: {right_precision:.2%}")

            left_recall = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
            right_recall = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
            print(f"Left Turn Recall: {left_recall:.2%}")
            print(f"Right Turn Recall: {right_recall:.2%}")

        return confusion_matrix

    def plot_confusion_matrix(self, save_path=None, title=None, notebook=False, overwrite=False, backend='matplotlib', font_size=None):
        """
        Plots the confusion matrix as a heatmap.

        :param backend: The backend to use for plotting ('matplotlib' or 'bokeh').
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        from bokeh.plotting import figure
        from bokeh.models import BasicTicker, ColorBar, LinearColorMapper, ColumnDataSource
        from bokeh.palettes import Blues8

        if backend == 'matplotlib':
            fig, ax = plt.subplots(figsize=(5, 4))
            sns.heatmap(self.confusion_matrix, annot=True, fmt='d', cmap='Blues',
                        xticklabels=['Left', 'Right'], yticklabels=['Left', 'Right'], ax=ax)
            ax.set_title('Confusion Matrix of Animal Turns')
            ax.set_ylabel('Actual Turn')
            ax.set_xlabel('Should Turn')

            return fig

        elif backend == 'bokeh':

            # Assuming self.confusion_matrix is a 2x2 numpy array
            confusion_matrix = self.confusion_matrix

            # Create data for the heatmap
            x_labels = ['Left', 'Right']
            y_labels = ['Left', 'Right']
            data = {
                'x': [x for x in x_labels for _ in y_labels],
                'y': y_labels * len(x_labels),
                'value': confusion_matrix.flatten().tolist(),
            }

            source = ColumnDataSource(data=data)

            # Create the color mapper
            color_mapper = LinearColorMapper(palette=Blues8[::-1], low=0, high=confusion_matrix.max())

            # Create the figure
            p = figure(title="Confusion Matrix of Animal Turns",
                       x_range=x_labels, y_range=list(reversed(y_labels)),
                       x_axis_label="Should Turn", y_axis_label="Actual Turn",
                       width=400, height=350, toolbar_location=None, tools="")

            # Create the heatmap
            p.rect(x='x', y='y', width=1, height=1, source=source,
                   line_color=None, fill_color={'field': 'value', 'transform': color_mapper})

            # Add color bar
            color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(desired_num_ticks=len(Blues8)),
                                 label_standoff=6, border_line_color=None, location=(0, 0))

            p.add_layout(color_bar, 'right')

            # Add text annotations
            p.text(x='x', y='y', text='value', source=source,
                   text_align="center", text_baseline="middle")

            VirmenTank.output_bokeh_plot(p, save_path=save_path, title=title, notebook=notebook, overwrite=overwrite, font_size=font_size)

            return p
        else:
            raise ValueError("Unsupported backend. Use 'matplotlib' or 'bokeh'.")

    def current_accuracy(self, save_path=None, title=None, notebook=False, overwrite=False):
        """
        Returns dictionary of trial as key and accurancy as item (as decimal) up to and including that trial
        """

        if len(self.correct_array) < 5:
            trials = {0: int(self.correct_array[0])}
            for i in range(1, len(self.correct_array)):
                trials[i] = (trials[i - 1] * i + int(self.correct_array[i])) / (i + 1)
        else:
            trials = {}
            for i in range(5, len(self.correct_array)):
                trials[i] = (sum(self.correct_array[i - 5:i])) / 5

        return trials


if __name__ == "__main__":
    print("Starting CivisServer")
