import pandas as pd
import numpy as np
import h5py
import json
from scipy.signal import find_peaks, savgol_filter


class VirmenTank:
    def __init__(self,
                 virmen_path,
                 maze_type=None,
                 threshold=None,
                 virmen_data_length=None,
                 vm_rate=20,
                 velocity_height=0.7,
                 velocity_distance=100,
                 session_duration=30 * 60,
                 window_length=31,
                 polyorder=3,
                 height=8):

        self.extend_data = None
        self.virmen_path = virmen_path
        self.vm_rate = vm_rate
        self.session_duration = session_duration
        self.threshold = threshold
        self.window_length = window_length  # Window length: number of coefficients (odd number)
        self.polyorder = polyorder  # Polynomial order
        self.velocity_height = velocity_height
        self.velocity_distance = velocity_distance
        self.maze_type = self.determine_maze_type(self.virmen_path) if maze_type is None else maze_type
        self.t = np.arange(0, self.session_duration, 1 / self.vm_rate)

        self.virmen_trials, self.virmen_data, self.trials_start_indices = self.read_and_process_data(self.virmen_path,
                                                                                                     threshold=self.threshold,
                                                                                                     length=virmen_data_length,
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

        # for analysis usage:
        self.velocity_peak_indices = find_peaks(self.smoothed_velocity, height=height,
                                                distance=self.velocity_distance)[0]
        self.lick_edge_indices = np.where(np.diff(self.lick) > 0)[0]
        self.movement_onset_indices = self.find_movement_onset(self.smoothed_velocity, self.velocity_peak_indices)

    def read_and_process_data(self, file_path, threshold=None, length=None, maze_type=None):
        if threshold is None:
            threshold = [70.0, -70.0]

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
            arr = data['x']
            mask = np.abs(arr) < 0.001
            if np.any(np.convolve(mask, np.ones(40), mode='valid') == 40):
                return 'turnv1'
            else:
                return 'turnv0'

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

        velocity_raw = velocity_raw[: self.session_duration * self.vm_rate]
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

    @staticmethod
    def find_movement_onset(velocity, velocity_peak_indices, threshold_ratio=0.1):
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

    @classmethod
    def output_bokeh_plot(cls, plot, save_path=None, title=None, notebook=False, overwrite=False):
        import os
        from bokeh.plotting import figure
        from bokeh.layouts import LayoutDOM
        from bokeh.io import output_notebook, output_file, reset_output, export_svg, save, show, curdoc

        if save_path is not None:
            reset_output()
            if os.path.exists(save_path) and not overwrite:
                print("File already exists and overwrite is set to False.")
            else:
                if save_path.split(".")[-1] == 'html':
                    output_file(save_path, title=title)
                    curdoc().clear()
                    curdoc().add_root(plot)
                    save(curdoc())
                    print("File saved as html.")
                    curdoc().clear()
                elif save_path.split(".")[-1] == 'svg':
                    if isinstance(plot, figure):
                        plot.output_backend = "svg"
                        export_svg(plot, filename=save_path)
                        print("Plot successfully saved as svg.")
                        plot.output_backend = "canvas"
                    elif isinstance(plot, LayoutDOM):
                        for i in range(len(plot.children)):
                            plot.children[i].output_backend = 'svg'
                        export_svg(plot, filename=save_path)
                        print("Plot successfully saved as svg.")
                        for i in range(len(plot.children)):
                            plot.children[i].output_backend = 'canvas'
                    else:
                        raise ValueError("Invalid plot element.")
                else:
                    raise ValueError("Invalid file type.")

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

    def plot_lick_and_velocity(self, title="Lick And Velocity", notebook=False, save_path=None, overwrite=False):
        """
        Plot the lick data and velocity data.
        :param notebook: Flag to indicate if the plot is for a Jupyter notebook.
        :param save_path:  Path to save the plot as an HTML file.
        :param overwrite: Flag to indicate whether to overwrite the existing plot
        :return: None
        """
        from bokeh.plotting import figure
        from bokeh.models import Span, HoverTool, CheckboxGroup, CustomJS
        from bokeh.layouts import column, row

        p = figure(width=800, height=400, y_range=[0, 1.2], active_drag='pan', active_scroll='wheel_zoom', title='Lick')
        p.line(self.t, self.lick_raw, line_color='navy', legend_label='raw', line_width=1)
        p.line(self.t, self.lick_raw_mask, line_color='palevioletred', legend_label='mask', line_width=2)
        p.line(self.t, self.lick, line_color='gold', legend_label='filtered', line_width=2)

        # Different groups of spans
        trials_end_spans = []
        velocity_peak_spans = []
        movement_onset_spans = []

        # Adding spans for trials end
        for idx in self.trials_end_indices:
            vline = Span(location=idx / self.vm_rate, dimension='height', line_color='green', line_width=2,
                         visible=True)
            trials_end_spans.append(vline)
            p.add_layout(vline, 'below')

        # Adding spans for velocity peaks
        for idx in self.velocity_peak_indices:
            vline = Span(location=idx / self.vm_rate, dimension='height', line_color='blue', line_width=2, visible=True)
            velocity_peak_spans.append(vline)
            p.add_layout(vline, 'below')

        # Adding spans for movement onsets
        for idx in self.movement_onset_indices:
            vline = Span(location=idx / self.vm_rate, dimension='height', line_color='brown', line_width=2,
                         visible=True)
            movement_onset_spans.append(vline)
            p.add_layout(vline, 'below')

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
        p_v.line(self.t, self.smoothed_velocity, line_color='navy', legend_label='smoothed_velocity', line_width=2)
        p_v.line(self.t, self.pstcr, line_color='red', legend_label='position change rate', line_width=2)
        p_v.line(self.t, self.velocity, line_color='purple', legend_label='velocity', line_width=2)

        # Copy spans to velocity plot
        for span in trials_end_spans + velocity_peak_spans + movement_onset_spans:
            p_v.add_layout(span)

        hover_v = HoverTool()
        hover_v.tooltips = [
            ("Index", "$index"),
            ("(x, y)", "($x, $y)"),
        ]
        p_v.add_tools(hover_v)
        p_v.legend.click_policy = "hide"

        # Checkboxes for toggling span visibility
        trials_checkbox = CheckboxGroup(labels=["Show Trial Ends"], active=[0], width=150)
        velocity_checkbox = CheckboxGroup(labels=["Show Velocity Peaks"], active=[0], width=150)
        movement_checkbox = CheckboxGroup(labels=["Show Movement Onsets"], active=[0], width=150)

        # Custom JS callbacks for each checkbox
        trials_callback = CustomJS(args={'spans': trials_end_spans, 'checkbox': trials_checkbox}, code="""
            for (let span of spans) {
                span.visible = checkbox.active.includes(0);
            }
        """)
        trials_checkbox.js_on_change('active', trials_callback)

        velocity_callback = CustomJS(args={'spans': velocity_peak_spans, 'checkbox': velocity_checkbox}, code="""
            for (let span of spans) {
                span.visible = checkbox.active.includes(0);
            }
        """)
        velocity_checkbox.js_on_change('active', velocity_callback)

        movement_callback = CustomJS(args={'spans': movement_onset_spans, 'checkbox': movement_checkbox}, code="""
            for (let span of spans) {
                span.visible = checkbox.active.includes(0);
            }
        """)
        movement_checkbox.js_on_change('active', movement_callback)

        # Layout the plot and the checkboxes
        checkbox_row = row(trials_checkbox, velocity_checkbox, movement_checkbox)
        layout = column(checkbox_row, p, p_v)

        self.output_bokeh_plot(layout, save_path=save_path, title=title, notebook=notebook, overwrite=overwrite)

        return layout


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
                    confusion_matrix[0, 0] += 1  # True Positive
                else:
                    confusion_matrix[0, 1] += 1  # False Negative
            else:  # Right turn
                if is_correct:
                    confusion_matrix[1, 1] += 1  # True Negative
                else:
                    confusion_matrix[1, 0] += 1  # False Positive

        if print_info:
            cm = confusion_matrix
            print("Confusion Matrix:")
            print(cm)
            print("\nInterpretation:")
            print(f"Correct Left Turns: {cm[0, 0]}")
            print(f"Incorrect Left Turns: {cm[0, 1]}")
            print(f"Incorrect Right Turns: {cm[1, 0]}")
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

    def plot_confusion_matrix(self, save_path=None, title=None, notebook=False, overwrite=False, backend='matplotlib'):
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
            ax.set_xlabel('Actual Turn')
            ax.set_ylabel('Predicted Turn')

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
                       x_axis_label="Actual Turn", y_axis_label="Should Turn",
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

            VirmenTank.output_bokeh_plot(p, save_path=save_path, title=title, notebook=notebook, overwrite=overwrite)

            return p
        else:
            raise ValueError("Unsupported backend. Use 'matplotlib' or 'bokeh'.")


if __name__ == "__main__":
    print("Starting CivisServer")
