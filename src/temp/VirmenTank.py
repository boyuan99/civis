import pandas as pd
import numpy as np
import h5py
import json
from scipy.signal import find_peaks, savgol_filter


class VirmenTank:
    def __init__(self,
                 virmen_path,
                 threshold,
                 maze_type,
                 virmen_data_length=None,
                 vm_rate=20,
                 velocity_height=0.7,
                 velocity_distance=100,
                 session_duration=30 * 60,
                 window_length=31,
                 polyorder=3,
                 height=8):

        self.virmen_path = virmen_path
        self.vm_rate = vm_rate
        self.session_duration = session_duration
        self.threshold = threshold
        self.window_length = window_length  # Window length: number of coefficients (odd number)
        self.polyorder = polyorder  # Polynomial order
        self.velocity_height = velocity_height
        self.velocity_distance = velocity_distance
        self.maze_type = maze_type
        self.t = np.arange(0, self.session_duration, 1 / self.vm_rate)

        self.virmen_trials, self.virmen_data = self.read_and_process_data(self.virmen_path, threshold=self.threshold,
                                                                          length=virmen_data_length, maze_type=maze_type)
        self.trials_end_indices = self.calculate_virmen_trials_end_indices(maze_type)
        self.lick_raw, self.lick_raw_mask, self.lick = self.find_lick_data()
        self.pstcr_raw, self.pstcr = self.find_position_movement_rate()  # position change rate
        self.velocity = self.find_velocity()
        self.smoothed_pstcr = self.butter_lowpass_filter(self.pstcr, 0.5, self.vm_rate)
        self.smoothed_pstcr[self.smoothed_pstcr <= 0] = 0
        self.smoothed_velocity = self.butter_lowpass_filter(self.velocity, 0.5, self.vm_rate)
        self.smoothed_velocity[self.smoothed_velocity <= 0] = 0
        self.dr, self.dr_raw = self.find_rotation()
        self.acceleration = self.find_acceleration()

        # for analysis usage:
        self.velocity_peak_indices = find_peaks(self.smoothed_velocity, height=height, distance=self.velocity_distance)[0]
        self.lick_edge_indices = np.where(np.diff(self.lick) > 0)[0]
        self.movement_onset_indices = self.find_movement_onset(self.smoothed_velocity, self.velocity_peak_indices)


    @staticmethod
    def read_and_process_data(file_path, threshold=None, length=None, maze_type=None):

        if threshold is None:
            threshold = [70.0, -70.0]

        if maze_type is None:
            maze_type = 'straight25'

        data = pd.read_csv(file_path, sep=r'\s+|,', engine='python', header=None)

        if length is not None:
            data = data.iloc[:length, :]
        potential_names = ['x', 'y', 'face_angle', 'dx', 'dy', 'lick', 'time_stamp', 'maze_type']
        data.columns = potential_names[:data.shape[1]]

        # Identifying trials
        trials = []
        start = 0

        if maze_type == 'straight25':
            for i in range(len(data)):
                if data.iloc[i]['y'] >= 25 or data.iloc[i]['y'] <= -25:
                    trials.append(data[start:i + 1].to_dict(orient='list'))
                    start = i + 1

        elif maze_type == 'straight50':
            for i in range(len(data)):
                if data.iloc[i]['y'] >= 50 or data.iloc[i]['y'] <= -50:
                    trials.append(data[start:i + 1].to_dict(orient='list'))
                    start = i + 1

        elif maze_type == 'TurnV0':
            for i in range(len(data)):
                if abs(data.iloc[i]['y']) + abs(data.iloc[i]['x']) >= 175:
                    trials.append(data[start:i + 1].to_dict(orient='list'))
                    start = i + 1

        return [trials, data]

    def calculate_virmen_trials_end_indices(self, maze_type=None):
        """
        Calculate the end indices for virmen trials based on the specified threshold.

        :return: List of indices where 'y' value exceeds the threshold.
        """
        if maze_type == 'straight25':
            indices = np.array(self.virmen_data.index[self.virmen_data['y'].abs() > 25].tolist())
            indices = indices[np.where(indices < self.vm_rate * self.session_duration)]

        elif maze_type == 'straight50':
            indices = np.array(self.virmen_data.index[self.virmen_data['y'].abs() > 50].tolist())
            indices = indices[np.where(indices < self.vm_rate * self.session_duration)]

        elif maze_type == 'TurnV0':
            indices = np.array(self.virmen_data.index[abs(self.virmen_data['y']) +
                                                      abs(self.virmen_data['x']) >= 175].tolist())
            indices = indices[np.where(indices < self.vm_rate * self.session_duration)]

        else:
            indices = []

        return indices

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
    def shift_signal(fluorescence):
        shifted = np.zeros_like(fluorescence)
        for i, signal in enumerate(fluorescence):
            shifted[i] = signal - np.mean(signal)

        return shifted

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
        dt = 1/self.vm_rate
        dv = np.diff(self.smoothed_velocity)

        acceleration = dv / dt
        acceleration = np.pad(acceleration, (0, len(self.t) - len(dv)), 'constant', constant_values=(0, ))

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
        from bokeh.models import Span, HoverTool
        from bokeh.layouts import column

        p = figure(width=800, height=400, y_range=[0, 1.2], active_drag='pan', active_scroll='wheel_zoom', title='Lick')
        p.line(self.t, self.lick_raw, line_color='navy', legend_label='raw', line_width=1)
        p.line(self.t, self.lick_raw_mask, line_color='palevioletred', legend_label='mask', line_width=2)
        p.line(self.t, self.lick, line_color='gold', legend_label='filtered', line_width=2)

        for idx in self.trials_end_indices:
            vline = Span(location=idx / self.vm_rate, dimension='height', line_color='green', line_width=2)
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
            vline = Span(location=idx / self.vm_rate, dimension='height', line_color='green', line_width=2)
            p_v.add_layout(vline)

        layout = column(p, p_v)

        self.output_bokeh_plot(layout, save_path=save_path, title=title, notebook=notebook, overwrite=overwrite)

        return layout


if __name__ == "__main__":
    print("Starting CivisServer")
