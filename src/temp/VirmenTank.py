import pandas as pd
import numpy as np
import h5py
import json
from scipy.signal import find_peaks, savgol_filter


class VirmenTank:
    def __init__(self,
                 virmen_path,
                 vm_rate=20,
                 window_length=31,
                 polyorder=3,
                 height=8):

        self.virmen_path = virmen_path
        self.vm_rate = vm_rate

        self.virmen_trials, self.virmen_data = self.read_and_process_data(self.virmen_path, threshold=self.threshold)
        self.trials_end_indices = self.calculate_virmen_trials_end_indices()
        self.lick_raw, self.lick_raw_mask, self.lick = self.find_lick_data()
        self.pstcr_raw, self.pstcr = self.find_position_movement_rate()  # position change rate
        self.velocity = self.find_velocity()
        self.window_length = window_length  # Window length: number of coefficients (odd number)
        self.polyorder = polyorder  # Polynomial order
        self.smoothed_pstcr = self.butter_lowpass_filter(self.pstcr, 0.5, self.vm_rate)
        self.smoothed_pstcr[self.smoothed_pstcr <= 0] = 0
        self.smoothed_velocity = self.butter_lowpass_filter(self.velocity, 0.5, self.ci_rate)
        self.smoothed_velocity[self.smoothed_velocity <= 0] = 0
        self.dr, self.dr_raw = self.find_rotation()
        self.acceleration = self.find_acceleration()


        # for analysis usage:
        self.velocity_peak_indices = find_peaks(self.smoothed_velocity, height=height, distance=self.velocity_distance)[0]
        self.lick_edge_indices = np.where(np.diff(self.lick) > 0)[0]
        self.movement_onset_indices = self.find_movement_onset(self.smoothed_velocity, self.velocity_peak_indices)


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

    def find_acceleration(self):
        dt = 1/self.ci_rate
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


if __name__ == "__main__":
    print("Starting CivisServer")
