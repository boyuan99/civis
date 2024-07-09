import pandas as pd
import numpy as np
import h5py
import json
import tdt
from .VirmenTank import VirmenTank


class ElecTankTmp(VirmenTank):
    def __init__(self, elec_path, virmen_path, vm_rate=20, resample_fs=200, session_duration=30 * 60, notch_fs=[60],
                 notch_Q=30, threshold=[25, -25]):

        super().__init__(virmen_path, vm_rate, threshold, session_duration)
        self.elec_path = elec_path
        self.session_duration = session_duration
        self.fs = resample_fs
        self.tdt_data = tdt.read_block(self.elec_path)
        self.tdt_fs = self.tdt_data.streams.Wav1.fs
        self.tdt_signal = self.tdt_data.streams.Wav1.data
        self.tdt_signal_filtered = self.butter_lowpass_filter(self.tdt_signal, 100, self.tdt_fs)
        self.time_tdt = np.linspace(0, len(self.tdt_signal) - 1, len(self.tdt_signal)) / self.tdt_fs - \
                        self.tdt_data.epocs.PC0_.onset[0] - 0.02
        self.t = np.arange(0, session_duration, 1 / self.fs)
        self.signal_raw = self.resample_data()
        self.signal = self.notch_filter(notch_freqs=notch_fs, notch_Q=notch_Q)


        # Virmen variables remove in the formal version
        self.threshold = threshold
        self.vm_rate = vm_rate
        self.vm_t = np.arange(0, session_duration, 1 / self.vm_rate)
        self.virmenPath = self.find_virmenpath()
        self.virmen_trials, self.virmen_data = self.read_and_process_data(self.virmenPath, threshold=self.threshold)
        self.trials_end_indices = self.calculate_virmen_trials_end_indices()
        self.lick_raw, self.lick_raw_mask, self.lick = self.find_lick_data()
        self.pstcr_raw, self.pstcr = self.find_position_movement_rate()  # position change rate
        self.velocity = self.find_velocity()

        self.smoothed_pstcr = self.butter_lowpass_filter(self.pstcr, 0.5, self.vm_rate)
        self.smoothed_pstcr[self.smoothed_pstcr <= 0] = 0
        self.smoothed_velocity = self.butter_lowpass_filter(self.velocity, 0.5, self.vm_rate)
        self.smoothed_velocity[self.smoothed_velocity <= 0] = 0
        self.dr, self.dr_raw = self.find_rotation()
        self.acceleration = self.find_acceleration()

    def resample_data(self):
        indices = np.searchsorted(self.time_tdt, self.t, side='left')
        LFP_data_raw = self.tdt_signal_filtered[indices]

        return LFP_data_raw

    def notch_filter(self, notch_freqs, notch_Q):
        from scipy.signal import iirnotch, lfilter

        LFP_data = self.signal_raw

        for f0 in notch_freqs:
            b, a = iirnotch(w0=f0 / (self.fs / 2), Q=notch_Q)
            LFP_data = lfilter(b, a, LFP_data)

        return LFP_data

    def find_virmenpath(self):
        with open("config.json", "r") as f:
            config = json.load(f)

        [animal_id, date, _] = self.elec_path.split("/")[-1].split("-")
        file_name = animal_id + "_" + date[2:]+"2024.txt"

        virmenpath = config['VirmenFilePath'] + file_name

        return virmenpath

    @staticmethod
    def convert_t_freq_domain(signal, fs, decimation_factor=1):
        from scipy.signal import get_window
        from scipy.fft import fft

        window = get_window('hann', len(signal))
        signal_windowed = signal * window

        fft_values = fft(signal_windowed)
        power = np.abs(fft_values) ** 2

        # Adjust the power spectrum to compensate for windowing
        power /= np.sum(window ** 2)

        freq = np.linspace(0, fs, len(power), endpoint=False)

        # Decimate the frequency and power arrays
        freq = freq[::decimation_factor]
        power = power[::decimation_factor]

        return freq[:len(freq) // 2], power[:len(freq) // 2]

    @staticmethod
    def normalize_signal(signal):
        min_val = np.min(signal)
        max_val = np.max(signal)
        normalized_signal = (signal - min_val) / (max_val - min_val)
        return normalized_signal


    @staticmethod
    def butter_lowpass_filter(data, cutoff, fs, order=4):
        from scipy.signal import butter, filtfilt

        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        [b, a] = butter(order, normal_cutoff, btype='low', analog=False)
        y = filtfilt(b, a, data)
        return y

    @staticmethod
    def read_and_process_data(file_path, threshold=None, length=None):
        if threshold is None:
            threshold = [70.0, -70.0]

        data = pd.read_csv(file_path, sep=r'\s+|,', engine='python', header=None)

        if length is not None:
            data = data.iloc[:length, :]
        potential_names = ['x', 'y', 'face_angle', 'dx', 'dy', 'lick', 'time_stamp', 'maze_type']
        data.columns = potential_names[:data.shape[1]]

        # Identifying trials
        trials = []
        start = 0
        for i in range(len(data)):
            if data.iloc[i]['y'] >= threshold[0] or data.iloc[i]['y'] <= threshold[1]:
                trials.append(data[start:i + 1].to_dict(orient='list'))
                start = i + 1

        return [trials, data]

    def calculate_virmen_trials_end_indices(self):
        """
        Calculate the end indices for virmen trials based on the specified threshold.

        :return: List of indices where 'y' value exceeds the threshold.
        """
        indices = np.array(self.virmen_data.index[self.virmen_data['y'].abs() > self.threshold[0]].tolist())
        indices = indices[np.where(indices < self.vm_rate * self.session_duration)]
        return indices

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


if __name__ == "__main__":
    ep = ElecTankTmp('Z:/BoData/Bo_electrophy-240615-124754/372-240619-123139')
