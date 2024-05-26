from src.CalciumTank import CalciumTank
import pandas as pd
import numpy as np


class TurnMazeTank(CalciumTank):
    def __init__(self, neuron_path,
                 ci_rate=20,
                 session_duration=30 * 60,
                 velocity_height=0.7,
                 velocity_distance=100,
                 threshold=[175.0, -175.0],
                 window_length=51,
                 polyorder=3,
                 height=8):
        # Pass all the necessary parameters to the parent class
        super().__init__(neuron_path, ci_rate, session_duration, velocity_height, velocity_distance, threshold,
                         window_length, polyorder, height)
        self.neuron_path = neuron_path

    @staticmethod
    def read_and_process_data(file_path, threshold=None):
        if threshold is None:
            threshold = [175.0, -175.0]
        data = pd.read_csv(file_path, sep=r'\s+|,', engine='python', header=None,
                           usecols=[0, 1, 2, 3, 4, 5, 6, 7],
                           names=['x', 'y', 'face_angle', 'dx', 'dy', 'lick', 'time_stamp', 'maze_type'])

        # Identifying trials
        trials = []
        start = 0
        for i in range(len(data)):
            if abs(data.iloc[i]['y']) + abs(data.iloc[i]['x']) >= threshold[0]:
                trials.append(data[start:i + 1].to_dict(orient='list'))
                start = i + 1

        return [trials, data]

    def calculate_virmen_trials_end_indices(self):
        """
        Calculate the end indices for virmen trials based on the specified threshold.

        :return: List of indices where 'y' value exceeds the threshold.
        """
        indices = np.array(self.virmen_data.index[abs(self.virmen_data['y']) + abs(self.virmen_data['x']) >=
                                                  self.threshold[0]].tolist())
        indices = indices[np.where(indices < self.ci_rate * self.session_duration)]
        return indices


if __name__ == '__main__':
    print("Starting CivisServer")
    # session_name = "366_04292024"
    # ci = TurnMazeTank("D:/Calcium Image Processing/ProcessedData/" + session_name + "/" + session_name + "_v7.mat")
