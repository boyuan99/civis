import numpy as np
import pandas as pd
import h5py
from CalciumTank import CalciumTank
import matplotlib.pyplot as plt 
from scipy.signal import find_peaks, savgol_filter, butter, sosfilt, sosfiltfilt, medfilt, peak_widths
from bokeh.plotting import figure, output_file, save, reset_output
from bokeh.io import output_notebook,show
from bokeh.models import Span, VStrip
from bokeh.layouts import column,row


class ClaciumTankVelocity(CalciumTank):
    def noramlize_processed_C(self):
        # Initialize the normalized array
        self.normalized_C = np.empty(shape=(len(self.C),len(self.C[0])))

        # Normalize the array by each neuron
        for x in range(len(self.C)):
            temp_max = 0
            temp_min = self.C_raw[x][0]
            # find min signal of each neuron
            for y in range(len(self.C_raw[0])):
                if temp_min > self.C_raw[x][y]:
                    temp_min = self.C_raw[x][y]
                if temp_max < self.C_raw[x][y]:
                    temp_max = self.C_raw[x][y] 
            # normalize the amplitude according to each neuron
            for y in range(len(self.C[0])):
                self.normalized_C[x][y] = (self.C_raw[x][y] - temp_min) / (temp_max - temp_min)

    def calcium_signal_velocity(self):
        # C is [713,36000]
        # Initializing empty arrays
        peak_triggered= []
        rising_edge_triggered = []
        falling_edge_triggered = []

        iterator = self.normalized_C
        for i, C_pick in enumerate(iterator):
            C_filtered = medfilt(C_pick,kernel_size=17)

            # Normalize the array by each neuron
            temp_max = 0
            temp_min = C_filtered[0]
            for y in range(len(C_filtered)):
                if temp_min > C_filtered[y]:
                    temp_min = C_filtered[y]
                if temp_max < C_filtered[y]:
                    temp_max = C_filtered[y] 
            # normalize the amplitude according to each neuron
            for y in range(len(self.C[0])):
                C_filtered[y] = (C_filtered[y] - temp_min) / (temp_max - temp_min)

            # Peak finding (can be improved)
            peak_calciums_filtered, _ = find_peaks(C_filtered, height = np.average(C_filtered),prominence=0.3,
                                                   distance=200)
            peak_width=peak_widths(C_filtered, peak_calciums_filtered, rel_height=0.75)
            peak_triggered.append(peak_calciums_filtered)

            # Lowpass each peak
            sparced_and_passed = np.zeros(shape=(len(self.C[0])))
            for y in range(np.size(peak_width[0])):
                temp_wave = [-0.2]*(round(peak_width[3][y])-round(peak_width[2][y]) + 400)
                temp_wave[200:200+round(peak_width[3][y])-round(peak_width[2][y])] = C_filtered[round(peak_width[2][y]):round(peak_width[3][y])]
                nyq = 0.5 * 20
                low = 0.75
                sos = butter(5,low, btype='low', fs = 20, analog=False,output='sos')
                sos_product = sosfiltfilt(sos, temp_wave)
                sos_product[sos_product<0] = 0
                sparced_and_passed[max(round(peak_width[2][y])-200,0):round(peak_width[3][y])+200] = sos_product
                
            # LeftEdge finding
            temp_rising_edge = []
            for y in range(0,np.size(peak_width[0])):
                x = peak_triggered[i][y] - 1
                while (sparced_and_passed[x] >= 0.05):
                    if (sparced_and_passed[x + 1] >= (sparced_and_passed[x])) or (sparced_and_passed[x] > 3/4 * sparced_and_passed[peak_triggered[i][y]]):
                        x -= 1
                    else:
                       break
                temp_rising_edge.append(x)
            rising_edge_triggered.append(temp_rising_edge)

            # RightEdge finding (can be improved)
            temp_falling_edge = []
            for y in range(0,np.size(peak_width[0])):
                temp_falling_edge.append(peak_width[3][y])
            falling_edge_triggered.append(temp_falling_edge)

    
    # not used
    def band_pass_C(self):
        for x in range(len(self.C_raw)):
            self.band_passed_C = np.empty(shape=(len(self.C),len(self.C[x])))
            # Center the average of the signal to an amplitude of 0
            temp_avg = sum(self.normalized_C[x]) / len(self.normalized_C[x])
            for y in range(len(self.normalized_C[x])):
                self.band_passed_C[x][y] = self.normalized_C[x][y] - temp_avg

            # Band pass filter
            nyq = 0.5 * 20
            low = 0.10 / nyq
            high = 0.60 / nyq
            sos = butter(10,[low,high], btype='band', fs = 20, analog=False,output='sos')
            self.band_passed_C[x] = sosfiltfilt(sos, self.band_passed_C[x])

            # Scale amplitde
            temp_max = 0
            temp_min = 0
            for y in range(len(self.band_passed_C[0])):
                if temp_max < self.band_passed_C[x][y]:
                    temp_max = self.band_passed_C[x][y]
                if temp_min > self.band_passed_C[x][y]:
                    temp_min = self.band_passed_C[x][y]
            for y in range(len(self.C[0])):
                if 0 < self.band_passed_C[x][y]:
                    self.band_passed_C[x][y] = self.band_passed_C[x][y] / temp_max
                else:
                    self.band_passed_C[x][y] = -1 * self.band_passed_C[x][y] / temp_min


def main():
    calciumImage = ClaciumTankVelocity("/Users/edward41803/Documents/CIS_Processed_Data/ProcessedData/366_04062024/366_04062024_v7.mat")
    calciumImage.noramlize_processed_C()
    # Peak finding
    calciumImage.calcium_signal_velocity()
    # Vectorization of signals
    

    

if __name__ == "__main__":
    main()
