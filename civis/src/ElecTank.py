import numpy as np
import tdt
import os
from .VirmenTank import VirmenTank


class ElecTank(VirmenTank):
    def __init__(self,
                 session_name,
                 elec_path=None,
                 virmen_path=None,
                 maze_type=None,
                 vm_rate=20,
                 resample_fs=200,
                 session_duration=30 * 60,
                 virmen_data_length=None,
                 velocity_height=0.7,
                 velocity_distance=100,
                 window_length=51,
                 polyorder=3,
                 height=6,
                 notch_fs=[60],
                 notch_Q=30,
                 threshold=None):

        self.session_name = session_name
        self.config = self.load_config()

        elec_path = os.path.join(self.config['ElecPath'], session_name) if elec_path is None else elec_path
        virmen_path = os.path.join(self.config['VirmenFilePath'],
                                   f"{session_name}.txt") if virmen_path is None else virmen_path

        super().__init__(
            session_name=session_name,
            virmen_path=virmen_path,
            threshold=threshold,
            maze_type=maze_type,
            virmen_data_length=virmen_data_length,
            vm_rate=vm_rate,
            velocity_height=velocity_height,
            velocity_distance=velocity_distance,
            session_duration=session_duration,
            window_length=window_length,
            polyorder=polyorder,
            height=height)

        self.elec_path = elec_path
        self.session_duration = session_duration
        self.fs = resample_fs
        self.tdt_data = tdt.read_block(self.elec_path)
        self.tdt_fs = self.tdt_data.streams.Wav1.fs
        self.tdt_signal = self.tdt_data.streams.Wav1.data
        self.tdt_signal_filtered = self.butter_lowpass_filter(self.tdt_signal, 100, self.tdt_fs)
        self.tdt_t = (np.linspace(0, len(self.tdt_signal) - 1, len(self.tdt_signal)) / self.tdt_fs -
                      self.tdt_data.epocs.PC0_.onset[0] - 0.02)
        self.elc_t = np.arange(0, session_duration, 1 / self.fs)
        self.signal_raw = self.resample_data()
        self.signal = self.notch_filter(notch_freqs=notch_fs, notch_Q=notch_Q)

    def resample_data(self):
        indices = np.searchsorted(self.tdt_t, self.elc_t, side='left')
        LFP_data_raw = self.tdt_signal_filtered[indices]

        return LFP_data_raw

    def notch_filter(self, notch_freqs, notch_Q):
        from scipy.signal import iirnotch, lfilter

        LFP_data = self.signal_raw

        for f0 in notch_freqs:
            b, a = iirnotch(w0=f0 / (self.fs / 2), Q=notch_Q)
            LFP_data = lfilter(b, a, LFP_data)

        return LFP_data

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

    def convert_t_freq_domain_welch(self, window=5):
        """
        Use mne to convert signal from time domain to frequency domain
        Args:
            window: how many seconds to use in multitaper

        Returns: [psd_welch, freq_welch]

        """
        import mne
        [psd_welch, freq_welch] = mne.time_frequency.psd_array_welch(self.signal, self.fs, fmin=1, fmax=100,
                                                                     n_fft=self.fs * window)

        return psd_welch, freq_welch

    def plot_resample_demo(self, start=70, stop=72,
                           notebook=False, save_path=None, overwrite=False):
        """
        Plot a short demo showing the resampling result

        :param start: Start index
        :param stop: Stop index
        :param notebook: Flag to indicate if the plot is for a Jupyter notebook.
        :param save_path: Path to save the plot as an HTML file.
        :param overwrite: Flag to overwrite the file or not
        :return: plotting element

        """
        from bokeh.plotting import figure

        sample_indices = np.where((self.tdt_t >= start) & (self.tdt_t <= stop))

        p = figure(width=800, height=300, active_scroll='wheel_zoom', title="Resampling Demo")
        p.line(self.tdt_t[sample_indices], self.tdt_signal[sample_indices], color='blue', alpha=0.7, legend_label="raw")
        p.line(self.tdt_t[sample_indices], self.tdt_signal_filtered[sample_indices], color='green', alpha=0.7,
               legend_label="band tdt")
        p.line(self.elc_t[start * self.fs:stop * self.fs], self.signal_raw[start * self.fs:stop * self.fs], color='red',
               alpha=0.7,
               legend_label="resample")
        p.line(self.elc_t[start * self.fs:stop * self.fs], self.signal[start * self.fs:stop * self.fs], color='orange',
               alpha=0.7,
               legend_label="resample filtered")

        p.legend.click_policy = 'hide'

        self.output_bokeh_plot(p, save_path=save_path, title=str(p.title.text), notebook=notebook, overwrite=overwrite)

        return p

    def plot_raw_vs_notch(self, start=10, stop=14, save_path=None, notebook=False, overwrite=False):
        """
        Plot raw resampled signal with notch filtered resampled signal

        :param start: Start index
        :param stop: Stop index
        :param notebook: Flag to indicate if the plot is for a Jupyter notebook.
        :param save_path: Path to save the plot as an HTML file.
        :param overwrite: Flag to overwrite the file or not
        :return: bokeh.plotting element

        """

        from bokeh.plotting import figure

        p = figure(width=800, height=300, x_range=(start, stop),
                   active_scroll='wheel_zoom', title="Raw vs Notch")
        p.line(self.elc_t, self.signal_raw, color='blue', alpha=0.7, line_width=2, legend_label="raw")
        p.line(self.elc_t, self.signal, color='red', alpha=0.7, line_width=2, legend_label="filtered")

        p.legend.click_policy = 'hide'

        self.output_bokeh_plot(p, save_path=save_path, title=str(p.title.text), notebook=notebook, overwrite=overwrite)

        return p

    def plot_velocity_with_signal(self, save_path=None, notebook=False, overwrite=False):
        """
        Plot velocity from virmen with resampled, filtered signal
        Args:
            save_path: (str) path to save
            notebook: (bool) whether to show in notebook or not
            overwrite: (bool) whether to overwrite existing file

        Returns:
            bokeh.plotting element

        """

        from bokeh.plotting import figure

        p = figure(width=800, height=300, active_scroll='wheel_zoom', title="velocity with signal (normalized)")
        p.line(self.elc_t, self.normalize_signal(self.signal) - 0.7, color='blue', alpha=0.7, legend_label="signal")
        p.line(self.t, self.lick, color='red', alpha=0.7, legend_label="lick")
        p.line(self.t, self.normalize_signal(self.smoothed_velocity), color='green', alpha=0.7,
               legend_label="sm_velocity")
        p.line(self.t, self.normalize_signal(self.velocity), color='purple', alpha=0.7, legend_label="velocity")

        p.legend.click_policy = 'hide'

        self.output_bokeh_plot(p, save_path=save_path, title=str(p.title.text), notebook=notebook, overwrite=overwrite)

        return p

    def generate_time_frequency_spectrogram(self, freq_start=1, freq_stop=100, freq_step=0.1):

        import mne

        freqs = np.arange(start=freq_start, stop=freq_stop, step=freq_step)
        tfr = mne.time_frequency.tfr_array_morlet(self.signal.reshape(1, 1, -1), self.fs, freqs, n_cycles=7,
                                                  zero_mean=True,
                                                  output='power')

        return freqs, tfr

    def plot_time_frequency_spectrogram(self, freqs=None, tfr=None,
                                        freq_start=1, freq_stop=100, freq_step=0.1,
                                        time_start=0, time_end=None,
                                        time_dec_factor=20, freq_dec_factor=10,
                                        palette="Turbo256",
                                        save_path=None, notebook=False, overwrite=False):
        import mne
        from bokeh.plotting import figure
        from bokeh.models import LinearColorMapper, ColorBar
        from bokeh.layouts import column

        if time_end is None:
            time_end = self.session_duration

        time_elc = self.elc_t[time_start * self.fs: time_end * self.fs]
        time_vm = self.t[time_start * self.vm_rate: time_end * self.vm_rate]
        signal = self.signal[time_start * self.fs: time_end * self.fs]
        velocity = self.smoothed_velocity[time_start * self.vm_rate: time_end * self.vm_rate]

        if (freqs is None) & (tfr is None):
            freqs = np.arange(start=freq_start, stop=freq_stop, step=freq_step)
            tfr = mne.time_frequency.tfr_array_morlet(signal.reshape(1, 1, -1), self.fs, freqs, n_cycles=7,
                                                      zero_mean=True,
                                                      output='power')

        # Decimate data for plotting
        time_dec = time_elc[::time_dec_factor]
        freqs_dec = freqs[::freq_dec_factor]
        tfr_dec = tfr[:, :, ::freq_dec_factor, ::time_dec_factor]

        power = 10 * np.log10(tfr_dec.squeeze())

        # Calculate dimensions for the image
        dw = time_dec[-1] - time_dec[0]  # Width of the image
        dh = freqs_dec[-1] - freqs_dec[0]

        # Create the first plot (Time-Frequency Representation)
        p1 = figure(width=1000, height=350, title="Time-Frequency Spectrogram",
                    x_axis_label="Time (s)", y_axis_label="Frequency (Hz)",
                    x_range=(time_dec[0], time_dec[-1]), y_range=(freqs_dec[0], 40),
                    active_scroll="wheel_zoom")
        color_mapper = LinearColorMapper(palette=palette, low=np.percentile(power.ravel(), 1), high=np.percentile(power.ravel(), 99))
        p1.image(image=[power], x=time_dec[0], y=freqs_dec[0], dw=dw, dh=dh,
                 color_mapper=color_mapper)

        # Add a color bar
        color_bar = ColorBar(color_mapper=color_mapper, label_standoff=12, border_line_color=None, location=(0, 0))
        p1.add_layout(color_bar, 'right')

        # Create the second plot (Normalized Velocity)
        p2 = figure(width=p1.width, height=350, title="Normalized Velocity and signal Over Time",
                    x_axis_label="Time (s)", y_axis_label="Velocity (normalized)", x_range=p1.x_range,
                    tools=p1.tools, toolbar_location=None)
        p2.line(time_vm, self.normalize_signal(velocity), legend_label="Normalized Velocity", color='blue',
                line_width=2)
        p2.line(time_elc, self.normalize_signal(signal), color='red', alpha=0.7, legend_label="signal")
        p2.legend.click_policy = 'hide'

        layout = column([p1, p2])

        self.output_bokeh_plot(layout, save_path=save_path, title=str(p1.title.text), notebook=notebook,
                               overwrite=overwrite)

        return layout


if __name__ == "__main__":
    ep = ElecTank('Z:/BoData/Bo_electrophy-240615-124754/372-240619-123139')
