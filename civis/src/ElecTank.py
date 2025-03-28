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
                                        freq_start=1, freq_stop=100, freq_step=0.5,
                                        time_start=0, time_end=None,
                                        time_dec_factor=20, freq_dec_factor=1,
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
        p2.line(time_vm, self.normalize_signal(velocity), legend_label="Normalized Velocity", color='skyblue',
                line_width=1)
        p2.line(time_elc, self.normalize_signal(signal), color='salmon', alpha=0.7, legend_label="signal")
        p2.line(time_vm, self.lick_raw, color='lightgreen', alpha=0.7, legend_label="lick")
        p2.legend.click_policy = 'hide'

        layout = column([p1, p2])

        self.output_bokeh_plot(layout, save_path=save_path, title=str(p1.title.text), notebook=notebook,
                               overwrite=overwrite)

        return layout

    def plot_time_frequency_with_beta_analysis(self, freqs=None, tfr=None,
                                               freq_start=1, freq_stop=100, freq_step=0.5,
                                               time_start=0, time_end=None,
                                               time_dec_factor=20, freq_dec_factor=1,
                                               beta_min=12, beta_max=30, threshold_std=1.5,
                                               palette="Turbo256",
                                               save_path=None, notebook=False, overwrite=False):
        import mne
        from bokeh.plotting import figure
        from bokeh.models import LinearColorMapper, ColorBar, BoxAnnotation, Span, Label
        from bokeh.layouts import gridplot

        if time_end is None:
            time_end = self.session_duration
        time_elc = self.elc_t[time_start * self.fs: time_end * self.fs]
        time_vm = self.t[time_start * self.vm_rate: time_end * self.vm_rate]
        signal = self.signal[time_start * self.fs: time_end * self.fs]
        velocity = self.smoothed_velocity[time_start * self.vm_rate: time_end * self.vm_rate]

        if (freqs is None) or (tfr is None):
            freqs = np.arange(start=freq_start, stop=freq_stop, step=freq_step)
            tfr = mne.time_frequency.tfr_array_morlet(signal.reshape(1, 1, -1),
                                                      self.fs,
                                                      freqs,
                                                      n_cycles=7,
                                                      zero_mean=True,
                                                      output='power')

        time_dec = time_elc[::time_dec_factor]
        freqs_dec = freqs[::freq_dec_factor]
        tfr_dec = tfr[:, :, ::freq_dec_factor, ::time_dec_factor]

        power = 10 * np.log10(tfr_dec.squeeze())

        dw = time_dec[-1] - time_dec[0]
        dh = freqs_dec[-1] - freqs_dec[0]

        p1 = figure(
            title="Time-Frequency Spectrogram",
            x_axis_label='Time (s)',
            y_axis_label='Frequency (Hz)',
            width=1000,
            height=350,
            x_range=(time_dec[0], time_dec[-1]),
            y_range=(freqs_dec[0], 40),
            active_scroll="wheel_zoom",
            tools="pan,wheel_zoom,box_zoom,reset,save"
        )

        color_mapper = LinearColorMapper(
            palette=palette,
            low=np.percentile(power.ravel(), 1),
            high=np.percentile(power.ravel(), 99)
        )

        p1.image(
            image=[power],
            x=time_dec[0],
            y=freqs_dec[0],
            dw=dw,
            dh=dh,
            color_mapper=color_mapper
        )

        color_bar = ColorBar(
            color_mapper=color_mapper,
            label_standoff=12,
            border_line_color=None,
            location=(0, 0)
        )
        p1.add_layout(color_bar, 'right')

        beta_lower_line = Span(
            location=beta_min,
            dimension='width',
            line_color='white',
            line_dash='dashed',
            line_width=2
        )
        p1.add_layout(beta_lower_line)

        beta_upper_line = Span(
            location=beta_max,
            dimension='width',
            line_color='white',
            line_dash='dashed',
            line_width=2
        )
        p1.add_layout(beta_upper_line)

        beta_label = Label(
            x=time_dec[0],
            y=(beta_min + beta_max) / 2,
            text=f'Beta Band ({beta_min}-{beta_max} Hz)',
            text_font_size="10pt",
            text_color="white",
            x_offset=10
        )
        p1.add_layout(beta_label)

        beta_indices = np.where((freqs_dec >= beta_min) & (freqs_dec <= beta_max))[0]
        beta_power = np.mean(power[beta_indices, :], axis=0)  # Shape: (n_times_ds,)

        mean_beta_power = np.mean(beta_power)
        std_beta_power = np.std(beta_power)
        threshold = mean_beta_power + threshold_std * std_beta_power

        significant_increases = beta_power > threshold

        def contiguous_regions(condition):
            """Find contiguous True regions of the boolean array "condition". Returns a list of (start, end) indices."""
            d = np.diff(condition)
            idx, = d.nonzero()

            idx += 1

            if condition[0]:
                start = [0]
            else:
                start = []
            start += idx.tolist()

            if condition[-1]:
                end = [len(condition)]
            else:
                end = []
            end += idx.tolist()

            regions = list(zip(start, end))
            return regions

        regions = contiguous_regions(significant_increases)

        p2 = figure(
            title="Beta Band Power with Significant Increases",
            x_axis_label='Time (s)',
            y_axis_label='Beta Power (dB)',
            tools="pan,wheel_zoom,box_zoom,reset,save",
            width=1000,
            height=200,
            x_range=p1.x_range  # Link x-axis with spectrogram
        )

        p2.line(time_dec, beta_power, line_width=2, line_color='navy', legend_label="Beta Power")

        for start, end in regions:
            p2.add_layout(BoxAnnotation(
                left=time_dec[start],
                right=time_dec[end - 1],
                fill_alpha=0.2,
                fill_color='red'
            ))

        threshold_line = Span(
            location=threshold,
            dimension='width',
            line_color='green',
            line_dash='dashed',
            line_width=1
        )
        p2.add_layout(threshold_line)

        threshold_label = Label(
            x=time_dec[0],
            y=threshold,
            text=f'Threshold: {threshold:.2f} dB',
            text_font_size="10pt",
            text_color="green",
            y_offset=5
        )
        p2.add_layout(threshold_label)

        p2.legend.location = "top_left"
        p2.legend.click_policy = 'hide'

        p3 = figure(
            title="Normalized Signals",
            x_axis_label="Time (s)",
            y_axis_label="Amplitude",
            width=1000,
            height=300,
            x_range=p1.x_range,  # Link x-axis with spectrogram
            tools=p1.tools
        )

        p3.line(time_elc, self.normalize_signal(signal), color="lightblue", legend_label="Signal")
        p3.line(time_vm, self.normalize_signal(velocity), color="orange", legend_label="Velocity", alpha=0.7)

        try:
            p3.line(time_vm, self.lick_raw[time_start * self.vm_rate: time_end * self.vm_rate],
                    color="MediumSeaGreen", legend_label="Lick", alpha=0.7)
        except (AttributeError, IndexError):
            pass

        p3.legend.location = "top_left"
        p3.legend.click_policy = 'hide'

        layout = gridplot([[p1], [p2], [p3]])

        self.output_bokeh_plot(layout, save_path=save_path, title="Time-Frequency Analysis with Beta Band",
                               notebook=notebook, overwrite=overwrite)

        return layout

    def plot_event_centered_spectrogram(self, event_indices, window_sec=5,
                                        analysis_padding=5,
                                        freq_start=1, freq_stop=100, freq_step=0.5,
                                        time_dec_factor=5, freq_dec_factor=1,
                                        palette="Turbo256",
                                        title="Event-centered Spectrogram", notebook=False,
                                        save_path=None, overwrite=False):
        """
        Plot spectrogram centered around specified events using Morlet wavelets

        Parameters:
        -----------
        event_indices : list or array
            Indices of events to center spectrograms around
        window_sec : float
            Half-width of the window in seconds (total window will be 2*window_sec)
        analysis_padding : float
            Additional padding in seconds for analysis to reduce edge artifacts
        freq_start : float
            Start frequency for analysis (Hz)
        freq_stop : float
            Stop frequency for analysis (Hz)
        freq_step : float
            Step size for frequency bins
        time_dec_factor : int
            Time decimation factor for visualization
        freq_dec_factor : int
            Frequency decimation factor for visualization
        palette : str
            Bokeh color palette for spectrogram
        title : str
            Title for the plot
        notebook : bool
            Whether to display in notebook
        save_path : str
            Path to save the plot
        overwrite : bool
            Whether to overwrite existing file

        Returns:
        --------
        layout : bokeh layout object
            The created plot layout
        """
        import numpy as np
        import mne
        from bokeh.plotting import figure
        from bokeh.models import ColorBar, LinearColorMapper
        from bokeh.layouts import column

        # Create extended analysis window with padding to reduce edge artifacts
        analysis_window_sec = window_sec + analysis_padding
        
        # Convert window seconds to samples
        window_samples = int(window_sec * self.fs)
        analysis_window_samples = int(analysis_window_sec * self.fs) 
        window_samples_vm = int(window_sec * self.vm_rate)

        # Ensure freq_stop is below Nyquist frequency (half the sampling rate)
        nyquist_freq = self.fs / 2
        if freq_stop >= nyquist_freq:
            freq_stop = nyquist_freq - freq_step
            print(
                f"Warning: Maximum frequency adjusted to {freq_stop} Hz to stay below Nyquist frequency ({nyquist_freq} Hz)")

        # Limit low frequencies if window is small
        min_freq = freq_start
        if analysis_window_sec < 8 and min_freq < 2:
            min_freq = 2
            print(f"Warning: Minimum frequency adjusted to {min_freq} Hz due to limited window size")

        freqs = np.arange(min_freq, freq_stop, freq_step)

        # Prepare to store spectrograms for all events
        all_tfrs = []
        valid_events = []

        # Process each event
        for event_idx in event_indices:
            # Convert to sample index in the resampled signal
            event_sample = int(event_idx / self.vm_rate * self.fs)

            # Check if we have enough data around this event with the extended window
            if (event_sample - analysis_window_samples >= 0 and
                    event_sample + analysis_window_samples < len(self.signal)):

                # Extract signal around event (with extended window)
                signal_segment = self.signal[event_sample - analysis_window_samples:
                                             event_sample + analysis_window_samples]

                # Reshape for MNE (using Morlet wavelets)
                signal_segment = signal_segment.reshape(1, 1, -1)

                # Calculate time-frequency representation with Morlet wavelets
                # For low frequencies, use fewer cycles to avoid wavelet length issues
                # Higher frequencies can use more cycles for better frequency resolution
                n_cycles = freqs / 2  # Adaptive n_cycles: fewer for low frequencies
                n_cycles = np.clip(n_cycles, 3, 20)  # Min 3, max 20 cycles

                # Try with adaptive cycles first
                try:
                    tfr = mne.time_frequency.tfr_array_morlet(
                        signal_segment,
                        sfreq=self.fs,
                        freqs=freqs,
                        n_cycles=n_cycles,
                        zero_mean=True,
                        output='power'
                    )
                except ValueError:
                    # If still too long, try with minimum cycles
                    print(f"Warning: Using minimum cycles (3) for analysis. Some frequency resolution may be lost.")
                    tfr = mne.time_frequency.tfr_array_morlet(
                        signal_segment,
                        sfreq=self.fs,
                        freqs=freqs,
                        n_cycles=3,
                        zero_mean=True,
                        output='power'
                    )

                # Convert to dB
                tfr_db = 10 * np.log10(tfr.squeeze())

                # Store this TFR
                all_tfrs.append(tfr_db)
                valid_events.append(event_idx)

        if not all_tfrs:
            print("No valid events found within signal bounds.")
            return None

        # Average all TFRs
        avg_tfr = np.mean(all_tfrs, axis=0)
        
        # Determine how many samples to crop on each side to get from analysis window to display window
        padding_samples = int(analysis_padding * self.fs)
        
        # Get the central portion of the time-frequency data (cropping out the padding)
        cropped_start_idx = padding_samples
        cropped_end_idx = avg_tfr.shape[1] - padding_samples
        
        # Crop the average TFR to the desired display window
        avg_tfr_cropped = avg_tfr[:, cropped_start_idx:cropped_end_idx]

        # Decimate data for plotting if needed
        freqs_dec = freqs[::freq_dec_factor]

        # Decimate in time if needed
        if time_dec_factor > 1:
            avg_tfr_dec = avg_tfr_cropped[:, ::time_dec_factor]
        else:
            avg_tfr_dec = avg_tfr_cropped

        # Create time axis centered around events (in seconds)
        times = np.linspace(-window_sec, window_sec, avg_tfr_cropped.shape[1])
        times_dec = times[::time_dec_factor] if time_dec_factor > 1 else times

        # Normalize each frequency row separately
        normalized_tfr = np.zeros_like(avg_tfr_dec)
        for i in range(avg_tfr_dec.shape[0]):
            row = avg_tfr_dec[i, :]
            row_min, row_max = np.min(row), np.max(row)
            if row_max > row_min:  # Avoid division by zero
                normalized_tfr[i, :] = (row - row_min) / (row_max - row_min)
            else:
                normalized_tfr[i, :] = 0.5  # Default value if row has no variation

        # Calculate dimensions for the image
        dw = times_dec[-1] - times_dec[0]
        dh = freqs_dec[-1] - freqs_dec[0]

        # Create Bokeh figure for spectrogram
        p1 = figure(
            title=f"{title} (n={len(valid_events)}) - Row-Normalized",
            x_axis_label='Time relative to event (s)',
            y_axis_label='Frequency (Hz)',
            tools="pan,wheel_zoom,box_zoom,reset,save",
            width=1000, height=350,
            x_range=(-window_sec, window_sec),
            y_range=(min_freq, min(40, freq_stop))  # Limit to 40Hz by default for better viewing
        )

        # Define the color mapper - fixed range from 0 to 1 for normalized data
        color_mapper = LinearColorMapper(
            palette=palette,
            low=0,
            high=1
        )

        # Add the normalized spectrogram image
        p1.image(
            image=[normalized_tfr],
            x=times_dec[0],
            y=freqs_dec[0],
            dw=dw,
            dh=dh,
            color_mapper=color_mapper
        )

        # Add a color bar
        color_bar = ColorBar(
            color_mapper=color_mapper,
            label_standoff=12,
            border_line_color=None,
            location=(0, 0),
            title="Normalized Power"
        )
        p1.add_layout(color_bar, 'right')

        # Calculate average LFP and velocity around events
        avg_signal = np.zeros(2 * window_samples)
        avg_velocity = np.zeros(2 * window_samples_vm)
        avg_lick = np.zeros(2 * window_samples_vm)
        valid_velocity_events = 0
        valid_lick_events = 0 

        for event_idx in valid_events:
            event_sample = int(event_idx / self.vm_rate * self.fs)

            # LFP segment
            if (event_sample - window_samples >= 0 and
                    event_sample + window_samples < len(self.signal)):
                signal_segment = self.signal[event_sample - window_samples:
                                             event_sample + window_samples]
                avg_signal += signal_segment

            # Velocity segment
            if (event_idx - window_samples_vm >= 0 and
                    event_idx + window_samples_vm < len(self.smoothed_velocity)):
                velocity_segment = self.smoothed_velocity[event_idx - window_samples_vm:
                                                          event_idx + window_samples_vm]
                avg_velocity += velocity_segment
                valid_velocity_events += 1
                
            # Lick segment
            if (event_idx - window_samples_vm >= 0 and
                    event_idx + window_samples_vm < len(self.lick_raw)):
                lick_segment = self.lick[event_idx - window_samples_vm:
                                            event_idx + window_samples_vm]
                avg_lick += lick_segment
                valid_lick_events += 1

        # Average and normalize
        avg_signal /= len(valid_events)
        avg_signal_norm = (avg_signal - np.min(avg_signal)) / (np.max(avg_signal) - np.min(avg_signal))

        # Create a second figure for signals
        p2 = figure(
            title="Normalized Signals",
            x_axis_label="Time relative to event (s)",
            y_axis_label="Amplitude",
            width=1000,
            height=300,
            x_range=p1.x_range,  # Link x-axis with spectrogram
            tools=p1.tools
        )

        # Plot the average LFP
        p2.line(
            np.linspace(-window_sec, window_sec, len(avg_signal)),
            avg_signal_norm,
            line_width=2,
            color="lightblue",
            legend_label="Avg. Signal"
        )

        # Plot velocity if available
        if valid_velocity_events > 0:
            avg_velocity /= valid_velocity_events
            avg_velocity_norm = (avg_velocity - np.min(avg_velocity)) / (np.max(avg_velocity) - np.min(avg_velocity))

            # Create velocity time axis
            velocity_times = np.linspace(-window_sec, window_sec, len(avg_velocity))

            # Plot velocity
            p2.line(
                velocity_times,
                avg_velocity_norm,
                line_width=2,
                color="orange",
                legend_label="Avg. Velocity",
                alpha=0.7
            )
            
        # Plot lick if available
        if valid_lick_events > 0:
            avg_lick /= valid_lick_events
            if np.max(avg_lick) > np.min(avg_lick):
                avg_lick_norm = (avg_lick - np.min(avg_lick)) / (np.max(avg_lick) - np.min(avg_lick))
            else:
                avg_lick_norm = avg_lick
                
            p2.line(
                velocity_times,
                avg_lick_norm,
                line_width=2,
                color="green",
                legend_label="Avg. Lick",
                alpha=0.7
            )

        # Add a vertical line at event time (t=0)
        p2.line([0, 0], [0, 1], line_width=2, color="red", line_dash="dashed")
        p2.legend.location = "top_left"
        p2.legend.click_policy = 'hide'

        # Combine the plots
        layout = column(p1, p2)

        # Use the class's output method
        self.output_bokeh_plot(layout, save_path=save_path, title=title, notebook=notebook, overwrite=overwrite)

        return layout

    def calculate_band_powers(self, event_indices, window_sec=5, notebook=False, title=None,
                              save_path=None, overwrite=False):
        """
        Calculate and visualize power in different frequency bands around events

        Parameters:
        -----------
        event_indices : list or array
            Indices of events to center band power analysis around
        window_sec : float
            Half-width of the window in seconds
        notebook : bool
            Whether to display in notebook
        title : str
            Title for the plot
        save_path : str
            Path to save the plot
        overwrite : bool
            Whether to overwrite existing file

        Returns:
        --------
        layout : bokeh layout object
            The created plot
        """
        import numpy as np
        from bokeh.plotting import figure
        from bokeh.models import Span
        from scipy.signal import welch

        # Define frequency bands
        bands = {
            'delta': (1, 4),
            'theta': (4, 8),
            'alpha': (8, 12),
            'beta': (12, 30),
            'gamma_low': (30, 60),
            'gamma_high': (60, 100)
        }

        # Convert window to samples
        window_samples = int(window_sec * self.fs)
        window_samples_vm = int(window_sec * self.vm_rate)
        times = np.linspace(-window_sec, window_sec, 2 * window_samples)
        velocity_times = np.linspace(-window_sec, window_sec, 2 * window_samples_vm)

        # Store band powers for each time point
        band_powers = {band: np.zeros(2 * window_samples) for band in bands}
        avg_velocity = np.zeros(2 * window_samples_vm)

        # Number of valid events
        valid_count = 0
        valid_velocity_count = 0

        # Process each event
        for event_idx in event_indices:
            event_sample = int(event_idx / self.vm_rate * self.fs)

            if (event_sample - window_samples >= 0 and
                    event_sample + window_samples < len(self.signal)):

                # Extract signal segment
                signal_segment = self.signal[event_sample - window_samples:
                                             event_sample + window_samples]

                # Use sliding window to calculate band powers
                window_size = int(self.fs * 0.2)  # 200ms window
                for i in range(len(signal_segment) - window_size):
                    window = signal_segment[i:i + window_size]

                    # Calculate power spectrum
                    freqs, psd = welch(window, fs=self.fs, nperseg=window_size)

                    # Calculate band powers
                    for band_name, (low, high) in bands.items():
                        band_mask = (freqs >= low) & (freqs <= high)
                        if np.any(band_mask):
                            band_power = np.mean(psd[band_mask])
                            band_powers[band_name][i] += 10 * np.log10(band_power)

                valid_count += 1

            # Extract velocity segment
            if (event_idx - window_samples_vm >= 0 and
                    event_idx + window_samples_vm < len(self.smoothed_velocity)):
                velocity_segment = self.smoothed_velocity[event_idx - window_samples_vm:
                                                          event_idx + window_samples_vm]
                avg_velocity += velocity_segment
                valid_velocity_count += 1

        # If no valid events, return None
        if valid_count == 0:
            print("No valid events found within signal bounds.")
            return None

        # Average band powers
        for band in bands:
            band_powers[band] /= valid_count
            # Normalize for better visualization
            band_powers[band] = (band_powers[band] - np.min(band_powers[band])) / \
                                (np.max(band_powers[band]) - np.min(band_powers[band]))

        # Average and normalize velocity
        if valid_velocity_count > 0:
            avg_velocity /= valid_velocity_count
            avg_velocity_norm = (avg_velocity - np.min(avg_velocity)) / \
                                (np.max(avg_velocity) - np.min(avg_velocity))

        # Set plot title
        if title is None:
            title = f"Frequency Band Powers around Events (n={valid_count})"

        # Create plot
        p = figure(
            title=title,
            x_axis_label='Time relative to event (s)',
            y_axis_label='Normalized Power',
            width=900,
            height=500,
            x_range=(-window_sec, window_sec),
            tools="pan,wheel_zoom,box_zoom,reset,save"
        )

        # Colors for bands
        colors = ['navy', 'forestgreen', 'darkred', 'purple', 'orange', 'darkturquoise']

        # Plot each band
        for (band, power), color in zip(band_powers.items(), colors):
            p.line(
                times,
                power,
                line_width=2,
                color=color,
                legend_label=f"{band} ({bands[band][0]}-{bands[band][1]} Hz)"
            )

        # Plot velocity if available
        if valid_velocity_count > 0:
            p.line(
                velocity_times,
                avg_velocity_norm,
                line_width=3,
                color="black",
                line_dash="dotted",
                legend_label="Avg. Velocity"
            )

        # Vertical line at event
        p.line([0, 0], [0, 1], line_width=2, color="black", line_dash="dashed")

        # Customize legend
        p.legend.location = "top_right"
        p.legend.click_policy = "hide"

        # Use the class's output method
        self.output_bokeh_plot(p, save_path=save_path, title=title, notebook=notebook, overwrite=overwrite)

        return p


if __name__ == "__main__":
    path=""
    ep = ElecTank(path)
