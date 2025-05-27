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
                 notch_fs=[60],
                 notch_Q=30
                 ):

        self.session_name = session_name
        self.config = self.load_config()

        elec_path = os.path.join(self.config['ElecPath'], session_name) if elec_path is None else elec_path
        virmen_path = os.path.join(self.config['VirmenFilePath'],
                                   f"{session_name}.txt") if virmen_path is None else virmen_path

        super().__init__(
            session_name=session_name,
            virmen_path=virmen_path,
            maze_type=maze_type,
            vm_rate=vm_rate,
            session_duration=session_duration)

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
                           notebook=False, save_path=None, overwrite=False, font_size=None):
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

        self.output_bokeh_plot(p, save_path=save_path, title=str(p.title.text), notebook=notebook, overwrite=overwrite, font_size=font_size)

        return p

    def plot_raw_vs_notch(self, start=10, stop=14, save_path=None, notebook=False, overwrite=False, font_size=None):
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

        self.output_bokeh_plot(p, save_path=save_path, title=str(p.title.text), notebook=notebook, overwrite=overwrite, font_size=font_size)

        return p

    def plot_velocity_with_signal(self, save_path=None, notebook=False, overwrite=False, font_size=None):
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

        self.output_bokeh_plot(p, save_path=save_path, title=str(p.title.text), notebook=notebook, overwrite=overwrite, font_size=font_size)

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
                                        save_path=None, notebook=False, overwrite=False, font_size=None):
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
                               overwrite=overwrite, font_size=font_size)

        return layout

    def plot_time_frequency_with_beta_analysis(self, freqs=None, tfr=None,
                                               freq_start=1, freq_stop=100, freq_step=0.5,
                                               time_start=0, time_end=None,
                                               time_dec_factor=20, freq_dec_factor=1,
                                               beta_min=12, beta_max=30, threshold_std=1.5,
                                               palette="Turbo256",
                                               save_path=None, notebook=False, overwrite=False, font_size=None):
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
                               notebook=notebook, overwrite=overwrite, font_size=font_size)

        return layout
    

    def plot_event_centered_spectrogram(self, event_indices, window_sec=5,
                                    analysis_padding=5,
                                    freq_start=1, freq_stop=100, freq_step=0.5,
                                    time_dec_factor=5, freq_dec_factor=1,
                                    palette="turbo",
                                    title="Event-centered Spectrogram", notebook=False,
                                    plot_signal=True, plot_velocity=True, plot_lick=True,
                                    show_legend=True,
                                    save_path=None, overwrite=False):
        """
        Improved version that uses z-score normalization and baseline correction rather than min-max normalization.
        Still uses Morlet wavelets with optimizations for low frequency issues.
        
        Parameters:
        -----------
        event_indices : list or array
            Indices of events to center analysis around
        window_sec : float
            Half-width of the window in seconds
        analysis_padding : float
            Extra padding in seconds for analysis window
        freq_start, freq_stop, freq_step : float
            Frequency range parameters in Hz
        time_dec_factor, freq_dec_factor : int
            Decimation factors for plotting
        palette : str
            Color palette for spectrogram
        title : str
            Title for the plot
        notebook : bool
            Whether to display in notebook
        plot_signal : bool
            Whether to plot the average signal
        plot_velocity : bool
            Whether to plot the average velocity
        plot_lick : bool
            Whether to plot the average lick data
        show_legend : bool
            Whether to display legends on the plots (default: True)
        save_path : str
            Path to save the plot
        overwrite : bool
            Whether to overwrite existing file
        """
        import numpy as np
        import mne
        from bokeh.plotting import figure
        from bokeh.models import ColorBar, LinearColorMapper
        from bokeh.layouts import column
        from bokeh.palettes import viridis, plasma, inferno, magma, Turbo256, Viridis256, Plasma256, Inferno256

        # Map palette string to actual Bokeh palette
        palette_map = {
            "viridis": Viridis256,
            "plasma": Plasma256,
            "inferno": Inferno256,
            "magma": magma(256),
            "turbo": Turbo256
        }
        
        # Get the actual palette or default to Viridis256
        actual_palette = palette_map.get(palette.lower(), Viridis256)

        # Check for valid events
        if len(event_indices) == 0:
            print("No valid events provided.")
            return None
            
        # Convert window lengths to sample counts
        window_samples = int(window_sec * self.fs)
        analysis_window_samples = int((window_sec + analysis_padding) * self.fs) 
        window_samples_vm = int(window_sec * self.vm_rate)

        # Ensure frequency upper limit doesn't exceed Nyquist frequency
        nyquist_freq = self.fs / 2
        if freq_stop >= nyquist_freq:
            freq_stop = nyquist_freq - freq_step
            print(f"Warning: Maximum frequency adjusted to {freq_stop} Hz to stay below Nyquist frequency ({nyquist_freq} Hz)")

        # Define frequency range
        freqs = np.arange(freq_start, freq_stop, freq_step)
        
        # Adjust analysis window size for low frequency wavelets
        min_wave_cycles = 3  # Minimum wavelet cycles
        # For 1Hz, need at least 3 seconds to accommodate 3 cycles
        min_window_for_lowest_freq = min_wave_cycles / freq_start
        
        if window_sec + analysis_padding < min_window_for_lowest_freq:
            print(f"Warning: Window may be too short ({window_sec + analysis_padding} s) for lowest frequency of {freq_start} Hz.")
            print(f"Recommended window length of at least {min_window_for_lowest_freq} s to capture complete waveforms.")
            print("Proceeding, but low frequency results may be inaccurate.")
        
        # Collect all valid event segments, processing individual events rather than global signal
        all_tfrs = []
        valid_events = []
        
        # Adjust wavelet cycles for each frequency
        n_cycles = freqs / 2  # Adaptive cycle count: fewer for low frequencies
        n_cycles = np.clip(n_cycles, 3, 10)  # Min 3 cycles, max 10 cycles
        
        for event_idx in event_indices:
            event_sample = int(event_idx / self.vm_rate * self.fs)
            
            # Check if we have sufficient data
            if (event_sample - analysis_window_samples >= 0 and 
                event_sample + analysis_window_samples < len(self.signal)):
                
                # Extract signal segment around event
                signal_segment = self.signal[event_sample - analysis_window_samples:
                                            event_sample + analysis_window_samples]
                
                # Reshape for MNE format
                signal_segment = signal_segment.reshape(1, 1, -1)
                
                try:
                    # Calculate time-frequency representation
                    tfr = mne.time_frequency.tfr_array_morlet(
                        signal_segment,
                        sfreq=self.fs,
                        freqs=freqs,
                        n_cycles=n_cycles,
                        zero_mean=True,
                        output='power'
                    )
                    
                    # Convert to dB
                    tfr_db = 10 * np.log10(tfr.squeeze())
                    
                    # Store this TFR
                    all_tfrs.append(tfr_db)
                    valid_events.append(event_idx)
                    
                except ValueError as e:
                    print(f"Warning: Error processing event {event_idx}: {e}")
                    continue
        
        if not all_tfrs:
            print("No valid events found.")
            return None
        
        # Stack all TFRs into one large array for global normalization
        # Shape will be [n_events, n_freqs, n_times]
        stacked_tfrs = np.stack(all_tfrs)
        
        # 1. Z-score normalization instead of min-max
        normalized_tfrs = np.zeros_like(stacked_tfrs)
        for freq_idx in range(stacked_tfrs.shape[1]):
            # Extract this frequency row for all events
            all_events_this_freq = stacked_tfrs[:, freq_idx, :]
            
            # Calculate mean and std for this frequency
            freq_mean = np.mean(all_events_this_freq)
            freq_std = np.std(all_events_this_freq)
            
            if freq_std > 0:  # Avoid division by zero
                # Apply z-score normalization to this frequency for all events
                normalized_tfrs[:, freq_idx, :] = (stacked_tfrs[:, freq_idx, :] - freq_mean) / freq_std
            else:
                normalized_tfrs[:, freq_idx, :] = 0  # Default value if no variation
        
        # Create time axis for baseline correction
        full_times = np.linspace(-window_sec-analysis_padding, window_sec+analysis_padding, 
                               normalized_tfrs.shape[2])
        
        # 2. Apply baseline correction
        # Define baseline period (e.g., -5 to -3 seconds)
        baseline_start = np.where(full_times >= -5)[0][0]
        baseline_end = np.where(full_times >= -3)[0][0]
        
        baseline_corrected_tfrs = np.zeros_like(normalized_tfrs)
        for event_idx in range(normalized_tfrs.shape[0]):
            for freq_idx in range(normalized_tfrs.shape[1]):
                # Get baseline period for this event and frequency
                baseline = normalized_tfrs[event_idx, freq_idx, baseline_start:baseline_end]
                baseline_mean = np.mean(baseline)
                
                # Subtract baseline mean
                baseline_corrected_tfrs[event_idx, freq_idx, :] = normalized_tfrs[event_idx, freq_idx, :] - baseline_mean
        
        # 3. Average the baseline-corrected TFRs
        avg_normalized_tfr = np.mean(baseline_corrected_tfrs, axis=0)
        
        # Determine how many samples to crop from analysis window to get display window
        padding_samples = int(analysis_padding * self.fs / 2)  # Crop half padding from each side
        
        # Get the central portion of the time-frequency data (crop padding)
        central_start = padding_samples
        central_end = avg_normalized_tfr.shape[1] - padding_samples
        
        # Crop the average TFR to desired display window
        avg_normalized_tfr_cropped = avg_normalized_tfr[:, central_start:central_end]
        
        # Decimate for plotting if needed
        freqs_dec = freqs[::freq_dec_factor]
        
        if time_dec_factor > 1:
            avg_normalized_tfr_dec = avg_normalized_tfr_cropped[:, ::time_dec_factor]
        else:
            avg_normalized_tfr_dec = avg_normalized_tfr_cropped
        
        # Create time axis centered around events (in seconds)
        times = np.linspace(-window_sec, window_sec, avg_normalized_tfr_cropped.shape[1])
        times_dec = times[::time_dec_factor] if time_dec_factor > 1 else times
        
        # Calculate dimensions for the image
        dw = times_dec[-1] - times_dec[0]
        dh = freqs_dec[-1] - freqs_dec[0]
        
        # Create Bokeh figure
        p1 = figure(
            title=f"{title} (n={len(valid_events)}) - Z-score & Baseline Corrected",
            x_axis_label='Time relative to event (s)',
            y_axis_label='Frequency (Hz)',
            tools="pan,wheel_zoom,box_zoom,reset,save",
            width=1000, height=700,
            x_range=(-window_sec, window_sec),
            y_range=(freqs[0], min(80, freq_stop))
        )
        
        # Remove grid
        p1.grid.grid_line_color = None
        
        # Define color mapper with improved settings - use percentile clipping for better contrast
        color_mapper = LinearColorMapper(
            palette=actual_palette,
            low=np.percentile(avg_normalized_tfr_dec.ravel(), 5),
            high=np.percentile(avg_normalized_tfr_dec.ravel(), 95)
        )
        
        # Add the normalized spectrogram image
        p1.image(
            image=[avg_normalized_tfr_dec],
            x=times_dec[0],
            y=freqs_dec[0],
            dw=dw,
            dh=dh,
            color_mapper=color_mapper
        )
        
        # Add color bar
        color_bar = ColorBar(
            color_mapper=color_mapper,
            label_standoff=12,
            border_line_color=None,
            location=(0, 0),
            title="Z-score Power"
        )
        p1.add_layout(color_bar, 'right')
        
        layout = [p1]
        
        # Only perform signal analysis and create second plot if needed
        if plot_signal or plot_velocity or plot_lick:
            # Calculate average LFP and velocity around events
            avg_signal = np.zeros(2 * window_samples)
            avg_velocity = np.zeros(2 * window_samples_vm)
            avg_lick = np.zeros(2 * window_samples_vm)
            valid_velocity_events = 0
            valid_lick_events = 0 

            for event_idx in valid_events:
                event_sample = int(event_idx / self.vm_rate * self.fs)

                # LFP segment
                if plot_signal and (event_sample - window_samples >= 0 and
                        event_sample + window_samples < len(self.signal)):
                    signal_segment = self.signal[event_sample - window_samples:
                                                event_sample + window_samples]
                    avg_signal += signal_segment

                # Velocity segment
                if plot_velocity and (event_idx - window_samples_vm >= 0 and
                        event_idx + window_samples_vm < len(self.smoothed_velocity)):
                    velocity_segment = self.smoothed_velocity[event_idx - window_samples_vm:
                                                            event_idx + window_samples_vm]
                    avg_velocity += velocity_segment
                    valid_velocity_events += 1
                    
                # Lick segment
                if plot_lick and (event_idx - window_samples_vm >= 0 and
                        event_idx + window_samples_vm < len(self.lick_raw)):
                    lick_segment = self.lick[event_idx - window_samples_vm:
                                                event_idx + window_samples_vm]
                    avg_lick += lick_segment
                    valid_lick_events += 1

            # Create a second figure for signals
            p2 = figure(
                title="Normalized Signals",
                x_axis_label="Time relative to event (s)",
                y_axis_label="Amplitude",
                width=1000,
                height=300,
                x_range=p1.x_range,
            )
            
            # Remove grid
            p2.grid.grid_line_color = None

            # Plot the average LFP
            if plot_signal:
                # Average and normalize
                avg_signal /= len(valid_events)
                avg_signal_norm = (avg_signal - np.min(avg_signal)) / (np.max(avg_signal) - np.min(avg_signal))
                
                p2.line(
                    np.linspace(-window_sec, window_sec, len(avg_signal)),
                    avg_signal_norm,
                    line_width=2,
                    color="lightblue",
                    legend_label="Avg. Signal"
                )

            # Plot velocity if available
            if plot_velocity and valid_velocity_events > 0:
                avg_velocity /= valid_velocity_events
                avg_velocity_norm = (avg_velocity - np.min(avg_velocity)) / (np.max(avg_velocity) - np.min(avg_velocity))

                # Create velocity time axis
                velocity_times = np.linspace(-window_sec, window_sec, len(avg_velocity))

                # Plot velocity
                p2.line(
                    velocity_times,
                    avg_velocity_norm,
                    line_width=2,
                    color="black",
                    legend_label="Avg. Velocity",
                    alpha=0.7
                )
                
            # Plot lick if available
            if plot_lick and valid_lick_events > 0:
                avg_lick /= valid_lick_events
                if np.max(avg_lick) > np.min(avg_lick):
                    avg_lick_norm = (avg_lick - np.min(avg_lick)) / (np.max(avg_lick) - np.min(avg_lick))
                else:
                    avg_lick_norm = avg_lick
                    
                p2.line(
                    velocity_times,
                    avg_lick_norm,
                    line_width=3,
                    color="green",
                    legend_label="Avg. Lick",
                    alpha=0.7
                )

            # Add vertical line at event time (t=0)
            p2.line([0, 0], [0, 1], line_width=2, color="red", line_dash="dashed")
            
            # Set legend location to a valid position only if show_legend is True
            if show_legend:
                p2.legend.location = "top_right"
                p2.legend.click_policy = 'hide'
            else:
                p2.legend.visible = False  # Hide the legend completely
            
            # Add the signal plot to the layout
            layout.append(p2)

        # Combine plots
        final_layout = column(*layout)

        # Use the class's output method
        self.output_bokeh_plot(final_layout, save_path=save_path, title=title, notebook=notebook, overwrite=overwrite)

        return final_layout
    


    def plot_event_centered_spectrogram_log_axis(self, event_indices, window_sec=5,
                                analysis_padding=5,
                                freq_bands=None, freq_scale='log',
                                time_dec_factor=5, freq_dec_factor=1,
                                palette="turbo",
                                title="Event-Centered Spectrogram", notebook=False,
                                plot_signal=True, plot_velocity=True, plot_lick=True,
                                show_legend=True, save_path=None, overwrite=False):
        """
        Improved version that uses z-score normalization, baseline correction,
        and custom frequency band representation with proper log-scale alignment.
        
        Parameters:
        -----------
        event_indices : list or array
            Indices of events to center analysis around
        window_sec : float
            Half-width of the window in seconds
        analysis_padding : float
            Extra padding in seconds for analysis window
        freq_bands : dict or None
            Custom frequency bands as dict with format {'band_name': (min_freq, max_freq, step)}
            If None, will use default bands (delta, theta, alpha, beta, gamma)
        freq_scale : str
            'log' for logarithmic frequency spacing, 'linear' for linear spacing
        time_dec_factor, freq_dec_factor : int
            Decimation factors for plotting
        palette : str
            Color palette for spectrogram
        title : str
            Title for the plot
        notebook : bool
            Whether to display in notebook
        plot_signal : bool
            Whether to plot the average signal
        plot_velocity : bool
            Whether to plot the average velocity
        plot_lick : bool
            Whether to plot the average lick data
        save_path : str
            Path to save the plot
        overwrite : bool
            Whether to overwrite existing file
        """
        import numpy as np
        import mne
        from bokeh.plotting import figure
        from bokeh.models import ColorBar, LinearColorMapper, LogTicker, FixedTicker
        from bokeh.layouts import column
        from bokeh.palettes import viridis, plasma, inferno, magma, Turbo256, Viridis256, Plasma256, Inferno256

        # Map palette string to actual Bokeh palette
        palette_map = {
            "viridis": Viridis256,
            "plasma": Plasma256,
            "inferno": Inferno256,
            "magma": magma(256),
            "turbo": Turbo256
        }
        
        # Get the actual palette or default to Viridis256
        actual_palette = palette_map.get(palette.lower(), Viridis256)

        # Check for valid events
        if len(event_indices) == 0:
            print("No valid events provided.")
            return None
            
        # Convert window lengths to sample counts
        window_samples = int(window_sec * self.fs)
        analysis_window_samples = int((window_sec + analysis_padding) * self.fs) 
        window_samples_vm = int(window_sec * self.vm_rate)

        # Ensure frequency upper limit doesn't exceed Nyquist frequency
        nyquist_freq = self.fs / 2
        
        # Define frequency bands if not provided
        if freq_bands is None:
            freq_bands = {
                'delta': (1, 4, 0.5),      # 1-4 Hz, 0.5 Hz steps
                'theta': (4, 8, 0.5),      # 4-8 Hz, 0.5 Hz steps
                'alpha': (8, 13, 0.5),     # 8-13 Hz, 0.5 Hz steps
                'beta': (13, 30, 1),       # 13-30 Hz, 1 Hz steps
                'gamma': (30, min(100, nyquist_freq - 1), 2)  # 30-100 Hz, 2 Hz steps (or below Nyquist)
            }
        
        # Create frequency array based on bands and scale
        freqs = []
        band_boundaries = []
        band_labels = []
        
        for band_name, (fmin, fmax, fstep) in freq_bands.items():
            # Adjust max freq to stay below Nyquist
            if fmax >= nyquist_freq:
                fmax = nyquist_freq - fstep
                print(f"Warning: Maximum frequency for {band_name} adjusted to {fmax} Hz to stay below Nyquist")
            
            # Create frequencies for this band
            if freq_scale == 'log':
                # Logarithmic spacing within each band
                # Use enough points to make the steps visually smooth
                num_points = max(10, int((np.log10(fmax) - np.log10(fmin)) / 0.05))
                band_freqs = np.logspace(np.log10(fmin), np.log10(fmax), num_points)
            else:
                # Linear spacing
                band_freqs = np.arange(fmin, fmax + fstep/2, fstep)
            
            # Store band boundaries for later plotting
            band_boundaries.append((fmin, fmax))
            band_labels.append(band_name)
            
            # Add these frequencies to the master list
            freqs.extend(band_freqs)
        
        # Convert to numpy array and remove duplicates
        freqs = np.array(sorted(set([round(f, 3) for f in freqs])))
        
        # Adjust analysis window size for low frequency wavelets
        min_wave_cycles = 3  # Minimum wavelet cycles
        # For 1Hz, need at least 3 seconds to accommodate 3 cycles
        min_window_for_lowest_freq = min_wave_cycles / freqs[0]
        
        if window_sec + analysis_padding < min_window_for_lowest_freq:
            print(f"Warning: Window may be too short ({window_sec + analysis_padding} s) for lowest frequency of {freqs[0]} Hz.")
            print(f"Recommended window length of at least {min_window_for_lowest_freq} s to capture complete waveforms.")
            print("Proceeding, but low frequency results may be inaccurate.")
        
        # Collect all valid event segments, processing individual events rather than global signal
        all_tfrs = []
        valid_events = []
        
        # Adjust wavelet cycles for each frequency (adaptive)
        # Lower frequencies get fewer cycles, higher frequencies get more
        n_cycles = freqs / 2  # Adaptive cycle count
        n_cycles = np.clip(n_cycles, 3, 10)  # Min 3 cycles, max 10 cycles
        
        for event_idx in event_indices:
            event_sample = int(event_idx / self.vm_rate * self.fs)
            
            # Check if we have sufficient data
            if (event_sample - analysis_window_samples >= 0 and 
                event_sample + analysis_window_samples < len(self.signal)):
                
                # Extract signal segment around event
                signal_segment = self.signal[event_sample - analysis_window_samples:
                                            event_sample + analysis_window_samples]
                
                # Reshape for MNE format
                signal_segment = signal_segment.reshape(1, 1, -1)
                
                try:
                    # Calculate time-frequency representation
                    tfr = mne.time_frequency.tfr_array_morlet(
                        signal_segment,
                        sfreq=self.fs,
                        freqs=freqs,
                        n_cycles=n_cycles,
                        zero_mean=True,
                        output='power'
                    )
                    
                    # Convert to dB
                    tfr_db = 10 * np.log10(tfr.squeeze())
                    
                    # Store this TFR
                    all_tfrs.append(tfr_db)
                    valid_events.append(event_idx)
                    
                except ValueError as e:
                    print(f"Warning: Error processing event {event_idx}: {e}")
                    continue
        
        if not all_tfrs:
            print("No valid events found.")
            return None
        
        # Stack all TFRs into one large array for global normalization
        # Shape will be [n_events, n_freqs, n_times]
        stacked_tfrs = np.stack(all_tfrs)
        
        # 1. Z-score normalization instead of min-max
        normalized_tfrs = np.zeros_like(stacked_tfrs)
        for freq_idx in range(stacked_tfrs.shape[1]):
            # Extract this frequency row for all events
            all_events_this_freq = stacked_tfrs[:, freq_idx, :]
            
            # Calculate mean and std for this frequency
            freq_mean = np.mean(all_events_this_freq)
            freq_std = np.std(all_events_this_freq)
            
            if freq_std > 0:  # Avoid division by zero
                # Apply z-score normalization to this frequency for all events
                normalized_tfrs[:, freq_idx, :] = (stacked_tfrs[:, freq_idx, :] - freq_mean) / freq_std
            else:
                normalized_tfrs[:, freq_idx, :] = 0  # Default value if no variation
        
        # Create time axis for baseline correction
        full_times = np.linspace(-window_sec-analysis_padding, window_sec+analysis_padding, 
                            normalized_tfrs.shape[2])
        
        # 2. Apply baseline correction
        # Define baseline period (e.g., -5 to -3 seconds)
        baseline_start = np.where(full_times >= -5)[0][0]
        baseline_end = np.where(full_times >= -3)[0][0]
        
        baseline_corrected_tfrs = np.zeros_like(normalized_tfrs)
        for event_idx in range(normalized_tfrs.shape[0]):
            for freq_idx in range(normalized_tfrs.shape[1]):
                # Get baseline period for this event and frequency
                baseline = normalized_tfrs[event_idx, freq_idx, baseline_start:baseline_end]
                baseline_mean = np.mean(baseline)
                
                # Subtract baseline mean
                baseline_corrected_tfrs[event_idx, freq_idx, :] = normalized_tfrs[event_idx, freq_idx, :] - baseline_mean
        
        # 3. Average the baseline-corrected TFRs
        avg_normalized_tfr = np.mean(baseline_corrected_tfrs, axis=0)
        
        # Determine how many samples to crop from analysis window to get display window
        padding_samples = int(analysis_padding * self.fs / 2)  # Crop half padding from each side
        
        # Get the central portion of the time-frequency data (crop padding)
        central_start = padding_samples
        central_end = avg_normalized_tfr.shape[1] - padding_samples
        
        # Crop the average TFR to desired display window
        avg_normalized_tfr_cropped = avg_normalized_tfr[:, central_start:central_end]
        
        # Create time axis centered around events (in seconds)
        times = np.linspace(-window_sec, window_sec, avg_normalized_tfr_cropped.shape[1])
        
        # Decimate for plotting if needed
        if freq_dec_factor > 1:
            # Ensure we preserve the frequency indices correctly
            freq_indices = list(range(0, len(freqs), freq_dec_factor))
            freqs_dec = freqs[freq_indices]
            avg_normalized_tfr_decimated = avg_normalized_tfr_cropped[freq_indices, :]
        else:
            freqs_dec = freqs
            avg_normalized_tfr_decimated = avg_normalized_tfr_cropped
        
        if time_dec_factor > 1:
            # Ensure we preserve the time indices correctly
            time_indices = list(range(0, len(times), time_dec_factor))
            times_dec = times[time_indices]
            avg_normalized_tfr_dec = avg_normalized_tfr_decimated[:, time_indices]
        else:
            times_dec = times
            avg_normalized_tfr_dec = avg_normalized_tfr_decimated
        
        # Create Bokeh figure with appropriate scaling for y-axis
        p1 = figure(
            title=f"{title} (n={len(valid_events)}) - Z-score & Baseline Corrected",
            x_axis_label='Time relative to event (s)',
            y_axis_label='Frequency (Hz)',
            tools="pan,wheel_zoom,box_zoom,reset,save",
            width=1000, height=500,  # Taller to accommodate better frequency axis
            x_range=(-window_sec, window_sec),
            y_axis_type="log" if freq_scale == 'log' else "linear"
        )
        
        # Make sure minimum frequency is not zero for log scale
        if freq_scale == 'log' and freqs_dec[0] <= 0:
            print("Warning: Adjusting minimum frequency to 0.1 Hz for log scale")
            p1.y_range.start = 0.1
        else:
            p1.y_range.start = freqs_dec[0]
        
        # Set the y-range end
        p1.y_range.end = freqs_dec[-1]
        
        # Remove grid
        p1.grid.grid_line_color = None
        
        # Define color mapper with improved settings - use percentile clipping for better contrast
        color_mapper = LinearColorMapper(
            palette=actual_palette,
            low=np.percentile(avg_normalized_tfr_dec.ravel(), 5),
            high=np.percentile(avg_normalized_tfr_dec.ravel(), 95)
        )
        
        # For proper image alignment, we need to use the correct coordinates
        # The key is understanding that in Bokeh's image method:
        # - x, y specify the BOTTOM-LEFT corner of the image
        # - dw, dh specify the width and height of the image in data coordinates
        
        # When using log scale for frequencies, the mapping is more complex
        # For correct alignment, we create the image with proper frequency boundaries
        
        # Reshape the TFR data for image rendering
        # Important: image shape must be [nrows, ncols] where
        # nrows = number of frequencies
        # ncols = number of time points
        
        # For log scale, we need to handle the image differently to ensure alignment
        if freq_scale == 'log':
            # For log scale, y = frequencies array values
            y_bottom = freqs_dec[0]  # Bottom edge (minimum frequency)
            y_top = freqs_dec[-1]    # Top edge (maximum frequency)
            
            # Use Bokeh's image method to render the spectrogram
            img = p1.image(
                image=[avg_normalized_tfr_dec],
                x=times_dec[0],         # Left edge (minimum time)
                y=y_bottom,             # Bottom edge (minimum frequency)
                dw=times_dec[-1] - times_dec[0],  # Width (time range)
                dh=y_top - y_bottom,    # Height (frequency range)
                color_mapper=color_mapper
            )
        else:
            # For linear scale, standard approach
            img = p1.image(
                image=[avg_normalized_tfr_dec],
                x=times_dec[0],         # Left edge
                y=freqs_dec[0],         # Bottom edge
                dw=times_dec[-1] - times_dec[0],  # Width
                dh=freqs_dec[-1] - freqs_dec[0],  # Height
                color_mapper=color_mapper
            )
        
        # Add color bar
        color_bar = ColorBar(
            color_mapper=color_mapper,
            label_standoff=12,
            border_line_color=None,
            location=(0, 0),
            title="Z-score Power"
        )
        p1.add_layout(color_bar, 'right')
        
        # Add horizontal lines to mark frequency band boundaries
        for fmin, fmax in band_boundaries:
            p1.line([-window_sec, window_sec], [fmin, fmin], line_color='white', line_width=1, line_alpha=0.5, line_dash='dashed')
            p1.line([-window_sec, window_sec], [fmax, fmax], line_color='white', line_width=1, line_alpha=0.5, line_dash='dashed')
        
        # Add band labels
        for i, (label, (fmin, fmax)) in enumerate(zip(band_labels, band_boundaries)):
            # Place text in the middle of each band
            mid_freq = (fmin + fmax) / 2 if freq_scale == 'linear' else np.sqrt(fmin * fmax)
            p1.text([-window_sec * 0.95], [mid_freq], [label], text_color='white', text_font_size='9pt')
        
        layout = [p1]
        
        # Only perform signal analysis and create second plot if needed
        if plot_signal or plot_velocity or plot_lick:
            # Calculate average LFP and velocity around events
            avg_signal = np.zeros(2 * window_samples)
            avg_velocity = np.zeros(2 * window_samples_vm)
            avg_lick = np.zeros(2 * window_samples_vm)
            valid_velocity_events = 0
            valid_lick_events = 0 

            for event_idx in valid_events:
                event_sample = int(event_idx / self.vm_rate * self.fs)

                # LFP segment
                if plot_signal and (event_sample - window_samples >= 0 and
                        event_sample + window_samples < len(self.signal)):
                    signal_segment = self.signal[event_sample - window_samples:
                                                event_sample + window_samples]
                    avg_signal += signal_segment

                # Velocity segment
                if plot_velocity and (event_idx - window_samples_vm >= 0 and
                        event_idx + window_samples_vm < len(self.smoothed_velocity)):
                    velocity_segment = self.smoothed_velocity[event_idx - window_samples_vm:
                                                            event_idx + window_samples_vm]
                    avg_velocity += velocity_segment
                    valid_velocity_events += 1
                    
                # Lick segment
                if plot_lick and (event_idx - window_samples_vm >= 0 and
                        event_idx + window_samples_vm < len(self.lick_raw)):
                    lick_segment = self.lick[event_idx - window_samples_vm:
                                                event_idx + window_samples_vm]
                    avg_lick += lick_segment
                    valid_lick_events += 1

            # Create a second figure for signals
            p2 = figure(
                title="Normalized Signals",
                x_axis_label="Time relative to event (s)",
                y_axis_label="Amplitude",
                width=1000,
                height=300,
                x_range=p1.x_range,
            )
            
            # Remove grid
            p2.grid.grid_line_color = None

            # Plot the average LFP
            if plot_signal:
                # Average and normalize
                avg_signal /= len(valid_events)
                avg_signal_norm = (avg_signal - np.min(avg_signal)) / (np.max(avg_signal) - np.min(avg_signal))
                
                p2.line(
                    np.linspace(-window_sec, window_sec, len(avg_signal)),
                    avg_signal_norm,
                    line_width=2,
                    color="lightblue",
                    legend_label="Avg. Signal"
                )

            # Plot velocity if available
            if plot_velocity and valid_velocity_events > 0:
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
            if plot_lick and valid_lick_events > 0:
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

            # Add vertical line at event time (t=0)
            p2.line([0, 0], [0, 1], line_width=2, color="red", line_dash="dashed")
            
            # Set legend location to a valid position only if show_legend is True
            if show_legend:
                p2.legend.location = "top_right"
                p2.legend.click_policy = 'hide'
            else:
                p2.legend.visible = False  # Hide the legend completely
            
            # Add the signal plot to the layout
            layout.append(p2)

        # Combine plots
        final_layout = column(*layout)

        # Use the class's output method
        self.output_bokeh_plot(final_layout, save_path=save_path, title=title, notebook=notebook, overwrite=overwrite)

        return final_layout
    
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

    def detect_spectrogram_artifacts(self, threshold_factor=3.0, min_freqs_affected=0.75, freq_range=(1, 80)):
        """
        Detect exact time indices of artifacts in the spectrogram.
        
        Parameters:
        -----------
        threshold_factor : float
            How many standard deviations above the mean to set the threshold
        min_freqs_affected : float
            Minimum proportion of frequency bands that must exceed threshold
        freq_range : tuple
            (min_freq, max_freq) range to consider for artifact detection
        
        Returns:
        --------
        artifact_time_indices : list
            List of time indices (in seconds) where artifacts occur
        """
        # Generate time-frequency representation
        freqs, tfr = self.generate_time_frequency_spectrogram(
            freq_start=freq_range[0], 
            freq_stop=freq_range[1], 
            freq_step=0.5
        )
        
        # Get time axis in seconds
        time_axis = self.elc_t[:tfr.shape[-1]]
        
        # Convert to dB scale if needed
        power_db = 10 * np.log10(tfr.squeeze()) if np.min(tfr) >= 0 else tfr.squeeze()
        
        # Get frequencies within our range
        freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        power_of_interest = power_db[freq_mask, :]
        
        # Calculate median and MAD for each frequency
        median_per_freq = np.median(power_of_interest, axis=1, keepdims=True)
        mad_per_freq = np.median(np.abs(power_of_interest - median_per_freq), axis=1, keepdims=True)
        
        # Set threshold using MAD (1.4826 factor converts MAD to STD-equivalent for normal distributions)
        thresholds = median_per_freq + threshold_factor * 1.4826 * mad_per_freq
        
        # Find where power exceeds threshold for each frequency
        exceeded_threshold = power_of_interest > thresholds
        
        # Calculate proportion of frequencies affected at each time point
        freqs_affected_ratio = np.mean(exceeded_threshold, axis=0)
        
        # Find time points where proportion exceeds minimum
        artifact_indices = np.where(freqs_affected_ratio >= min_freqs_affected)[0]
        
        # Convert indices to actual time values (in seconds)
        artifact_time_points = time_axis[artifact_indices]
        
        print(f"Detected {len(artifact_time_points)} artifact time points")
        if len(artifact_time_points) > 0:
            print(f"Artifact times range from {artifact_time_points[0]:.2f}s to {artifact_time_points[-1]:.2f}s")
        
        return artifact_time_points

    def filter_events_with_window(self, event_indices, artifact_window_sec=0.5, 
                                  artifact_times=None, threshold_factor=3.0):
        """
        Filter event indices by checking if any artifacts fall within a window around each event.
        
        Parameters:
        -----------
        event_indices : list or array
            Indices of events to filter (e.g., self.movement_onset_indices)
        artifact_window_sec : float
            Size of window around each event to check for artifacts (±seconds)
        artifact_times : array or None
            Precomputed artifact time points (in seconds), if None will be detected
        threshold_factor : float
            Passed to detect_spectrogram_artifacts if artifact_times is None
            
        Returns:
        --------
        filtered_indices : list
            Event indices that don't have artifacts in their windows
        excluded_indices : list
            Event indices that were excluded due to artifacts
        """
        # Convert event indices to seconds
        event_times = event_indices / self.vm_rate
        
        # Get artifact time points if not provided
        if artifact_times is None:
            artifact_times = self.detect_spectrogram_artifacts(threshold_factor=threshold_factor)
        
        if len(artifact_times) == 0:
            # No artifacts detected
            return event_indices, []
        
        # Initialize lists for filtered and excluded events
        filtered_indices = []
        excluded_indices = []
        
        # For each event, check if any artifact falls within its window
        for idx, event_time in zip(event_indices, event_times):
            window_start = event_time - artifact_window_sec
            window_end = event_time + artifact_window_sec
            
            # Check if any artifact falls in this window
            artifacts_in_window = np.any((artifact_times >= window_start) & 
                                        (artifact_times <= window_end))
            
            if artifacts_in_window:
                excluded_indices.append(idx)
            else:
                filtered_indices.append(idx)
        
        print(f"Excluded {len(excluded_indices)} out of {len(event_indices)} events:")
        print(f"  - {len(filtered_indices)} clean events retained")
        print(f"  - {len(excluded_indices)} events had artifacts within ±{artifact_window_sec}s window")
        
        return filtered_indices, excluded_indices


if __name__ == "__main__":
    path=""
    ep = ElecTank(path)
