import os
import numpy as np
import pandas as pd
from .CITank import CITank

class CellTypeTank(CITank):
    """
    A class for analyzing and visualizing different cell types (D1, D2, and Cholinergic neurons) in 
    calcium imaging data.
    Inherits from CITank and adds cell type classification and analysis capabilities.
    """
    
    def __init__(self,
                 session_name,
                 cell_type_label_file=None,
                 ci_path=None,
                 virmen_path=None,
                 gcamp_path=None,
                 tdt_org_path=None,
                 tdt_adjusted_path=None,
                 maze_type=None,
                 ci_rate=20,
                 vm_rate=20,
                 session_duration=30 * 60):
        """
        Initialize CellTypeTank with cell type classification capabilities.
        
        Parameters:
        -----------
        session_name : str
            Name of the session
        cell_type_label_file : str, optional
            Path to a JSON/CSV file containing the labels
        ci_path : str, optional
            Path to the calcium imaging data
        virmen_path : str, optional
            Path to the Virmen data
        gcamp_path : str, optional
            Path to the GCaMP data
        tdt_org_path : str, optional
            Path to the original TDT data
        tdt_adjusted_path : str, optional
            Path to the adjusted TDT data
        maze_type : str, optional
            Type of maze used in the experiment
        ci_rate : int, optional
            Sampling rate of calcium imaging data
        vm_rate : int, optional
            Sampling rate of Virmen data
        session_duration : int, optional
            Duration of the session in seconds
        """

        super().__init__(
            session_name=session_name,
            ci_path=ci_path,
            virmen_path=virmen_path,
            gcamp_path=gcamp_path,
            tdt_org_path=tdt_org_path,
            tdt_adjusted_path=tdt_adjusted_path,
            maze_type=maze_type,
            ci_rate=ci_rate,
            vm_rate=vm_rate,
            session_duration=session_duration
        )
        
        # Get cell type indices
        [self.d1_indices, self.d2_indices, self.chi_indices, _] = self._load_cell_type_labels(cell_type_label_file)
        
        # Process signals for each cell type
        # Raw calcium signals
        self.d1_raw = self.C_raw[self.d1_indices]
        self.d2_raw = self.C_raw[self.d2_indices]
        self.chi_raw = self.C_raw[self.chi_indices]
        
        # Denoised signals
        self.d1_denoised = self.C_denoised[self.d1_indices]
        self.d2_denoised = self.C_denoised[self.d2_indices]
        self.chi_denoised = self.C_denoised[self.chi_indices]
        
        # Deconvolved signals
        self.d1_deconvolved = self.C_deconvolved[self.d1_indices]
        self.d2_deconvolved = self.C_deconvolved[self.d2_indices]
        self.chi_deconvolved = self.C_deconvolved[self.chi_indices]
        
        # Baseline signals
        self.d1_baseline = self.C_baseline[self.d1_indices]
        self.d2_baseline = self.C_baseline[self.d2_indices]
        self.chi_baseline = self.C_baseline[self.chi_indices]
        
        # Reraw signals
        self.d1_reraw = self.C_reraw[self.d1_indices]
        self.d2_reraw = self.C_reraw[self.d2_indices]
        self.chi_reraw = self.C_reraw[self.chi_indices]
        
        # Spatial components
        self.d1_A = self.A[self.d1_indices]
        self.d2_A = self.A[self.d2_indices]
        self.chi_A = self.A[self.chi_indices]
        
        # Centroids
        self.d1_centroids = self.centroids[self.d1_indices]
        self.d2_centroids = self.centroids[self.d2_indices]
        self.chi_centroids = self.centroids[self.chi_indices]
        
        # Coordinates
        self.d1_Coor = [self.Coor[i] for i in self.d1_indices]
        self.d2_Coor = [self.Coor[i] for i in self.d2_indices]
        self.chi_Coor = [self.Coor[i] for i in self.chi_indices]
        
        # Calculate average signals for each cell type
        self.d1_ca_all = self.normalize_signal(self.shift_signal_single(np.mean(self.d1_raw, axis=0)))
        self.d2_ca_all = self.normalize_signal(self.shift_signal_single(np.mean(self.d2_raw, axis=0)))
        self.chi_ca_all = self.normalize_signal(self.shift_signal_single(np.mean(self.chi_raw, axis=0)))

        # Z-scored signals
        self.d1_zsc = self.C_zsc[self.d1_indices]
        self.d2_zsc = self.C_zsc[self.d2_indices]
        self.chi_zsc = self.C_zsc[self.chi_indices]
        
        
    def _load_cell_type_labels(self, cell_type_label_file):
        """
        Load cell type labels from either a dictionary or a file.
        
        Parameters:
        -----------
        cell_type_label_file : str
            Path to a JSON/CSV file containing the labels
            
        Returns:
        --------
        tuple
            Tuple containing arrays of indices for D1, D2, CHI, and unknown neurons
        """
        if cell_type_label_file is None:
            cell_type_label_file = os.path.join(self.config['ProcessedFilePath'], self.session_name, f"{self.session_name}.json")
            
        if isinstance(cell_type_label_file, str):
            # Load from file based on extension
            if cell_type_label_file.endswith('.json'):
                import json
                with open(cell_type_label_file, 'r') as f:
                    data = json.load(f)
            else:
                raise ValueError("Unsupported file format. Use .json")
                
            mask = np.zeros(len(data), dtype=int)
    
            # Fill in the mask values
            for id_num, value in data.items():
                idx = int(id_num)
                if value == "D1":
                    mask[idx] = 1
                elif value == "D2":
                    mask[idx] = 2
                elif value == "cholinergic":
                    mask[idx] = 3
                # unknown remains 0

            unknown_indices = np.where(mask == 0)[0]
            d1_indices = np.where(mask == 1)[0]
            d2_indices = np.where(mask == 2)[0]
            chi_indices = np.where(mask == 3)[0]

            return d1_indices, d2_indices, chi_indices, unknown_indices
        else:
            raise ValueError("cell_type_label_file must be a string path to a JSON file")
        

    def plot_cell_activity_at_events(self, event_type='movement_onset', 
                                     show_d1=True, show_d2=True, show_chi=True,
                                     d1_signal=None, d2_signal=None, chi_signal=None,
                                     cut_interval=50, save_path=None, title=None, notebook=False, overwrite=False):
        """
        Plot average calcium traces for D1, D2, and/or CHI neuron populations, along with velocity, around specific events.
        Uses dual y-axes to separately scale calcium and velocity signals, with optimized scaling.
        
        Parameters:
        -----------
        event_type : str
            Type of event to plot around. Options: 'movement_onset', 'velocity_peak', 'movement_offset'
        show_d1 : bool
            Whether to show D1 neuron signals
        show_d2 : bool
            Whether to show D2 neuron signals
        show_chi : bool
            Whether to show CHI neuron signals
        d1_signal : numpy.ndarray, optional
            Signal from D1 neurons. If None, will use self.d1_raw
        d2_signal : numpy.ndarray, optional
            Signal from D2 neurons. If None, will use self.d2_raw
        chi_signal : numpy.ndarray, optional
            Signal from CHI neurons. If None, will use self.chi_raw
        cut_interval : int
            Interval to cut around the indices (in samples)
        save_path : str, optional
            Path to save the plot as an HTML file
        title : str, optional
            Title for the plot. If None, will generate based on event_type
        notebook : bool
            Flag to indicate if the plot is for a Jupyter notebook
        overwrite : bool
            Flag to indicate whether to overwrite existing file
        
        Returns:
        --------
        bokeh.plotting.figure
            The created plot
        """
        from bokeh.plotting import figure
        from bokeh.models import LinearAxis, Range1d, Legend
        
        # Select indices based on event type
        if event_type == 'movement_onset':
            indices = self.movement_onset_indices
            default_title = "Average Calcium Trace around Movement Onsets"
        elif event_type == 'velocity_peak':
            indices = self.velocity_peak_indices
            default_title = "Average Calcium Trace around Velocity Peaks"
        elif event_type == 'movement_offset':
            indices = self.movement_offset_indices
            default_title = "Average Calcium Trace around Movement Offsets"
        else:
            raise ValueError("event_type must be one of: 'movement_onset', 'velocity_peak', 'movement_offset'")
        
        # Check if we need to show at least one cell type
        if not (show_d1 or show_d2 or show_chi):
            raise ValueError("At least one of show_d1, show_d2, or show_chi must be True")
        
        # Set default signals if not provided
        if show_d1 and d1_signal is None:
            d1_signal = self.d1_zsc
        if show_d2 and d2_signal is None:
            d2_signal = self.d2_zsc
        if show_chi and chi_signal is None:
            chi_signal = self.chi_zsc
        
        # Use provided title or generate default
        plot_title = title or default_title
        
        # Calculate average signals around the event for each cell type
        C_avg_mean_d1 = None
        C_avg_mean_d2 = None
        C_avg_mean_chi = None
        
        if show_d1:
            [C_avg_d1, _, _, C_avg_mean_d1, _] = self.average_across_indices(indices, signal=d1_signal, cut_interval=cut_interval)
            
        if show_d2:
            [_, _, _, C_avg_mean_d2, _] = self.average_across_indices(indices, signal=d2_signal, cut_interval=cut_interval)
            
        if show_chi:
            [_, _, _, C_avg_mean_chi, _] = self.average_across_indices(indices, signal=chi_signal, cut_interval=cut_interval)
        
        # Always include velocity
        [_, _, _, velocity_mean, _] = self.average_across_indices(indices, signal=self.smoothed_velocity, cut_interval=cut_interval)
        
        # Create time axis in seconds
        time_axis = (np.array(range(2 * cut_interval)) - cut_interval) / self.ci_rate
        
        # Calculate optimal y-axis ranges for calcium signals with padding
        # Collect all the signals we're going to plot
        all_ca_signals = []
        if show_d1:
            all_ca_signals.append(C_avg_mean_d1)
        if show_d2:
            all_ca_signals.append(C_avg_mean_d2)
        if show_chi:
            all_ca_signals.append(C_avg_mean_chi)
        
        # Calculate min and max across all cell types
        ca_min = min(signal.min() for signal in all_ca_signals)
        ca_max = max(signal.max() for signal in all_ca_signals)
        
        # Add padding to ensure the signals are well-positioned in the plot (10% padding)
        ca_range = ca_max - ca_min
        ca_min = ca_min - (ca_range * 0.1)
        ca_max = ca_max + (ca_range * 0.1)
        
        # Create plot with optimized y-axis for calcium signals
        p = figure(width=800, height=400, active_scroll="wheel_zoom", title=plot_title,
                   x_axis_label='Time (s)', y_axis_label='Calcium Signal (ΔF/F)',
                   y_range=Range1d(ca_min, ca_max))
        
        # Add secondary y-axis for velocity with its own optimized range
        vel_range = velocity_mean.max() - velocity_mean.min()
        vel_min = velocity_mean.min() - (vel_range * 0.1)
        vel_max = velocity_mean.max() + (vel_range * 0.1)
        
        p.extra_y_ranges = {"velocity_axis": Range1d(start=vel_min, end=vel_max)}
        p.add_layout(LinearAxis(y_range_name="velocity_axis", axis_label="Velocity"), 'right')
        
        # Add calcium traces and velocity to plot (without adding to legend)
        legend_items = []
        
        if show_d1:
            d1_line = p.line(time_axis, C_avg_mean_d1, line_width=3, color='navy', alpha=0.6)
            legend_items.append(("D1", [d1_line]))
        
        if show_d2:
            d2_line = p.line(time_axis, C_avg_mean_d2, line_width=3, color='red', alpha=0.6)
            legend_items.append(("D2", [d2_line]))
        
        if show_chi:
            chi_line = p.line(time_axis, C_avg_mean_chi, line_width=3, color='green', alpha=0.6)
            legend_items.append(("CHI", [chi_line]))
        
        vel_line = p.line(time_axis, velocity_mean, line_width=3, color='gray', 
                         y_range_name="velocity_axis", alpha=0.6)
        legend_items.append(("Velocity", [vel_line]))
        
        # Create a legend outside the plot
        legend = Legend(
            items=legend_items,
            location="center",
            orientation="horizontal",
            click_policy="hide"
        )
        
        # Add the legend to the top of the plot (outside the plot area)
        p.add_layout(legend, 'above')
        
        # Style the plot
        p.xgrid.grid_line_color = None
        p.ygrid.grid_line_color = None
        
        # Add vertical line at time=0 (the event)
        p.line([0, 0], [ca_min, ca_max], line_width=1, line_dash='dashed', color='black', alpha=0.7)
        
        # Output the plot
        cell_types = []
        if show_d1: cell_types.append("D1")
        if show_d2: cell_types.append("D2")
        if show_chi: cell_types.append("CHI")
        cell_type_str = "_".join(cell_types)
        
        output_title = f"{plot_title} ({cell_type_str})" if cell_types else plot_title
        self.output_bokeh_plot(p, save_path=save_path, title=output_title, notebook=notebook, overwrite=overwrite)
        
        return p

    def plot_average_celltype_signals(self, show_d1=True, show_d2=True, show_chi=True, 
                                    show_velocity=True, show_movement_onset=True,
                                    d1_color='skyblue', d2_color='salmon', chi_color='limegreen',
                                    velocity_color='plum', movement_onset_color='green',
                                    x_range=(0, 60), y_range=(-0.05, 1.5), width=800, height=400,
                                    split_plots=False, 
                                    save_path=None, title=None, notebook=False, overwrite=False):
        """
        Plot average signals for different cell types (D1, D2, CHI) along with behavioral markers.
        
        Parameters:
        -----------
        show_d1 : bool
            Whether to show D1 neuron signals
        show_d2 : bool
            Whether to show D2 neuron signals
        show_chi : bool
            Whether to show CHI neuron signals
        show_velocity : bool
            Whether to show velocity signal
        show_movement_onset : bool
            Whether to show movement onset markers
        d1_color : str
            Color for D1 signal
        d2_color : str
            Color for D2 signal
        chi_color : str
            Color for CHI signal
        velocity_color : str
            Color for velocity signal
        movement_onset_color : str
            Color for movement onset markers
        y_range : tuple
            Y-axis range for the plot
        width : int
            Width of the plot
        height : int
            Height of the plot
        split_plots : bool
            Whether to split neural signals and velocity into two separate plots
        save_path : str, optional
            Path to save the plot as an HTML file. If None, will use the default path
        title : str, optional
            Title for the plot. If None, will use a default title
        notebook : bool
            Flag to indicate if the plot is for a Jupyter notebook
        overwrite : bool
            Flag to indicate whether to overwrite existing file
        
        Returns:
        --------
        bokeh.plotting.figure or bokeh.layouts.layout
            The created plot or layout
        """
        from bokeh.plotting import figure
        from bokeh.layouts import column
        
        # Compute normalized average signals
        d1_avg = np.mean(self.d1_zsc, axis=0) if show_d1 else None
        d2_avg = np.mean(self.d2_zsc, axis=0) if show_d2 else None
        chi_avg = np.mean(self.chi_zsc, axis=0) if show_chi else None
        
        # Set default title
        if title is None:
            cell_types = []
            if show_d1: cell_types.append("D1")
            if show_d2: cell_types.append("D2")
            if show_chi: cell_types.append("CHI")
            cell_type_str = " and ".join(cell_types)
            title = f"{cell_type_str} Average Signals"
        
        if split_plots and show_velocity:
            # Create two separate plots with shared x-range
            # Neural signals plot
            p_neurons = figure(width=width, height=height, x_range=x_range, title=title,
                       output_backend="webgl", tools="pan,box_zoom,wheel_zoom,reset,save")
            
            # Add cell type signals to neural plot
            if show_d1:
                p_neurons.line(self.t, d1_avg, color=d1_color, alpha=0.7,
                       line_width=2, legend_label='D1')
            
            if show_d2:
                p_neurons.line(self.t, d2_avg, color=d2_color, alpha=0.7,
                       line_width=2, legend_label='D2')
            
            if show_chi:
                p_neurons.line(self.t, chi_avg, color=chi_color, alpha=0.7,
                       line_width=2, legend_label='CHI')
                
            # Configure the legend for neural plot
            p_neurons.legend.click_policy = 'hide'
            
            # Add neural axis labels
            p_neurons.xaxis.axis_label = "Time (s)"
            p_neurons.yaxis.axis_label = "Neural Signal (z-score)"
            
            # Configure the grid for neural plot
            p_neurons.xgrid.grid_line_color = "lightgray"
            p_neurons.xgrid.grid_line_alpha = 0.4
            p_neurons.ygrid.grid_line_color = "lightgray"
            p_neurons.ygrid.grid_line_alpha = 0.4
            
            # Velocity plot with shared x-range
            p_vel = figure(width=width, height=height//2, x_range=p_neurons.x_range,
                       output_backend="webgl", tools="pan,box_zoom,wheel_zoom,reset,save")
            
            # Add velocity signal
            p_vel.line(self.t, self.smoothed_velocity, color=velocity_color, 
                   line_width=2, legend_label='Velocity')
            
            # Add movement onset markers to velocity plot
            if show_movement_onset:
                p_vel.multi_line(
                    xs=[[x/self.ci_rate, x/self.ci_rate] for x in self.movement_onset_indices], 
                    ys=[[0, 50] for _ in self.movement_onset_indices], 
                    line_color=movement_onset_color,
                    alpha=0.5,
                    line_width=2,
                    legend_label="Movement Onset"
                )
            
            # Configure the legend for velocity plot
            p_vel.legend.click_policy = 'hide'
            
            # Add velocity axis labels
            p_vel.xaxis.axis_label = "Time (s)"
            p_vel.yaxis.axis_label = "Velocity"
            
            # Configure the grid for velocity plot
            p_vel.xgrid.grid_line_color = "lightgray"
            p_vel.xgrid.grid_line_alpha = 0.4
            p_vel.ygrid.grid_line_color = "lightgray"
            p_vel.ygrid.grid_line_alpha = 0.4
            
            # Create a column layout
            layout = column(p_neurons, p_vel)
            
            # Set default save path if none provided
            if save_path is None:
                cell_types_filename = "_".join(
                    [ct for show, ct in [(show_d1, "D1"), (show_d2, "D2"), (show_chi, "CHI")] if show]
                )
                save_path = os.path.join(self.config["SummaryPlotsPath"], 
                                       self.session_name, 
                                       f'Average{cell_types_filename}Signals_Split.html')
            
            # Output the layout
            self.output_bokeh_plot(layout, save_path=save_path, title=title, 
                                  notebook=notebook, overwrite=overwrite)
            
            return layout
        else:
            # Original single plot implementation
            p = figure(width=width, height=height, x_range=x_range, title=title,
                     output_backend="webgl", tools="pan,box_zoom,wheel_zoom,reset,save")
            
            # Add cell type signals
            if show_d1:
                p.line(self.t, d1_avg, color=d1_color, alpha=0.7,
                      line_width=2, legend_label='D1')
            
            if show_d2:
                p.line(self.t, d2_avg, color=d2_color, alpha=0.7,
                      line_width=2, legend_label='D2')
            
            if show_chi:
                p.line(self.t, chi_avg, color=chi_color, alpha=0.7,
                      line_width=2, legend_label='CHI')
            
            # Add velocity signal
            if show_velocity:
                p.line(self.t, self.smoothed_velocity, color=velocity_color, 
                      line_width=2, legend_label='Velocity')
            
            # Add movement onset markers
            if show_movement_onset and hasattr(self, 'movement_onset_indices'):
                p.multi_line(
                    xs=[[x/self.ci_rate, x/self.ci_rate] for x in self.movement_onset_indices], 
                    ys=[[y_range[0], y_range[1]] for _ in self.movement_onset_indices], 
                    line_color=movement_onset_color,
                    alpha=0.5,
                    line_width=2,
                    legend_label="Movement Onset"
                )
            
            # Configure the legend
            p.legend.click_policy = 'hide'
            
            # Add axis labels
            p.xaxis.axis_label = "Time (s)"
            p.yaxis.axis_label = "Normalized Signal"
            
            # Configure the grid
            p.xgrid.grid_line_color = "lightgray"
            p.xgrid.grid_line_alpha = 0.4
            p.ygrid.grid_line_color = "lightgray"
            p.ygrid.grid_line_alpha = 0.4
            
            # Set default save path if none provided
            if save_path is None:
                cell_types_filename = "_".join(
                    [ct for show, ct in [(show_d1, "D1"), (show_d2, "D2"), (show_chi, "CHI")] if show]
                )
                save_path = os.path.join(self.config["SummaryPlotsPath"], 
                                       self.session_name, 
                                       f'Average{cell_types_filename}Signals.html')
            
            # Output the plot
            self.output_bokeh_plot(p, save_path=save_path, title=title, 
                                  notebook=notebook, overwrite=overwrite)
            
            return p

    def _compare_neuron_peaks_with_signal(self, signal=None, signal_name="velocity", 
                                     d1_peak_indices=None, d2_peak_indices=None, chi_peak_indices=None,
                                     show_d1=True, show_d2=True, show_chi=True,
                                     add_kde_contours=True, add_population_means=True,
                                     split_scatter_plots=True,
                                     save_path=None, title=None, notebook=False, overwrite=False):
        """
        Compare peak activity of different neuron types (D1, D2, CHI) with a specified signal.
        Creates visualizations including histograms of signal values at neural peaks,
        firing rate vs. mean signal scatter plots, and signal distribution box plots.
        
        Parameters:
        -----------
        signal : numpy.ndarray, optional
            Signal to analyze (e.g., self.smoothed_velocity, self.acceleration).
            If None, defaults to self.smoothed_velocity.
        signal_name : str
            Name of signal for plot labels (e.g., "Velocity", "Acceleration")
        d1_peak_indices : list of arrays, optional
            List of arrays containing peak indices for each D1 neuron.
            If None and show_d1 is True, will use self.peak_indices for D1 neurons.
        d2_peak_indices : list of arrays, optional
            List of arrays containing peak indices for each D2 neuron.
            If None and show_d2 is True, will use self.peak_indices for D2 neurons.
        chi_peak_indices : list of arrays, optional
            List of arrays containing peak indices for each CHI neuron.
            If None and show_chi is True, will use self.peak_indices for CHI neurons.
        show_d1 : bool
            Whether to include D1 neurons in the analysis
        show_d2 : bool
            Whether to include D2 neurons in the analysis
        show_chi : bool
            Whether to include CHI neurons in the analysis
        add_kde_contours : bool
            Whether to add KDE contours to show population clusters
        add_population_means : bool
            Whether to add population mean markers with crosshairs
        split_scatter_plots : bool
            Whether to create separate side-by-side scatter plots for each cell type
        save_path : str, optional
            Path to save the plot as an HTML file
        title : str, optional
            Title for the plot
        notebook : bool
            Flag to indicate if the plot is for a Jupyter notebook
        overwrite : bool
            Flag to indicate whether to overwrite existing file
            
        Returns:
        --------
        list
            List of bokeh figures created
        """
        import numpy as np
        from bokeh.plotting import figure
        from bokeh.layouts import column, row
        from bokeh.models import ColumnDataSource, Span
        from scipy import stats
        from skimage import measure
        
        # Use default signal if none provided
        if signal is None:
            signal = self.smoothed_velocity
        
        # Check if at least one cell type is selected to show
        if not (show_d1 or show_d2 or show_chi):
            raise ValueError("At least one cell type (show_d1, show_d2, or show_chi) must be set to True")
        
        # Use pre-computed peak indices for each cell type if not provided
        if show_d1 and d1_peak_indices is None:
            d1_peak_indices = [self.peak_indices[i] for i in self.d1_indices]
            
        if show_d2 and d2_peak_indices is None:
            d2_peak_indices = [self.peak_indices[i] for i in self.d2_indices]
            
        if show_chi and chi_peak_indices is None:
            chi_peak_indices = [self.peak_indices[i] for i in self.chi_indices]
        
        # Get signal values corresponding to each peak for each neuron type
        d1_peak_signals = []
        d2_peak_signals = []
        chi_peak_signals = []
        
        if show_d1 and d1_peak_indices is not None:
            d1_peak_signals = [signal[peaks] for peaks in d1_peak_indices]
        
        if show_d2 and d2_peak_indices is not None:
            d2_peak_signals = [signal[peaks] for peaks in d2_peak_indices]
        
        if show_chi and chi_peak_indices is not None:
            chi_peak_signals = [signal[peaks] for peaks in chi_peak_indices]
        
        
        # Calculate total recording time in seconds
        total_time = len(signal) / self.ci_rate
        
        # Calculate firing rates for each neuron
        d1_firing_rates = []
        d2_firing_rates = []
        chi_firing_rates = []
        
        if show_d1 and d1_peak_indices is not None:
            d1_firing_rates = [len(peaks)/total_time for peaks in d1_peak_indices]  # Hz
        
        if show_d2 and d2_peak_indices is not None:
            d2_firing_rates = [len(peaks)/total_time for peaks in d2_peak_indices]  # Hz
        
        if show_chi and chi_peak_indices is not None:
            chi_firing_rates = [len(peaks)/total_time for peaks in chi_peak_indices]  # Hz
        
        # Calculate mean firing rates and SEM
        if d1_firing_rates:
            d1_mean_rate = np.mean(d1_firing_rates)
            d1_rate_sem = np.std(d1_firing_rates) / np.sqrt(len(d1_firing_rates))
            print(f"\nD1 Neurons (n={len(d1_peak_indices)}):")
            print(f"- Mean firing rate: {d1_mean_rate:.2f} ± {d1_rate_sem:.2f} Hz")
            print(f"- Range: {np.min(d1_firing_rates):.2f} to {np.max(d1_firing_rates):.2f} Hz")
        
        if d2_firing_rates:
            d2_mean_rate = np.mean(d2_firing_rates)
            d2_rate_sem = np.std(d2_firing_rates) / np.sqrt(len(d2_firing_rates))
            print(f"\nD2 Neurons (n={len(d2_peak_indices)}):")
            print(f"- Mean firing rate: {d2_mean_rate:.2f} ± {d2_rate_sem:.2f} Hz")
            print(f"- Range: {np.min(d2_firing_rates):.2f} to {np.max(d2_firing_rates):.2f} Hz")

        if chi_firing_rates:
            chi_mean_rate = np.mean(chi_firing_rates)
            chi_rate_sem = np.std(chi_firing_rates) / np.sqrt(len(chi_firing_rates))
            print(f"\nCHI Neurons (n={len(chi_peak_indices)}):")
            print(f"- Mean firing rate: {chi_mean_rate:.2f} ± {chi_rate_sem:.2f} Hz")
            print(f"- Range: {np.min(chi_firing_rates):.2f} to {np.max(chi_firing_rates):.2f} Hz")
        
        # Create histogram plot
        p1 = figure(width=800, height=400, title=f"Normalized Distribution of {signal_name.capitalize()} at Neural Peaks")
        # Remove grid
        p1.grid.grid_line_color = None
        
        # Combine all signals to determine overall range
        all_signals = np.array([])
        
        if show_d1 and len(d1_peak_signals) > 0:
            all_signals = np.concatenate([all_signals, np.concatenate(d1_peak_signals)]) if all_signals.size > 0 else np.concatenate(d1_peak_signals)
        
        if show_d2 and len(d2_peak_signals) > 0:
            all_signals = np.concatenate([all_signals, np.concatenate(d2_peak_signals)]) if all_signals.size > 0 else np.concatenate(d2_peak_signals)
        
        if show_chi and len(chi_peak_signals) > 0:
            all_signals = np.concatenate([all_signals, np.concatenate(chi_peak_signals)]) if all_signals.size > 0 else np.concatenate(chi_peak_signals)
        
        # If there are no signals to analyze, return early with a warning
        if all_signals.size == 0:
            print(f"Warning: No peaks found for the selected cell types with {signal_name} data.")
            return []
            
        signal_range = (np.min(all_signals), np.max(all_signals))
        
        # Create and plot histograms for each cell type
        if show_d1 and len(d1_peak_signals) > 0:
            hist1, edges1 = np.histogram(np.concatenate(d1_peak_signals), bins=10, 
                                        range=signal_range, density=True)
            p1.quad(top=hist1, bottom=0, left=edges1[:-1], right=edges1[1:],
                    fill_color='navy', alpha=0.5, line_color='navy', legend_label='D1')
        
        if show_d2 and len(d2_peak_signals) > 0:
            hist2, edges2 = np.histogram(np.concatenate(d2_peak_signals), bins=10,
                                        range=signal_range, density=True)
            p1.quad(top=hist2, bottom=0, left=edges2[:-1], right=edges2[1:],
                    fill_color='crimson', alpha=0.5, line_color='crimson', legend_label='D2')
        
        if show_chi and len(chi_peak_signals) > 0:
            hist3, edges3 = np.histogram(np.concatenate(chi_peak_signals), bins=10,
                                        range=signal_range, density=True)
            p1.quad(top=hist3, bottom=0, left=edges3[:-1], right=edges3[1:],
                    fill_color='green', alpha=0.5, line_color='green', legend_label='CHI')
        
        # Customize histogram plot
        p1.xaxis.axis_label = signal_name.capitalize()
        p1.yaxis.axis_label = 'Density'
        p1.legend.location = "top_right"
        p1.legend.click_policy = "hide"
        
        # Create line plot version of the histogram for better visibility when histograms overlap
        p1_line = figure(width=800, height=400, 
                       title=f"Distribution of {signal_name.capitalize()} at Neural Peaks (Line Plot)",
                       x_axis_label=signal_name.capitalize(),
                       y_axis_label='Density')
        
        # Remove grid
        p1_line.grid.grid_line_color = None
        
        # Create line versions of histograms using the same data and bins
        if show_d1 and len(d1_peak_signals) > 0:
            hist1_line, edges1_line = np.histogram(np.concatenate(d1_peak_signals), bins=30, 
                                                 range=signal_range, density=True)
            # Convert bin edges to centers for line plot
            centers1 = (edges1_line[:-1] + edges1_line[1:]) / 2
            p1_line.line(centers1, hist1_line, line_width=3, color='navy', 
                        alpha=0.8, legend_label='D1')
            p1_line.scatter(centers1, hist1_line, size=5, color='navy', alpha=0.6)
        
        if show_d2 and len(d2_peak_signals) > 0:
            hist2_line, edges2_line = np.histogram(np.concatenate(d2_peak_signals), bins=30, 
                                                 range=signal_range, density=True)
            centers2 = (edges2_line[:-1] + edges2_line[1:]) / 2
            p1_line.line(centers2, hist2_line, line_width=3, color='crimson', 
                        alpha=0.8, legend_label='D2')
            p1_line.scatter(centers2, hist2_line, size=5, color='crimson', alpha=0.6)
        
        if show_chi and len(chi_peak_signals) > 0:
            hist3_line, edges3_line = np.histogram(np.concatenate(chi_peak_signals), bins=30, 
                                                 range=signal_range, density=True)
            centers3 = (edges3_line[:-1] + edges3_line[1:]) / 2
            p1_line.line(centers3, hist3_line, line_width=3, color='green', 
                        alpha=0.8, legend_label='CHI')
            p1_line.scatter(centers3, hist3_line, size=5, color='green', alpha=0.6)
        
        # Configure legend
        p1_line.legend.location = "top_right"
        p1_line.legend.click_policy = "hide"
        
        # Create line plot for peak count distribution
        peak_counts_fig = figure(width=800, height=400, 
                                title=f"Distribution of Peak Counts for {signal_name.capitalize()} Analysis",
                                x_axis_label="Number of Peaks per Neuron",
                                y_axis_label="Count")
        
        # Remove grid for cleaner look
        peak_counts_fig.grid.grid_line_color = None
        
        # Calculate peak counts for each cell type
        d1_counts = [len(peaks) for peaks in d1_peak_indices] if show_d1 and d1_peak_indices is not None else []
        d2_counts = [len(peaks) for peaks in d2_peak_indices] if show_d2 and d2_peak_indices is not None else []
        chi_counts = [len(peaks) for peaks in chi_peak_indices] if show_chi and chi_peak_indices is not None else []
        
        # Find the overall range for bins
        all_counts = []
        if d1_counts: all_counts.extend(d1_counts)
        if d2_counts: all_counts.extend(d2_counts)
        if chi_counts: all_counts.extend(chi_counts)
        
        if all_counts:
            # Calculate histogram bins
            min_count = min(all_counts)
            max_count = max(all_counts)
            bins = np.linspace(min_count, max_count, 30)  # 30 bins for smooth curve
            
            # Draw line plots for each cell type
            if d1_counts:
                hist, edges = np.histogram(d1_counts, bins=bins)
                # Convert bin edges to centers for line plot
                centers = (edges[:-1] + edges[1:]) / 2
                peak_counts_fig.line(centers, hist, line_width=3, color='navy', 
                                    alpha=0.7, legend_label='D1')
                # Add scatter points for more visibility
                peak_counts_fig.scatter(centers, hist, size=6, color='navy', alpha=0.7)
            
            if d2_counts:
                hist, edges = np.histogram(d2_counts, bins=bins)
                centers = (edges[:-1] + edges[1:]) / 2
                peak_counts_fig.line(centers, hist, line_width=3, color='crimson', 
                                   alpha=0.7, legend_label='D2')
                peak_counts_fig.scatter(centers, hist, size=6, color='crimson', alpha=0.7)
            
            if chi_counts:
                hist, edges = np.histogram(chi_counts, bins=bins)
                centers = (edges[:-1] + edges[1:]) / 2
                peak_counts_fig.line(centers, hist, line_width=3, color='green', 
                                   alpha=0.7, legend_label='CHI')
                peak_counts_fig.scatter(centers, hist, size=6, color='green', alpha=0.7)
            
            # Configure legend
            peak_counts_fig.legend.location = "top_right"
            peak_counts_fig.legend.click_policy = "hide"
        
        # Calculate mean signal for each neuron's peaks
        d1_mean_signals = []
        d2_mean_signals = []
        chi_mean_signals = []
        
        if show_d1 and len(d1_peak_signals) > 0:
            d1_mean_signals = [np.mean(signals) for signals in d1_peak_signals]
        
        if show_d2 and len(d2_peak_signals) > 0:
            d2_mean_signals = [np.mean(signals) for signals in d2_peak_signals]
        
        if show_chi and len(chi_peak_signals) > 0:
            chi_mean_signals = [np.mean(signals) for signals in chi_peak_signals]
        
        # Define function to add KDE contours
        def add_kde_contours(plot, x, y, color, num_levels=3, alpha=0.05, bandwidth=None, cell_type=None):
            from scipy.stats import gaussian_kde
            
            # Need at least 5 points for a reasonable KDE
            if len(x) < 5:
                return
            
            # Create a grid over which to evaluate the KDE
            x_min, x_max = min(x), max(x)
            y_min, y_max = min(y), max(y)
            
            # Add some margin
            x_margin = (x_max - x_min) * 0.1
            y_margin = (y_max - y_min) * 0.1
            
            x_grid = np.linspace(x_min - x_margin, x_max + x_margin, 100)
            y_grid = np.linspace(y_min - y_margin, y_max + y_margin, 100)
            xx, yy = np.meshgrid(x_grid, y_grid)
            
            # Stack the meshgrid to position vectors
            positions = np.vstack([xx.ravel(), yy.ravel()])
            
            # Get the values and reshape them
            try:
                # Apply custom bandwidth for CHI neurons
                if cell_type == 'CHI':
                    # Use a larger bandwidth for CHI to get smoother contours
                    bw_method = bandwidth if bandwidth is not None else 0.5
                    kernel = gaussian_kde(np.vstack([x, y]), bw_method=bw_method)
                    # Increase contour levels for smoother appearance
                    num_levels = 5
                    alpha = 0.07  # Slightly higher alpha for better visibility
                else:
                    # Default for other cell types
                    kernel = gaussian_kde(np.vstack([x, y]), bw_method=bandwidth)
                
                zz = np.reshape(kernel(positions), xx.shape)
                
                # Compute contour levels
                levels = np.linspace(zz.min(), zz.max(), num_levels+2)[1:-1]
                
                # Plot contours
                for level in levels:
                    # Find contours at this level
                    contours = measure.find_contours(zz, level)
                    
                    # Plot each contour as a patch
                    for contour in contours:
                        # Scale contour back to original coordinate system
                        scaled_contour_x = contour[:, 1] * (x_max - x_min + 2*x_margin) / 100 + (x_min - x_margin)
                        scaled_contour_y = contour[:, 0] * (y_max - y_min + 2*y_margin) / 100 + (y_min - y_margin)
                        
                        plot.patch(scaled_contour_x, scaled_contour_y, 
                                  fill_color=color, fill_alpha=alpha, line_color=color, line_alpha=0.2)
            except:
                # If KDE fails (e.g., not enough unique points), just skip it
                pass
        
        # Create scatter plots
        scatter_plots = []
        
        if split_scatter_plots:
            # Create separate side-by-side scatter plots for each cell type
            width = 400  # Half the original width
            height = 400
            
            # Determine which types to show
            types_to_show = []
            if show_d1 and len(d1_mean_signals) > 0:
                types_to_show.append(('D1', 'navy', d1_firing_rates, d1_mean_signals))
            if show_d2 and len(d2_mean_signals) > 0:
                types_to_show.append(('D2', 'crimson', d2_firing_rates, d2_mean_signals))
            if show_chi and len(chi_mean_signals) > 0:
                types_to_show.append(('CHI', 'green', chi_firing_rates, chi_mean_signals))
            
            # Calculate common x and y ranges for all plots
            all_rates = []
            all_means = []
            for _, _, rates, means in types_to_show:
                all_rates.extend(rates)
                all_means.extend(means)
                
            # Add padding to ranges (10%)
            if all_rates and all_means:
                min_rate, max_rate = min(all_rates), max(all_rates)
                min_mean, max_mean = min(all_means), max(all_means)
                
                rate_range = max_rate - min_rate
                mean_range = max_mean - min_mean
                
                x_min = min_rate - (rate_range * 0.1)
                x_max = max_rate + (rate_range * 0.1)
                y_min = min_mean - (mean_range * 0.1)
                y_max = max_mean + (mean_range * 0.1)
            
            # Create a plot for each type that needs to be shown
            for i, (label, color, rates, means) in enumerate(types_to_show):
                # Create figure
                title_label = f"{label} Neurons"
                p_scatter = figure(width=width, height=height, 
                                 title=title_label,
                                 tools="pan,box_zoom,wheel_zoom,reset,save")
                
                # If not the first plot, share y-range with first plot
                if i > 0:
                    p_scatter.y_range = scatter_plots[0].y_range
                    p_scatter.x_range = scatter_plots[0].x_range
                elif all_rates and all_means:
                    # For first plot, set the calculated ranges
                    p_scatter.x_range.start = x_min
                    p_scatter.x_range.end = x_max
                    p_scatter.y_range.start = y_min
                    p_scatter.y_range.end = y_max
                
                # Remove grid
                p_scatter.grid.grid_line_color = None
                
                # Create data source
                source = ColumnDataSource(data=dict(
                    x=rates,
                    y=means
                ))
                
                # Plot scatter points
                p_scatter.scatter('x', 'y', source=source, color=color, size=8, alpha=0.6)
                
                # Add KDE contours if requested
                if add_kde_contours and len(rates) > 5:
                    add_kde_contours(p_scatter, rates, means, color, cell_type=label)
                
                # Add population mean markers with crosshairs if requested
                if add_population_means:
                    mean_x = np.mean(rates)
                    mean_y = np.mean(means)
                    
                    # Add mean marker
                    p_scatter.scatter([mean_x], [mean_y], color=color, size=15, alpha=1.0, 
                                  marker='diamond', line_width=2, line_color='white')
                    
                    # Add crosshair lines
                    p_scatter.line([mean_x, mean_x], [p_scatter.y_range.start, p_scatter.y_range.end], 
                               line_color=color, line_width=1, line_dash='dashed', alpha=0.5)
                    p_scatter.line([p_scatter.x_range.start, p_scatter.x_range.end], [mean_y, mean_y], 
                               line_color=color, line_width=1, line_dash='dashed', alpha=0.5)
                    
                    # Add text annotation
                    p_scatter.text([mean_x], [mean_y], [f"{label} Mean"], text_font_size='8pt',
                               text_color=color, x_offset=10, y_offset=10)
                
                # Add correlation coefficient
                if len(rates) > 1:
                    corr = np.corrcoef(rates, means)[0,1]
                    p_scatter.text([p_scatter.x_range.start + (p_scatter.x_range.end - p_scatter.x_range.start)*0.05], 
                               [p_scatter.y_range.start + (p_scatter.y_range.end - p_scatter.y_range.start)*0.95],
                               [f"r = {corr:.3f}"],
                               text_font_size='8pt', text_font_style='bold', text_color=color)
                
                # Customize plot
                p_scatter.xaxis.axis_label = 'Firing Rate (Hz)'
                if i == 0:  # Only first plot gets y-axis label
                    p_scatter.yaxis.axis_label = f'Mean {signal_name.capitalize()} at Peaks'
                
                scatter_plots.append(p_scatter)
            
            # Add t-test comparison if we have both D1 and D2
            if show_d1 and show_d2 and d1_mean_signals and d2_mean_signals:
                stat_result = stats.ttest_ind(d1_mean_signals, d2_mean_signals, equal_var=False)
                p_value = stat_result.pvalue
                print(f"\nD1 vs D2 mean {signal_name} t-test: p={p_value:.4f}")
        else:
            # Create a single scatter plot with all cell types
            p2 = figure(width=800, height=400, 
                       title=f"Firing Rate vs Mean {signal_name.capitalize()}: Population Comparison",
                       tools="pan,box_zoom,wheel_zoom,reset,save")
            # Remove grid
            p2.grid.grid_line_color = None
            
            # Add scatter plots for each cell type
            if show_d1 and len(d1_mean_signals) > 0:
                d1_source = ColumnDataSource(data=dict(
                    x=d1_firing_rates,
                    y=d1_mean_signals
                ))
                d1_scatter = p2.scatter('x', 'y', source=d1_source, 
                                      color='navy', size=8, alpha=0.6, legend_label='D1 Neurons')
                
                # Add KDE contours if requested
                if add_kde_contours and len(d1_firing_rates) > 5:
                    add_kde_contours(p2, d1_firing_rates, d1_mean_signals, 'navy', cell_type='D1')
                
                # Add mean marker and crosshairs if requested
                if add_population_means:
                    d1_mean_x = np.mean(d1_firing_rates)
                    d1_mean_y = np.mean(d1_mean_signals)
                    
                    p2.scatter([d1_mean_x], [d1_mean_y], color='navy', size=15, alpha=1.0, 
                              marker='diamond', line_width=2, line_color='white')
                    
                    p2.line([d1_mean_x, d1_mean_x], [p2.y_range.start, p2.y_range.end], 
                           line_color='navy', line_width=1, line_dash='dashed', alpha=0.5)
                    p2.line([p2.x_range.start, p2.x_range.end], [d1_mean_y, d1_mean_y], 
                           line_color='navy', line_width=1, line_dash='dashed', alpha=0.5)
                    
                    p2.text([d1_mean_x], [d1_mean_y], ['D1 Mean'], text_font_size='8pt',
                           text_color='navy', x_offset=10, y_offset=10)
            
            if show_d2 and len(d2_mean_signals) > 0:
                d2_source = ColumnDataSource(data=dict(
                    x=d2_firing_rates,
                    y=d2_mean_signals
                ))
                d2_scatter = p2.scatter('x', 'y', source=d2_source, 
                                      color='crimson', size=8, alpha=0.6, legend_label='D2 Neurons')
                
                # Add KDE contours if requested
                if add_kde_contours and len(d2_firing_rates) > 5:
                    add_kde_contours(p2, d2_firing_rates, d2_mean_signals, 'crimson', cell_type='D2')
                
                # Add mean marker and crosshairs if requested
                if add_population_means:
                    d2_mean_x = np.mean(d2_firing_rates)
                    d2_mean_y = np.mean(d2_mean_signals)
                    
                    p2.scatter([d2_mean_x], [d2_mean_y], color='crimson', size=15, alpha=1.0, 
                              marker='diamond', line_width=2, line_color='white')
                    
                    p2.line([d2_mean_x, d2_mean_x], [p2.y_range.start, p2.y_range.end], 
                           line_color='crimson', line_width=1, line_dash='dashed', alpha=0.5)
                    p2.line([p2.x_range.start, p2.x_range.end], [d2_mean_y, d2_mean_y], 
                           line_color='crimson', line_width=1, line_dash='dashed', alpha=0.5)
                    
                    p2.text([d2_mean_x], [d2_mean_y], ['D2 Mean'], text_font_size='8pt',
                           text_color='crimson', x_offset=10, y_offset=10)
            
            if show_chi and len(chi_mean_signals) > 0:
                chi_source = ColumnDataSource(data=dict(
                    x=chi_firing_rates,
                    y=chi_mean_signals
                ))
                chi_scatter = p2.scatter('x', 'y', source=chi_source, 
                                       color='green', size=8, alpha=0.6, legend_label='CHI Neurons')
                
                # Add KDE contours if requested
                if add_kde_contours and len(chi_firing_rates) > 5:
                    add_kde_contours(p2, chi_firing_rates, chi_mean_signals, 'green', cell_type='CHI')
                
                # Add mean marker and crosshairs if requested
                if add_population_means:
                    chi_mean_x = np.mean(chi_firing_rates)
                    chi_mean_y = np.mean(chi_mean_signals)
                    
                    p2.scatter([chi_mean_x], [chi_mean_y], color='green', size=15, alpha=1.0, 
                              marker='diamond', line_width=2, line_color='white')
                    
                    p2.line([chi_mean_x, chi_mean_x], [p2.y_range.start, p2.y_range.end], 
                           line_color='green', line_width=1, line_dash='dashed', alpha=0.5)
                    p2.line([p2.x_range.start, p2.x_range.end], [chi_mean_y, chi_mean_y], 
                           line_color='green', line_width=1, line_dash='dashed', alpha=0.5)
                    
                    p2.text([chi_mean_x], [chi_mean_y], ['CHI Mean'], text_font_size='8pt',
                           text_color='green', x_offset=10, y_offset=10)
            
            # Add statistical comparison (t-test) between D1 and D2 if both are shown
            if show_d1 and show_d2 and d1_mean_signals and d2_mean_signals:
                stat_result = stats.ttest_ind(d1_mean_signals, d2_mean_signals, equal_var=False)
                p_value = stat_result.pvalue
                
                # Add p-value annotation
                p2.text([p2.x_range.start + (p2.x_range.end - p2.x_range.start)*0.05], 
                       [p2.y_range.start + (p2.y_range.end - p2.y_range.start)*0.95],
                       [f"t-test: p = {p_value:.4f}"],
                       text_font_size='10pt', text_font_style='bold')
                
                print(f"\nD1 vs D2 mean {signal_name} t-test: p={p_value:.4f}")
            
            # Customize plot
            p2.xaxis.axis_label = 'Firing Rate (Hz)'
            p2.yaxis.axis_label = f'Mean {signal_name.capitalize()} at Peaks'
            p2.legend.location = "top_right"
            p2.legend.click_policy = "hide"  # Allow toggling visibility
            
            scatter_plots.append(p2)
        
        # Create box plots to show signal distributions
        p3 = figure(width=800, height=400, title=f"{signal_name.capitalize()} Distribution by Neuron Type")
        
        # Remove grid
        p3.grid.grid_line_color = None
        
        # Helper function to calculate box plot data
        def get_box_data(signals_list):
            all_signals = np.concatenate(signals_list)
            q1 = np.percentile(all_signals, 25)
            q2 = np.percentile(all_signals, 50)
            q3 = np.percentile(all_signals, 75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            return lower, q1, q2, q3, upper
        
        # Helper function to plot box
        def plot_box(p, x_center, box_data, width=0.1, color='navy'):
            lower, q1, q2, q3, upper = box_data
            # Box
            p.vbar(x=x_center, width=width, top=q3, bottom=q1, fill_color=color, alpha=0.5)
            # Median line
            p.line([x_center-width/2, x_center+width/2], [q2, q2], line_color='black', line_width=2)
            # Whiskers
            p.line([x_center, x_center], [lower, q1], line_color='black')
            p.line([x_center, x_center], [q3, upper], line_color='black')
            # Caps on whiskers
            p.line([x_center-width/4, x_center+width/4], [lower, lower], line_color='black')
            p.line([x_center-width/4, x_center+width/4], [upper, upper], line_color='black')
        
        # Calculate x positions and box data based on which cell types are shown
        positions = []
        labels = {}
        cell_types_shown = []
        
        if show_d1 and len(d1_peak_signals) > 0:
            cell_types_shown.append(('D1', 'navy', d1_peak_signals))
        
        if show_d2 and len(d2_peak_signals) > 0:
            cell_types_shown.append(('D2', 'crimson', d2_peak_signals))
        
        if show_chi and len(chi_peak_signals) > 0:
            cell_types_shown.append(('CHI', 'green', chi_peak_signals))
        
        # Set spacing based on number of types shown
        num_types = len(cell_types_shown)
        spacing = 0.2
        start_pos = 1.0 + 0.5 * spacing * (num_types - 1)
        
        # Plot the boxes
        for i, (label, color, data) in enumerate(cell_types_shown):
            pos = start_pos + spacing * (i - (num_types - 1) / 2)
            positions.append(pos)
            labels[pos] = label
            box_data = get_box_data(data)
            plot_box(p3, pos, box_data, color=color)
        
        # Customize box plot
        p3.xaxis.ticker = positions
        p3.xaxis.major_label_overrides = labels
        p3.xaxis.axis_label = 'Neuron Type'
        p3.yaxis.axis_label = signal_name.capitalize()
        
        if positions:
            p3.x_range.start = positions[0] - 0.5
            p3.x_range.end = positions[-1] + 0.5
        
        # Create the layout based on the scatter plot configuration
        if split_scatter_plots and len(scatter_plots) > 1:
            # Create a row layout for the scatter plots
            scatter_row = row(scatter_plots)
            all_plots = column(p1, p1_line, peak_counts_fig, scatter_row, p3)
        else:
            # Create a column layout for single scatter plot or just one type
            all_plots = column(p1, p1_line, peak_counts_fig, *scatter_plots, p3)
            
        # Set default title if not provided
        if title is None:
            cell_types = [label for label, _, _ in cell_types_shown]
            title = f"Neural {signal_name.capitalize()} Analysis: {', '.join(cell_types)}"
        
        # Set default save path if none provided
        if save_path is None:
            cell_types_filename = "_".join([label for label, _, _ in cell_types_shown])
            save_path = os.path.join(self.config["SummaryPlotsPath"], 
                                     self.session_name, 
                                     f'{cell_types_filename}_{signal_name.capitalize()}_Analysis.html')
        
        # Output the plot
        self.output_bokeh_plot(all_plots, save_path=save_path, title=title, 
                              notebook=notebook, overwrite=overwrite)
        
        # Return all created plots
        plot_list = [p1, p1_line, peak_counts_fig]
        plot_list.extend(scatter_plots)
        plot_list.append(p3)
        
        return plot_list

    def compare_neuron_peaks_with_velocity(self, d1_peak_indices=None, d2_peak_indices=None, chi_peak_indices=None, 
                                           show_d1=True, show_d2=True, show_chi=True, 
                                           split_scatter_plots=True,
                                           save_path=None, title=None, 
                                           notebook=False, overwrite=False):
        """
        Compare peak activity of different neuron types (D1, D2, CHI) with velocity data.
        Creates visualizations including histograms of velocities at neural peaks,
        firing rate vs. mean velocity scatter plots, and velocity distribution box plots.
        
        This is a wrapper around _compare_neuron_peaks_with_signal for velocity analysis.
        
        Parameters:
        -----------
        d1_peak_indices : list of arrays, optional
            List of arrays containing peak indices for each D1 neuron.
            If None and show_d1 is True, will use self.peak_indices for D1 neurons.
        d2_peak_indices : list of arrays, optional
            List of arrays containing peak indices for each D2 neuron.
            If None and show_d2 is True, will use self.peak_indices for D2 neurons.
        chi_peak_indices : list of arrays, optional
            List of arrays containing peak indices for each CHI neuron.
            If None and show_chi is True, will use self.peak_indices for CHI neurons.
        show_d1 : bool
            Whether to include D1 neurons in the analysis
        show_d2 : bool
            Whether to include D2 neurons in the analysis
        show_chi : bool
            Whether to include CHI neurons in the analysis
        split_scatter_plots : bool
            Whether to create separate side-by-side scatter plots for each cell type
        save_path : str, optional
            Path to save the plot as an HTML file
        title : str, optional
            Title for the plot
        notebook : bool
            Flag to indicate if the plot is for a Jupyter notebook
        overwrite : bool
            Flag to indicate whether to overwrite existing file
            
        Returns:
        --------
        list
            List of bokeh figures created
        """
        return self._compare_neuron_peaks_with_signal(
            signal=self.smoothed_velocity, 
            signal_name="velocity",
            d1_peak_indices=d1_peak_indices,
            d2_peak_indices=d2_peak_indices,
            chi_peak_indices=chi_peak_indices,
            show_d1=show_d1,
            show_d2=show_d2,
            show_chi=show_chi,
            split_scatter_plots=split_scatter_plots,
            save_path=save_path,
            title=title,
            notebook=notebook,
            overwrite=overwrite
        )
    
    def compare_neuron_peaks_with_acceleration(self, d1_peak_indices=None, d2_peak_indices=None, chi_peak_indices=None, 
                                              show_d1=True, show_d2=True, show_chi=True, 
                                              split_scatter_plots=True,
                                              save_path=None, title=None, 
                                              notebook=False, overwrite=False):
        """
        Compare peak activity of different neuron types (D1, D2, CHI) with acceleration data.
        Creates visualizations including histograms of accelerations at neural peaks,
        firing rate vs. mean acceleration scatter plots, and acceleration distribution box plots.
        
        This is a wrapper around _compare_neuron_peaks_with_signal for acceleration analysis.
        
        Parameters:
        -----------
        d1_peak_indices : list of arrays, optional
            List of arrays containing peak indices for each D1 neuron.
            If None and show_d1 is True, will use self.peak_indices for D1 neurons.
        d2_peak_indices : list of arrays, optional
            List of arrays containing peak indices for each D2 neuron.
            If None and show_d2 is True, will use self.peak_indices for D2 neurons.
        chi_peak_indices : list of arrays, optional
            List of arrays containing peak indices for each CHI neuron.
            If None and show_chi is True, will use self.peak_indices for CHI neurons.
        show_d1 : bool
            Whether to include D1 neurons in the analysis
        show_d2 : bool
            Whether to include D2 neurons in the analysis
        show_chi : bool
            Whether to include CHI neurons in the analysis
        split_scatter_plots : bool
            Whether to create separate side-by-side scatter plots for each cell type
        save_path : str, optional
            Path to save the plot as an HTML file
        title : str, optional
            Title for the plot
        notebook : bool
            Flag to indicate if the plot is for a Jupyter notebook
        overwrite : bool
            Flag to indicate whether to overwrite existing file
            
        Returns:
        --------
        list
            List of bokeh figures created
        """
        return self._compare_neuron_peaks_with_signal(
            signal=self.acceleration,
            signal_name="acceleration",
            d1_peak_indices=d1_peak_indices,
            d2_peak_indices=d2_peak_indices,
            chi_peak_indices=chi_peak_indices,
            show_d1=show_d1,
            show_d2=show_d2,
            show_chi=show_chi,
            split_scatter_plots=split_scatter_plots,
            save_path=save_path,
            title=title,
            notebook=notebook,
            overwrite=overwrite
        )

    def create_cell_type_spike_visualizations(self, d1_peak_indices=None, d2_peak_indices=None, chi_peak_indices=None,
                                       show_d1=True, show_d2=True, show_chi=True, 
                                       save_path=None, title="Neural Spike Analysis", notebook=False, overwrite=False):
        """
        Create comprehensive visualizations of D1, D2, and CHI neural spike data.
        
        Parameters:
        -----------
        d1_peak_indices : list of arrays, optional
            List of peak indices for each D1 neuron. If None and show_d1 is True, will use self.peak_indices for D1 neurons.
        d2_peak_indices : list of arrays, optional
            List of peak indices for each D2 neuron. If None and show_d2 is True, will use self.peak_indices for D2 neurons.
        chi_peak_indices : list of arrays, optional
            List of peak indices for each CHI neuron. If None and show_chi is True, will use self.peak_indices for CHI neurons.
        show_d1 : bool
            Whether to include D1 neurons in the analysis
        show_d2 : bool
            Whether to include D2 neurons in the analysis
        show_chi : bool
            Whether to include CHI neurons in the analysis
        save_path : str, optional
            Path to save the visualization
        title : str
            Title for the visualization
        notebook : bool
            Flag to indicate if the visualization is for a Jupyter notebook
        overwrite : bool
            Flag to indicate whether to overwrite existing file
        
        Returns:
        --------
        bokeh.layouts.layout
            The created visualization layout
        """
        from bokeh.plotting import figure
        from bokeh.layouts import column, row
        from bokeh.models import ColumnDataSource, HoverTool, Legend, LegendItem
        
        # Check if at least one cell type is selected
        if not any([show_d1, show_d2, show_chi]):
            raise ValueError("At least one cell type (show_d1, show_d2, or show_chi) must be set to True")
            
        # Use pre-computed peak indices for each cell type if not provided
        if show_d1 and d1_peak_indices is None:
            d1_peak_indices = [self.peak_indices[i] for i in self.d1_indices]
            
        if show_d2 and d2_peak_indices is None:
            d2_peak_indices = [self.peak_indices[i] for i in self.d2_indices]
            
        if show_chi and chi_peak_indices is None:
            chi_peak_indices = [self.peak_indices[i] for i in self.chi_indices]
            
        def convert_to_seconds(indices):
            """Convert sample indices to seconds"""
            return np.array(indices) / self.ci_rate
        
        # Create figure for firing rate histogram
        p1 = figure(width=500, height=300, title='Distribution of Spike Counts per Neuron')
        
        # Remove grid
        p1.grid.grid_line_color = None
        
        # Calculate spike counts per neuron
        d1_counts = [len(peaks) for peaks in d1_peak_indices] if show_d1 and d1_peak_indices else []
        d2_counts = [len(peaks) for peaks in d2_peak_indices] if show_d2 and d2_peak_indices else []
        chi_counts = [len(peaks) for peaks in chi_peak_indices] if show_chi and chi_peak_indices else []
        
        # Collect all counts to determine overall range
        all_counts = []
        if d1_counts: all_counts.extend(d1_counts)
        if d2_counts: all_counts.extend(d2_counts)
        if chi_counts: all_counts.extend(chi_counts)
        
        if not all_counts:
            raise ValueError("No spikes found in any of the selected cell types")
            
        # Create histograms with shared bins
        min_count = min(all_counts)
        max_count = max(all_counts)
        shared_edges = np.linspace(min_count, max_count, 21)  # 20 bins
        
        # Create legend items list
        legend_items1 = []
        
        # Plot histograms for each cell type
        if show_d1 and d1_counts:
            hist1, _ = np.histogram(d1_counts, bins=shared_edges)
            d1_hist_plot = p1.quad(top=hist1, bottom=0, left=shared_edges[:-1], right=shared_edges[1:],
                    fill_color='navy', alpha=0.5, line_color='navy')
            legend_items1.append(LegendItem(label="D1", renderers=[d1_hist_plot]))
        
        if show_d2 and d2_counts:
            hist2, _ = np.histogram(d2_counts, bins=shared_edges)
            d2_hist_plot = p1.quad(top=hist2, bottom=0, left=shared_edges[:-1], right=shared_edges[1:],
                    fill_color='crimson', alpha=0.5, line_color='crimson')
            legend_items1.append(LegendItem(label="D2", renderers=[d2_hist_plot]))
            
        if show_chi and chi_counts:
            hist3, _ = np.histogram(chi_counts, bins=shared_edges)
            chi_hist_plot = p1.quad(top=hist3, bottom=0, left=shared_edges[:-1], right=shared_edges[1:],
                    fill_color='green', alpha=0.5, line_color='green')
            legend_items1.append(LegendItem(label="CHI", renderers=[chi_hist_plot]))
        
        # Create legend and place it outside the plot
        legend1 = Legend(items=legend_items1, location="center")
        p1.add_layout(legend1, 'right')
        
        # Customize first histogram
        p1.xaxis.axis_label = 'Number of Spikes per Neuron'
        p1.yaxis.axis_label = 'Count'
        p1.legend.click_policy = "hide"
        
        # Create figure for inter-spike interval histogram
        p2 = figure(width=500, height=300, title='Distribution of Inter-spike Intervals')
        
        # Remove grid
        p2.grid.grid_line_color = None
        
        # Calculate inter-spike intervals in seconds
        d1_isis = []
        d2_isis = []
        chi_isis = []
        
        if show_d1:
            for peaks in d1_peak_indices:
                if len(peaks) > 1:
                    isis = np.diff(np.sort(peaks)) / self.ci_rate  # Convert to seconds
                    d1_isis.extend(isis)
                    
        if show_d2:
            for peaks in d2_peak_indices:
                if len(peaks) > 1:
                    isis = np.diff(np.sort(peaks)) / self.ci_rate  # Convert to seconds
                    d2_isis.extend(isis)
                    
        if show_chi:
            for peaks in chi_peak_indices:
                if len(peaks) > 1:
                    isis = np.diff(np.sort(peaks)) / self.ci_rate  # Convert to seconds
                    chi_isis.extend(isis)
        
        # Collect all ISIs to determine overall range
        all_isis = []
        if d1_isis: all_isis.extend(d1_isis)
        if d2_isis: all_isis.extend(d2_isis)
        if chi_isis: all_isis.extend(chi_isis)
        
        if all_isis:
            # Create histograms for ISIs with shared bins
            min_isi = min(all_isis)
            max_isi = max(all_isis)
            shared_isi_edges = np.linspace(min_isi, max_isi, 31)  # 30 bins
            
            # Create legend items list
            legend_items2 = []
            
            # Plot ISI histograms for each cell type
            if d1_isis:
                hist_d1_isi, _ = np.histogram(d1_isis, bins=shared_isi_edges)
                d1_isi_plot = p2.quad(top=hist_d1_isi, bottom=0, left=shared_isi_edges[:-1], right=shared_isi_edges[1:],
                        fill_color='navy', alpha=0.5, line_color='navy')
                legend_items2.append(LegendItem(label="D1", renderers=[d1_isi_plot]))
            
            if d2_isis:
                hist_d2_isi, _ = np.histogram(d2_isis, bins=shared_isi_edges)
                d2_isi_plot = p2.quad(top=hist_d2_isi, bottom=0, left=shared_isi_edges[:-1], right=shared_isi_edges[1:],
                        fill_color='crimson', alpha=0.5, line_color='crimson')
                legend_items2.append(LegendItem(label="D2", renderers=[d2_isi_plot]))
                
            if chi_isis:
                hist_chi_isi, _ = np.histogram(chi_isis, bins=shared_isi_edges)
                chi_isi_plot = p2.quad(top=hist_chi_isi, bottom=0, left=shared_isi_edges[:-1], right=shared_isi_edges[1:],
                        fill_color='green', alpha=0.5, line_color='green')
                legend_items2.append(LegendItem(label="CHI", renderers=[chi_isi_plot]))
            
            # Create legend and place it outside the plot
            legend2 = Legend(items=legend_items2, location="center")
            p2.add_layout(legend2, 'right')
            
            # Customize second histogram
            p2.xaxis.axis_label = 'Inter-spike Interval (seconds)'
            p2.yaxis.axis_label = 'Count'
            p2.legend.click_policy = "hide"
        # Create raster plot
        p3 = figure(width=800, height=500, title='Neural Activity Patterns')
        
        # Remove grid
        p3.grid.grid_line_color = None
        
        # Initialize legend items for raster plot
        legend_items3 = []
        
        # Track vertical position offset
        y_offset = 0
        
        # Create data for D1 neurons with time in seconds
        if show_d1 and d1_peak_indices:
            d1_data = []
            for i, peaks in enumerate(d1_peak_indices):
                times = convert_to_seconds(peaks)
                y_values = [i + y_offset for _ in peaks]
                d1_data.extend(list(zip(times, y_values, [i+1]*len(peaks))))
            
            d1_source = ColumnDataSource(data=dict(
                x=[d[0] for d in d1_data],
                y=[d[1] for d in d1_data],
                neuron=[f"D1-{d[2]}" for d in d1_data]
            ))
            
            d1_scatter = p3.scatter('x', 'y', source=d1_source, color='navy', alpha=0.6, size=3)
            legend_items3.append(LegendItem(label="D1 Neurons", renderers=[d1_scatter]))
            
            # Update offset for next cell type
            y_offset += len(d1_peak_indices) + 2
        
        # Create data for D2 neurons with time in seconds
        if show_d2 and d2_peak_indices:
            d2_data = []
            for i, peaks in enumerate(d2_peak_indices):
                times = convert_to_seconds(peaks)
                y_values = [i + y_offset for _ in peaks]
                d2_data.extend(list(zip(times, y_values, [i+1]*len(peaks))))
            
            d2_source = ColumnDataSource(data=dict(
                x=[d[0] for d in d2_data],
                y=[d[1] for d in d2_data],
                neuron=[f"D2-{d[2]}" for d in d2_data]
            ))
            
            d2_scatter = p3.scatter('x', 'y', source=d2_source, color='crimson', alpha=0.6, size=3)
            legend_items3.append(LegendItem(label="D2 Neurons", renderers=[d2_scatter]))
            
            # Update offset for next cell type
            y_offset += len(d2_peak_indices) + 2
        
        # Create data for CHI neurons with time in seconds
        if show_chi and chi_peak_indices:
            chi_data = []
            for i, peaks in enumerate(chi_peak_indices):
                times = convert_to_seconds(peaks)
                y_values = [i + y_offset for _ in peaks]
                chi_data.extend(list(zip(times, y_values, [i+1]*len(peaks))))
            
            chi_source = ColumnDataSource(data=dict(
                x=[d[0] for d in chi_data],
                y=[d[1] for d in chi_data],
                neuron=[f"CHI-{d[2]}" for d in chi_data]
            ))
            
            chi_scatter = p3.scatter('x', 'y', source=chi_source, color='green', alpha=0.6, size=3)
            legend_items3.append(LegendItem(label="CHI Neurons", renderers=[chi_scatter]))
        
        # Create legend and place it outside the plot
        legend3 = Legend(items=legend_items3, location="center")
        p3.add_layout(legend3, 'right')
        legend3.click_policy = "hide"
        
        # Customize raster plot
        p3.xaxis.axis_label = 'Time (seconds)'
        p3.yaxis.axis_label = 'Neuron ID'
        
        # Add hover tool
        hover = HoverTool(tooltips=[
            ('Time', '@x{0.00} s'),
            ('Neuron', '@neuron')
        ])
        p3.add_tools(hover)

        # Create velocity plot with same x-range as raster plot
        p4 = figure(width=800, height=200, x_range=p3.x_range, 
                    title='Velocity')
        
        # Remove grid
        p4.grid.grid_line_color = None

        velocity_line = p4.line(self.t, self.smoothed_velocity, color="black", line_width=2)
        
        # Create legend item for velocity plot
        legend_items4 = [
            LegendItem(label="Velocity", renderers=[velocity_line])
        ]
        
        # Create legend and place it outside the plot
        legend4 = Legend(items=legend_items4, location="center")
        p4.add_layout(legend4, 'right')
        
        # Calculate and print summary statistics
        print(f"\nNeural Spike Statistics:")
        
        if show_d1 and d1_counts:
            d1_mean_count = np.mean(d1_counts)
            d1_mean_isi = np.mean(d1_isis) if d1_isis else 0
            print(f"D1 Neurons (n={len(d1_peak_indices)}):")
            print(f"- Average spikes per neuron: {d1_mean_count:.2f}")
            print(f"- Average inter-spike interval: {d1_mean_isi:.2f} seconds")
        
        if show_d2 and d2_counts:
            d2_mean_count = np.mean(d2_counts)
            d2_mean_isi = np.mean(d2_isis) if d2_isis else 0
            print(f"D2 Neurons (n={len(d2_peak_indices)}):")
            print(f"- Average spikes per neuron: {d2_mean_count:.2f}")
            print(f"- Average inter-spike interval: {d2_mean_isi:.2f} seconds")
            
        if show_chi and chi_counts:
            chi_mean_count = np.mean(chi_counts)
            chi_mean_isi = np.mean(chi_isis) if chi_isis else 0
            print(f"CHI Neurons (n={len(chi_peak_indices)}):")
            print(f"- Average spikes per neuron: {chi_mean_count:.2f}")
            print(f"- Average inter-spike interval: {chi_mean_isi:.2f} seconds")
        
        # Combine the plots into a layout
        layout = column(
            row(p1, p2),
            p3,
            p4
        )
        
        # Output the plot
        self.output_bokeh_plot(layout, save_path=save_path, title=title, notebook=notebook, overwrite=overwrite)
        
        return layout
