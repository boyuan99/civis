import os
import numpy as np
import pandas as pd
from civis.src.CITank import CITank

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

        # Rising edges starts
        self.d1_rising_edges_starts = [self.rising_edges_starts[i] for i in self.d1_indices]
        self.d2_rising_edges_starts = [self.rising_edges_starts[i] for i in self.d2_indices]
        self.chi_rising_edges_starts = [self.rising_edges_starts[i] for i in self.chi_indices]
        
        # Peak indices for each cell type
        self.d1_peak_indices = [self.peak_indices[i] for i in self.d1_indices]
        self.d2_peak_indices = [self.peak_indices[i] for i in self.d2_indices]
        self.chi_peak_indices = [self.peak_indices[i] for i in self.chi_indices]
        
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
                                     cut_interval=50, save_path=None, title=None, notebook=False, overwrite=False, font_size=None):
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
        font_size : str, optional
            Font size for all text elements in the plot (e.g., "16pt", "24pt"). 
            If None, uses Bokeh's default font sizes.
        
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
        self.output_bokeh_plot(p, save_path=save_path, title=output_title, notebook=notebook, overwrite=overwrite, font_size=font_size)
        
        return p

    def plot_average_celltype_signals(self, show_d1=True, show_d2=True, show_chi=True, 
                                    show_velocity=True, show_movement_onset=True,
                                    d1_color='skyblue', d2_color='salmon', chi_color='limegreen',
                                    velocity_color='plum', movement_onset_color='green',
                                    x_range=(0, 60), y_range=(-0.05, 1.5), width=800, height=400,
                                    split_plots=False, 
                                    save_path=None, title=None, notebook=False, overwrite=False, font_size=None):
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
                                  notebook=notebook, overwrite=overwrite, font_size=font_size)
            
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
                                  notebook=notebook, overwrite=overwrite, font_size=font_size)
            
            return p

    def _compare_neuron_peaks_with_signal(self, signal=None, signal_name="velocity", 
                                     d1_peak_indices=None, d2_peak_indices=None, chi_peak_indices=None,
                                     show_d1=True, show_d2=True, show_chi=True,
                                     add_kde_contours=True, add_population_means=True,
                                     split_scatter_plots=True,
                                     save_path=None, title=None, notebook=False, overwrite=False, font_size=None):
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
            d1_peak_indices = self.d1_peak_indices
            
        if show_d2 and d2_peak_indices is None:
            d2_peak_indices = self.d2_peak_indices
            
        if show_chi and chi_peak_indices is None:
            chi_peak_indices = self.chi_peak_indices
        
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
                              notebook=notebook, overwrite=overwrite, font_size=font_size)
        
        # Return all created plots
        plot_list = [p1, p1_line, peak_counts_fig]
        plot_list.extend(scatter_plots)
        plot_list.append(p3)
        
        return plot_list

    def compare_neuron_peaks_with_velocity(self, d1_peak_indices=None, d2_peak_indices=None, chi_peak_indices=None, 
                                           show_d1=True, show_d2=True, show_chi=True, 
                                           split_scatter_plots=True,
                                           save_path=None, title=None, 
                                           notebook=False, overwrite=False, font_size=None):
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
            overwrite=overwrite,
            font_size=font_size
        )
    
    def compare_neuron_peaks_with_acceleration(self, d1_peak_indices=None, d2_peak_indices=None, chi_peak_indices=None, 
                                              show_d1=True, show_d2=True, show_chi=True, 
                                              split_scatter_plots=True,
                                              save_path=None, title=None, 
                                              notebook=False, overwrite=False, font_size=None):
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
            overwrite=overwrite,
            font_size=font_size
        )

    def create_cell_type_spike_visualizations(self, d1_peak_indices=None, d2_peak_indices=None, chi_peak_indices=None,
                                       show_d1=True, show_d2=True, show_chi=True, 
                                       save_path=None, title="Neural Spike Analysis", notebook=False, overwrite=False, font_size=None):
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
            d1_peak_indices = self.d1_peak_indices
            
        if show_d2 and d2_peak_indices is None:
            d2_peak_indices = self.d2_peak_indices
            
        if show_chi and chi_peak_indices is None:
            chi_peak_indices = self.chi_peak_indices
            
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
        self.output_bokeh_plot(layout, save_path=save_path, title=title, notebook=notebook, overwrite=overwrite, font_size=font_size)

        return layout


    def movement_related_activity_analysis(self, event_indices=None, event_type='movement_onset',
                                        d1_peak_indices=None, d2_peak_indices=None, chi_peak_indices=None,
                                        show_d1=True, show_d2=True, show_chi=True,
                                        pre_window=-3, post_window=3, bin_width=0.1,
                                        max_neurons_to_plot=20,
                                        save_path=None, title=None, notebook=False, overwrite=False, font_size=None):
        """
        Analyze neural activity in relation to any specified event indices. Creates multiple visualizations
        showing how different neuron types (D1, D2, CHI) activate around these events.

        Parameters:
        -----------
        event_indices : array-like, optional
            Indices of events to analyze around. If None, will use predefined indices based on event_type.
        event_type : str, optional
            Type of event to use if event_indices is None. Options: 'movement_onset', 'velocity_peak', 'movement_offset'
            Only used when event_indices is None.
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
        pre_window : float
            Time window before event (in seconds, negative value)
        post_window : float
            Time window after event (in seconds, positive value)
        bin_width : float
            Width of time bins for PSTH histogram (in seconds)
        max_neurons_to_plot : int
            Maximum number of neurons to include in the raster plot
        save_path : str, optional
            Path to save the visualization
        title : str, optional
            Title for the visualization. If None, will generate based on event_type.
        notebook : bool
            Flag to indicate if the visualization is for a Jupyter notebook
        overwrite : bool
            Flag to indicate whether to overwrite existing file

        Returns:
        --------
        bokeh.layouts.layout
            The created visualization layout
        """
        import numpy as np
        from scipy import stats
        from bokeh.plotting import figure
        from bokeh.layouts import column, row
        from bokeh.models import ColumnDataSource, Span, Legend, LegendItem, BoxAnnotation

        # Determine event indices to use
        if event_indices is None:
            # Use indices based on event_type if no explicit indices are provided
            if event_type == 'movement_onset':
                if not hasattr(self, 'movement_onset_indices') or len(self.movement_onset_indices) == 0:
                    raise ValueError("No movement onset indices available. Please run movement detection first.")
                event_indices = self.movement_onset_indices
                default_title = "Movement Onset-Related Neural Activity"
            elif event_type == 'velocity_peak':
                if not hasattr(self, 'velocity_peak_indices') or len(self.velocity_peak_indices) == 0:
                    raise ValueError("No velocity peak indices available.")
                event_indices = self.velocity_peak_indices
                default_title = "Velocity Peak-Related Neural Activity"
            elif event_type == 'movement_offset':
                if not hasattr(self, 'movement_offset_indices') or len(self.movement_offset_indices) == 0:
                    raise ValueError("No movement offset indices available.")
                event_indices = self.movement_offset_indices
                default_title = "Movement Offset-Related Neural Activity"
            else:
                raise ValueError("event_type must be one of: 'movement_onset', 'velocity_peak', 'movement_offset'")
        else:
            # Use the provided event indices
            if len(event_indices) == 0:
                raise ValueError("event_indices cannot be empty")
            default_title = "Event-Related Neural Activity"

        # Set title if not provided
        if title is None:
            title = default_title

        # Check if at least one cell type is selected
        if not any([show_d1, show_d2, show_chi]):
            raise ValueError("At least one cell type (show_d1, show_d2, or show_chi) must be set to True")

        # Use pre-computed peak indices for each cell type if not provided
        if show_d1 and d1_peak_indices is None:
            d1_peak_indices = self.d1_peak_indices

        if show_d2 and d2_peak_indices is None:
            d2_peak_indices = self.d2_peak_indices

        if show_chi and chi_peak_indices is None:
            chi_peak_indices = self.chi_peak_indices

        # Convert time window to samples
        pre_samples = int(pre_window * self.ci_rate)
        post_samples = int(post_window * self.ci_rate)
        window_size = post_samples - pre_samples

        # Function to align spikes to events
        def align_spikes_to_events(spike_indices, event_indices, pre_samples, post_samples, ci_rate):
            aligned_spikes = []
            event_windows = []

            # For each event, create a window
            for event in event_indices:
                window_start = event + pre_samples  # negative pre_samples
                window_end = event + post_samples
                event_windows.append((window_start, window_end))

            # For each neuron's spikes, find those within event windows
            for neuron_spikes in spike_indices:
                neuron_aligned = []

                for event_idx, (window_start, window_end) in enumerate(event_windows):
                    # Find spikes within this window
                    in_window = [spike for spike in neuron_spikes
                                if window_start <= spike < window_end]

                    # Convert to time relative to event (in seconds)
                    relative_times = [(spike - event_indices[event_idx]) / ci_rate
                                    for spike in in_window]

                    # Add to this neuron's aligned spikes
                    neuron_aligned.extend([(event_idx, t) for t in relative_times])

                aligned_spikes.append(neuron_aligned)

            return aligned_spikes

        # Calculate average velocity aligned to events
        def align_velocity_to_events(velocity, event_indices, pre_samples, post_samples, ci_rate):
            window_size = post_samples - pre_samples
            aligned_velocity = np.zeros((len(event_indices), window_size))

            for i, event in enumerate(event_indices):
                window_start = event + pre_samples
                window_end = event + post_samples

                # Check if window is within the velocity array bounds
                if window_start >= 0 and window_end < len(velocity):
                    aligned_velocity[i, :] = velocity[window_start:window_end]

            # Average across all events
            avg_velocity = np.mean(aligned_velocity, axis=0)

            # Create time points
            time_points = np.linspace(pre_window, post_window, len(avg_velocity))

            return time_points, avg_velocity

        # Align spikes to events
        d1_aligned = align_spikes_to_events(d1_peak_indices, event_indices,
                                            pre_samples, post_samples, self.ci_rate) if show_d1 else []
        d2_aligned = align_spikes_to_events(d2_peak_indices, event_indices,
                                            pre_samples, post_samples, self.ci_rate) if show_d2 else []
        chi_aligned = align_spikes_to_events(chi_peak_indices, event_indices,
                                            pre_samples, post_samples, self.ci_rate) if show_chi else []

        # Get average velocity aligned to events
        time_points, avg_velocity = align_velocity_to_events(self.smoothed_velocity,
                                                            event_indices,
                                                            pre_samples, post_samples,
                                                            self.ci_rate)

        # Calculate spike density over time (PSTH)
        bins = np.arange(pre_window, post_window + bin_width, bin_width)
        bin_centers = bins[:-1] + bin_width / 2

        # Aggregate spikes across neurons
        d1_all_times = []
        d2_all_times = []
        chi_all_times = []

        if show_d1 and d1_aligned:
            # Collect all spike times across all D1 neurons
            spikes_list = [[t for _, t in neuron_spikes] for neuron_spikes in d1_aligned if neuron_spikes]
            if spikes_list:  # Make sure there's data
                d1_all_times = np.concatenate(spikes_list)

        if show_d2 and d2_aligned:
            spikes_list = [[t for _, t in neuron_spikes] for neuron_spikes in d2_aligned if neuron_spikes]
            if spikes_list:
                d2_all_times = np.concatenate(spikes_list)

        if show_chi and chi_aligned:
            spikes_list = [[t for _, t in neuron_spikes] for neuron_spikes in chi_aligned if neuron_spikes]
            if spikes_list:
                chi_all_times = np.concatenate(spikes_list)

        # Create histograms
        d1_hist = np.zeros_like(bin_centers)
        d2_hist = np.zeros_like(bin_centers)
        chi_hist = np.zeros_like(bin_centers)

        if len(d1_all_times) > 0:
            d1_hist, _ = np.histogram(d1_all_times, bins=bins)
            # Normalize by number of neurons and bin width to get firing rate (spikes/s)
            d1_hist = d1_hist / (len(d1_aligned) * bin_width) if len(d1_aligned) > 0 else d1_hist

        if len(d2_all_times) > 0:
            d2_hist, _ = np.histogram(d2_all_times, bins=bins)
            d2_hist = d2_hist / (len(d2_aligned) * bin_width) if len(d2_aligned) > 0 else d2_hist

        if len(chi_all_times) > 0:
            chi_hist, _ = np.histogram(chi_all_times, bins=bins)
            chi_hist = chi_hist / (len(chi_aligned) * bin_width) if len(chi_aligned) > 0 else chi_hist

        # Normalize velocity for plotting (scale to match firing rate range)
        max_firing_rate = max(np.max(d1_hist) if len(d1_hist) > 0 else 0,
                            np.max(d2_hist) if len(d2_hist) > 0 else 0,
                            np.max(chi_hist) if len(chi_hist) > 0 else 0)

        if max_firing_rate > 0 and np.max(avg_velocity) > 0:
            normalized_velocity = avg_velocity / np.max(
                avg_velocity) * max_firing_rate * 0.8  # Scale to 80% of max firing rate
        else:
            normalized_velocity = avg_velocity  # Fallback if no firing rate data

        # 1. PSTH (Peri-Stimulus Time Histogram) plot with velocity overlay
        # Determine appropriate title based on event type
        psth_title = f"Peri-Event Time Histogram"
        if event_type == 'movement_onset':
            x_axis_label = "Time relative to movement onset (s)"
        elif event_type == 'velocity_peak':
            x_axis_label = "Time relative to velocity peak (s)"
        elif event_type == 'movement_offset':
            x_axis_label = "Time relative to movement offset (s)"
        else:
            x_axis_label = "Time relative to event (s)"

        p1 = figure(width=900, height=400,
                    title=psth_title,
                    x_axis_label=x_axis_label,
                    y_axis_label="Firing Rate (Hz)",
                    tools="pan,box_zoom,wheel_zoom,reset,save")

        # Remove grid
        p1.grid.grid_line_color = None

        # Add vertical line at event time
        event_line = Span(location=0, dimension='height', line_color='black',
                        line_dash='dashed', line_width=2)
        p1.add_layout(event_line)

        # Create ColumnDataSource for the histograms
        source_hist = ColumnDataSource(data=dict(
            bin_centers=bin_centers,
            d1_hist=d1_hist,
            d2_hist=d2_hist,
            chi_hist=chi_hist
        ))

        # Create ColumnDataSource for velocity
        source_velocity = ColumnDataSource(data=dict(
            time=time_points,
            velocity=normalized_velocity
        ))

        # Plot histograms as bars without legend labels
        legend_items1 = []

        if show_d1 and len(d1_all_times) > 0:
            d1_hist_plot = p1.vbar(x='bin_centers', top='d1_hist', width=bin_width * 0.8, source=source_hist,
                                color='navy', alpha=0.5)
            legend_items1.append(LegendItem(label="D1 Neurons", renderers=[d1_hist_plot]))

        if show_d2 and len(d2_all_times) > 0:
            d2_hist_plot = p1.vbar(x='bin_centers', top='d2_hist', width=bin_width * 0.8, source=source_hist,
                                color='crimson', alpha=0.5)
            legend_items1.append(LegendItem(label="D2 Neurons", renderers=[d2_hist_plot]))

        if show_chi and len(chi_all_times) > 0:
            chi_hist_plot = p1.vbar(x='bin_centers', top='chi_hist', width=bin_width * 0.8, source=source_hist,
                                    color='green', alpha=0.5)
            legend_items1.append(LegendItem(label="CHI Neurons", renderers=[chi_hist_plot]))

        # Plot velocity as a line
        velocity_line = p1.line('time', 'velocity', source=source_velocity, color='black',
                                line_width=3, alpha=0.8)
        legend_items1.append(LegendItem(label="Avg. Velocity", renderers=[velocity_line]))

        # Create and add legend outside the plot
        legend1 = Legend(items=legend_items1, location="center")

        # Set legend outside the plot
        p1.add_layout(legend1, 'right')
        legend1.click_policy = "hide"  # Allow toggling visibility
        legend1.background_fill_alpha = 0.7

        # Add pre/post event regions
        pre_box = BoxAnnotation(left=pre_window, right=0, fill_color='lightgray', fill_alpha=0.2)
        post_box = BoxAnnotation(left=0, right=post_window, fill_color='lightyellow', fill_alpha=0.2)
        p1.add_layout(pre_box)
        p1.add_layout(post_box)

        # 2. Raster plot of sample neurons
        raster_title = "Spike Raster Plot"
        if event_type == 'movement_onset':
            raster_title += " Relative to Movement Onset"
        elif event_type == 'velocity_peak':
            raster_title += " Relative to Velocity Peak"
        elif event_type == 'movement_offset':
            raster_title += " Relative to Movement Offset"
        else:
            raster_title += " Relative to Event"

        p2 = figure(width=900, height=500,
                    title=raster_title,
                    x_axis_label=x_axis_label,
                    y_axis_label="Neuron #",
                    tools="pan,box_zoom,wheel_zoom,reset,save")

        # Remove grid
        p2.grid.grid_line_color = None

        # Add vertical line at event time
        event_line = Span(location=0, dimension='height', line_color='black',
                        line_dash='dashed', line_width=2)
        p2.add_layout(event_line)

        # Plot raster for a subset of neurons (limited by max_neurons_to_plot)
        # Calculate max neurons to show for each type
        max_d1 = min(max_neurons_to_plot, len(d1_aligned)) if show_d1 else 0
        max_d2 = min(max_neurons_to_plot, len(d2_aligned)) if show_d2 else 0
        max_chi = min(max_neurons_to_plot, len(chi_aligned)) if show_chi else 0

        # Prepare data for raster plot
        d1_raster_data = {'x': [], 'y': []}
        d2_raster_data = {'x': [], 'y': []}
        chi_raster_data = {'x': [], 'y': []}

        # Add D1 neurons
        for i in range(max_d1):
            if i < len(d1_aligned) and d1_aligned[i]:
                for _, spike_time in d1_aligned[i]:
                    d1_raster_data['x'].append(spike_time)
                    d1_raster_data['y'].append(i)

        # Add D2 neurons (offset y position)
        d2_offset = max_d1 + 1 if max_d1 > 0 else 0  # Add a gap between groups
        for i in range(max_d2):
            if i < len(d2_aligned) and d2_aligned[i]:
                for _, spike_time in d2_aligned[i]:
                    d2_raster_data['x'].append(spike_time)
                    d2_raster_data['y'].append(i + d2_offset)

        # Add CHI neurons (offset y position further)
        chi_offset = d2_offset + max_d2 + 1 if max_d2 > 0 else d2_offset  # Add another gap
        for i in range(max_chi):
            if i < len(chi_aligned) and chi_aligned[i]:
                for _, spike_time in chi_aligned[i]:
                    chi_raster_data['x'].append(spike_time)
                    chi_raster_data['y'].append(i + chi_offset)

        # Create ColumnDataSource for rasters
        d1_source_raster = ColumnDataSource(data=d1_raster_data)
        d2_source_raster = ColumnDataSource(data=d2_raster_data)
        chi_source_raster = ColumnDataSource(data=chi_raster_data)

        # Create legend items for raster plot
        raster_legend_items = []

        # Plot raster points for each cell type
        if show_d1 and d1_raster_data['x']:
            d1_scatter = p2.scatter(x='x', y='y', source=d1_source_raster, size=4, color='navy', alpha=0.2)
            raster_legend_items.append(LegendItem(label="D1 Neurons", renderers=[d1_scatter]))

        if show_d2 and d2_raster_data['x']:
            d2_scatter = p2.scatter(x='x', y='y', source=d2_source_raster, size=4, color='crimson', alpha=0.2)
            raster_legend_items.append(LegendItem(label="D2 Neurons", renderers=[d2_scatter]))

        if show_chi and chi_raster_data['x']:
            chi_scatter = p2.scatter(x='x', y='y', source=chi_source_raster, size=4, color='green', alpha=0.2)
            raster_legend_items.append(LegendItem(label="CHI Neurons", renderers=[chi_scatter]))

        # Add velocity to raster plot
        # Calculate the vertical position for the velocity line
        middle_point = (chi_offset + max_chi) / 2 if max_chi > 0 else (d2_offset + max_d2) / 2 if max_d2 > 0 else max_d1 / 2
        max_height = chi_offset + max_chi if max_chi > 0 else d2_offset + max_d2 if max_d2 > 0 else max_d1

        # Scale velocity to fit within the plot's vertical range (25% of max height)
        if max_height > 0 and np.max(avg_velocity) > 0:
            scale_factor = max_height * 0.25 / np.max(avg_velocity)
            scaled_velocity = avg_velocity * scale_factor
            # Center the velocity line vertically
            velocity_offset = middle_point

            velocity_line2 = p2.line(time_points, scaled_velocity + velocity_offset,
                                    color='gold', line_width=3, alpha=0.8)
            raster_legend_items.append(LegendItem(label="Avg. Velocity", renderers=[velocity_line2]))

        # Add dividing lines between neuron types
        if max_d1 > 0 and (max_d2 > 0 or max_chi > 0):
            div_line1 = Span(location=max_d1, dimension='width',
                            line_color='gray', line_width=1, line_alpha=0.5)
            p2.add_layout(div_line1)

        if max_d2 > 0 and max_chi > 0:
            div_line2 = Span(location=d2_offset + max_d2, dimension='width',
                            line_color='gray', line_width=1, line_alpha=0.5)
            p2.add_layout(div_line2)

        # Create and add legend outside the plot
        legend2 = Legend(items=raster_legend_items, location="center")
        p2.add_layout(legend2, 'right')
        legend2.click_policy = "hide"  # Allow toggling visibility

        # Set y-range to accommodate all plotted neurons
        p2.y_range.start = -1
        p2.y_range.end = chi_offset + max_chi + 1 if max_chi > 0 else d2_offset + max_d2 + 1 if max_d2 > 0 else max_d1 + 1

        # 3. Modulation Analysis
        # Calculate phase preferences
        pre_phase = (pre_window, 0)  # Before event
        post_phase = (0, post_window)  # After event

        # Count spikes in each phase
        def count_phase_spikes(aligned_spikes, phase_range):
            phase_start, phase_end = phase_range
            counts = []

            for neuron_spikes in aligned_spikes:
                phase_count = sum(1 for _, t in neuron_spikes if phase_start <= t < phase_end)
                counts.append(phase_count)

            return counts

        # Calculate modulation index: (post - pre) / (post + pre)
        def calc_modulation_index(pre_counts, post_counts):
            indices = []

            for pre, post in zip(pre_counts, post_counts):
                if pre + post > 0:  # Avoid division by zero
                    index = (post - pre) / (post + pre)
                else:
                    index = 0
                indices.append(index)

            return indices

        # Calculate pre/post counts and modulation indices for each cell type
        d1_pre_counts = count_phase_spikes(d1_aligned, pre_phase) if show_d1 else []
        d1_post_counts = count_phase_spikes(d1_aligned, post_phase) if show_d1 else []
        d1_mod_indices = calc_modulation_index(d1_pre_counts, d1_post_counts) if show_d1 and d1_pre_counts else []

        d2_pre_counts = count_phase_spikes(d2_aligned, pre_phase) if show_d2 else []
        d2_post_counts = count_phase_spikes(d2_aligned, post_phase) if show_d2 else []
        d2_mod_indices = calc_modulation_index(d2_pre_counts, d2_post_counts) if show_d2 and d2_pre_counts else []

        chi_pre_counts = count_phase_spikes(chi_aligned, pre_phase) if show_chi else []
        chi_post_counts = count_phase_spikes(chi_aligned, post_phase) if show_chi else []
        chi_mod_indices = calc_modulation_index(chi_pre_counts, chi_post_counts) if show_chi and chi_pre_counts else []

        # Check if we have enough data for modulation analysis
        if any([d1_mod_indices, d2_mod_indices, chi_mod_indices]):
            # Create box plot for modulation indices
            mod_title = "Event-Related Modulation"
            if event_type == 'movement_onset':
                mod_title = "Movement Onset-Related Modulation"
            elif event_type == 'velocity_peak':
                mod_title = "Velocity Peak-Related Modulation"
            elif event_type == 'movement_offset':
                mod_title = "Movement Offset-Related Modulation"

            p3 = figure(width=900, height=400,
                        title=mod_title,
                        x_axis_label="Neuron Type",
                        y_axis_label="Modulation Index (post-pre)/(post+pre)",
                        tools="pan,box_zoom,wheel_zoom,reset,save")

            # Remove grid
            p3.grid.grid_line_color = None

            # Add horizontal line at 0
            zero_line = Span(location=0, dimension='width', line_color='black',
                            line_dash='dashed', line_width=1)
            p3.add_layout(zero_line)

            # Function to calculate box plot statistics
            def calc_box_stats(data):
                if not data:
                    return (0, 0, 0, 0, 0)  # Return zeros if no data

                q1 = np.percentile(data, 25)
                q2 = np.percentile(data, 50)  # median
                q3 = np.percentile(data, 75)
                iqr = q3 - q1
                upper = min(q3 + 1.5 * iqr, np.max(data))
                lower = max(q1 - 1.5 * iqr, np.min(data))
                return lower, q1, q2, q3, upper

            # Calculate box plot data for each cell type
            d1_box = calc_box_stats(d1_mod_indices)
            d2_box = calc_box_stats(d2_mod_indices)
            chi_box = calc_box_stats(chi_mod_indices)

            # Helper function to draw boxplot
            def add_boxplot(plot, x_pos, stats, width=0.3, color='navy'):
                lower, q1, q2, q3, upper = stats

                # Only draw if we have valid data
                if q1 == q3 == 0:
                    return None

                # Box
                box = plot.vbar(x=x_pos, width=width, top=q3, bottom=q1,
                                fill_color=color, line_color="black", alpha=0.7)

                # Median line
                plot.line([x_pos - width / 2, x_pos + width / 2], [q2, q2], line_color="white", line_width=2)

                # Whiskers
                plot.segment(x0=x_pos, y0=q3, x1=x_pos, y1=upper, line_color="black")
                plot.segment(x0=x_pos, y0=q1, x1=x_pos, y1=lower, line_color="black")

                # Caps on whiskers
                plot.segment(x0=x_pos - width / 4, y0=upper, x1=x_pos + width / 4, y1=upper, line_color="black")
                plot.segment(x0=x_pos - width / 4, y0=lower, x1=x_pos + width / 4, y1=lower, line_color="black")

                return box

            # Function to create jittered x-coordinates
            def jitter(x_pos, n, width=0.2):
                return np.random.normal(x_pos, width / 4, size=n)

            # Create legend items list
            mod_legend_items = []

            # Plot D1 modulation data if available
            if show_d1 and d1_mod_indices:
                # Add D1 box plot
                d1_box_plot = add_boxplot(p3, 1, d1_box, color='navy')

                # Add scatter points with jitter
                d1_x = jitter(1, len(d1_mod_indices), width=0.2)
                d1_source = ColumnDataSource(data=dict(x=d1_x, y=d1_mod_indices))
                d1_circles = p3.scatter(x='x', y='y', source=d1_source, size=6, color='navy', alpha=0.5)

                mod_legend_items.append(LegendItem(label="D1", renderers=[d1_box_plot, d1_circles]))

            # Plot D2 modulation data if available
            if show_d2 and d2_mod_indices:
                # Add D2 box plot
                d2_box_plot = add_boxplot(p3, 2, d2_box, color='crimson')

                # Add scatter points with jitter
                d2_x = jitter(2, len(d2_mod_indices), width=0.2)
                d2_source = ColumnDataSource(data=dict(x=d2_x, y=d2_mod_indices))
                d2_circles = p3.scatter(x='x', y='y', source=d2_source, size=6, color='crimson', alpha=0.5)

                mod_legend_items.append(LegendItem(label="D2", renderers=[d2_box_plot, d2_circles]))

            # Plot CHI modulation data if available
            if show_chi and chi_mod_indices:
                # Add CHI box plot
                chi_box_plot = add_boxplot(p3, 3, chi_box, color='green')

                # Add scatter points with jitter
                chi_x = jitter(3, len(chi_mod_indices), width=0.2)
                chi_source = ColumnDataSource(data=dict(x=chi_x, y=chi_mod_indices))
                chi_circles = p3.scatter(x='x', y='y', source=chi_source, size=6, color='green', alpha=0.5)

                mod_legend_items.append(LegendItem(label="CHI", renderers=[chi_box_plot, chi_circles]))

            # Create legend for the modulation plot
            legend3 = Legend(items=mod_legend_items, location="center")
            p3.add_layout(legend3, 'right')

            # Set x-axis ticks and range
            positions = []
            labels = {}

            if show_d1 and d1_mod_indices:
                positions.append(1)
                labels[1] = 'D1'

            if show_d2 and d2_mod_indices:
                positions.append(2)
                labels[2] = 'D2'

            if show_chi and chi_mod_indices:
                positions.append(3)
                labels[3] = 'CHI'

            p3.xaxis.ticker = positions
            p3.xaxis.major_label_overrides = labels
            p3.x_range.start = min(positions) - 0.5 if positions else 0.5
            p3.x_range.end = max(positions) + 0.5 if positions else 3.5

            # Calculate statistics for printing
            event_str = event_type.replace('_', ' ').title() if event_type else "Event"

            if d1_mod_indices:
                d1_mean = np.mean(d1_mod_indices)
                d1_sem = np.std(d1_mod_indices) / np.sqrt(len(d1_mod_indices))
                d1_median = np.median(d1_mod_indices)
                d1_pos_pct = np.sum(np.array(d1_mod_indices) > 0) / len(d1_mod_indices) * 100

                print(f"\nEvent-Related Activity Statistics:")
                print(f"D1 Neurons (n={len(d1_aligned)}):")
                print(f"- Mean modulation index: {d1_mean:.3f} ± {d1_sem:.3f}")
                print(f"- Median modulation index: {d1_median:.3f}")
                print(f"- % positive modulation: {d1_pos_pct:.1f}%")

            if d2_mod_indices:
                d2_mean = np.mean(d2_mod_indices)
                d2_sem = np.std(d2_mod_indices) / np.sqrt(len(d2_mod_indices))
                d2_median = np.median(d2_mod_indices)
                d2_pos_pct = np.sum(np.array(d2_mod_indices) > 0) / len(d2_mod_indices) * 100

                if not d1_mod_indices:  # Print header if not already printed
                    print(f"\n{event_str}-Related Activity Statistics:")
                print(f"\nD2 Neurons (n={len(d2_aligned)}):")
                print(f"- Mean modulation index: {d2_mean:.3f} ± {d2_sem:.3f}")
                print(f"- Median modulation index: {d2_median:.3f}")
                print(f"- % positive modulation: {d2_pos_pct:.1f}%")

            if chi_mod_indices:
                chi_mean = np.mean(chi_mod_indices)
                chi_sem = np.std(chi_mod_indices) / np.sqrt(len(chi_mod_indices))
                chi_median = np.median(chi_mod_indices)
                chi_pos_pct = np.sum(np.array(chi_mod_indices) > 0) / len(chi_mod_indices) * 100

                if not d1_mod_indices and not d2_mod_indices:  # Print header if not already printed
                    print(f"\n{event_str}-Related Activity Statistics:")
                print(f"\nCHI Neurons (n={len(chi_aligned)}):")
                print(f"- Mean modulation index: {chi_mean:.3f} ± {chi_sem:.3f}")
                print(f"- Median modulation index: {chi_median:.3f}")
                print(f"- % positive modulation: {chi_pos_pct:.1f}%")

            # Statistical comparison if we have multiple cell types
            if show_d1 and show_d2 and d1_mod_indices and d2_mod_indices:
                t_stat, p_val = stats.ttest_ind(d1_mod_indices, d2_mod_indices, equal_var=False)
                print(f"\nStatistical Comparison:")
                print(f"- D1 vs D2 t-test: t={t_stat:.3f}, p={p_val:.5f}")

            if show_d1 and show_chi and d1_mod_indices and chi_mod_indices:
                t_stat, p_val = stats.ttest_ind(d1_mod_indices, chi_mod_indices, equal_var=False)
                print(f"- D1 vs CHI t-test: t={t_stat:.3f}, p={p_val:.5f}")

            if show_d2 and show_chi and d2_mod_indices and chi_mod_indices:
                t_stat, p_val = stats.ttest_ind(d2_mod_indices, chi_mod_indices, equal_var=False)
                print(f"- D2 vs CHI t-test: t={t_stat:.3f}, p={p_val:.5f}")
            else:
                # Skip modulation analysis if no data
                p3 = figure(width=900, height=400,
                            title="Event-Related Modulation (insufficient data)",
                            x_axis_label="Neuron Type",
                            y_axis_label="Modulation Index")
                p3.grid.grid_line_color = None

            # 4. Heatmap visualization
            # Create a heatmap showing activity over time for each neuron

            # Function to create neuron heatmap data
            def create_neuron_heatmap_data(aligned_spikes, bins):
                n_neurons = len(aligned_spikes)
                n_bins = len(bins) - 1
                heatmap = np.zeros((n_neurons, n_bins))

                for i, neuron_spikes in enumerate(aligned_spikes):
                    if neuron_spikes:
                        spike_times = [t for _, t in neuron_spikes]
                        hist, _ = np.histogram(spike_times, bins=bins)
                        heatmap[i, :] = hist

                return heatmap

            # Function to normalize heatmap values
            def normalize_heatmap(heatmap):
                normalized = np.zeros_like(heatmap)
                for i in range(heatmap.shape[0]):
                    if np.max(heatmap[i, :]) > 0:
                        normalized[i, :] = heatmap[i, :] / np.max(heatmap[i, :])
                return normalized

            # Create heatmap bins
            heatmap_bins = np.linspace(pre_window, post_window, 61)  # 0.1s bins
            heatmap_bin_centers = (heatmap_bins[:-1] + heatmap_bins[1:]) / 2

            # List to store heatmap plots
            heatmap_plots = []

            # Determine heatmap titles
            heatmap_title_base = "Neuron Activity Heat Map"
            if event_type == 'movement_onset':
                heatmap_title_suffix = "around Movement Onset"
            elif event_type == 'velocity_peak':
                heatmap_title_suffix = "around Velocity Peak"
            elif event_type == 'movement_offset':
                heatmap_title_suffix = "around Movement Offset"
            else:
                heatmap_title_suffix = "around Event"

            # Create D1 heatmap if applicable
            if show_d1 and len(d1_aligned) > 0:
                d1_heatmap = create_neuron_heatmap_data(d1_aligned, heatmap_bins)
                d1_heatmap_norm = normalize_heatmap(d1_heatmap)

                if np.sum(d1_heatmap_norm) > 0:  # Only create plot if there's activity
                    p4 = figure(width=900, height=300, title=f"D1 {heatmap_title_base} {heatmap_title_suffix}",
                                x_axis_label="Time (s)", y_axis_label="Neuron #",
                                tools="pan,box_zoom,wheel_zoom,reset,save")

                    # Remove grid
                    p4.grid.grid_line_color = None

                    # Flip heatmap vertically for proper orientation
                    d1_img = np.flipud(d1_heatmap_norm)

                    # Calculate dimensions for plotting
                    d1_dh = len(d1_aligned)

                    # Plot the heatmap
                    p4.image(image=[d1_img], x=pre_window, y=0,
                            dw=post_window - pre_window, dh=d1_dh,
                            palette="Viridis256", level="image")

                    # Add event line
                    p4.add_layout(Span(location=0, dimension='height',
                                    line_color='white', line_dash='dashed', line_width=2))

                    # Add velocity overlay if possible
                    if len(time_points) > 0 and len(avg_velocity) > 0:
                        scaled_velocity = avg_velocity / np.max(avg_velocity) * d1_dh * 0.8
                        p4.line(time_points, scaled_velocity, color='gold', line_width=3, alpha=0.8)

                    # Set y-range
                    p4.y_range.start = 0
                    p4.y_range.end = d1_dh

                    heatmap_plots.append(p4)

            # Create D2 heatmap if applicable
            if show_d2 and len(d2_aligned) > 0:
                d2_heatmap = create_neuron_heatmap_data(d2_aligned, heatmap_bins)
                d2_heatmap_norm = normalize_heatmap(d2_heatmap)

                if np.sum(d2_heatmap_norm) > 0:  # Only create plot if there's activity
                    p5 = figure(width=900, height=300, title=f"D2 {heatmap_title_base} {heatmap_title_suffix}",
                                x_axis_label="Time (s)", y_axis_label="Neuron #",
                                tools="pan,box_zoom,wheel_zoom,reset,save")

                    # Remove grid
                    p5.grid.grid_line_color = None

                    # Flip heatmap vertically for proper orientation
                    d2_img = np.flipud(d2_heatmap_norm)

                    # Calculate dimensions for plotting
                    d2_dh = len(d2_aligned)

                    # Plot the heatmap
                    p5.image(image=[d2_img], x=pre_window, y=0,
                            dw=post_window - pre_window, dh=d2_dh,
                            palette="Viridis256", level="image")

                    # Add event line
                    p5.add_layout(Span(location=0, dimension='height',
                                    line_color='white', line_dash='dashed', line_width=2))

                    # Add velocity overlay if possible
                    if len(time_points) > 0 and len(avg_velocity) > 0:
                        scaled_velocity = avg_velocity / np.max(avg_velocity) * d2_dh * 0.8
                        p5.line(time_points, scaled_velocity, color='gold', line_width=3, alpha=0.8)

                    # Set y-range
                    p5.y_range.start = 0
                    p5.y_range.end = d2_dh

                    heatmap_plots.append(p5)

            # Create CHI heatmap if applicable
            if show_chi and len(chi_aligned) > 0:
                chi_heatmap = create_neuron_heatmap_data(chi_aligned, heatmap_bins)
                chi_heatmap_norm = normalize_heatmap(chi_heatmap)

                if np.sum(chi_heatmap_norm) > 0:  # Only create plot if there's activity
                    p6 = figure(width=900, height=300, title=f"CHI {heatmap_title_base} {heatmap_title_suffix}",
                                x_axis_label="Time (s)", y_axis_label="Neuron #",
                                tools="pan,box_zoom,wheel_zoom,reset,save")

                    # Remove grid
                    p6.grid.grid_line_color = None

                    # Flip heatmap vertically for proper orientation
                    chi_img = np.flipud(chi_heatmap_norm)

                    # Calculate dimensions for plotting
                    chi_dh = len(chi_aligned)

                    # Plot the heatmap
                    p6.image(image=[chi_img], x=pre_window, y=0,
                            dw=post_window - pre_window, dh=chi_dh,
                            palette="Viridis256", level="image")

                    # Add event line
                    p6.add_layout(Span(location=0, dimension='height',
                                    line_color='white', line_dash='dashed', line_width=2))

                    # Add velocity overlay if possible
                    if len(time_points) > 0 and len(avg_velocity) > 0:
                        scaled_velocity = avg_velocity / np.max(avg_velocity) * chi_dh * 0.8
                        p6.line(time_points, scaled_velocity, color='gold', line_width=3, alpha=0.8)

                    # Set y-range
                    p6.y_range.start = 0
                    p6.y_range.end = chi_dh

                    heatmap_plots.append(p6)

            # Create a layout with the plots stacked vertically
            plots = [p1, p2, p3] + heatmap_plots
            layout = column(*plots)

            # Set default save path if none provided
            if save_path is None:
                cell_types = []
                if show_d1: cell_types.append("D1")
                if show_d2: cell_types.append("D2")
                if show_chi: cell_types.append("CHI")
                cell_type_str = "_".join(cell_types)

                # Include event type in filename if provided
                event_name = event_type if event_type else "Event"

                save_path = os.path.join(self.config["SummaryPlotsPath"],
                                        self.session_name,
                                        f'{event_name}_Related_{cell_type_str}_Activity.html')

            # Output the plot
            self.output_bokeh_plot(layout, save_path=save_path, title=title, notebook=notebook, overwrite=overwrite, font_size=font_size)

            return layout

    def analyze_signal_around_cell_type_spikes(self, d1_indices=None, d2_indices=None, chi_indices=None, 
                                              signal=None, signal_name="Signal", time_window=4.0, height=0.5,
                                              show_d1=True, show_d2=True, show_chi=True,
                                              save_path=None, title=None, notebook=False, overwrite=False, font_size=None):
        """
        Analyze any signal before and after spikes in D1, D2, and CHI neurons.
        
        Parameters:
        -----------
        d1_indices : array-like, optional
            Indices of D1 neurons. If None, uses self.d1_indices
        d2_indices : array-like, optional
            Indices of D2 neurons. If None, uses self.d2_indices
        chi_indices : array-like, optional
            Indices of CHI neurons. If None, uses self.chi_indices
        signal : numpy.ndarray, optional
            Signal array to analyze (e.g., self.smoothed_velocity, self.acceleration)
            If None, uses self.smoothed_velocity
        signal_name : str
            Name of signal for plot labels (e.g., "Velocity", "Acceleration")
        time_window : float
            Time window in seconds before and after spike to analyze
        height : float
            Height threshold for peak detection
        show_d1 : bool
            Whether to analyze and show D1 neurons
        show_d2 : bool
            Whether to analyze and show D2 neurons
        show_chi : bool
            Whether to analyze and show CHI neurons
        save_path : str, optional
            Path to save the plot
        title : str, optional
            Title for the plot (if None, will be generated based on signal_name)
        notebook : bool
            Whether to display in notebook
        overwrite : bool
            Whether to overwrite existing file
        font_size : str, optional
            Font size for plot text elements
        
        Returns:
        --------
        dict : Dictionary containing analysis results and plot objects
        """
        from bokeh.plotting import figure
        from bokeh.layouts import gridplot
        from bokeh.models import ColumnDataSource, Span
        from scipy.signal import find_peaks
        
        # Use default indices if not provided
        if d1_indices is None:
            d1_indices = self.d1_indices
        if d2_indices is None:
            d2_indices = self.d2_indices
        if chi_indices is None:
            chi_indices = self.chi_indices
            
        # Use default signal if not provided
        if signal is None:
            signal = self.smoothed_velocity
        
        # Set default title if not provided
        if title is None:
            cell_types = []
            if show_d1: cell_types.append("D1")
            if show_d2: cell_types.append("D2") 
            if show_chi: cell_types.append("CHI")
            title = f"{signal_name} around {', '.join(cell_types)} Neuron Spikes"
        
        # Convert time window to samples
        samples_window = int(time_window * self.ci_rate)
        
        # Initialize results dictionary
        results = {
            'time_axis': np.linspace(-time_window, time_window, 2 * samples_window + 1),
            'signal_name': signal_name
        }
        
        plots = []
        
        # Process D1 neurons if requested
        if show_d1 and len(d1_indices) > 0:
            d1_peak_indices = []
            for i in d1_indices:
                x, _ = find_peaks(self.C_denoised[i], height=height)
                d1_peak_indices.append(x)
            
            d1_signals = []
            for neuron_spikes in d1_peak_indices:
                for spike_idx in neuron_spikes:
                    if spike_idx - samples_window >= 0 and spike_idx + samples_window < len(signal):
                        signal_segment = signal[spike_idx - samples_window:spike_idx + samples_window + 1]
                        d1_signals.append(signal_segment)
            
            if len(d1_signals) > 0:
                d1_signals = np.array(d1_signals)
                d1_avg_signal = np.mean(d1_signals, axis=0)
                d1_sem_signal = np.std(d1_signals, axis=0) / np.sqrt(len(d1_signals))
                
                # Store results
                results['d1_avg_signal'] = d1_avg_signal
                results['d1_sem_signal'] = d1_sem_signal
                results['d1_sample_count'] = len(d1_signals)
                
                # Create D1 plot
                p1 = figure(width=900, height=400, 
                        title=f"{signal_name} around D1 Neuron Spikes (n={len(d1_signals)})",
                        x_axis_label="Time relative to spike (s)",
                        y_axis_label=signal_name)
                
                p1.line(results['time_axis'], d1_avg_signal, line_width=3, color='navy', legend_label="D1 Average")
                p1.patch(
                    np.concatenate([results['time_axis'], results['time_axis'][::-1]]),
                    np.concatenate([
                        d1_avg_signal + d1_sem_signal,
                        (d1_avg_signal - d1_sem_signal)[::-1]
                    ]),
                    color='navy', alpha=0.2
                )
                
                # Add vertical line at spike time
                p1.line([0, 0], [np.min(d1_avg_signal)-1, np.max(d1_avg_signal) * 1.2], 
                       line_width=2, color='black', line_dash='dashed')
                
                # Add horizontal line at y=0 for acceleration signals
                if signal_name.lower() == "acceleration":
                    zero_line = Span(location=0, dimension='width', line_color='black', line_width=2, line_dash='dotted')
                    p1.add_layout(zero_line)
                
                p1.xgrid.grid_line_color = None
                p1.ygrid.grid_line_color = None
                p1.legend.location = "top_right"
                p1.legend.click_policy = "hide"
                
                plots.append(p1)
        
        # Process D2 neurons if requested
        if show_d2 and len(d2_indices) > 0:
            d2_peak_indices = []
            for i in d2_indices:
                x, _ = find_peaks(self.C_denoised[i], height=height)
                d2_peak_indices.append(x)
            
            d2_signals = []
            for neuron_spikes in d2_peak_indices:
                for spike_idx in neuron_spikes:
                    if spike_idx - samples_window >= 0 and spike_idx + samples_window < len(signal):
                        signal_segment = signal[spike_idx - samples_window:spike_idx + samples_window + 1]
                        d2_signals.append(signal_segment)
            
            if len(d2_signals) > 0:
                d2_signals = np.array(d2_signals)
                d2_avg_signal = np.mean(d2_signals, axis=0)
                d2_sem_signal = np.std(d2_signals, axis=0) / np.sqrt(len(d2_signals))
                
                # Store results
                results['d2_avg_signal'] = d2_avg_signal
                results['d2_sem_signal'] = d2_sem_signal
                results['d2_sample_count'] = len(d2_signals)
                
                # Create D2 plot
                p2 = figure(width=900, height=400, 
                        title=f"{signal_name} around D2 Neuron Spikes (n={len(d2_signals)})",
                        x_axis_label="Time relative to spike (s)",
                        y_axis_label=signal_name,
                        x_range=plots[0].x_range if plots else None)
                
                p2.line(results['time_axis'], d2_avg_signal, line_width=3, color='crimson', legend_label="D2 Average")
                p2.patch(
                    np.concatenate([results['time_axis'], results['time_axis'][::-1]]),
                    np.concatenate([
                        d2_avg_signal + d2_sem_signal,
                        (d2_avg_signal - d2_sem_signal)[::-1]
                    ]),
                    color='crimson', alpha=0.2
                )
                
                # Add vertical line at spike time
                p2.line([0, 0], [np.min(d2_avg_signal)-1, np.max(d2_avg_signal) * 1.2], 
                       line_width=2, color='black', line_dash='dashed')
                
                # Add horizontal line at y=0 for acceleration signals
                if signal_name.lower() == "acceleration":
                    zero_line = Span(location=0, dimension='width', line_color='black', line_width=2, line_dash='dotted')
                    p2.add_layout(zero_line)
                
                p2.xgrid.grid_line_color = None
                p2.ygrid.grid_line_color = None
                p2.legend.location = "top_right"
                p2.legend.click_policy = "hide"
                
                plots.append(p2)
        
        # Process CHI neurons if requested
        if show_chi and len(chi_indices) > 0:
            chi_peak_indices = []
            for i in chi_indices:
                x, _ = find_peaks(self.C_denoised[i], height=height)
                chi_peak_indices.append(x)
            
            chi_signals = []
            for neuron_spikes in chi_peak_indices:
                for spike_idx in neuron_spikes:
                    if spike_idx - samples_window >= 0 and spike_idx + samples_window < len(signal):
                        signal_segment = signal[spike_idx - samples_window:spike_idx + samples_window + 1]
                        chi_signals.append(signal_segment)
            
            if len(chi_signals) > 0:
                chi_signals = np.array(chi_signals)
                chi_avg_signal = np.mean(chi_signals, axis=0)
                chi_sem_signal = np.std(chi_signals, axis=0) / np.sqrt(len(chi_signals))
                
                # Store results
                results['chi_avg_signal'] = chi_avg_signal
                results['chi_sem_signal'] = chi_sem_signal
                results['chi_sample_count'] = len(chi_signals)
                
                # Create CHI plot
                p3 = figure(width=900, height=400, 
                        title=f"{signal_name} around CHI Neuron Spikes (n={len(chi_signals)})",
                        x_axis_label="Time relative to spike (s)",
                        y_axis_label=signal_name,
                        x_range=plots[0].x_range if plots else None)
                
                p3.line(results['time_axis'], chi_avg_signal, line_width=3, color='limegreen', legend_label="CHI Average")
                p3.patch(
                    np.concatenate([results['time_axis'], results['time_axis'][::-1]]),
                    np.concatenate([
                        chi_avg_signal + chi_sem_signal,
                        (chi_avg_signal - chi_sem_signal)[::-1]
                    ]),
                    color='limegreen', alpha=0.2
                )
                
                # Add vertical line at spike time
                p3.line([0, 0], [np.min(chi_avg_signal)-1, np.max(chi_avg_signal) * 1.2], 
                       line_width=2, color='black', line_dash='dashed')
                
                # Add horizontal line at y=0 for acceleration signals
                if signal_name.lower() == "acceleration":
                    zero_line = Span(location=0, dimension='width', line_color='black', line_width=2, line_dash='dotted')
                    p3.add_layout(zero_line)
                
                p3.xgrid.grid_line_color = None
                p3.ygrid.grid_line_color = None
                p3.legend.location = "top_right"
                p3.legend.click_policy = "hide"
                
                plots.append(p3)
        
        # Create comparison plot if multiple cell types are shown
        if len(plots) > 1:
            comparison_plot = figure(width=900, height=500, 
                    title=f"Comparison of {signal_name} around Cell Type Spikes",
                    x_axis_label="Time relative to spike (s)",
                    y_axis_label=signal_name,
                    x_range=plots[0].x_range if plots else None)
            
            # Add lines for each cell type
            if show_d1 and 'd1_avg_signal' in results:
                comparison_plot.line(results['time_axis'], results['d1_avg_signal'], 
                                   line_width=3, color='navy', legend_label="D1 Average")
            if show_d2 and 'd2_avg_signal' in results:
                comparison_plot.line(results['time_axis'], results['d2_avg_signal'], 
                                   line_width=3, color='crimson', legend_label="D2 Average")
            if show_chi and 'chi_avg_signal' in results:
                comparison_plot.line(results['time_axis'], results['chi_avg_signal'], 
                                   line_width=3, color='limegreen', legend_label="CHI Average")
            
            # Add vertical line at spike time
            y_min = min([np.min(results.get(f'{ct}_avg_signal', [0])) for ct in ['d1', 'd2', 'chi']])
            y_max = max([np.max(results.get(f'{ct}_avg_signal', [0])) for ct in ['d1', 'd2', 'chi']])
            comparison_plot.line([0, 0], [y_min - 1, y_max * 1.2], 
                               line_width=2, color='black', line_dash='dashed')
            
            # Add horizontal line at y=0 for acceleration signals
            if signal_name.lower() == "acceleration":
                zero_line = Span(location=0, dimension='width', line_color='black', line_width=2, line_dash='dotted')
                comparison_plot.add_layout(zero_line)
            
            comparison_plot.xgrid.grid_line_color = None
            comparison_plot.ygrid.grid_line_color = None
            comparison_plot.legend.location = "top_right"
            comparison_plot.legend.click_policy = "hide"
            
            plots.append(comparison_plot)
        
        # Create the layout
        if plots:
            layout = gridplot([[p] for p in plots])
            
            # Output the plot
            self.output_bokeh_plot(layout, save_path=save_path, title=title, notebook=notebook, 
                                 overwrite=overwrite, font_size=font_size)
            
            results['plot'] = layout
        else:
            print("No valid spikes found for the selected cell types.")
            results['plot'] = None
        
        return results

    def analyze_cell_type_spikes_signal(self, signal=None, signal_name="Signal", time_window=4, height=0.0,
                                       show_d1=True, show_d2=True, show_chi=True,
                                       save_path=None, title=None, notebook=False, overwrite=False, font_size=None):
        """
        Wrapper function to analyze and plot any signal around D1, D2, and CHI neuron spikes.
        
        Parameters:
        -----------
        signal : numpy.ndarray, optional
            Signal array to analyze (e.g., self.smoothed_velocity, self.acceleration)
            If None, uses self.smoothed_velocity
        signal_name : str
            Name of signal for plot labels (e.g., "Velocity", "Acceleration")
        time_window : float
            Time window in seconds before and after spike to analyze
        height : float
            Height threshold for peak detection
        show_d1 : bool
            Whether to analyze and show D1 neurons
        show_d2 : bool
            Whether to analyze and show D2 neurons
        show_chi : bool
            Whether to analyze and show CHI neurons
        save_path : str, optional
            Path to save the figure
        title : str, optional
            Title for the plot (if None, will be generated based on signal_name)
        notebook : bool
            Whether to display in notebook
        overwrite : bool
            Whether to overwrite existing file
        font_size : str, optional
            Font size for plot text elements
        
        Returns:
        --------
        dict : Results dictionary
        """
        # Use default signal if not provided
        if signal is None:
            signal = self.smoothed_velocity
        
        # Call the analysis function
        results = self.analyze_signal_around_cell_type_spikes(
            d1_indices=self.d1_indices,
            d2_indices=self.d2_indices,
            chi_indices=self.chi_indices,
            signal=signal,
            signal_name=signal_name,
            time_window=time_window,
            height=height,
            show_d1=show_d1,
            show_d2=show_d2,
            show_chi=show_chi,
            save_path=save_path,
            title=title,
            notebook=notebook,
            overwrite=overwrite,
            font_size=font_size
        )
        
        return results

    def _plot_neuron_centered_activity_core(self, center_cell_type, center_signals, center_peak_indices,
                                           other_signals_dict, other_peak_indices_dict,
                                           ci_rate=None, time_window=3.0, activity_window=1.0,
                                           save_path=None, title=None, notebook=False, overwrite=False, font_size=None):
        """
        Core function for plotting neuron-centered activity analysis with active neurons from other cell types.
        This reduces code redundancy across D1, D2, and CHI centered analyses.
        
        Parameters:
        -----------
        center_cell_type : str
            The cell type being centered ('D1', 'D2', or 'CHI')
        center_signals : numpy.ndarray
            Signals for the center cell type (shape: n_neurons x n_timepoints)
        center_peak_indices : list
            List of arrays containing peak indices for each center neuron
        other_signals_dict : dict
            Dictionary with other cell type signals {'D1': signals, 'D2': signals, 'CHI': signals}
        other_peak_indices_dict : dict
            Dictionary with other cell type peak indices {'D1': indices, 'D2': indices, 'CHI': indices}
        ci_rate : float, optional
            Sampling rate of calcium imaging data (Hz). If None, uses self.ci_rate
        time_window : float
            Time window in seconds before and after each peak for plotting
        activity_window : float
            Time window in seconds to look for other cell type activities around center peaks
        save_path : str, optional
            Path to save the HTML file
        title : str, optional
            Title for the plot
        notebook : bool
            Whether to display in notebook
        overwrite : bool
            Whether to overwrite existing file
        font_size : str, optional
            Font size for plot text elements
        
        Returns:
        --------
        tuple
            (bokeh.plotting.figure, dict) - The created figure and summary statistics
        """
        import numpy as np
        from bokeh.plotting import figure
        
        if ci_rate is None:
            ci_rate = self.ci_rate
        
        # Convert time windows to samples
        plot_window_samples = int(time_window * ci_rate)
        activity_window_samples = int(activity_window * ci_rate)
        
        def find_active_neurons_around_peak(peak_idx, neuron_peak_indices, window_samples):
            """Find neurons that have peaks within the specified window around a given peak."""
            active_neurons = []
            for neuron_idx, peaks in enumerate(neuron_peak_indices):
                # Check if any peak from this neuron falls within the window
                for p in peaks:
                    if abs(p - peak_idx) <= window_samples:
                        active_neurons.append(neuron_idx)
                        break  # Found at least one peak, no need to check more
            return active_neurons
        
        # Collect all valid traces
        all_center_traces = []
        all_other_traces = {cell_type: [] for cell_type in other_signals_dict.keys()}
        valid_peaks_info = []
        
        total_center_peaks = 0
        peaks_with_activity = {cell_type: 0 for cell_type in other_signals_dict.keys()}
        peaks_with_any_activity = 0
        peaks_with_multiple_activity = 0
        
        # Process each center neuron
        for neuron_idx, (center_signal, peaks) in enumerate(zip(center_signals, center_peak_indices)):
            if len(peaks) == 0:
                continue
                
            # Process each peak for this neuron
            for peak_idx in peaks:
                total_center_peaks += 1
                
                start_idx = peak_idx - plot_window_samples
                end_idx = peak_idx + plot_window_samples + 1
                
                # Check if the window is within bounds
                if start_idx >= 0 and end_idx < len(center_signal):
                    # Find active neurons from other cell types around this center peak
                    active_neurons_dict = {}
                    has_any_activity = False
                    active_cell_types = []
                    
                    for cell_type, peak_indices in other_peak_indices_dict.items():
                        active_neurons = find_active_neurons_around_peak(
                            peak_idx, peak_indices, activity_window_samples
                        )
                        active_neurons_dict[cell_type] = active_neurons
                        
                        if len(active_neurons) > 0:
                            peaks_with_activity[cell_type] += 1
                            has_any_activity = True
                            active_cell_types.append(cell_type)
                    
                    # Only include peaks that have activity from other cell types
                    if has_any_activity:
                        peaks_with_any_activity += 1
                        
                        if len(active_cell_types) > 1:
                            peaks_with_multiple_activity += 1
                        
                        # Extract center trace (main signal)
                        center_trace = center_signal[start_idx:end_idx]
                        all_center_traces.append(center_trace)
                        
                        # Extract and average traces from active neurons of other cell types
                        peak_info = {
                            'neuron_idx': neuron_idx,
                            'peak_idx': peak_idx,
                            'peak_time': peak_idx / ci_rate,
                            'active_cell_types': active_cell_types
                        }
                        
                        for cell_type, active_neurons in active_neurons_dict.items():
                            peak_info[f'n_active_{cell_type.lower()}'] = len(active_neurons)
                            peak_info[f'has_{cell_type.lower()}_activity'] = len(active_neurons) > 0
                            
                            # Extract and average traces from active neurons
                            traces_at_peak = []
                            if len(active_neurons) > 0 and cell_type in other_signals_dict:
                                other_signals = other_signals_dict[cell_type]
                                for other_neuron_idx in active_neurons:
                                    if other_neuron_idx < len(other_signals):
                                        other_signal = other_signals[other_neuron_idx]
                                        if end_idx < len(other_signal):
                                            other_trace = other_signal[start_idx:end_idx]
                                            traces_at_peak.append(other_trace)
                            
                            avg_trace = np.mean(traces_at_peak, axis=0) if traces_at_peak else np.zeros_like(center_trace)
                            all_other_traces[cell_type].append(avg_trace)
                        
                        valid_peaks_info.append(peak_info)
        
        if not all_center_traces:
            print(f"No {center_cell_type} peaks with activity from other cell types found for plotting.")
            return None, None
        
        # Calculate average traces
        avg_center_trace = np.mean(all_center_traces, axis=0)
        avg_other_traces = {}
        sem_other_traces = {}
        
        for cell_type in other_signals_dict.keys():
            if all_other_traces[cell_type]:
                avg_other_traces[cell_type] = np.mean(all_other_traces[cell_type], axis=0)
                sem_other_traces[cell_type] = np.std(all_other_traces[cell_type], axis=0) / np.sqrt(len(all_other_traces[cell_type]))
            else:
                avg_other_traces[cell_type] = np.zeros_like(avg_center_trace)
                sem_other_traces[cell_type] = np.zeros_like(avg_center_trace)
        
        # Calculate standard error for center trace
        sem_center_trace = np.std(all_center_traces, axis=0) / np.sqrt(len(all_center_traces))
        
        # Create time axis
        time_axis = np.linspace(-time_window, time_window, 2 * plot_window_samples + 1)
        
        # Create summary statistics
        stats = {
            f'total_{center_cell_type.lower()}_peaks': total_center_peaks,
            'peaks_with_any_activity': peaks_with_any_activity,
            'peaks_with_multiple_activity': peaks_with_multiple_activity,
            'percentage_with_activity': (peaks_with_any_activity / total_center_peaks) * 100 if total_center_peaks > 0 else 0,
            'n_traces_averaged': len(all_center_traces)
        }
        
        # Add individual cell type activity stats
        for cell_type in other_signals_dict.keys():
            stats[f'peaks_with_{cell_type.lower()}_activity'] = peaks_with_activity[cell_type]
        
        # Create the plot
        other_types_str = "/".join([ct for ct in other_signals_dict.keys() if ct != center_cell_type])
        plot_title = (f"{title or f'{center_cell_type}-Centered Analysis'}\n"
                      f"n={stats['n_traces_averaged']} {center_cell_type} peaks with {other_types_str} activity "
                      f"({stats['percentage_with_activity']:.1f}% of all {center_cell_type} peaks)")
        
        p = figure(width=1000, height=400,
                   title=plot_title,
                   x_axis_label=f"Time relative to {center_cell_type} peak (s)",
                   y_axis_label="Calcium Signal",
                   tools="pan,box_zoom,wheel_zoom,reset,save")
        
        # Remove grid for cleaner look
        p.grid.grid_line_color = None
        
        # Define colors for each cell type
        colors = {'D1': 'blue', 'D2': 'red', 'CHI': 'green'}
        
        # Plot the center trace (main signal) with thicker line
        center_color = colors.get(center_cell_type, 'black')
        p.line(time_axis, avg_center_trace, line_width=3, color=center_color, 
               legend_label=f"{center_cell_type} Average (n={len(all_center_traces)})")
        p.patch(np.concatenate([time_axis, time_axis[::-1]]), 
                np.concatenate([avg_center_trace - sem_center_trace, (avg_center_trace + sem_center_trace)[::-1]]),
                alpha=0.2, color=center_color)
        
        # Plot traces from other cell types (averaged from active neurons)
        for cell_type in other_signals_dict.keys():
            if cell_type != center_cell_type:
                other_color = colors.get(cell_type, 'gray')
                p.line(time_axis, avg_other_traces[cell_type], line_width=2, color=other_color, 
                       legend_label=f"{cell_type} Average (active neurons only)")
                p.patch(np.concatenate([time_axis, time_axis[::-1]]), 
                        np.concatenate([avg_other_traces[cell_type] - sem_other_traces[cell_type], 
                                      (avg_other_traces[cell_type] + sem_other_traces[cell_type])[::-1]]),
                        alpha=0.2, color=other_color)
        
        # Add vertical line at peak time (t=0)
        p.line([0, 0], [p.y_range.start if p.y_range.start else -1, 
                        p.y_range.end if p.y_range.end else 1], 
               line_width=2, color='black', line_dash='dashed', alpha=0.7)
        
        # Configure legend
        p.legend.location = "top_right"
        p.legend.click_policy = "hide"
        
        # Print summary statistics
        print(f"\n{center_cell_type}-Centered Analysis Summary Statistics:")
        print(f"Total {center_cell_type} peaks analyzed: {stats[f'total_{center_cell_type.lower()}_peaks']}")
        print(f"{center_cell_type} peaks with {other_types_str} activity: {stats['peaks_with_any_activity']} ({stats['percentage_with_activity']:.1f}%)")
        for cell_type in other_signals_dict.keys():
            if cell_type != center_cell_type:
                print(f"{center_cell_type} peaks with {cell_type} activity: {stats[f'peaks_with_{cell_type.lower()}_activity']}")
        print(f"{center_cell_type} peaks with multiple cell type activity: {stats['peaks_with_multiple_activity']}")
        print(f"Number of traces averaged: {stats['n_traces_averaged']}")
        
        # Output the plot
        self.output_bokeh_plot(p, save_path=save_path, title=plot_title, notebook=notebook, 
                             overwrite=overwrite, font_size=font_size)
        
        return p, stats

    def plot_neuron_centered_activity(self, center_cell_type='D1', signal_type='zsc',
                                     time_window=3.0, activity_window=1.0,
                                     save_path=None, title=None, notebook=False, overwrite=False, font_size=None):
        """
        Plot neuron-centered activity analysis showing average activity of the center cell type
        along with corresponding signals from other active cell types.
        
        Parameters:
        -----------
        center_cell_type : str
            The cell type to center the analysis on ('D1', 'D2', or 'CHI')
        signal_type : str
            Type of signal to use ('zsc' for z-score normalized, 'denoised' for denoised signals)
        time_window : float
            Time window in seconds before and after each peak for plotting
        activity_window : float
            Time window in seconds to look for activities from other cell types around center peaks
        save_path : str, optional
            Path to save the HTML file
        title : str, optional
            Title for the plot
        notebook : bool
            Whether to display in notebook
        overwrite : bool
            Whether to overwrite existing file
        font_size : str, optional
            Font size for plot text elements
        
        Returns:
        --------
        tuple
            (bokeh.plotting.figure, dict) - The created figure and summary statistics
        """
        # Validate inputs
        if center_cell_type not in ['D1', 'D2', 'CHI']:
            raise ValueError("center_cell_type must be one of: 'D1', 'D2', 'CHI'")
        
        if signal_type not in ['zsc', 'denoised']:
            raise ValueError("signal_type must be one of: 'zsc', 'denoised'")
        
        # Get signals based on signal type
        if signal_type == 'zsc':
            signal_suffix = '_zsc'
            signal_description = 'Z-score'
        else:
            signal_suffix = '_denoised'
            signal_description = 'Denoised'
        
        # Get center signals and peak indices
        center_signals = getattr(self, f'{center_cell_type.lower()}{signal_suffix}')
        center_peak_indices = getattr(self, f'{center_cell_type.lower()}_peak_indices')
        
        # Get other cell type signals and peak indices
        other_cell_types = [ct for ct in ['D1', 'D2', 'CHI'] if ct != center_cell_type]
        other_signals_dict = {}
        other_peak_indices_dict = {}
        
        for cell_type in other_cell_types:
            other_signals_dict[cell_type] = getattr(self, f'{cell_type.lower()}{signal_suffix}')
            other_peak_indices_dict[cell_type] = getattr(self, f'{cell_type.lower()}_peak_indices')
        
        # Generate title if not provided
        if title is None:
            other_types_str = "/".join(other_cell_types)
            title = f"{center_cell_type}-Centered Analysis: Average {center_cell_type} Activity with Active {other_types_str} ({signal_description})"
        
        # Call the core function
        return self._plot_neuron_centered_activity_core(
            center_cell_type=center_cell_type,
            center_signals=center_signals,
            center_peak_indices=center_peak_indices,
            other_signals_dict=other_signals_dict,
            other_peak_indices_dict=other_peak_indices_dict,
            ci_rate=self.ci_rate,
            time_window=time_window,
            activity_window=activity_window,
            save_path=save_path,
            title=title,
            notebook=notebook,
            overwrite=overwrite,
            font_size=font_size
        )

    def plot_all_neuron_centered_analyses(self, signal_type='zsc', activity_window=1.0, 
                                         save_path=None, notebook=False, overwrite=False, font_size=None):
        """
        Comprehensive function to plot all three neuron-centered analyses (D1, D2, CHI) for comparison.
        
        Parameters:
        -----------
        signal_type : str
            Type of signal to use ('zsc' for z-score normalized, 'denoised' for denoised signals)
        activity_window : float
            Time window in seconds to look for activities around peaks
        save_path : str, optional
            Path to save the combined HTML file
        notebook : bool
            Whether to display in notebook
        overwrite : bool
            Whether to overwrite existing file
        font_size : str, optional
            Font size for plot text elements
        
        Returns:
        --------
        dict
            Dictionary containing plots and statistics for all three analyses
        """
        from bokeh.layouts import column
        from bokeh.io import output_file, save
        
        results = {}
        signal_description = 'Z-score' if signal_type == 'zsc' else 'Denoised'
        
        # D1-centered analysis
        print(f"Running D1-centered analysis ({signal_description})...")
        d1_plot, d1_stats = self.plot_neuron_centered_activity(
            center_cell_type='D1',
            signal_type=signal_type,
            time_window=3.0,
            activity_window=activity_window,
            save_path=None,  # Don't save individual plots
            title=f"D1-Centered Analysis: Average D1 Activity with Active D2/CHI ({signal_description})",
            notebook=False,  # Don't display individual plots
            overwrite=overwrite,
            font_size=font_size
        )
        results['d1_centered'] = {'plot': d1_plot, 'stats': d1_stats}
        
        # D2-centered analysis
        print(f"\nRunning D2-centered analysis ({signal_description})...")
        d2_plot, d2_stats = self.plot_neuron_centered_activity(
            center_cell_type='D2',
            signal_type=signal_type,
            time_window=3.0,
            activity_window=activity_window,
            save_path=None,  # Don't save individual plots
            title=f"D2-Centered Analysis: Average D2 Activity with Active D1/CHI ({signal_description})",
            notebook=False,  # Don't display individual plots
            overwrite=overwrite,
            font_size=font_size
        )
        results['d2_centered'] = {'plot': d2_plot, 'stats': d2_stats}
        
        # CHI-centered analysis
        print(f"\nRunning CHI-centered analysis ({signal_description})...")
        chi_plot, chi_stats = self.plot_neuron_centered_activity(
            center_cell_type='CHI',
            signal_type=signal_type,
            time_window=3.0,
            activity_window=activity_window,
            save_path=None,  # Don't save individual plots
            title=f"CHI-Centered Analysis: Average CHI Activity with Active D1/D2 ({signal_description})",
            notebook=False,  # Don't display individual plots
            overwrite=overwrite,
            font_size=font_size
        )
        results['chi_centered'] = {'plot': chi_plot, 'stats': chi_stats}
        
        # Print comparative summary
        print("\n" + "="*80)
        print(f"COMPARATIVE ANALYSIS SUMMARY ({signal_description})")
        print("="*80)
        
        if d1_stats and d2_stats and chi_stats:
            print(f"D1-centered: {d1_stats['percentage_with_activity']:.1f}% of D1 peaks have D2/CHI activity")
            print(f"D2-centered: {d2_stats['percentage_with_activity']:.1f}% of D2 peaks have D1/CHI activity")
            print(f"CHI-centered: {chi_stats['percentage_with_activity']:.1f}% of CHI peaks have D1/D2 activity")
            
            print(f"\nTotal peaks analyzed:")
            print(f"  D1: {d1_stats['total_d1_peaks']} peaks")
            print(f"  D2: {d2_stats['total_d2_peaks']} peaks")
            print(f"  CHI: {chi_stats['total_chi_peaks']} peaks")
            
            print(f"\nCo-activation patterns:")
            print(f"  D1 peaks with D2 activity: {d1_stats['peaks_with_d2_activity']}")
            print(f"  D1 peaks with CHI activity: {d1_stats['peaks_with_chi_activity']}")
            print(f"  D2 peaks with D1 activity: {d2_stats['peaks_with_d1_activity']}")
            print(f"  D2 peaks with CHI activity: {d2_stats['peaks_with_chi_activity']}")
            print(f"  CHI peaks with D1 activity: {chi_stats['peaks_with_d1_activity']}")
            print(f"  CHI peaks with D2 activity: {chi_stats['peaks_with_d2_activity']}")
        
        # Create and save combined analysis
        if all([d1_plot, d2_plot, chi_plot]):
            combined_layout = column(d1_plot, d2_plot, chi_plot)
            combined_title = f"All Neuron-Centered Analyses ({signal_description})"
            
            # Output the combined plot
            self.output_bokeh_plot(combined_layout, save_path=save_path, title=combined_title, 
                                 notebook=notebook, overwrite=overwrite, font_size=font_size)
            
            results['combined_plot'] = combined_layout
            
            if save_path:
                print(f"\nCombined analysis saved to: {save_path}")
        
        return results

    # Convenience wrapper functions for backward compatibility and ease of use
    def plot_d1_centered_analysis(self, signal_type='zsc', save_path=None, activity_window=1.0, 
                                 notebook=False, overwrite=False, font_size=None):
        """
        Convenience function to plot D1-centered analysis.
        
        Parameters:
        -----------
        signal_type : str
            Type of signal to use ('zsc' for z-score normalized, 'denoised' for denoised signals)
        save_path : str, optional
            Path to save the HTML file
        activity_window : float
            Time window in seconds to look for D2/CHI activities around D1 peaks
        notebook : bool
            Whether to display in notebook
        overwrite : bool
            Whether to overwrite existing file
        font_size : str, optional
            Font size for plot text elements
        
        Returns:
        --------
        tuple
            (bokeh.plotting.figure, dict) - The created figure and summary statistics
        """
        return self.plot_neuron_centered_activity(
            center_cell_type='D1',
            signal_type=signal_type,
            time_window=3.0,
            activity_window=activity_window,
            save_path=save_path,
            title=f"Average D1 Activity with Active D2 and CHI Neurons ({'Z-score' if signal_type == 'zsc' else 'Denoised'})",
            notebook=notebook,
            overwrite=overwrite,
            font_size=font_size
        )

    def plot_d2_centered_analysis(self, signal_type='zsc', save_path=None, activity_window=1.0,
                                 notebook=False, overwrite=False, font_size=None):
        """
        Convenience function to plot D2-centered analysis.
        
        Parameters:
        -----------
        signal_type : str
            Type of signal to use ('zsc' for z-score normalized, 'denoised' for denoised signals)
        save_path : str, optional
            Path to save the HTML file
        activity_window : float
            Time window in seconds to look for D1/CHI activities around D2 peaks
        notebook : bool
            Whether to display in notebook
        overwrite : bool
            Whether to overwrite existing file
        font_size : str, optional
            Font size for plot text elements
        
        Returns:
        --------
        tuple
            (bokeh.plotting.figure, dict) - The created figure and summary statistics
        """
        return self.plot_neuron_centered_activity(
            center_cell_type='D2',
            signal_type=signal_type,
            time_window=3.0,
            activity_window=activity_window,
            save_path=save_path,
            title=f"Average D2 Activity with Active D1 and CHI Neurons ({'Z-score' if signal_type == 'zsc' else 'Denoised'})",
            notebook=notebook,
            overwrite=overwrite,
            font_size=font_size
        )

    def plot_chi_centered_analysis(self, signal_type='zsc', save_path=None, activity_window=1.0,
                                  notebook=False, overwrite=False, font_size=None):
        """
        Convenience function to plot CHI-centered analysis.
        
        Parameters:
        -----------
        signal_type : str
            Type of signal to use ('zsc' for z-score normalized, 'denoised' for denoised signals)
        save_path : str, optional
            Path to save the HTML file
        activity_window : float
            Time window in seconds to look for D1/D2 activities around CHI peaks
        notebook : bool
            Whether to display in notebook
        overwrite : bool
            Whether to overwrite existing file
        font_size : str, optional
            Font size for plot text elements
        
        Returns:
        --------
        tuple
            (bokeh.plotting.figure, dict) - The created figure and summary statistics
        """
        return self.plot_neuron_centered_activity(
            center_cell_type='CHI',
            signal_type=signal_type,
            time_window=3.0,
            activity_window=activity_window,
            save_path=save_path,
            title=f"Average CHI Activity with Active D1 and D2 Neurons ({'Z-score' if signal_type == 'zsc' else 'Denoised'})",
            notebook=notebook,
            overwrite=overwrite,
            font_size=font_size
        )

    class NeuralConnectivityAnalyzer:
        """
        Nested class for analyzing functional connectivity between neural types.
        Directly accesses parent CellTypeTank data without redundant property declarations.
        """
        
        def __init__(self, parent_tank, save_dir=None, use_rising_edges=True):
            """
            Initialize analyzer with reference to parent CellTypeTank and optionally run full analysis
            
            Parameters:
            -----------
            parent_tank : CellTypeTank
                Parent CellTypeTank object containing neural data
            save_dir : str, optional
                Directory to save results
            use_rising_edges : bool
                Whether to use rising edges instead of peak indices
            run_analysis : bool
                Whether to run the full analysis during initialization
            """
            self.tank = parent_tank

            print(f"Neural Connectivity Analyzer initialized:")
            print(f"D1 neurons: {len(self.tank.d1_peak_indices)} cells, total {sum(len(peaks) for peaks in self.tank.d1_peak_indices)} peaks")
            print(f"D2 neurons: {len(self.tank.d2_peak_indices)} cells, total {sum(len(peaks) for peaks in self.tank.d2_peak_indices)} peaks")
            print(f"CHI neurons: {len(self.tank.chi_peak_indices)} cells, total {sum(len(peaks) for peaks in self.tank.chi_peak_indices)} peaks")

            # Store analysis results - initialize all properties that will be set by analysis methods
            self.binary_signals = None
            self.conditional_probs = None
            self.cross_correlations = None
            self.coactivation_patterns = None
            self.peak_timing_relationships = None
            self.mutual_information = None
            self.network = None
            self.analysis_results = None
            
            # Store parameters for later use
            self.use_rising_edges = use_rising_edges
            self.save_dir = save_dir
            
            print("Neural Connectivity Analyzer ready. Call individual analysis methods to generate results.")
        
        def get_analysis_results(self):
            """
            Get all analysis results as a dictionary
            
            Returns:
            --------
            dict : Analysis results
            """
            return {
                'binary_signals': self.binary_signals,
                'conditional_probs': self.conditional_probs,
                'cross_correlations': self.cross_correlations,
                'coactivation_patterns': self.coactivation_patterns,
                'peak_timing_relationships': self.peak_timing_relationships,
                'mutual_information': self.mutual_information,
                'network': self.network
            }
        
        def run_binary_signals_analysis(self, window_size=5):
            """
            Run binary signals analysis step
            
            Parameters:
            -----------
            window_size : int
                Duration window size for activation events (in samples)
            
            Returns:
            --------
            dict : Binary signals results
            """
            print("Running binary signals analysis...")
            self.binary_signals = self.create_binary_signals_from_peaks(window_size=window_size, use_rising_edges=self.use_rising_edges)
            print("Binary signals analysis completed!")
            return self.binary_signals
        
        def run_conditional_probabilities_analysis(self, time_windows=[5, 10, 20, 50], method='individual'):
            """
            Run conditional probabilities analysis step
            
            Parameters:
            -----------
            time_windows : list
                Different time window sizes (in samples)
            method : str
                'individual' - calculate probabilities between individual neurons
                'population' - calculate probabilities between cell type populations
            
            Returns:
            --------
            dict : Conditional probability results
            """
            print("Running conditional probabilities analysis...")
            if self.binary_signals is None:
                print("Binary signals not available. Running binary signals analysis first...")
                self.run_binary_signals_analysis()
            
            self.conditional_probs = self.calculate_conditional_probabilities(time_windows=time_windows, method=method)
            print("Conditional probabilities analysis completed!")
            return self.conditional_probs
        
        def run_cross_correlations_analysis(self, max_lag=100):
            """
            Run cross-correlations analysis step
            
            Parameters:
            -----------
            max_lag : int
                Maximum lag for cross-correlation analysis
            
            Returns:
            --------
            dict : Cross-correlation results
            """
            print("Running cross-correlations analysis...")
            if self.binary_signals is None:
                print("Binary signals not available. Running binary signals analysis first...")
                self.run_binary_signals_analysis()
            
            self.cross_correlations = self.calculate_cross_correlations_from_peaks(max_lag=max_lag)
            print("Cross-correlations analysis completed!")
            return self.cross_correlations
        
        def run_coactivation_patterns_analysis(self, min_duration=3):
            """
            Run coactivation patterns analysis step
            
            Parameters:
            -----------
            min_duration : int
                Minimum duration for coactivation patterns
            
            Returns:
            --------
            dict : Coactivation patterns results
            """
            print("Running coactivation patterns analysis...")
            if self.binary_signals is None:
                print("Binary signals not available. Running binary signals analysis first...")
                self.run_binary_signals_analysis()
            
            self.coactivation_patterns = self.identify_coactivation_patterns_from_peaks(min_duration=min_duration)
            print("Coactivation patterns analysis completed!")
            return self.coactivation_patterns
        
        def run_peak_timing_analysis(self):
            """
            Run peak timing relationships analysis step
            
            Returns:
            --------
            dict : Peak timing relationships results
            """
            print("Running peak timing relationships analysis...")
            self.peak_timing_relationships = self.calculate_peak_timing_relationships()
            print("Peak timing relationships analysis completed!")
            return self.peak_timing_relationships
        
        def run_mutual_information_analysis(self, bins=10):
            """
            Run mutual information analysis step
            
            Parameters:
            -----------
            bins : int
                Number of bins for mutual information calculation
            
            Returns:
            --------
            dict : Mutual information results
            """
            print("Running mutual information analysis...")
            if self.binary_signals is None:
                print("Binary signals not available. Running binary signals analysis first...")
                self.run_binary_signals_analysis()
            
            self.mutual_information = self.calculate_mutual_information_from_peaks(bins=bins)
            print("Mutual information analysis completed!")
            return self.mutual_information
        
        def run_connectivity_network_analysis(self, threshold=0.1, use_individual_neurons=False):
            """
            Run connectivity network analysis step
            
            Parameters:
            -----------
            threshold : float
                Connection strength threshold
            use_individual_neurons : bool
                If True, create network with individual neurons as nodes
                If False, create network with cell types as nodes (default)
            
            Returns:
            --------
            networkx.Graph : Network graph object
            """
            print("Running connectivity network analysis...")
            if self.conditional_probs is None:
                print("Conditional probabilities not available. Running conditional probabilities analysis first...")
                self.run_conditional_probabilities_analysis()
            
            self.network = self.create_connectivity_network(threshold=threshold, use_individual_neurons=use_individual_neurons)
            print("Connectivity network analysis completed!")
            return self.network
        
        def run_all_analyses(self, window_size=5, time_windows=[5, 10, 20, 50], max_lag=100, 
                           min_duration=3, bins=10, threshold=0.1, use_individual_neurons=False):
            """
            Run all analysis steps in sequence
            
            Parameters:
            -----------
            window_size : int
                Duration window size for activation events
            time_windows : list
                Different time window sizes for conditional probabilities
            max_lag : int
                Maximum lag for cross-correlation analysis
            min_duration : int
                Minimum duration for coactivation patterns
            bins : int
                Number of bins for mutual information calculation
            threshold : float
                Connection strength threshold
            use_individual_neurons : bool
                Whether to use individual neurons in network analysis
            
            Returns:
            --------
            dict : Complete analysis results
            """
            print("Running all neural connectivity analyses...")
            
            # Run all analysis steps
            self.run_binary_signals_analysis(window_size=window_size)
            self.run_conditional_probabilities_analysis(time_windows=time_windows)
            self.run_cross_correlations_analysis(max_lag=max_lag)
            self.run_coactivation_patterns_analysis(min_duration=min_duration)
            self.run_peak_timing_analysis()
            self.run_mutual_information_analysis(bins=bins)
            self.run_connectivity_network_analysis(threshold=threshold, use_individual_neurons=use_individual_neurons)
            
            # Generate summary report
            self.generate_summary_report()
            self.analysis_results = self.get_analysis_results()
            
            print("All neural connectivity analyses completed!")
            return self.analysis_results
        
        def create_timing_plot(self, save_path=None, notebook=False, overwrite=False, font_size=None):
            """
            Create and save timing relationships plot
            
            Parameters:
            -----------
            save_path : str, optional
                Path to save the plot
            notebook : bool
                Whether to display in notebook
            overwrite : bool
                Whether to overwrite existing file
            font_size : str, optional
                Font size for plot text elements
            
            Returns:
            --------
            bokeh.layouts.layout : The created plot layout
            """
            print("Creating timing relationships plot...")
            
            if self.peak_timing_relationships is None:
                print("Peak timing relationships not available. Running peak timing analysis first...")
                self.run_peak_timing_analysis()
            
            # Set default save path if none provided
            if save_path is None and self.save_dir:
                save_path = os.path.join(self.save_dir, 'peak_timing_relationships.html')
            
            plot = self.plot_peak_timing_relationships(
                save_path=save_path,
                notebook=notebook,
                overwrite=overwrite,
                font_size=font_size
            )
            
            print("Timing relationships plot created!")
            return plot
        
        def create_summary_plot(self, save_path=None, notebook=False, overwrite=False, font_size=None):
            """
            Create and save connectivity analysis summary plot
            
            Parameters:
            -----------
            save_path : str, optional
                Path to save the plot
            notebook : bool
                Whether to display in notebook
            overwrite : bool
                Whether to overwrite existing file
            font_size : str, optional
                Font size for plot text elements
            
            Returns:
            --------
            bokeh.layouts.layout : The created plot layout
            """
            print("Creating connectivity analysis summary plot...")
            
            # Ensure all analyses are completed
            if not hasattr(self, 'analysis_results') or self.analysis_results is None:
                print("Analysis results not available. Running all analyses first...")
                self.run_all_analyses()
            
            # Set default save path if none provided
            if save_path is None and self.save_dir:
                save_path = os.path.join(self.save_dir, 'connectivity_summary.html')
            
            plot = self.plot_connectivity_analysis_summary(
                save_path=save_path,
                notebook=notebook,
                overwrite=overwrite,
                font_size=font_size
            )
            
            print("Connectivity analysis summary plot created!")
            return plot
        
        def create_binary_signals_from_peaks(self, window_size=5, use_rising_edges=True):
            """
            Create binary activation signals based on existing peak indices
            
            Parameters:
            -----------
            window_size : int
                Duration window size for activation events (in samples)
            use_rising_edges : bool
                Whether to use rising edges instead of peak indices
            
            Returns:
            --------
            dict : Contains binary signals for each neural type
            """
            print("Creating binary activation signals from existing peak indices...")
            
            # Choose between peak indices or rising edges
            if use_rising_edges:
                d1_events = self.tank.d1_rising_edges_starts
                d2_events = self.tank.d2_rising_edges_starts
                chi_events = self.tank.chi_rising_edges_starts
                print("Using rising edges data")
            else:
                d1_events = self.tank.d1_peak_indices
                d2_events = self.tank.d2_peak_indices
                chi_events = self.tank.chi_peak_indices
                print("Using peak indices data")
            
            signal_length = len(self.tank.d1_denoised[0])
            
            # Create D1 binary signals
            d1_binary = np.zeros((len(d1_events), signal_length))
            for i, events in enumerate(d1_events):
                for event in events:
                    start = max(0, event - window_size//2)
                    end = min(signal_length, event + window_size//2 + 1)
                    d1_binary[i, start:end] = 1
            
            # Create D2 binary signals
            d2_binary = np.zeros((len(d2_events), signal_length))
            for i, events in enumerate(d2_events):
                for event in events:
                    start = max(0, event - window_size//2)
                    end = min(signal_length, event + window_size//2 + 1)
                    d2_binary[i, start:end] = 1
            
            # Create CHI binary signals
            chi_binary = np.zeros((len(chi_events), signal_length))
            for i, events in enumerate(chi_events):
                for event in events:
                    start = max(0, event - window_size//2)
                    end = min(signal_length, event + window_size//2 + 1)
                    chi_binary[i, start:end] = 1
            
            binary_signals = {
                'D1': d1_binary,
                'D2': d2_binary,
                'CHI': chi_binary
            }
            
            return binary_signals
        
        def calculate_conditional_probabilities(self, time_windows=[5, 10, 20, 50], method='individual'):
            """
            Calculate conditional activation probabilities
            
            Parameters:
            -----------
            time_windows : list
                Different time window sizes (in samples)
            method : str
                'individual' - calculate probabilities between individual neurons
                'population' - calculate probabilities between cell type populations (legacy)
                'cell_type_proportion' - calculate using average proportion of target neurons activated per source activation
            
            Returns:
            --------
            dict : Conditional probability results
            """
            print(f"Calculating conditional activation probabilities using {method} method...")
            
            if method == 'individual':
                return self._calculate_individual_conditional_probabilities(time_windows)
            elif method == 'population':
                return self._calculate_population_conditional_probabilities(time_windows)
            elif method == 'cell_type_proportion':
                return self._calculate_cell_type_conditional_probabilities(time_windows)
            else:
                raise ValueError("Method must be 'individual', 'population', or 'cell_type_proportion'")
        
        def _calculate_individual_conditional_probabilities(self, time_windows):
            """Calculate conditional probabilities between individual neurons"""
            results = {}
            
            # Get all neuron signals and their cell type labels
            all_signals = []
            neuron_labels = []
            cell_types = []
            
            for cell_type in ['D1', 'D2', 'CHI']:
                signals = self.binary_signals[cell_type]
                for i in range(signals.shape[0]):
                    all_signals.append(signals[i])
                    neuron_labels.append(f"{cell_type}_{i}")
                    cell_types.append(cell_type)
            
            all_signals = np.array(all_signals)
            n_neurons = len(all_signals)
            
            # Also calculate cell type level conditional probabilities using new method
            cell_type_probs = self._calculate_cell_type_conditional_probabilities(time_windows)
            
            def calc_conditional_prob_individual(signal_a, signal_b, window_size):
                """Calculate probability of B activation within window_size time after A activation"""
                a_events = np.where(signal_a == 1)[0]
                if len(a_events) == 0:
                    return 0.0
                
                conditional_activations = 0
                for event in a_events:
                    # Check time window after the event
                    start = event + 1
                    end = min(len(signal_b), event + window_size + 1)
                    if start < len(signal_b) and np.any(signal_b[start:end]):
                        conditional_activations += 1
                
                return conditional_activations / len(a_events)
            
            for window in time_windows:
                print(f"Processing window size: {window}")
                
                # Initialize probability matrix
                prob_matrix = np.zeros((n_neurons, n_neurons))
                
                # Calculate conditional probabilities for all neuron pairs
                for i in range(n_neurons):
                    for j in range(n_neurons):
                        if i != j:  # Skip self-connections
                            prob_matrix[i, j] = calc_conditional_prob_individual(
                                all_signals[i], all_signals[j], window
                            )
                
                # Store results
                window_results = {
                    'probability_matrix': prob_matrix,
                    'neuron_labels': neuron_labels,
                    'cell_types': cell_types,
                    'n_neurons': n_neurons
                }
                
                # Add cell type aggregated results for compatibility
                window_results.update(self._aggregate_by_cell_type(prob_matrix, cell_types))
                
                # Add new cell type level results
                if f'window_{window}' in cell_type_probs:
                    window_results.update(cell_type_probs[f'window_{window}'])
                
                results[f'window_{window}'] = window_results
            
            return results
        
        def _calculate_cell_type_conditional_probabilities(self, time_windows):
            """Calculate conditional probabilities between cell types using new method:
            Average proportion of target cell type neurons activated per source cell type activation"""
            results = {}
            
            # Get cell type signals
            d1_signals = self.binary_signals['D1']  # Shape: (n_d1_neurons, time_points)
            d2_signals = self.binary_signals['D2']  # Shape: (n_d2_neurons, time_points)
            chi_signals = self.binary_signals['CHI']  # Shape: (n_chi_neurons, time_points)
            
            cell_type_signals = {
                'D1': d1_signals,
                'D2': d2_signals, 
                'CHI': chi_signals
            }
            
            def calc_cell_type_conditional_prob(source_signals, target_signals, window_size):
                """Calculate P(target_cell_type_activated | source_cell_type_activated)
                using average proportion of target neurons activated per source activation"""
                
                # Find all activation events of source cell type (any neuron in source type)
                source_population = np.sum(source_signals, axis=0)  # Sum across all source neurons
                source_events = np.where(source_population > 0)[0]  # Times when any source neuron is active
                
                if len(source_events) == 0:
                    return 0.0
                
                total_target_proportion = 0.0
                for event in source_events:
                    # Check time window after the event
                    start = event + 1
                    end = min(target_signals.shape[1], event + window_size + 1)
                    if start < target_signals.shape[1]:
                        # Count how many target neurons are activated in the window
                        target_activations_in_window = np.sum(target_signals[:, start:end])
                        # Calculate proportion of target neurons activated
                        target_proportion = target_activations_in_window / target_signals.shape[0] if target_signals.shape[0] > 0 else 0
                        total_target_proportion += target_proportion
                
                # Return average proportion across all source activations
                return total_target_proportion / len(source_events)
            
            for window in time_windows:
                print(f"Processing cell type level window size: {window}")
                
                window_results = {}
                
                # Calculate conditional probabilities for all cell type pairs
                for source_type in ['D1', 'D2', 'CHI']:
                    for target_type in ['D1', 'D2', 'CHI']:
                        if source_type != target_type:
                            prob = calc_cell_type_conditional_prob(
                                cell_type_signals[source_type], 
                                cell_type_signals[target_type], 
                                window
                            )
                            window_results[f'{source_type}_to_{target_type}'] = prob
                
                results[f'window_{window}'] = window_results
            
            return results
        
        def _aggregate_by_cell_type(self, prob_matrix, cell_types):
            """Aggregate individual neuron probabilities by cell type"""
            aggregated = {}
            
            # Create indices for each cell type
            cell_type_indices = {}
            for i, cell_type in enumerate(cell_types):
                if cell_type not in cell_type_indices:
                    cell_type_indices[cell_type] = []
                cell_type_indices[cell_type].append(i)
            
            # Calculate mean probabilities between cell types
            for source_type in ['D1', 'D2', 'CHI']:
                for target_type in ['D1', 'D2', 'CHI']:
                    if source_type != target_type:
                        source_indices = cell_type_indices[source_type]
                        target_indices = cell_type_indices[target_type]
                        
                        # Extract submatrix and calculate mean
                        submatrix = prob_matrix[np.ix_(source_indices, target_indices)]
                        mean_prob = np.mean(submatrix)
                        aggregated[f'{source_type}_to_{target_type}'] = mean_prob
            
            return aggregated
        
        def _calculate_population_conditional_probabilities(self, time_windows):
            """Legacy method: Calculate conditional probabilities between cell type populations"""
            results = {}
            
            for window in time_windows:
                window_results = {}
                
                # Calculate population activation signals (how many neurons are active at each time point)
                d1_population = np.sum(self.binary_signals['D1'], axis=0)
                d2_population = np.sum(self.binary_signals['D2'], axis=0)
                chi_population = np.sum(self.binary_signals['CHI'], axis=0)
                
                # Binarize population signals (at least one neuron active)
                d1_active = (d1_population > 0).astype(int)
                d2_active = (d2_population > 0).astype(int)
                chi_active = (chi_population > 0).astype(int)
                
                # Calculate conditional probability P(B activated | A activated)
                def calc_conditional_prob(signal_a, signal_b, window_size):
                    """Calculate probability of B activation within window_size time after A activation"""
                    a_events = np.where(signal_a == 1)[0]
                    if len(a_events) == 0:
                        return 0.0
                    
                    conditional_activations = 0
                    for event in a_events:
                        # Check time window after the event
                        start = event + 1
                        end = min(len(signal_b), event + window_size + 1)
                        if start < len(signal_b) and np.any(signal_b[start:end]):
                            conditional_activations += 1
                    
                    return conditional_activations / len(a_events)
                
                # Calculate conditional probabilities for all pairs
                pairs = [('D1', 'D2'), ('D1', 'CHI'), ('D2', 'D1'), 
                        ('D2', 'CHI'), ('CHI', 'D1'), ('CHI', 'D2')]
                
                signals = {'D1': d1_active, 'D2': d2_active, 'CHI': chi_active}
                
                for source, target in pairs:
                    prob = calc_conditional_prob(signals[source], signals[target], window)
                    window_results[f'{source}_to_{target}'] = prob
                
                results[f'window_{window}'] = window_results
            
            return results
        
        def calculate_cross_correlations_from_peaks(self, max_lag=100):
            """
            Calculate time-lagged cross-correlations based on existing peak indices
            
            Parameters:
            -----------
            max_lag : int
                Maximum lag time (in samples)
            
            Returns:
            --------
            dict : Cross-correlation results
            """
            print("Calculating time-lagged cross-correlations from peak indices...")
            
            # Calculate population activation signals
            d1_population = np.sum(self.binary_signals['D1'], axis=0)
            d2_population = np.sum(self.binary_signals['D2'], axis=0)
            chi_population = np.sum(self.binary_signals['CHI'], axis=0)
            
            signals = {
                'D1': d1_population,
                'D2': d2_population,
                'CHI': chi_population
            }
            
            results = {}
            lags = np.arange(-max_lag, max_lag + 1)
            
            # Calculate cross-correlations for all pairs
            pairs = [('D1', 'D2'), ('D1', 'CHI'), ('D2', 'CHI')]
            
            for source, target in pairs:
                correlation = np.correlate(signals[source], signals[target], mode='full')
                
                # Normalize
                n = len(signals[source])
                std_source = np.std(signals[source])
                std_target = np.std(signals[target])
                
                if std_source > 0 and std_target > 0:
                    correlation = correlation / (std_source * std_target * n)
                else:
                    correlation = np.zeros_like(correlation)
                
                # Extract lag range of interest
                center = len(correlation) // 2
                start = center - max_lag
                end = center + max_lag + 1
                correlation = correlation[start:end]
                
                results[f'{source}_vs_{target}'] = {
                    'lags': lags,
                    'correlation': correlation,
                    'peak_lag': lags[np.argmax(np.abs(correlation))],
                    'peak_correlation': np.max(np.abs(correlation))
                }
            
            return results
        
        def identify_coactivation_patterns_from_peaks(self, min_duration=3):
            """
            Identify co-activation patterns based on existing peak indices
            
            Parameters:
            -----------
            min_duration : int
                Minimum co-activation duration (in samples)
            
            Returns:
            --------
            dict : Co-activation pattern statistics
            """
            print("Identifying co-activation patterns from peak indices...")
            
            # Calculate population activation signals
            d1_active = (np.sum(self.binary_signals['D1'], axis=0) > 0).astype(int)
            d2_active = (np.sum(self.binary_signals['D2'], axis=0) > 0).astype(int)
            chi_active = (np.sum(self.binary_signals['CHI'], axis=0) > 0).astype(int)
            
            # Create combination activation patterns
            patterns = {
                'D1_only': d1_active * (1 - d2_active) * (1 - chi_active),
                'D2_only': (1 - d1_active) * d2_active * (1 - chi_active),
                'CHI_only': (1 - d1_active) * (1 - d2_active) * chi_active,
                'D1_D2': d1_active * d2_active * (1 - chi_active),
                'D1_CHI': d1_active * (1 - d2_active) * chi_active,
                'D2_CHI': (1 - d1_active) * d2_active * chi_active,
                'D1_D2_CHI': d1_active * d2_active * chi_active,
                'None': (1 - d1_active) * (1 - d2_active) * (1 - chi_active)
            }
            
            # Calculate duration and frequency statistics for each pattern
            results = {}
            
            for pattern_name, pattern_signal in patterns.items():
                # Find continuous activation segments
                diff = np.diff(np.concatenate(([0], pattern_signal, [0])))
                starts = np.where(diff == 1)[0]
                ends = np.where(diff == -1)[0]
                
                durations = ends - starts
                valid_episodes = durations >= min_duration
                
                results[pattern_name] = {
                    'total_episodes': len(durations),
                    'valid_episodes': np.sum(valid_episodes),
                    'total_duration': np.sum(durations),
                    'valid_duration': np.sum(durations[valid_episodes]),
                    'mean_duration': np.mean(durations) if len(durations) > 0 else 0,
                    'proportion': np.sum(pattern_signal) / len(pattern_signal)
                }
            
            return results
        
        def calculate_peak_timing_relationships(self):
            """
            Analyze peak timing relationships between different neural types
            
            Returns:
            --------
            dict : Peak timing relationship analysis results
            """
            print("Analyzing peak timing relationships...")
            
            # Convert all peaks to time series
            def peaks_to_times(peak_indices_list):
                """Convert peak indices to time points (seconds)"""
                all_times = []
                for peaks in peak_indices_list:
                    times = peaks / self.tank.ci_rate
                    all_times.extend(times)
                return np.array(sorted(all_times))
            
            d1_times = peaks_to_times(self.tank.d1_peak_indices)
            d2_times = peaks_to_times(self.tank.d2_peak_indices)
            chi_times = peaks_to_times(self.tank.chi_peak_indices)
            
            results = {}
            
            # Calculate inter-peak interval distributions
            def calc_inter_peak_intervals(times):
                if len(times) > 1:
                    return np.diff(times)
                return np.array([])
            
            results['inter_peak_intervals'] = {
                'D1': calc_inter_peak_intervals(d1_times),
                'D2': calc_inter_peak_intervals(d2_times),
                'CHI': calc_inter_peak_intervals(chi_times)
            }
            
            # Calculate temporal proximity between different types
            def calc_peak_proximity(times1, times2, max_distance=2.0):
                """Calculate temporal proximity between two peak time groups (within seconds)"""
                proximities = []
                for t1 in times1:
                    distances = np.abs(times2 - t1)
                    min_distance = np.min(distances) if len(distances) > 0 else np.inf
                    if min_distance <= max_distance:
                        proximities.append(min_distance)
                return np.array(proximities)
            
            results['peak_proximities'] = {
                'D1_to_D2': calc_peak_proximity(d1_times, d2_times),
                'D1_to_CHI': calc_peak_proximity(d1_times, chi_times),
                'D2_to_D1': calc_peak_proximity(d2_times, d1_times),
                'D2_to_CHI': calc_peak_proximity(d2_times, chi_times),
                'CHI_to_D1': calc_peak_proximity(chi_times, d1_times),
                'CHI_to_D2': calc_peak_proximity(chi_times, d2_times)
            }
            
            # Calculate sequential activation patterns (activation order within time window) - optimized version
            def find_sequential_activations(times1, times2, times3, window=1.0, max_sequence_length=5):
                """Find sequential activation patterns within time window, limiting sequence length"""
                sequences = []
                
                # Combine all time points and mark types
                all_events = []
                for t in times1:
                    all_events.append((t, 'D1'))
                for t in times2:
                    all_events.append((t, 'D2'))
                for t in times3:
                    all_events.append((t, 'CHI'))
                
                # Sort by time
                all_events.sort(key=lambda x: x[0])
                
                # Find sequences within window, limiting sequence length
                for i, (start_time, start_type) in enumerate(all_events):
                    window_events = [(start_time, start_type)]
                    
                    for j in range(i+1, len(all_events)):
                        event_time, event_type = all_events[j]
                        if event_time - start_time <= window:
                            window_events.append((event_time, event_type))
                            # Limit sequence length
                            if len(window_events) >= max_sequence_length:
                                break
                        else:
                            break
                    
                    if len(window_events) >= 2:
                        sequence = tuple([event[1] for event in window_events])
                        sequences.append(sequence)
                
                return sequences
            
            sequences = find_sequential_activations(d1_times, d2_times, chi_times, 
                                                  window=1.0, max_sequence_length=5)
            
            # Count sequence patterns
            from collections import Counter
            sequence_counts = Counter(sequences)
            results['sequential_patterns'] = dict(sequence_counts)
            
            return results
        
        def calculate_mutual_information_from_peaks(self, bins=10):
            """
            Calculate mutual information between neural types based on existing peak indices
            
            Parameters:
            -----------
            bins : int
                Number of bins for discretization
            
            Returns:
            --------
            dict : Mutual information matrix
            """
            print("Calculating mutual information from peak data...")
            
            # Calculate population activation strength
            d1_strength = np.sum(self.binary_signals['D1'], axis=0)
            d2_strength = np.sum(self.binary_signals['D2'], axis=0)
            chi_strength = np.sum(self.binary_signals['CHI'], axis=0)
            
            # Discretize signals
            def discretize(signal, bins):
                if np.max(signal) > 0:
                    return np.digitize(signal, np.linspace(0, np.max(signal), bins))
                else:
                    return np.zeros_like(signal, dtype=int)
            
            d1_discrete = discretize(d1_strength, bins)
            d2_discrete = discretize(d2_strength, bins)
            chi_discrete = discretize(chi_strength, bins)
            
            signals = {
                'D1': d1_discrete,
                'D2': d2_discrete,
                'CHI': chi_discrete
            }
            
            # Calculate mutual information for all pairs
            from sklearn.metrics import mutual_info_score
            results = {}
            pairs = [('D1', 'D2'), ('D1', 'CHI'), ('D2', 'CHI')]
            
            for source, target in pairs:
                mi = mutual_info_score(signals[source], signals[target])
                results[f'{source}_vs_{target}'] = mi
            
            self.mutual_information = results
            return results
        
        def create_connectivity_network(self, threshold=0.1, use_individual_neurons=False):
            """
            Create functional connectivity network
            
            Parameters:
            -----------
            threshold : float
                Connection strength threshold
            use_individual_neurons : bool
                If True, create network with individual neurons as nodes
                If False, create network with cell types as nodes (default)
            
            Returns:
            --------
            networkx.Graph : Network graph object
            """
            print("Creating functional connectivity network...")
            
            import networkx as nx
            G = nx.DiGraph()
            
            # Use shortest time window results
            window_key = list(self.conditional_probs.keys())[0]
            probs = self.conditional_probs[window_key]
            
            if use_individual_neurons and 'probability_matrix' in probs:
                # Individual neuron network
                return self._create_individual_neuron_network(probs, threshold)
            else:
                # Cell type population network (default)
                return self._create_population_network(probs, threshold)
        
        def _create_individual_neuron_network(self, probs, threshold):
            """Create network with individual neurons as nodes"""
            import networkx as nx
            G = nx.DiGraph()
            
            prob_matrix = probs['probability_matrix']
            neuron_labels = probs['neuron_labels']
            cell_types = probs['cell_types']
            
            # Add nodes for each neuron
            color_map = {'D1': 'navy', 'D2': 'crimson', 'CHI': 'green'}
            for i, (label, cell_type) in enumerate(zip(neuron_labels, cell_types)):
                G.add_node(label, type=cell_type, color=color_map[cell_type], index=i)
            
            # Add edges based on probability matrix
            for i in range(len(neuron_labels)):
                for j in range(len(neuron_labels)):
                    if i != j and prob_matrix[i, j] > threshold:
                        source = neuron_labels[i]
                        target = neuron_labels[j]
                        G.add_edge(source, target, weight=prob_matrix[i, j], type='conditional_prob')
            
            return G
        
        def _create_population_network(self, probs, threshold):
            """Create network with cell types as nodes"""
            import networkx as nx
            G = nx.DiGraph()
            
            # Add nodes
            G.add_node('D1', type='D1', color='navy')
            G.add_node('D2', type='D2', color='crimson')
            G.add_node('CHI', type='CHI', color='green')
            
            # Add directed edges based on conditional probabilities
            for connection, prob in probs.items():
                if connection.endswith('_to_D1') or connection.endswith('_to_D2') or connection.endswith('_to_CHI'):
                    # Handle scalar probabilities (both individual aggregated and population methods)
                    if isinstance(prob, (int, float, np.integer, np.floating)) and prob > threshold:
                        source, target = connection.split('_to_')
                        G.add_edge(source, target, weight=prob, type='conditional_prob')
            
            # Add undirected edge weights based on mutual information
            for connection, mi in self.mutual_information.items():
                source, target = connection.split('_vs_')
                if G.has_edge(source, target):
                    G[source][target]['mutual_info'] = mi
                elif G.has_edge(target, source):
                    G[target][source]['mutual_info'] = mi
            
            return G
        
        def generate_summary_report(self):
            """
            Generate analysis summary report
            """
            print("\n" + "="*60)
            print("Neural Functional Connectivity Analysis Report")
            print("="*60)
            
            # Basic statistics
            print("\n1. Peak Event Statistics:")
            d1_total = sum(len(peaks) for peaks in self.tank.d1_peak_indices)
            d2_total = sum(len(peaks) for peaks in self.tank.d2_peak_indices)
            chi_total = sum(len(peaks) for peaks in self.tank.chi_peak_indices)
            
            print(f"   D1: {len(self.tank.d1_peak_indices)} neurons, total {d1_total} peaks, average {d1_total/len(self.tank.d1_peak_indices):.1f} peaks/neuron")
            print(f"   D2: {len(self.tank.d2_peak_indices)} neurons, total {d2_total} peaks, average {d2_total/len(self.tank.d2_peak_indices):.1f} peaks/neuron")
            print(f"   CHI: {len(self.tank.chi_peak_indices)} neurons, total {chi_total} peaks, average {chi_total/len(self.tank.chi_peak_indices):.1f} peaks/neuron")
            
            # Conditional probabilities
            print("\n2. Conditional Activation Probabilities (shortest time window):")
            window_key = list(self.conditional_probs.keys())[0]
            probs = self.conditional_probs[window_key]
            
            # Handle both individual neuron and population data structures
            if 'probability_matrix' in probs:
                # Individual neuron analysis - show aggregated cell type results
                print("   Cell type aggregated results:")
                for connection, prob in probs.items():
                    if connection.endswith('_to_D1') or connection.endswith('_to_D2') or connection.endswith('_to_CHI'):
                        if isinstance(prob, (int, float, np.integer, np.floating)):
                            source, target = connection.split('_to_')
                            print(f"   P({target} activated|{source} activated) = {prob:.3f}")
                
                # Show individual neuron matrix summary
                prob_matrix = probs['probability_matrix']
                print(f"   Individual neuron matrix: {prob_matrix.shape[0]}x{prob_matrix.shape[1]} neurons")
                print(f"   Mean probability: {np.mean(prob_matrix):.3f}")
                print(f"   Max probability: {np.max(prob_matrix):.3f}")
                print(f"   Connections > 0.1: {np.sum(prob_matrix > 0.1)}")
            else:
                # Population analysis - original format
                for connection, prob in probs.items():
                    if '_to_' in connection:
                        source, target = connection.split('_to_')
                        print(f"   P({target} activated|{source} activated) = {prob:.3f}")
            
            # Cross-correlation peaks
            print("\n3. Cross-correlation Peaks:")
            for pair, data in self.cross_correlations.items():
                print(f"   {pair}: peak correlation={data['peak_correlation']:.3f}, "
                      f"lag={data['peak_lag']} samples")
            
            # Co-activation patterns
            print("\n4. Major Co-activation Patterns:")
            patterns = self.coactivation_patterns
            sorted_patterns = sorted(patterns.items(), 
                                   key=lambda x: x[1]['proportion'], reverse=True)
            for pattern, stats in sorted_patterns[:5]:
                print(f"   {pattern}: {stats['proportion']:.3f} "
                      f"({stats['valid_episodes']} valid episodes)")
            
            # Peak timing relationships
            print("\n5. Peak Timing Relationships:")
            proximities = self.peak_timing_relationships['peak_proximities']
            for pair, prox_array in proximities.items():
                if len(prox_array) > 0:
                    mean_prox = np.mean(prox_array)
                    print(f"   {pair}: average proximity={mean_prox:.3f}s ({len(prox_array)} proximity events)")
            
            # Mutual information
            print("\n6. Mutual Information:")
            for pair, mi in self.mutual_information.items():
                print(f"   {pair}: {mi:.3f}")
            
            print("\n" + "="*60)
        
        
        def plot_peak_timing_relationships(self, save_path=None, title="Peak Timing Relationships Analysis", 
                                         notebook=False, overwrite=False, font_size=None):
            """
            Plot peak timing relationship graphs
            
            Parameters:
            -----------
            save_path : str, optional
                Path to save the plot as an HTML file
            title : str, optional
                Title for the plot
            notebook : bool
                Flag to indicate if the plot is for a Jupyter notebook
            overwrite : bool
                Flag to indicate whether to overwrite existing file
            font_size : str, optional
                Font size for all text elements in the plot
            
            Returns:
            --------
            bokeh.layouts.layout
                The created plot layout
            """
            from bokeh.plotting import figure
            from bokeh.layouts import column
            from bokeh.models import Range1d
            
            results = self.peak_timing_relationships
            
            # 1. Inter-peak intervals distribution with FIXED AXES
            p1 = figure(width=800, height=400, 
                       title="Inter-Peak Interval Distribution",
                       x_axis_label="Interval Time (seconds)",
                       y_axis_label="Density")
            
            # Remove grid
            p1.grid.grid_line_color = None
            
            colors = ['navy', 'crimson', 'green']
            cell_types = ['D1', 'D2', 'CHI']
            
            has_interval_data = False
            all_intervals = []
            
            for i, cell_type in enumerate(cell_types):
                intervals = results['inter_peak_intervals'][cell_type]
                if len(intervals) > 0:
                    has_interval_data = True
                    all_intervals.extend(intervals)
                    
                    # Use more bins for better visualization and density=True for proper scaling
                    hist, edges = np.histogram(intervals, bins=50, density=True)
                    centers = (edges[:-1] + edges[1:]) / 2
                    
                    p1.line(centers, hist, line_width=3, color=colors[i], 
                           legend_label=f'{cell_type} (n={len(intervals)})')
                    
                    # Add circles to make lines more visible
                    p1.scatter(centers, hist, size=4, color=colors[i], alpha=0.6)
            
            if has_interval_data and len(all_intervals) > 0:
                # Set explicit axis ranges based on data (use 95th percentile to avoid outliers)
                x_min, x_max = 0, np.percentile(all_intervals, 95)
                p1.x_range = Range1d(x_min, x_max)
            else:
                p1.text([0.5], [0.5], ["No inter-peak interval data available"], 
                       text_align="center", text_baseline="middle")
            
            p1.legend.click_policy = "hide"
            
            # 2. Peak proximity distribution with FIXED AXES
            p2 = figure(width=800, height=400,
                       title="Peak Proximity Between Different Types",
                       x_axis_label="Minimum Distance (seconds)",
                       y_axis_label="Frequency")
            
            # Remove grid
            p2.grid.grid_line_color = None
            
            proximities = results['peak_proximities']
            pair_colors = ['purple', 'orange', 'brown', 'pink', 'gray', 'cyan']
            
            has_proximity_data = False
            all_proximities = []
            max_frequency = 0
            
            for i, (pair, prox_array) in enumerate(proximities.items()):
                if len(prox_array) > 0:
                    has_proximity_data = True
                    all_proximities.extend(prox_array)
                    
                    # Use fewer bins for better visibility
                    hist, edges = np.histogram(prox_array, bins=25)
                    max_frequency = max(max_frequency, hist.max())
                    
                    p2.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:],
                           fill_color=pair_colors[i % len(pair_colors)], alpha=0.7,
                           legend_label=f'{pair} (n={len(prox_array)})')
            
            if has_proximity_data and len(all_proximities) > 0:
                # Set explicit axis ranges
                x_min, x_max = 0, np.percentile(all_proximities, 95)
                y_max = max_frequency * 1.1  # Add 10% padding
                p2.x_range = Range1d(x_min, x_max)
                p2.y_range = Range1d(0, y_max)
            else:
                p2.text([0.5], [0.5], ["No peak proximity data available"], 
                       text_align="center", text_baseline="middle")
            
            p2.legend.click_policy = "hide"
            
            # 3. Sequential activation patterns with FIXED AXES
            if 'sequential_patterns' in results:
                patterns = results['sequential_patterns']
                if patterns:
                    # Convert tuples to strings and limit display to most common patterns
                    pattern_items = list(patterns.items())
                    # Sort by frequency, take top 15 most common patterns
                    pattern_items.sort(key=lambda x: x[1], reverse=True)
                    top_patterns = pattern_items[:15]  # Show more patterns
                    
                    # Convert tuples to strings
                    pattern_names = [' → '.join(pattern) for pattern, _ in top_patterns]
                    pattern_counts = [count for _, count in top_patterns]
                    
                    p3 = figure(width=1200, height=500,  # Made wider and taller
                               title="Sequential Activation Pattern Frequency (Top 15 Most Common)",
                               x_axis_label="Activation Sequence",
                               y_axis_label="Occurrence Count")
                    
                    # Remove grid
                    p3.grid.grid_line_color = None
                    
                    # Create bar chart with explicit positioning (FIX: don't use categorical strings directly)
                    x_pos = list(range(len(pattern_names)))
                    p3.vbar(x=x_pos, top=pattern_counts, width=0.8, 
                           color='steelblue', alpha=0.7)
                    
                    # Set explicit axis ranges
                    p3.x_range = Range1d(-0.5, len(pattern_names) - 0.5)
                    p3.y_range = Range1d(0, max(pattern_counts) * 1.1)
                    
                    # Override x-axis labels (FIX: use explicit label overrides)
                    p3.xaxis.ticker = x_pos
                    p3.xaxis.major_label_overrides = {i: name for i, name in enumerate(pattern_names)}
                    p3.xaxis.major_label_orientation = 45
                else:
                    p3 = figure(width=800, height=400, title="No Sequential Activation Patterns Found")
                    p3.grid.grid_line_color = None
                    p3.text([0.5], [0.5], ["No Sequential Activation Patterns Found"], 
                           text_align="center", text_baseline="middle")
            else:
                p3 = figure(width=800, height=400, title="Sequential Activation Pattern Analysis Not Run")
                p3.grid.grid_line_color = None
                p3.text([0.5], [0.5], ["Sequential Activation Pattern Analysis Not Run"], 
                       text_align="center", text_baseline="middle")
            
            layout = column(p1, p2, p3)
            
            # Set default save path if none provided
            if save_path is None:
                save_path = os.path.join(self.tank.config["SummaryPlotsPath"], 
                                       self.tank.session_name, 
                                       'Peak_Timing_Relationships.html')
            
            # Use tank's output_bokeh_plot method
            self.tank.output_bokeh_plot(layout, save_path=save_path, title=title, 
                                      notebook=notebook, overwrite=overwrite, font_size=font_size)
            
            return layout
        
        def plot_connectivity_analysis_summary(self, save_path=None, title="Neural Connectivity Analysis Summary",
                                             notebook=False, overwrite=False, font_size=None):
            """
            Create comprehensive visualization of connectivity analysis results
            
            Parameters:
            -----------
            save_path : str, optional
                Path to save the plot as an HTML file
            title : str, optional
                Title for the plot
            notebook : bool
                Flag to indicate if the plot is for a Jupyter notebook
            overwrite : bool
                Flag to indicate whether to overwrite existing file
            font_size : str, optional
                Font size for all text elements in the plot
            
            Returns:
            --------
            bokeh.layouts.layout
                The created plot layout
            """
            def create_enhanced_connectivity_visualizations(analyzer):
                """
                Create enhanced connectivity visualizations to better display analysis results
                """
                from bokeh.plotting import figure
                from bokeh.layouts import column, row, gridplot
                from bokeh.models import ColumnDataSource, HoverTool, Legend, LegendItem, Arrow, VeeHead
                from bokeh.palettes import RdYlBu11, Spectral6
                import networkx as nx
                import pandas as pd
                import numpy as np
                
                plots = []
                
                # 1. Conditional Probability Heatmap Matrix
                def create_conditional_probability_heatmap():
                    if not hasattr(analyzer, 'conditional_probs'):
                        return None
                        
                    # Use the shortest time window data
                    window_key = list(analyzer.conditional_probs.keys())[0]
                    probs = analyzer.conditional_probs[window_key]
                    
                    # Create 3x3 matrix - corrected coordinate system
                    cell_types = ['CHI', 'D1', 'D2']
                    matrix_data = []
                    
                    for i, source in enumerate(cell_types):
                        for j, target in enumerate(cell_types):
                            if source != target:
                                key = f'{source}_to_{target}'
                                prob_value = probs.get(key, 0)
                                # Handle both numeric and non-numeric values
                                if isinstance(prob_value, (int, float, np.integer, np.floating)):
                                    prob = prob_value
                                else:
                                    prob = 0
                            else:
                                prob = 0  # Self-to-self set to 0
                            
                            # Format probability value to 3 decimal places
                            prob_formatted = f'{prob:.3f}' if prob > 0 else '0'
                            
                            matrix_data.append({
                                'source': source,
                                'target': target,
                                'probability': prob,
                                'probability_text': prob_formatted,
                                'x': j,      # Column index - Target
                                'y': 2-i,    # Row index - Source (flipped: CHI=2, D1=1, D2=0)
                                'color_value': prob
                            })
                    
                    df = pd.DataFrame(matrix_data)
                    
                    # Create HoverTool
                    hover = HoverTool(tooltips=[
                        ('From', '@source'),
                        ('To', '@target'),
                        ('Probability', '@probability{0.000}')
                    ])
                    
                    p = figure(width=450, height=450,
                              title="Conditional Activation Probability Matrix P(Target|Source)",
                              x_range=[-0.5, 2.5],           # Extended range to center squares
                              y_range=[-0.5, 2.5],           # Extended range to center squares
                              tools=[hover, 'pan', 'wheel_zoom', 'box_zoom', 'reset', 'save'])
                    
                    # Create color mapping
                    from bokeh.transform import linear_cmap
                    from bokeh.palettes import RdYlBu11
                    
                    source_data = ColumnDataSource(df)
                    
                    # Draw heatmap squares
                    p.rect(x='x', y='y', width=0.9, height=0.9, source=source_data,
                           fill_color=linear_cmap('color_value', RdYlBu11, 0, df['probability'].max()),
                           line_color='white', line_width=3)
                    
                    # Add probability value text
                    p.text(x='x', y='y', text='probability_text', source=source_data,
                           text_align='center', text_baseline='middle', text_font_size='16pt',
                           text_color='black', text_font_style='bold')
                    
                    # Set axis labels
                    p.xaxis.ticker = [0, 1, 2]
                    p.yaxis.ticker = [0, 1, 2]
                    p.xaxis.major_label_overrides = {0: "CHI", 1: "D1", 2: "D2"}
                    p.yaxis.major_label_overrides = {0: "D2", 1: "D1", 2: "CHI"}  # Flipped labels
                    
                    # Correct axis labels
                    p.xaxis.axis_label = "Target (Activated Neuron Type)"
                    p.yaxis.axis_label = "Source (Activating Neuron Type)"
                    
                    # Add axis label styling
                    p.xaxis.axis_label_text_font_size = "12pt"
                    p.yaxis.axis_label_text_font_size = "12pt"
                    p.xaxis.major_label_text_font_size = "12pt"
                    p.yaxis.major_label_text_font_size = "12pt"
                    
                    return p
                
                # 2. Network Connection Strength Diagram
                def create_network_strength_diagram():
                    if not hasattr(analyzer, 'conditional_probs'):
                        return None
                        
                    window_key = list(analyzer.conditional_probs.keys())[0]
                    probs = analyzer.conditional_probs[window_key]
                    
                    # Extract only numeric probabilities for cell type connections
                    numeric_probs = {}
                    for connection, prob in probs.items():
                        if (connection.endswith('_to_D1') or connection.endswith('_to_D2') or 
                            connection.endswith('_to_CHI')) and isinstance(prob, (int, float, np.integer, np.floating)):
                            numeric_probs[connection] = prob
                    
                    if not numeric_probs:
                        return None
                    
                    p = figure(width=600, height=600,
                              title="Neural Network Connection Strength Diagram",
                              tools=['pan', 'wheel_zoom', 'box_zoom', 'reset'])
                    
                    # Node positions - triangular layout highlighting hierarchical relationships
                    positions = {
                        'CHI': (0.5, 0.8),    # Top - regulator
                        'D1': (0.2, 0.3),     # Bottom left - relay
                        'D2': (0.8, 0.3)      # Bottom right - executor
                    }
                    
                    # Draw nodes
                    node_colors = {'CHI': 'green', 'D1': 'navy', 'D2': 'crimson'}
                    node_sizes = {'CHI': 60, 'D1': 80, 'D2': 70}  # D1 largest, emphasizing relay role
                    
                    for node, (x, y) in positions.items():
                        p.scatter([x], [y], size=node_sizes[node], color=node_colors[node], 
                                alpha=0.8, line_width=3, line_color='white')
                        p.text([x], [y], [node], text_align='center', text_baseline='middle',
                              text_font_size='14pt', text_color='white', text_font_style='bold')
                    
                    # Draw connection lines - line width represents connection strength
                    max_prob = max(numeric_probs.values()) if numeric_probs.values() else 1
                    
                    for connection, prob in numeric_probs.items():
                        source, target = connection.split('_to_')
                        if prob > 0.1:  # Only show stronger connections
                            x1, y1 = positions[source]
                            x2, y2 = positions[target]
                            
                            # Line width proportional to probability
                            line_width = max(2, prob / max_prob * 15)
                            
                            # Color depth represents strength
                            alpha = 0.3 + (prob / max_prob) * 0.7
                            
                            p.line([x1, x2], [y1, y2], line_width=line_width, 
                                  color='gray', alpha=alpha)
                            
                            # Add probability labels
                            mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
                            p.text([mid_x], [mid_y], [f'{prob:.2f}'], 
                                  text_align='center', text_baseline='middle',
                                  text_font_size='10pt', text_color='black',
                                  background_fill_color='white', background_fill_alpha=0.8)
                    
                    p.axis.visible = False
                    p.grid.visible = False
                    p.x_range.start, p.x_range.end = -0.1, 1.1
                    p.y_range.start, p.y_range.end = 0, 1
                    
                    return p
                
                # 3. Activation Cascade Timeline Plot
                def create_activation_cascade_plot():
                    if not hasattr(analyzer, 'peak_timing_relationships'):
                        return None
                        
                    proximities = analyzer.peak_timing_relationships['peak_proximities']
                    
                    p = figure(width=800, height=400,
                              title="Neural Activation Temporal Relationships (Average Proximity)",
                              x_axis_label="Average Time Delay (seconds)",
                              y_axis_label="Connection Type",
                              tools=['pan', 'wheel_zoom', 'box_zoom', 'reset'])
                    
                    # Prepare data
                    connections = []
                    delays = []
                    colors = []
                    
                    color_map = {
                        'CHI_to_D1': 'green',
                        'CHI_to_D2': 'lightgreen', 
                        'D1_to_D2': 'blue',
                        'D2_to_D1': 'red',
                        'D1_to_CHI': 'orange',
                        'D2_to_CHI': 'pink'
                    }
                    
                    for connection, prox_array in proximities.items():
                        if len(prox_array) > 0:
                            connections.append(connection)
                            delays.append(np.mean(prox_array))
                            colors.append(color_map.get(connection, 'gray'))
                    
                    if not connections:
                        return None
                        
                    # Sort by delay
                    sorted_data = sorted(zip(connections, delays, colors), key=lambda x: x[1])
                    connections, delays, colors = zip(*sorted_data)
                    
                    y_positions = list(range(len(connections)))
                    
                    p.scatter(delays, y_positions, size=15, color=colors, alpha=0.8)
                    
                    # Add connecting lines showing activation sequence
                    for i in range(len(delays)-1):
                        p.line([delays[i], delays[i+1]], [y_positions[i], y_positions[i+1]],
                              line_dash='dashed', color='gray', alpha=0.5)
                    
                    p.yaxis.ticker = y_positions
                    p.yaxis.major_label_overrides = {i: conn for i, conn in enumerate(connections)}
                    
                    return p
                
                # 4. Co-activation Pattern Pie Chart
                def create_coactivation_pie_chart():
                    if not hasattr(analyzer, 'coactivation_patterns'):
                        return None
                        
                    patterns = analyzer.coactivation_patterns
                    
                    # Prepare data
                    pattern_names = []
                    proportions = []
                    colors = []
                    
                    color_scheme = {
                        'None': 'lightgray',
                        'D1_only': 'navy',
                        'D2_only': 'crimson', 
                        'CHI_only': 'green',
                        'D1_D2': 'purple',
                        'D1_CHI': 'teal',
                        'D2_CHI': 'orange',
                        'D1_D2_CHI': 'gold'
                    }
                    
                    for pattern, stats in patterns.items():
                        if stats['proportion'] > 0.01:  # Only show patterns >1%
                            pattern_names.append(pattern)
                            proportions.append(stats['proportion'])
                            colors.append(color_scheme.get(pattern, 'gray'))
                    
                    if not pattern_names:
                        return None
                        
                    # Calculate angles
                    angles = [p * 2 * np.pi for p in proportions]
                    
                    p = figure(width=500, height=500,
                              title="Co-activation Pattern Distribution",
                              tools=['pan', 'wheel_zoom', 'box_zoom', 'reset'])
                    
                    # Draw pie chart
                    start_angle = 0
                    for i, (angle, color, name, prop) in enumerate(zip(angles, colors, pattern_names, proportions)):
                        end_angle = start_angle + angle
                        
                        p.wedge(x=0, y=0, radius=0.8, start_angle=start_angle, end_angle=end_angle,
                               color=color, alpha=0.8, legend_label=f'{name}: {prop:.1%}')
                        
                        # Add labels
                        mid_angle = (start_angle + end_angle) / 2
                        label_x = 0.6 * np.cos(mid_angle)
                        label_y = 0.6 * np.sin(mid_angle)
                        
                        if prop > 0.05:  # Only add labels for large segments
                            p.text([label_x], [label_y], [f'{prop:.1%}'], 
                                  text_align='center', text_baseline='middle',
                                  text_font_size='10pt', text_color='white', text_font_style='bold')
                        
                        start_angle = end_angle
                    
                    p.axis.visible = False
                    p.grid.visible = False
                    p.legend.location = "center_right"
                    p.legend.click_policy = "hide"
                    
                    return p
                
                # 5. Information Flow Diagram
                def create_information_flow_diagram():
                    if not hasattr(analyzer, 'mutual_information') or not hasattr(analyzer, 'conditional_probs'):
                        return None
                        
                    # Create HoverTool
                    hover = HoverTool(tooltips=[
                        ('Connection', '@connection'),
                        ('Conditional Prob', '@cond_prob{0.000}'),
                        ('Mutual Info', '@mutual_info{0.000}')
                    ])
                    
                    p = figure(width=600, height=400,
                              title="Information Flow Strength Analysis",
                              x_axis_label="Conditional Probability P(Target|Source)",
                              y_axis_label="Mutual Information",
                              tools=[hover, 'pan', 'wheel_zoom', 'box_zoom', 'reset'])
                    
                    # Prepare data
                    window_key = list(analyzer.conditional_probs.keys())[0]
                    cond_probs = analyzer.conditional_probs[window_key]
                    mutual_info = analyzer.mutual_information
                    
                    # Map conditional probabilities to mutual information
                    flow_data = []
                    
                    for connection, cond_prob in cond_probs.items():
                        # Only process cell type connections with numeric probabilities
                        if (connection.endswith('_to_D1') or connection.endswith('_to_D2') or 
                            connection.endswith('_to_CHI')) and isinstance(cond_prob, (int, float, np.integer, np.floating)):
                            
                            source, target = connection.split('_to_')
                            
                            # Find corresponding mutual information
                            mi_key1 = f'{source}_vs_{target}'
                            mi_key2 = f'{target}_vs_{source}'
                            
                            mi_value = mutual_info.get(mi_key1, mutual_info.get(mi_key2, 0))
                            
                            flow_data.append({
                                'connection': connection,
                                'cond_prob': cond_prob,
                                'mutual_info': mi_value,
                                'source': source,
                                'target': target
                            })
                    
                    if not flow_data:
                        return None
                        
                    df = pd.DataFrame(flow_data)
                    
                    # Different connection types use different colors
                    colors = []
                    for _, row in df.iterrows():
                        if 'CHI' in row['source']:
                            colors.append('green')
                        elif 'D1' in row['source']:
                            colors.append('navy')
                        else:
                            colors.append('crimson')
                    
                    df['colors'] = colors
                    
                    source_data = ColumnDataSource(df)
                    
                    p.scatter('cond_prob', 'mutual_info', size=12, color='colors', 
                            alpha=0.7, source=source_data)
                    
                    # Add connection labels
                    p.text('cond_prob', 'mutual_info', 'connection', source=source_data,
                          text_font_size='8pt', x_offset=5, y_offset=5)
                    
                    return p
                
                # Create all plots
                p1 = create_conditional_probability_heatmap()
                p2 = create_network_strength_diagram()
                p3 = create_activation_cascade_plot()
                p4 = create_coactivation_pie_chart()
                p5 = create_information_flow_diagram()
                
                # Filter out None plots
                valid_plots = [p for p in [p1, p2, p3, p4, p5] if p is not None]
                
                if not valid_plots:
                    print("No available data to create visualization plots")
                    return None
                
                # Combine layout
                if len(valid_plots) >= 5:
                    layout = column(
                        row(valid_plots[0], valid_plots[1]),
                        valid_plots[2],
                        row(valid_plots[3], valid_plots[4])
                    )
                elif len(valid_plots) >= 3:
                    layout = column(
                        row(valid_plots[0], valid_plots[1]),
                        *valid_plots[2:]
                    )
                else:
                    layout = column(*valid_plots)
                
                return layout
            
            # Create enhanced plots using the analyzer instance (self)
            layout = create_enhanced_connectivity_visualizations(self)
            
            if layout is None:
                # Fallback to simple message if no data available
                from bokeh.plotting import figure
                from bokeh.layouts import column
                
                p = figure(width=800, height=400, title="No Connectivity Data Available")
                p.grid.grid_line_color = None
                p.text([0.5], [0.5], ["No connectivity analysis data available"], 
                       text_align="center", text_baseline="middle")
                p.axis.visible = False
                layout = column(p)
            
            # Set default save path if none provided
            if save_path is None:
                save_path = os.path.join(self.tank.config["SummaryPlotsPath"], 
                                       self.tank.session_name, 
                                       'Connectivity_Analysis_Summary.html')
            
            # Use tank's output_bokeh_plot method
            self.tank.output_bokeh_plot(layout, save_path=save_path, title=title, 
                                      notebook=notebook, overwrite=overwrite, font_size=font_size)
            
            return layout
        
        def analyze_and_plot_connectivity(self, save_dir=None, timing_filename=None, summary_filename=None,
                                        timing_save_path=None, summary_save_path=None,
                                        use_rising_edges=True, create_plots=True, 
                                        notebook=False, overwrite=False, font_size=None):
            """
            Complete connectivity analysis workflow with optional plotting
            (Equivalent to the standalone analyze_neural_connectivity_optimized function)
            
            Parameters:
            -----------
            save_dir : str, optional
                Directory to save results and plots (used if individual save_path parameters are not provided)
            timing_filename : str, optional
                Custom filename for timing relationships plot (e.g., 'my_timing_analysis.html')
                If None, defaults to 'peak_timing_relationships.html'
            summary_filename : str, optional
                Custom filename for connectivity summary plot (e.g., 'my_connectivity_summary.html')
                If None, defaults to 'connectivity_summary.html'
            timing_save_path : str, optional
                Full path for timing relationships plot (overrides save_dir and timing_filename)
            summary_save_path : str, optional
                Full path for connectivity summary plot (overrides save_dir and summary_filename)
            use_rising_edges : bool
                Whether to use rising edges instead of peak indices
            create_plots : bool
                Whether to create and save plots
            notebook : bool
                Flag to indicate if plots are for Jupyter notebook
            overwrite : bool
                Flag to indicate whether to overwrite existing files
            font_size : str, optional
                Font size for plot text elements
            
            Returns:
            --------
            dict : Complete analysis results including plots if created
            """
            print("Running complete neural connectivity analysis workflow...")
            
            # Update use_rising_edges parameter if provided
            if use_rising_edges != self.use_rising_edges:
                self.use_rising_edges = use_rising_edges
                print(f"Updated use_rising_edges parameter to: {use_rising_edges}")
            
            # Update save_dir if provided
            if save_dir is not None:
                self.save_dir = save_dir
            
            # Run all analyses if not already completed
            if not hasattr(self, 'analysis_results') or self.analysis_results is None:
                results = self.run_all_analyses()
            else:
                results = self.analysis_results
            
            # Create plots if requested
            if create_plots:
                print("Creating connectivity analysis plots...")
                
                # Set save paths with flexible naming options
                if timing_save_path:
                    # Use provided full path
                    timing_path = timing_save_path
                elif save_dir:
                    # Use directory with custom or default filename
                    filename = timing_filename if timing_filename else 'peak_timing_relationships.html'
                    timing_path = os.path.join(save_dir, filename)
                else:
                    timing_path = None
                
                if summary_save_path:
                    # Use provided full path
                    summary_path = summary_save_path
                elif save_dir:
                    # Use directory with custom or default filename
                    filename = summary_filename if summary_filename else 'connectivity_summary.html'
                    summary_path = os.path.join(save_dir, filename)
                else:
                    summary_path = None
                
                # Create timing relationships plot
                timing_plot = self.create_timing_plot(
                    save_path=timing_path, 
                    notebook=notebook, 
                    overwrite=overwrite, 
                    font_size=font_size
                )
                
                # Create summary plot
                summary_plot = self.create_summary_plot(
                    save_path=summary_path, 
                    notebook=notebook, 
                    overwrite=overwrite, 
                    font_size=font_size
                )
                
                results['timing_plot'] = timing_plot
                results['summary_plot'] = summary_plot
                
                print("Connectivity analysis and plotting complete!")
            
            return results

    def create_connectivity_analyzer(self):
        """
        Create and return a NeuralConnectivityAnalyzer instance for this tank
        
        Returns:
        --------
        NeuralConnectivityAnalyzer : Analyzer instance
        """
        return self.NeuralConnectivityAnalyzer(self)
    
    def plot_connectivity_timing_analysis(self, save_path=None, notebook=False, overwrite=False, font_size=None):
        """
        Generate timing relationships analysis plot
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the HTML file
        notebook : bool
            Whether to display in notebook
        overwrite : bool
            Whether to overwrite existing file
        font_size : str, optional
            Font size for plot text elements
        
        Returns:
        --------
        bokeh.layouts.layout : The created plot layout
        """
        connectivity = self.create_connectivity_analyzer()
        
        # Set default save path if none provided
        if save_path is None:
            save_path = os.path.join(self.config["SummaryPlotsPath"], 
                                   self.session_name, 
                                   'connectivity_timing_analysis.html')
        
        plot = connectivity.create_timing_plot(
            save_path=save_path,
            notebook=notebook,
            overwrite=overwrite,
            font_size=font_size
        )
        
        return plot
    
    def plot_connectivity_summary_analysis(self, save_path=None, notebook=False, overwrite=False, font_size=None):
        """
        Generate connectivity analysis summary plot
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the HTML file
        notebook : bool
            Whether to display in notebook
        overwrite : bool
            Whether to overwrite existing file
        font_size : str, optional
            Font size for plot text elements
        
        Returns:
        --------
        bokeh.layouts.layout : The created plot layout
        """
        connectivity = self.create_connectivity_analyzer()
        
        # Set default save path if none provided
        if save_path is None:
            save_path = os.path.join(self.config["SummaryPlotsPath"], 
                                   self.session_name, 
                                   'connectivity_summary_analysis.html')
        
        plot = connectivity.create_summary_plot(
            save_path=save_path,
            notebook=notebook,
            overwrite=overwrite,
            font_size=font_size
        )
        
        return plot
    
    def plot_connectivity_conditional_probabilities(self, time_windows=[5, 10, 20, 50], method='individual',
                                                 save_path=None, notebook=False, overwrite=False, font_size=None):
            """
            Generate conditional probabilities analysis plot
            
            Parameters:
            -----------
            time_windows : list
                Different time window sizes (in samples)
            method : str
                'individual' - calculate probabilities between individual neurons
                'population' - calculate probabilities between cell type populations
                'cell_type_proportion' - calculate using average proportion of target neurons activated per source activation
            save_path : str, optional
                Path to save the HTML file
            notebook : bool
                Whether to display in notebook
            overwrite : bool
                Whether to overwrite existing file
            font_size : str, optional
                Font size for plot text elements
            
            Returns:
            --------
            bokeh.layouts.layout : The created plot layout
            """
            connectivity = self.create_connectivity_analyzer()
            
            # Run conditional probabilities analysis
            cond_prob_results = connectivity.run_conditional_probabilities_analysis(
                time_windows=time_windows, 
                method=method
            )
            
            # Set default save path if none provided
            if save_path is None:
                save_path = os.path.join(self.config["SummaryPlotsPath"], 
                                       self.session_name, 
                                       'connectivity_conditional_probabilities.html')
            
            # Create a comprehensive visualization showing all time windows
            from bokeh.plotting import figure
            from bokeh.layouts import column, row, gridplot
            import numpy as np
            
            # Create subplots for each time window
            plots = []
            
            for window_key in cond_prob_results.keys():
                probs = cond_prob_results[window_key]
                
                # Extract cell type probabilities
                cell_type_probs = {}
                for connection, prob in probs.items():
                    if (connection.endswith('_to_D1') or connection.endswith('_to_D2') or 
                        connection.endswith('_to_CHI')) and isinstance(prob, (int, float, np.integer, np.floating)):
                        cell_type_probs[connection] = prob
                
                if cell_type_probs:
                    connections = list(cell_type_probs.keys())
                    probabilities = list(cell_type_probs.values())
                    
                    # Convert window size to time (assuming 20Hz sampling rate)
                    time_seconds = int(window_key.split('_')[1]) / 20.0
                    
                    p = figure(width=400, height=300,
                              title=f"Window: {window_key} ({time_seconds:.1f}s)",
                              x_axis_label="Connection Type",
                              y_axis_label="Probability",
                              tools="pan,box_zoom,wheel_zoom,reset,save")
                    
                    # Color code by source cell type
                    colors = []
                    for conn in connections:
                        if 'CHI_to_' in conn:
                            colors.append('green')
                        elif 'D1_to_' in conn:
                            colors.append('navy')
                        else:
                            colors.append('crimson')
                    
                    p.vbar(x=connections, top=probabilities, width=0.5, color=colors, alpha=0.7)
                    p.xaxis.major_label_orientation = 45
                    
                    plots.append(p)
                else:
                    # Create empty plot if no data
                    p = figure(width=400, height=300, title=f"Window: {window_key} (No Data)")
                    p.grid.grid_line_color = None
                    p.text([0.5], [0.5], ["No data available"], 
                           text_align="center", text_baseline="middle")
                    p.axis.visible = False
                    plots.append(p)
            
            # Arrange plots in a grid
            if len(plots) == 1:
                layout = column(plots[0])
            elif len(plots) == 2:
                layout = row(plots[0], plots[1])
            elif len(plots) == 3:
                layout = row(plots[0], plots[1], plots[2])
            elif len(plots) == 4:
                layout = gridplot([[plots[0], plots[1]], [plots[2], plots[3]]])
            else:
                layout = column(*plots)
            
            # Save the plot
            self.output_bokeh_plot(layout, save_path=save_path, 
                                  title="Conditional Probabilities Analysis - All Time Windows",
                                  notebook=notebook, overwrite=overwrite, font_size=font_size)
            
            return layout
    
    def plot_connectivity_cross_correlations(self, max_lag=100, save_path=None, notebook=False, 
                                           overwrite=False, font_size=None):
        """
        Generate cross-correlations analysis plot
        
        Parameters:
        -----------
        max_lag : int
            Maximum lag for cross-correlation analysis
        save_path : str, optional
            Path to save the HTML file
        notebook : bool
            Whether to display in notebook
        overwrite : bool
            Whether to overwrite existing file
        font_size : str, optional
            Font size for plot text elements
        
        Returns:
        --------
        bokeh.layouts.layout : The created plot layout
        """
        connectivity = self.create_connectivity_analyzer()
        
        # Run cross-correlations analysis
        cross_corr_results = connectivity.run_cross_correlations_analysis(max_lag=max_lag)
        
        # Set default save path if none provided
        if save_path is None:
            save_path = os.path.join(self.config["SummaryPlotsPath"], 
                                   self.session_name, 
                                   'connectivity_cross_correlations.html')
        
        # Create visualization of cross-correlations
        from bokeh.plotting import figure
        from bokeh.layouts import column
        import numpy as np
        
        if not cross_corr_results:
            # Create empty plot if no data
            p = figure(width=600, height=400, title="No Cross-Correlation Data Available")
            p.grid.grid_line_color = None
            p.text([0.5], [0.5], ["No cross-correlation data available"], 
                   text_align="center", text_baseline="middle")
            p.axis.visible = False
            layout = column(p)
        else:
            # Create bar plot of peak correlations
            pairs = list(cross_corr_results.keys())
            peak_correlations = [cross_corr_results[pair]['peak_correlation'] for pair in pairs]
            peak_lags = [cross_corr_results[pair]['peak_lag'] for pair in pairs]
            
            p = figure(width=800, height=400,
                      title=f"Cross-Correlation Peak Analysis (Max Lag: {max_lag})",
                      x_axis_label="Neuron Pair",
                      y_axis_label="Peak Correlation",
                      tools="pan,box_zoom,wheel_zoom,reset,save")
            
            # Color code by correlation strength
            colors = ['red' if corr < 0 else 'blue' for corr in peak_correlations]
            
            p.vbar(x=pairs, top=peak_correlations, width=0.5, color=colors, alpha=0.7)
            p.xaxis.major_label_orientation = 45
            
            # Add lag information as text
            p.text(x=pairs, y=peak_correlations, text=[f"lag={lag}" for lag in peak_lags],
                  text_align="center", text_baseline="bottom", text_font_size="8pt")
            
            layout = column(p)
        
        # Save the plot
        self.output_bokeh_plot(layout, save_path=save_path, 
                              title="Cross-Correlations Analysis",
                              notebook=notebook, overwrite=overwrite, font_size=font_size)
        
        return layout
    
    def plot_connectivity_coactivation_patterns(self, min_duration=3, save_path=None, notebook=False,
                                              overwrite=False, font_size=None):
        """
        Generate coactivation patterns analysis plot
        
        Parameters:
        -----------
        min_duration : int
            Minimum duration for coactivation patterns
        save_path : str, optional
            Path to save the HTML file
        notebook : bool
            Whether to display in notebook
        overwrite : bool
            Whether to overwrite existing file
        font_size : str, optional
            Font size for plot text elements
        
        Returns:
        --------
        bokeh.layouts.layout : The created plot layout
        """
        connectivity = self.create_connectivity_analyzer()
        
        # Run coactivation patterns analysis
        coactivation_results = connectivity.run_coactivation_patterns_analysis(min_duration=min_duration)
        
        # Set default save path if none provided
        if save_path is None:
            save_path = os.path.join(self.config["SummaryPlotsPath"], 
                                   self.session_name, 
                                   'connectivity_coactivation_patterns.html')
        
        # Create visualization of coactivation patterns
        from bokeh.plotting import figure
        from bokeh.layouts import column
        import numpy as np
        
        if not coactivation_results:
            # Create empty plot if no data
            p = figure(width=600, height=400, title="No Coactivation Pattern Data Available")
            p.grid.grid_line_color = None
            p.text([0.5], [0.5], ["No coactivation pattern data available"], 
                   text_align="center", text_baseline="middle")
            p.axis.visible = False
            layout = column(p)
        else:
            # Create pie chart of coactivation patterns
            patterns = list(coactivation_results.keys())
            proportions = [coactivation_results[pattern]['proportion'] for pattern in patterns]
            
            # Filter patterns with proportion > 1%
            significant_patterns = [(p, prop) for p, prop in zip(patterns, proportions) if prop > 0.01]
            
            if not significant_patterns:
                p = figure(width=600, height=400, title="No Significant Coactivation Patterns")
                p.grid.grid_line_color = None
                p.text([0.5], [0.5], ["No significant coactivation patterns found"], 
                       text_align="center", text_baseline="middle")
                p.axis.visible = False
                layout = column(p)
            else:
                pattern_names, pattern_props = zip(*significant_patterns)
                
                p = figure(width=800, height=400,
                          title=f"Coactivation Patterns (Min Duration: {min_duration})",
                          x_axis_label="Pattern Type",
                          y_axis_label="Proportion",
                          tools="pan,box_zoom,wheel_zoom,reset,save")
                
                # Color code patterns
                colors = ['green', 'blue', 'red', 'orange', 'purple', 'brown', 'pink', 'gray']
                bar_colors = [colors[i % len(colors)] for i in range(len(pattern_names))]
                
                p.vbar(x=pattern_names, top=pattern_props, width=0.5, color=bar_colors, alpha=0.7)
                p.xaxis.major_label_orientation = 45
                
                layout = column(p)
        
        # Save the plot
        self.output_bokeh_plot(layout, save_path=save_path, 
                              title="Coactivation Patterns Analysis",
                              notebook=notebook, overwrite=overwrite, font_size=font_size)
        
        return layout
    
    def plot_connectivity_mutual_information(self, bins=10, save_path=None, notebook=False,
                                           overwrite=False, font_size=None):
        """
        Generate mutual information analysis plot
        
        Parameters:
        -----------
        bins : int
            Number of bins for mutual information calculation
        save_path : str, optional
            Path to save the HTML file
        notebook : bool
            Whether to display in notebook
        overwrite : bool
            Whether to overwrite existing file
        font_size : str, optional
            Font size for plot text elements
        
        Returns:
        --------
        bokeh.layouts.layout : The created plot layout
        """
        connectivity = self.create_connectivity_analyzer()
        
        # Run mutual information analysis
        mi_results = connectivity.run_mutual_information_analysis(bins=bins)
        
        # Set default save path if none provided
        if save_path is None:
            save_path = os.path.join(self.config["SummaryPlotsPath"], 
                                   self.session_name, 
                                   'connectivity_mutual_information.html')
        
        # Create visualization of mutual information
        from bokeh.plotting import figure
        from bokeh.layouts import column
        import numpy as np
        
        if not mi_results:
            # Create empty plot if no data
            p = figure(width=600, height=400, title="No Mutual Information Data Available")
            p.grid.grid_line_color = None
            p.text([0.5], [0.5], ["No mutual information data available"], 
                   text_align="center", text_baseline="middle")
            p.axis.visible = False
            layout = column(p)
        else:
            # Create bar plot of mutual information
            pairs = list(mi_results.keys())
            mi_values = list(mi_results.values())
            
            p = figure(width=800, height=400,
                      title=f"Mutual Information Analysis (Bins: {bins})",
                      x_axis_label="Neuron Pair",
                      y_axis_label="Mutual Information",
                      tools="pan,box_zoom,wheel_zoom,reset,save")
            
            # Color code by MI value
            colors = ['red' if mi < 0 else 'blue' for mi in mi_values]
            
            p.vbar(x=pairs, top=mi_values, width=0.5, color=colors, alpha=0.7)
            p.xaxis.major_label_orientation = 45
            
            layout = column(p)
        
        # Save the plot
        self.output_bokeh_plot(layout, save_path=save_path, 
                              title="Mutual Information Analysis",
                              notebook=notebook, overwrite=overwrite, font_size=font_size)
        
        return layout
    
    def plot_connectivity_time_window_comparison(self, time_windows=[5, 10, 20, 50], method='individual',
                                               save_path=None, notebook=False, overwrite=False, font_size=None):
        """
        Generate a comparison plot showing conditional probabilities across different time windows
        
        Parameters:
        -----------
        time_windows : list
            Different time window sizes (in samples)
        method : str
            'individual' - calculate probabilities between individual neurons
            'population' - calculate probabilities between cell type populations
            'cell_type_proportion' - calculate using average proportion of target neurons activated per source activation
        save_path : str, optional
            Path to save the HTML file
        notebook : bool
            Whether to display in notebook
        overwrite : bool
            Whether to overwrite existing file
        font_size : str, optional
            Font size for plot text elements
        
        Returns:
        --------
        bokeh.layouts.layout : The created plot layout
        """
        connectivity = self.create_connectivity_analyzer()
        
        # Run conditional probabilities analysis
        cond_prob_results = connectivity.run_conditional_probabilities_analysis(
            time_windows=time_windows, 
            method=method
        )
        
        # Set default save path if none provided
        if save_path is None:
            save_path = os.path.join(self.config["SummaryPlotsPath"], 
                                   self.session_name, 
                                   'connectivity_time_window_comparison.html')
        
        # Create comparison visualization
        from bokeh.plotting import figure
        from bokeh.layouts import column
        import numpy as np
        
        # Extract data for comparison
        window_sizes = []
        connection_types = set()
        all_probabilities = {}
        
        for window_key in cond_prob_results.keys():
            window_size = int(window_key.split('_')[1])
            window_sizes.append(window_size)
            probs = cond_prob_results[window_key]
            
            for connection, prob in probs.items():
                if (connection.endswith('_to_D1') or connection.endswith('_to_D2') or 
                    connection.endswith('_to_CHI')) and isinstance(prob, (int, float, np.integer, np.floating)):
                    connection_types.add(connection)
                    if connection not in all_probabilities:
                        all_probabilities[connection] = []
                    all_probabilities[connection].append(prob)
        
        if not all_probabilities:
            # Create empty plot if no data
            p = figure(width=600, height=400, title="No Conditional Probability Data Available")
            p.grid.grid_line_color = None
            p.text([0.5], [0.5], ["No conditional probability data available"], 
                   text_align="center", text_baseline="middle")
            p.axis.visible = False
            layout = column(p)
        else:
            # Create line plot showing how probabilities change with window size
            p = figure(width=1000, height=600,
                      title="Conditional Probabilities vs Time Window Size",
                      x_axis_label="Time Window Size (samples)",
                      y_axis_label="Conditional Probability",
                      tools="pan,box_zoom,wheel_zoom,reset,save")
            
            # Color scheme for different connection types
            colors = ['green', 'navy', 'crimson', 'orange', 'purple', 'brown']
            color_map = {}
            
            for i, conn_type in enumerate(sorted(connection_types)):
                color_map[conn_type] = colors[i % len(colors)]
                
                if conn_type in all_probabilities and len(all_probabilities[conn_type]) == len(window_sizes):
                    # Convert window sizes to time (assuming 20Hz sampling rate)
                    time_seconds = [ws / 20.0 for ws in window_sizes]
                    
                    p.line(time_seconds, all_probabilities[conn_type], 
                          line_width=3, color=color_map[conn_type], 
                          legend_label=conn_type, alpha=0.8)
                    p.circle(time_seconds, all_probabilities[conn_type], 
                           size=8, color=color_map[conn_type], alpha=0.8)
            
            p.legend.location = "top_left"
            p.legend.click_policy = "hide"
            
            layout = column(p)
        
        # Save the plot
        self.output_bokeh_plot(layout, save_path=save_path, 
                              title="Time Window Comparison Analysis",
                              notebook=notebook, overwrite=overwrite, font_size=font_size)
        
        return layout
    
    def _plot_population_centered_activity_core(self, center_cell_type, center_signals, center_peak_indices,
                                               all_signals_dict, ci_rate=None, time_window=3.0,
                                               save_path=None, title=None, notebook=False, overwrite=False, font_size=None,
                                               baseline_correct=False, correction_factors=None, exclude_indices=None):
        """
        Core function for plotting population-centered activity analysis where ALL neurons from each cell type
        are averaged at the center cell type peak times (no activity filtering).
        
        This is different from the original approach which only averaged "active" neurons.
        Here we average ALL neurons of each type at the same time points.
        
        Parameters:
        -----------
        center_cell_type : str
            The cell type being centered ('D1', 'D2', or 'CHI')
        center_signals : numpy.ndarray
            Signals for the center cell type (shape: n_neurons x n_timepoints)
        center_peak_indices : list
            List of arrays containing peak indices for each center neuron
        all_signals_dict : dict
            Dictionary with all cell type signals {'D1': signals, 'D2': signals, 'CHI': signals}
        ci_rate : float, optional
            Sampling rate of calcium imaging data (Hz). If None, uses self.ci_rate
        time_window : float
            Time window in seconds before and after each peak for plotting
        save_path : str, optional
            Path to save the HTML file
        title : str, optional
            Title for the plot
        notebook : bool
            Whether to display in notebook
        overwrite : bool
            Whether to overwrite existing file
        font_size : str, optional
            Font size for plot text elements
        baseline_correct : bool
            Whether to apply baseline correction
        correction_factors : dict, optional
            Dictionary of correction factors for each cell type {'D1': factor, 'D2': factor, 'CHI': factor}
        exclude_indices : array-like, optional
            Binary array or list of indices to exclude from analysis. If binary array, indices where value is 1 (or True) will be excluded.
            If list of indices, those specific indices will be excluded. Can be used to exclude timepoints during licking, movement, etc.
        
        Returns:
        --------
        tuple
            (bokeh.plotting.figure, dict) - The created figure and summary statistics
        """
        import numpy as np
        from bokeh.plotting import figure
        from bokeh.models import LinearAxis, Range1d
        
        if ci_rate is None:
            ci_rate = self.ci_rate
        
        if correction_factors is None:
            correction_factors = {'D1': 0, 'D2': 0, 'CHI': 0}
        
        # Convert time window to samples
        plot_window_samples = int(time_window * ci_rate)
        
        # Process exclude_indices to create exclusion set
        exclusion_set = set()
        if exclude_indices is not None:
            exclude_indices = np.array(exclude_indices)
            if exclude_indices.dtype == bool or all(val in [0, 1, True, False] for val in exclude_indices):
                # Binary array - get indices where value is True/1
                exclusion_set = set(np.where(exclude_indices)[0])
            else:
                # List of indices
                exclusion_set = set(exclude_indices)
            print(f"Excluding {len(exclusion_set)} time indices from {center_cell_type} analysis")
        
        # Collect all center peak time indices, excluding those in exclusion_set
        all_center_peak_times = []
        for neuron_idx, peaks in enumerate(center_peak_indices):
            for peak_idx in peaks:
                if peak_idx not in exclusion_set:
                    all_center_peak_times.append(peak_idx)
        
        if not all_center_peak_times:
            print(f"No {center_cell_type} peaks found for analysis.")
            return None, None
        
        print(f"Found {len(all_center_peak_times)} total {center_cell_type} peaks for population analysis")
        
        # Collect traces for each cell type at all center peak times
        all_cell_type_traces = {cell_type: [] for cell_type in all_signals_dict.keys()}
        all_velocity_traces = []
        valid_peak_count = 0
        
        # Check if velocity data is available
        has_velocity = hasattr(self, 'smoothed_velocity') and self.smoothed_velocity is not None
        if has_velocity:
            velocity_data = self.smoothed_velocity
            # Convert velocity sampling rate if needed (assuming velocity is at vm_rate)
            if hasattr(self, 'vm_rate') and self.vm_rate != ci_rate:
                # Resample velocity to match ci_rate if needed
                # For now, assume they're aligned or close enough
                pass
        
        for peak_idx in all_center_peak_times:
            start_idx = peak_idx - plot_window_samples
            end_idx = peak_idx + plot_window_samples + 1
            
            # Check if the window is within bounds for all signal types
            valid_window = True
            for cell_type, signals in all_signals_dict.items():
                if len(signals) > 0:
                    signal_length = len(signals[0])  # Assuming all neurons have same length
                    if start_idx < 0 or end_idx >= signal_length:
                        valid_window = False
                        break
            
            # Also check velocity bounds if available
            if has_velocity and valid_window:
                if start_idx < 0 or end_idx >= len(velocity_data):
                    valid_window = False
            
            if valid_window:
                valid_peak_count += 1
                
                # For each cell type, average ALL neurons at this time point
                for cell_type, signals in all_signals_dict.items():
                    if len(signals) > 0:
                        # Extract traces from ALL neurons of this cell type
                        traces_at_peak = []
                        for neuron_signal in signals:
                            trace = neuron_signal[start_idx:end_idx]
                            traces_at_peak.append(trace)
                        
                        # Average across all neurons of this cell type at this peak time
                        avg_trace_at_peak = np.mean(traces_at_peak, axis=0)
                        all_cell_type_traces[cell_type].append(avg_trace_at_peak)
                    else:
                        # If no neurons of this type, add zeros
                        trace_length = 2 * plot_window_samples + 1
                        all_cell_type_traces[cell_type].append(np.zeros(trace_length))
                
                # Extract velocity trace at this peak time
                if has_velocity:
                    velocity_trace = velocity_data[start_idx:end_idx]
                    all_velocity_traces.append(velocity_trace)
        
        if valid_peak_count == 0:
            print(f"No valid {center_cell_type} peaks found within signal bounds.")
            return None, None
        
        # Calculate final average traces across all peak events
        final_avg_traces = {}
        final_sem_traces = {}
        
        for cell_type in all_signals_dict.keys():
            if all_cell_type_traces[cell_type]:
                final_avg_traces[cell_type] = np.mean(all_cell_type_traces[cell_type], axis=0)
                final_sem_traces[cell_type] = np.std(all_cell_type_traces[cell_type], axis=0) / np.sqrt(len(all_cell_type_traces[cell_type]))
                
                # Apply baseline correction if requested
                if baseline_correct and cell_type in correction_factors:
                    correction_factor = correction_factors[cell_type]
                    final_avg_traces[cell_type] += correction_factor
                    # Note: SEM doesn't need correction as it's a relative measure
                    print(f"Applied baseline correction to {cell_type}: {correction_factor:.4f}")
            else:
                trace_length = 2 * plot_window_samples + 1
                final_avg_traces[cell_type] = np.zeros(trace_length)
                final_sem_traces[cell_type] = np.zeros(trace_length)
        
        # Calculate average velocity trace
        avg_velocity_trace = None
        sem_velocity_trace = None
        if has_velocity and all_velocity_traces:
            avg_velocity_trace = np.mean(all_velocity_traces, axis=0)
            sem_velocity_trace = np.std(all_velocity_traces, axis=0) / np.sqrt(len(all_velocity_traces))
        
        # Create time axis
        time_axis = np.linspace(-time_window, time_window, 2 * plot_window_samples + 1)
        
        # Create summary statistics
        stats = {
            f'total_{center_cell_type.lower()}_peaks': len(all_center_peak_times),
            'valid_peaks_analyzed': valid_peak_count,
            'n_traces_averaged': valid_peak_count,
            'has_velocity': has_velocity,
            'baseline_corrected': baseline_correct,
            'excluded_indices_count': len(exclusion_set) if exclude_indices is not None else 0
        }
        
        # Add baseline correction info to stats
        if baseline_correct:
            stats['correction_factors'] = correction_factors.copy()
        
        # Add neuron counts for each cell type
        for cell_type, signals in all_signals_dict.items():
            stats[f'n_{cell_type.lower()}_neurons'] = len(signals)
        
        # Create the plot
        plot_title = (f"{title or f'{center_cell_type}-Centered Population Analysis'}\n"
                      f"All neurons averaged at {center_cell_type} peak times (n={valid_peak_count} peaks)")
        if baseline_correct:
            plot_title += " - Baseline Corrected"
        
        # Calculate neuron signal range for left y-axis
        all_neuron_values = []
        for cell_type in final_avg_traces:
            all_neuron_values.extend(final_avg_traces[cell_type])
            # Also include SEM bounds
            all_neuron_values.extend(final_avg_traces[cell_type] + final_sem_traces[cell_type])
            all_neuron_values.extend(final_avg_traces[cell_type] - final_sem_traces[cell_type])
        
        if all_neuron_values:
            neuron_min = min(all_neuron_values)
            neuron_max = max(all_neuron_values)
            # Add padding for better visualization
            neuron_padding = (neuron_max - neuron_min) * 0.15
            neuron_y_min = neuron_min - neuron_padding
            neuron_y_max = neuron_max + neuron_padding
        else:
            neuron_y_min, neuron_y_max = -1, 1
        
        p = figure(width=1000, height=400,
                   title=plot_title,
                   x_axis_label=f"Time relative to {center_cell_type} peak (s)",
                   y_axis_label="Average Calcium Signal",
                   y_range=(neuron_y_min, neuron_y_max),  # Explicitly set left y-axis range
                   tools="pan,box_zoom,wheel_zoom,reset,save")
        
        # Remove grid for cleaner look
        p.grid.grid_line_color = None
        
        # Define colors for each cell type
        colors = {'D1': 'blue', 'D2': 'red', 'CHI': 'green'}
        
        # Plot traces for each cell type (on left y-axis)
        for cell_type in all_signals_dict.keys():
            if cell_type in final_avg_traces:
                color = colors.get(cell_type, 'black')
                line_width = 3 if cell_type == center_cell_type else 2
                
                # Create legend label with baseline correction info
                legend_label = f"{cell_type} Population Avg (n={stats[f'n_{cell_type.lower()}_neurons']} neurons)"
                if baseline_correct and cell_type in correction_factors:
                    correction = correction_factors[cell_type]
                    if correction != 0:
                        legend_label += f" [Corrected: {correction:+.3f}]"
                
                # Main line (explicitly on default y-axis)
                p.line(time_axis, final_avg_traces[cell_type], line_width=line_width, color=color, 
                       legend_label=legend_label)
                
                # SEM patch (explicitly on default y-axis)
                p.patch(np.concatenate([time_axis, time_axis[::-1]]), 
                        np.concatenate([final_avg_traces[cell_type] - final_sem_traces[cell_type], 
                                      (final_avg_traces[cell_type] + final_sem_traces[cell_type])[::-1]]),
                        alpha=0.2, color=color)
        
        # Add velocity trace with secondary y-axis if available
        if has_velocity and avg_velocity_trace is not None:
            # Calculate velocity range for secondary y-axis
            velocity_values = list(avg_velocity_trace)
            velocity_values.extend(avg_velocity_trace + sem_velocity_trace)
            velocity_values.extend(avg_velocity_trace - sem_velocity_trace)
            
            vel_min = min(velocity_values)
            vel_max = max(velocity_values)
            vel_padding = (vel_max - vel_min) * 0.15
            vel_y_min = vel_min - vel_padding
            vel_y_max = vel_max + vel_padding
            
            # Add secondary y-axis for velocity
            p.extra_y_ranges = {"velocity_axis": Range1d(start=vel_y_min, end=vel_y_max)}
            velocity_axis = LinearAxis(y_range_name="velocity_axis", axis_label="Velocity (cm/s)")
            velocity_axis.axis_label_text_color = "orange"
            velocity_axis.major_label_text_color = "orange"
            velocity_axis.axis_line_color = "orange"
            velocity_axis.major_tick_line_color = "orange"
            velocity_axis.minor_tick_line_color = "orange"
            p.add_layout(velocity_axis, 'right')
            
            # Plot velocity trace (explicitly on velocity y-axis)
            p.line(time_axis, avg_velocity_trace, line_width=2, color='orange', 
                   y_range_name="velocity_axis", alpha=0.8,
                   legend_label="Smoothed Velocity")
            
            # Add velocity SEM patch (explicitly on velocity y-axis)
            p.patch(np.concatenate([time_axis, time_axis[::-1]]), 
                    np.concatenate([avg_velocity_trace - sem_velocity_trace, 
                                  (avg_velocity_trace + sem_velocity_trace)[::-1]]),
                    alpha=0.1, color='orange', y_range_name="velocity_axis")
        
        # Add vertical line at peak time (t=0) - spans the neuron y-axis range
        p.line([0, 0], [neuron_y_min, neuron_y_max], 
               line_width=3, color='black', line_dash='dashed', alpha=0.8,
               legend_label="Peak Time (t=0)")
        
        # Configure legend
        p.legend.location = "top_right"
        p.legend.click_policy = "hide"
        
        # Print summary statistics
        print(f"\n{center_cell_type}-Centered Population Analysis Summary:")
        print(f"Total {center_cell_type} peaks found: {stats[f'total_{center_cell_type.lower()}_peaks']}")
        print(f"Valid peaks analyzed: {stats['valid_peaks_analyzed']}")
        for cell_type, signals in all_signals_dict.items():
            print(f"{cell_type} neurons included: {len(signals)}")
        if has_velocity:
            print(f"Velocity data included: Yes (smoothed)")
        else:
            print(f"Velocity data included: No")
        
        # Output the plot
        self.output_bokeh_plot(p, save_path=save_path, title=plot_title, notebook=notebook, 
                             overwrite=overwrite, font_size=font_size)
        
        return p, stats

    def plot_population_centered_activity(self, center_cell_type='D1', signal_type='zsc',
                                         time_window=3.0, save_path=None, title=None, 
                                         notebook=False, overwrite=False, font_size=None,
                                         baseline_correct=False, correction_factors=None, exclude_indices=None):
        """
        Plot population-centered activity analysis where ALL neurons from each cell type
        are averaged at the center cell type peak times (no activity filtering).
        
        This function averages ALL neurons of each type at the same time points (center peaks),
        providing a population-level view of neural activity.
        
        Parameters:
        -----------
        center_cell_type : str
            The cell type to center analysis on ('D1', 'D2', or 'CHI')
        signal_type : str
            Type of signal to use ('zsc' for z-score normalized, 'denoised' for denoised signals)
        time_window : float
            Time window in seconds before and after each peak for plotting
        save_path : str, optional
            Path to save the HTML file
        title : str, optional
            Custom title for the plot
        notebook : bool
            Whether to display in notebook
        overwrite : bool
            Whether to overwrite existing file
        font_size : str, optional
            Font size for plot text elements
        baseline_correct : bool
            Whether to apply baseline correction
        correction_factors : dict, optional
            Dictionary of correction factors for each cell type {'D1': factor, 'D2': factor, 'CHI': factor}
        exclude_indices : array-like, optional
            Binary array or list of indices to exclude from analysis. If binary array, indices where value is 1 (or True) will be excluded.
            If list of indices, those specific indices will be excluded. Can be used to exclude timepoints during licking, movement, etc.
        
        Returns:
        --------
        tuple
            (bokeh.plotting.figure, dict) - The created figure and summary statistics
        """
        # Get signals based on type
        if signal_type == 'zsc':
            d1_signals = self.d1_zsc
            d2_signals = self.d2_zsc
            chi_signals = self.chi_zsc
            signal_description = 'Z-score Normalized'
        elif signal_type == 'denoised':
            d1_signals = self.d1_denoised
            d2_signals = self.d2_denoised
            chi_signals = self.chi_denoised
            signal_description = 'Denoised'
        else:
            raise ValueError("signal_type must be 'zsc' or 'denoised'")
        
        # Get peak indices
        d1_peak_indices = self.d1_peak_indices if hasattr(self, 'd1_peak_indices') else []
        d2_peak_indices = self.d2_peak_indices if hasattr(self, 'd2_peak_indices') else []
        chi_peak_indices = self.chi_peak_indices if hasattr(self, 'chi_peak_indices') else []
        
        # Prepare signals and peak indices based on center cell type
        all_signals_dict = {
            'D1': d1_signals,
            'D2': d2_signals, 
            'CHI': chi_signals
        }
        
        if center_cell_type == 'D1':
            center_signals = d1_signals
            center_peak_indices = d1_peak_indices
        elif center_cell_type == 'D2':
            center_signals = d2_signals
            center_peak_indices = d2_peak_indices
        elif center_cell_type == 'CHI':
            center_signals = chi_signals
            center_peak_indices = chi_peak_indices
        else:
            raise ValueError("center_cell_type must be 'D1', 'D2', or 'CHI'")
        
        # Generate title if not provided
        if title is None:
            title = f"{center_cell_type}-Centered Population Analysis: All Neurons Averaged ({signal_description})"
        
        # Call the core function
        return self._plot_population_centered_activity_core(
            center_cell_type=center_cell_type,
            center_signals=center_signals,
            center_peak_indices=center_peak_indices,
            all_signals_dict=all_signals_dict,
            ci_rate=self.ci_rate,
            time_window=time_window,
            save_path=save_path,
            title=title,
            notebook=notebook,
            overwrite=overwrite,
            font_size=font_size,
            baseline_correct=baseline_correct,
            correction_factors=correction_factors,
            exclude_indices=exclude_indices
        )

    def plot_all_population_centered_analyses(self, signal_type='zsc', time_window=3.0,
                                             save_path=None, notebook=False, overwrite=False, font_size=None,
                                             baseline_correct=True, baseline_window=(-3.0, -1.0), exclude_indices=None):
        """
        Comprehensive function to plot all three population-centered analyses (D1, D2, CHI) for comparison.
        
        This version averages ALL neurons of each type (no activity filtering) at the center cell type peak times,
        providing a population-level view of neural coordination.
        
        Parameters:
        -----------
        signal_type : str
            Type of signal to use ('zsc' for z-score normalized, 'denoised' for denoised signals)
        time_window : float
            Time window in seconds before and after each peak for plotting
        save_path : str, optional
            Path to save the combined HTML file
        notebook : bool
            Whether to display in notebook
        overwrite : bool
            Whether to overwrite existing file
        font_size : str, optional
            Font size for plot text elements
        baseline_correct : bool
            Whether to apply baseline correction to unify baseline levels across cell types
        baseline_window : tuple
            Time window (start, end) in seconds for baseline calculation (relative to peak time)
        exclude_indices : array-like, optional
            Binary array or list of indices to exclude from analysis. If binary array, indices where value is 1 (or True) will be excluded.
            If list of indices, those specific indices will be excluded. Can be used to exclude timepoints during licking, movement, etc.
        
        Returns:
        --------
        dict
            Dictionary containing plots and statistics for all three analyses
        """
        from bokeh.layouts import column
        from bokeh.io import output_file, save
        import numpy as np
        
        results = {}
        signal_description = 'Z-score' if signal_type == 'zsc' else 'Denoised'
        
        # Process exclude_indices to create exclusion set
        exclusion_set = set()
        if exclude_indices is not None:
            exclude_indices = np.array(exclude_indices)
            if exclude_indices.dtype == bool or all(val in [0, 1, True, False] for val in exclude_indices):
                # Binary array - get indices where value is True/1
                exclusion_set = set(np.where(exclude_indices)[0])
            else:
                # List of indices
                exclusion_set = set(exclude_indices)
            print(f"Excluding {len(exclusion_set)} time indices from analysis")
        
        # First pass: collect all traces to calculate global baseline correction
        all_traces = {'D1': [], 'D2': [], 'CHI': []}
        
        if baseline_correct:
            print(f"Calculating baseline correction factors...")
            
            # Get signals based on type
            if signal_type == 'zsc':
                d1_signals = self.d1_zsc
                d2_signals = self.d2_zsc
                chi_signals = self.chi_zsc
            elif signal_type == 'denoised':
                d1_signals = self.d1_denoised
                d2_signals = self.d2_denoised
                chi_signals = self.chi_denoised
            else:
                raise ValueError("signal_type must be 'zsc' or 'denoised'")
            
            # Get peak indices
            d1_peak_indices = self.d1_peak_indices if hasattr(self, 'd1_peak_indices') else []
            d2_peak_indices = self.d2_peak_indices if hasattr(self, 'd2_peak_indices') else []
            chi_peak_indices = self.chi_peak_indices if hasattr(self, 'chi_peak_indices') else []
            
            all_signals_dict = {
                'D1': d1_signals,
                'D2': d2_signals, 
                'CHI': chi_signals
            }
            
            all_peak_indices_dict = {
                'D1': d1_peak_indices,
                'D2': d2_peak_indices,
                'CHI': chi_peak_indices
            }
            
            # Calculate average traces for each cell type across all analysis types
            for center_type in ['D1', 'D2', 'CHI']:
                center_peak_indices = all_peak_indices_dict[center_type]
                
                # Collect all center peak times, excluding those in exclusion_set
                all_center_peak_times = []
                for neuron_idx, peaks in enumerate(center_peak_indices):
                    for peak_idx in peaks:
                        if peak_idx not in exclusion_set:
                            all_center_peak_times.append(peak_idx)
                
                if all_center_peak_times:
                    plot_window_samples = int(time_window * self.ci_rate)
                    
                    # Collect traces for each cell type at center peak times
                    for cell_type, signals in all_signals_dict.items():
                        if len(signals) > 0:
                            cell_type_traces = []
                            for peak_idx in all_center_peak_times:
                                start_idx = peak_idx - plot_window_samples
                                end_idx = peak_idx + plot_window_samples + 1
                                
                                # Check bounds
                                if start_idx >= 0 and end_idx < len(signals[0]):
                                    # Average all neurons of this cell type at this peak time
                                    traces_at_peak = [neuron_signal[start_idx:end_idx] for neuron_signal in signals]
                                    avg_trace_at_peak = np.mean(traces_at_peak, axis=0)
                                    cell_type_traces.append(avg_trace_at_peak)
                            
                            if cell_type_traces:
                                # Calculate final average trace for this cell type in this analysis
                                final_avg_trace = np.mean(cell_type_traces, axis=0)
                                all_traces[cell_type].append(final_avg_trace)
            
            # Calculate baseline values for each cell type
            baseline_corrections = {}
            time_axis = np.linspace(-time_window, time_window, 2 * int(time_window * self.ci_rate) + 1)
            baseline_mask = (time_axis >= baseline_window[0]) & (time_axis <= baseline_window[1])
            
            for cell_type in ['D1', 'D2', 'CHI']:
                if all_traces[cell_type]:
                    # Calculate average baseline across all analysis types for this cell type
                    all_baselines = []
                    for trace in all_traces[cell_type]:
                        baseline_value = np.mean(trace[baseline_mask])
                        all_baselines.append(baseline_value)
                    baseline_corrections[cell_type] = np.mean(all_baselines)
                else:
                    baseline_corrections[cell_type] = 0.0
            
            # Calculate the target baseline (minimum baseline to shift all to)
            target_baseline = min(baseline_corrections.values()) if baseline_corrections else 0.0
            
            print(f"Baseline values: D1={baseline_corrections.get('D1', 0):.4f}, "
                  f"D2={baseline_corrections.get('D2', 0):.4f}, CHI={baseline_corrections.get('CHI', 0):.4f}")
            print(f"Target baseline: {target_baseline:.4f}")
            
            # Calculate correction factors
            correction_factors = {}
            for cell_type in ['D1', 'D2', 'CHI']:
                correction_factors[cell_type] = target_baseline - baseline_corrections.get(cell_type, 0)
                print(f"{cell_type} correction factor: {correction_factors[cell_type]:.4f}")
        else:
            correction_factors = {'D1': 0, 'D2': 0, 'CHI': 0}
        
        # D1-centered population analysis
        print(f"\nRunning D1-centered population analysis ({signal_description})...")
        d1_plot, d1_stats = self.plot_population_centered_activity(
            center_cell_type='D1',
            signal_type=signal_type,
            time_window=time_window,
            save_path=None,  # Don't save individual plots
            title=f"D1-Centered Population Analysis: All Neurons Averaged ({signal_description})",
            notebook=False,  # Don't display individual plots
            overwrite=overwrite,
            font_size=font_size,
            baseline_correct=baseline_correct,
            correction_factors=correction_factors,
            exclude_indices=exclude_indices
        )
        results['d1_population'] = {'plot': d1_plot, 'stats': d1_stats}
        
        # D2-centered population analysis
        print(f"\nRunning D2-centered population analysis ({signal_description})...")
        d2_plot, d2_stats = self.plot_population_centered_activity(
            center_cell_type='D2',
            signal_type=signal_type,
            time_window=time_window,
            save_path=None,  # Don't save individual plots
            title=f"D2-Centered Population Analysis: All Neurons Averaged ({signal_description})",
            notebook=False,  # Don't display individual plots
            overwrite=overwrite,
            font_size=font_size,
            baseline_correct=baseline_correct,
            correction_factors=correction_factors,
            exclude_indices=exclude_indices
        )
        results['d2_population'] = {'plot': d2_plot, 'stats': d2_stats}
        
        # CHI-centered population analysis
        print(f"\nRunning CHI-centered population analysis ({signal_description})...")
        chi_plot, chi_stats = self.plot_population_centered_activity(
            center_cell_type='CHI',
            signal_type=signal_type,
            time_window=time_window,
            save_path=None,  # Don't save individual plots
            title=f"CHI-Centered Population Analysis: All Neurons Averaged ({signal_description})",
            notebook=False,  # Don't display individual plots
            overwrite=overwrite,
            font_size=font_size,
            baseline_correct=baseline_correct,
            correction_factors=correction_factors,
            exclude_indices=exclude_indices
        )
        results['chi_population'] = {'plot': chi_plot, 'stats': chi_stats}
        
        # Print comparative summary
        print("\n" + "="*80)
        print(f"POPULATION-LEVEL COMPARATIVE ANALYSIS SUMMARY ({signal_description})")
        if baseline_correct:
            print("*** BASELINE CORRECTED FOR UNIFIED COMPARISON ***")
        if exclude_indices is not None:
            print(f"*** {len(exclusion_set)} TIME INDICES EXCLUDED FROM ANALYSIS ***")
        print("="*80)
        
        if d1_stats and d2_stats and chi_stats:
            print(f"D1-centered: {d1_stats['valid_peaks_analyzed']} D1 peaks analyzed")
            print(f"D2-centered: {d2_stats['valid_peaks_analyzed']} D2 peaks analyzed")
            print(f"CHI-centered: {chi_stats['valid_peaks_analyzed']} CHI peaks analyzed")
            
            print(f"\nNeuron counts:")
            print(f"  D1 neurons: {d1_stats['n_d1_neurons']}")
            print(f"  D2 neurons: {d1_stats['n_d2_neurons']}")
            print(f"  CHI neurons: {d1_stats['n_chi_neurons']}")
            
            print(f"\nNote: All neurons of each type were included in averaging (no activity filtering)")
            if baseline_correct:
                print(f"Baseline correction applied using window {baseline_window} seconds relative to peak")
            if exclude_indices is not None:
                print(f"Excluded {len(exclusion_set)} time indices from peak analysis")
        
        # Create and save combined analysis
        if all([d1_plot, d2_plot, chi_plot]):
            combined_layout = column(d1_plot, d2_plot, chi_plot)
            combined_title = f"All Population-Centered Analyses ({signal_description})"
            if baseline_correct:
                combined_title += " - Baseline Corrected"
            if exclude_indices is not None:
                combined_title += f" - Excluded {len(exclusion_set)} indices"
            
            # Output the combined plot
            self.output_bokeh_plot(combined_layout, save_path=save_path, title=combined_title, 
                                 notebook=notebook, overwrite=overwrite, font_size=font_size)
            
            results['combined_plot'] = combined_layout
            
            if save_path:
                print(f"\nCombined population analysis saved to: {save_path}")
        
        return results

    # Convenience wrapper functions for population analysis
    def plot_d1_population_analysis(self, signal_type='zsc', save_path=None, time_window=3.0,
                                   notebook=False, overwrite=False, font_size=None):
        """
        Convenience function to plot D1-centered population analysis.
        
        Parameters:
        -----------
        signal_type : str
            Type of signal to use ('zsc' for z-score normalized, 'denoised' for denoised signals)
        save_path : str, optional
            Path to save the HTML file
        time_window : float
            Time window in seconds before and after each peak for plotting
        notebook : bool
            Whether to display in notebook
        overwrite : bool
            Whether to overwrite existing file
        font_size : str, optional
            Font size for plot text elements
        
        Returns:
        --------
        tuple
            (bokeh.plotting.figure, dict) - The created figure and summary statistics
        """
        return self.plot_population_centered_activity(
            center_cell_type='D1',
            signal_type=signal_type,
            time_window=time_window,
            save_path=save_path,
            title=f"D1-Centered Population Analysis: All Neurons Averaged ({'Z-score' if signal_type == 'zsc' else 'Denoised'})",
            notebook=notebook,
            overwrite=overwrite,
            font_size=font_size
        )

    def plot_d2_population_analysis(self, signal_type='zsc', save_path=None, time_window=3.0,
                                   notebook=False, overwrite=False, font_size=None):
        """
        Convenience function to plot D2-centered population analysis.
        
        Parameters:
        -----------
        signal_type : str
            Type of signal to use ('zsc' for z-score normalized, 'denoised' for denoised signals)
        save_path : str, optional
            Path to save the HTML file
        time_window : float
            Time window in seconds before and after each peak for plotting
        notebook : bool
            Whether to display in notebook
        overwrite : bool
            Whether to overwrite existing file
        font_size : str, optional
            Font size for plot text elements
        
        Returns:
        --------
        tuple
            (bokeh.plotting.figure, dict) - The created figure and summary statistics
        """
        return self.plot_population_centered_activity(
            center_cell_type='D2',
            signal_type=signal_type,
            time_window=time_window,
            save_path=save_path,
            title=f"D2-Centered Population Analysis: All Neurons Averaged ({'Z-score' if signal_type == 'zsc' else 'Denoised'})",
            notebook=notebook,
            overwrite=overwrite,
            font_size=font_size
        )

    def plot_chi_population_analysis(self, signal_type='zsc', save_path=None, time_window=3.0,
                                    notebook=False, overwrite=False, font_size=None):
        """
        Convenience function to plot CHI-centered population analysis.
        
        Parameters:
        -----------
        signal_type : str
            Type of signal to use ('zsc' for z-score normalized, 'denoised' for denoised signals)
        save_path : str, optional
            Path to save the HTML file
        time_window : float
            Time window in seconds before and after each peak for plotting
        notebook : bool
            Whether to display in notebook
        overwrite : bool
            Whether to overwrite existing file
        font_size : str, optional
            Font size for plot text elements
        
        Returns:
        --------
        tuple
            (bokeh.plotting.figure, dict) - The created figure and summary statistics
        """
        return self.plot_population_centered_activity(
            center_cell_type='CHI',
            signal_type=signal_type,
            time_window=time_window,
            save_path=save_path,
            title=f"CHI-Centered Population Analysis: All Neurons Averaged ({'Z-score' if signal_type == 'zsc' else 'Denoised'})",
            notebook=notebook,
            overwrite=overwrite,
            font_size=font_size
        )

    def plot_all_population_centered_analyses_unified_baseline(self, signal_type='zsc', time_window=3.0,
                                                              baseline_window=(-3.0, -1.0),
                                                              save_path=None, notebook=False, overwrite=False, font_size=None, exclude_indices=None):
        """
        Convenience function to plot all population-centered analyses with unified baseline correction.
        
        This function automatically applies baseline correction to ensure all cell types (D1, D2, CHI) 
        have the same baseline level for easy comparison.
        
        Parameters:
        -----------
        signal_type : str
            Type of signal to use ('zsc' for z-score normalized, 'denoised' for denoised signals)
        time_window : float
            Time window in seconds before and after each peak for plotting
        baseline_window : tuple
            Time window (start, end) in seconds for baseline calculation (relative to peak time)
        save_path : str, optional
            Path to save the combined HTML file
        notebook : bool
            Whether to display in notebook
        overwrite : bool
            Whether to overwrite existing file
        font_size : str, optional
            Font size for plot text elements
        exclude_indices : array-like, optional
            Binary array or list of indices to exclude from analysis. If binary array, indices where value is 1 (or True) will be excluded.
            If list of indices, those specific indices will be excluded. Can be used to exclude timepoints during licking, movement, etc.
        
        Returns:
        --------
        dict
            Dictionary containing plots and statistics for all three analyses with unified baselines
        """
        print("Creating population-centered analyses with unified baseline...")
        return self.plot_all_population_centered_analyses(
            signal_type=signal_type,
            time_window=time_window,
            save_path=save_path,
            notebook=notebook,
            overwrite=overwrite,
            font_size=font_size,
            baseline_correct=True,
            baseline_window=baseline_window,
            exclude_indices=exclude_indices
        )

    def plot_all_population_centered_analyses_no_baseline_correction(self, signal_type='zsc', time_window=3.0,
                                                                   save_path=None, notebook=False, overwrite=False, font_size=None, exclude_indices=None):
        """
        Convenience function to plot all population-centered analyses without baseline correction.
        
        This function creates the original analyses without any baseline modification.
        
        Parameters:
        -----------
        signal_type : str
            Type of signal to use ('zsc' for z-score normalized, 'denoised' for denoised signals)
        time_window : float
            Time window in seconds before and after each peak for plotting
        save_path : str, optional
            Path to save the combined HTML file
        notebook : bool
            Whether to display in notebook
        overwrite : bool
            Whether to overwrite existing file
        font_size : str, optional
            Font size for plot text elements
        exclude_indices : array-like, optional
            Binary array or list of indices to exclude from analysis. If binary array, indices where value is 1 (or True) will be excluded.
            If list of indices, those specific indices will be excluded. Can be used to exclude timepoints during licking, movement, etc.
        
        Returns:
        --------
        dict
            Dictionary containing plots and statistics for all three analyses without baseline correction
        """
        print("Creating population-centered analyses without baseline correction...")
        return self.plot_all_population_centered_analyses(
            signal_type=signal_type,
            time_window=time_window,
            save_path=save_path,
            notebook=notebook,
            overwrite=overwrite,
            font_size=font_size,
            baseline_correct=False,
            exclude_indices=exclude_indices
        )
