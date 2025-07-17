import os
import json
from typing import Optional, Dict, Any
from civis.src.CITank import CITank
from civis.src.ElecTank import ElecTank
from civis.src.CellTypeTank import CellTypeTank


class CompositeTank:
    """
    Composite Tank that combines CITank, ElecTank, and CellTypeTank for comprehensive analysis of 
    calcium imaging, electrophysiology, and cell-type-specific data.
    
    This class provides a three-way composition pattern where all analyzers share common behavioral data
    while maintaining their independent functionality and enabling sophisticated multi-modal analysis.
    
    Attributes:
        ci (CITank): Basic calcium imaging analyzer
        elec (ElecTank): Electrophysiology analyzer  
        celltype (CellTypeTank): Cell-type-specific calcium imaging analyzer
    """
    
    def __init__(self,
                 session_name: str,
                 # Shared parameters
                 virmen_path: Optional[str] = None,
                 maze_type: Optional[str] = None,
                 vm_rate: int = 20,
                 session_duration: int = 30 * 60,
                 
                 # CI-specific parameters
                 ci_path: Optional[str] = None,
                 gcamp_path: Optional[str] = None,
                 tdt_org_path: Optional[str] = None,
                 tdt_adjusted_path: Optional[str] = None,
                 ci_rate: int = 20,
                 
                 # Elec-specific parameters
                 elec_path: Optional[str] = None,
                 resample_fs: int = 200,
                 notch_fs: list = None,
                 notch_Q: float = 30,
                 
                 # CellType-specific parameters
                 cell_type_label_file: Optional[str] = None,
                 
                 # Configuration
                 config: Optional[Dict[str, Any]] = None):
        """
        Initialize CompositeTank with both CI and Elec analyzers.
        
        Parameters:
        -----------
        session_name : str
            Name of the session
        virmen_path : str, optional
            Path to Virmen behavioral data (shared between both analyzers)
        maze_type : str, optional
            Type of maze used in the experiment
        vm_rate : int
            Sampling rate for Virmen data
        session_duration : int
            Duration of session in seconds
        ci_path : str, optional
            Path to calcium imaging data
        gcamp_path : str, optional
            Path to GCaMP image data
        tdt_org_path : str, optional
            Path to original TDT data
        tdt_adjusted_path : str, optional
            Path to adjusted TDT data
        ci_rate : int
            Sampling rate for calcium imaging
        elec_path : str, optional
            Path to electrophysiology data
        resample_fs : int
            Resampling frequency for electrophysiology
        notch_fs : list, optional
            Notch filter frequencies (default: [60])
        notch_Q : float
            Q factor for notch filter
        cell_type_label_file : str, optional
            Path to cell type label file for CellTypeTank
        config : dict, optional
            Configuration dictionary override
        """
        
        self.session_name = session_name
        self.session_duration = session_duration
        self.vm_rate = vm_rate
        self.maze_type = maze_type
        
        # Load default config if not provided
        if config is None:
            self.config = self._load_config()
        else:
            self.config = config
        
        # Set default notch frequencies
        if notch_fs is None:
            notch_fs = [60]
        
        # Resolve paths using config if not provided
        virmen_path = self._resolve_path(virmen_path, 'VirmenFilePath', f"{session_name}.txt")
        ci_path = self._resolve_path(ci_path, 'ProcessedFilePath', session_name, f"{session_name}_v7.mat")
        elec_path = self._resolve_path(elec_path, 'ElecPath', session_name)
        
        if gcamp_path is None and 'ProcessedFilePath' in self.config:
            gcamp_path = os.path.join(self.config['ProcessedFilePath'], session_name,
                                     f"{session_name}_tiff_projections", f"{session_name}_max.tif")
        
        if tdt_org_path is None and 'ProcessedFilePath' in self.config:
            tdt_org_path = os.path.join(self.config['ProcessedFilePath'], session_name,
                                       f"{session_name}_alignment_check", f"{session_name}_tdt_original_16bit.tif")
        
        if tdt_adjusted_path is None and 'ProcessedFilePath' in self.config:
            tdt_adjusted_path = os.path.join(self.config['ProcessedFilePath'], session_name,
                                           f"{session_name}_tdt_adjustment", f"{session_name}_tdt_adjusted_16bit.tif")
        
        # Store paths for reference
        self.virmen_path = virmen_path
        self.ci_path = ci_path
        self.elec_path = elec_path
        
        # Initialize CI analyzer
        print(f"Initializing CI analyzer for {session_name}...")
        self.ci = CITank(
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
        self.ci_loaded = True
        print("CI analyzer loaded successfully")
        
        # Initialize Elec analyzer
        print(f"Initializing Elec analyzer for {session_name}...")
        self.elec = ElecTank(
            session_name=session_name,
            elec_path=elec_path,
            virmen_path=virmen_path,
            maze_type=maze_type,
            vm_rate=vm_rate,
            resample_fs=resample_fs,
            session_duration=session_duration,
            notch_fs=notch_fs,
            notch_Q=notch_Q
        )
        self.elec_loaded = True
        print("✓ Elec analyzer loaded successfully")
        
        # Initialize CellType analyzer
        print(f"Initializing CellType analyzer for {session_name}...")
        self.celltype = CellTypeTank(
            session_name=session_name,
            cell_type_label_file=cell_type_label_file,
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
        self.celltype_loaded = True
        print("✓ CellType analyzer loaded successfully")
        
        # Verify at least one analyzer loaded
        if not self.ci_loaded and not self.elec_loaded and not self.celltype_loaded:
            raise RuntimeError(f"Failed to load all analyzers for session {session_name}")
        
        # Store paths for CellType
        self.cell_type_label_file = cell_type_label_file
        
        print(f"CompositeTank initialized for {session_name} (CI: {'✓' if self.ci_loaded else '✗'}, Elec: {'✓' if self.elec_loaded else '✗'}, CellType: {'✓' if self.celltype_loaded else '✗'})")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from config.json file."""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        config_path = os.path.join(project_root, 'config.json')
        
        with open(config_path, 'r') as f:
            return json.load(f)
    
    def _resolve_path(self, provided_path: Optional[str], config_key: str, *path_parts) -> Optional[str]:
        """Resolve file path using config if not provided."""
        if provided_path is not None:
            return provided_path
        
        if config_key in self.config and self.config[config_key]:
            return os.path.join(self.config[config_key], *path_parts)
        
        return None
    
    @property
    def analyzers_loaded(self) -> Dict[str, bool]:
        """Return status of loaded analyzers."""
        return {
            'ci': self.ci_loaded,
            'elec': self.elec_loaded,
            'celltype': self.celltype_loaded
        }
    
    def get_session_info(self) -> Dict[str, Any]:
        """Get information about the composite session."""
        info = {
            'session_name': self.session_name,
            'session_duration_min': self.session_duration / 60,
            'analyzers_loaded': self.analyzers_loaded,
            'ci_info': {},
            'elec_info': {},
            'celltype_info': {}
        }
        
        # CI information
        if self.ci_loaded and self.ci:
            info['ci_info'] = {
                'num_neurons': self.ci.neuron_num if hasattr(self.ci, 'neuron_num') else 0,
                'ci_rate': self.ci.ci_rate if hasattr(self.ci, 'ci_rate') else 'N/A',
                'recording_length': self.ci.C_raw.shape[1] if hasattr(self.ci, 'C_raw') and self.ci.C_raw is not None else 0
            }
        
        # Elec information
        if self.elec_loaded and self.elec:
            info['elec_info'] = {
                'sampling_rate': self.elec.fs if hasattr(self.elec, 'fs') else 'N/A',
                'recording_length': len(self.elec.signal) if hasattr(self.elec, 'signal') and self.elec.signal is not None else 0,
                'frequency_bands': list(self.elec.frequency_bands.keys()) if hasattr(self.elec, 'frequency_bands') else []
            }
        
        # CellType information
        if self.celltype_loaded and self.celltype:
            info['celltype_info'] = {
                'num_neurons': self.celltype.neuron_num if hasattr(self.celltype, 'neuron_num') else 0,
                'ci_rate': self.celltype.ci_rate if hasattr(self.celltype, 'ci_rate') else 'N/A',
                'd1_neurons': len(self.celltype.d1_indices) if hasattr(self.celltype, 'd1_indices') and self.celltype.d1_indices is not None else 0,
                'd2_neurons': len(self.celltype.d2_indices) if hasattr(self.celltype, 'd2_indices') and self.celltype.d2_indices is not None else 0,
                'chi_neurons': len(self.celltype.chi_indices) if hasattr(self.celltype, 'chi_indices') and self.celltype.chi_indices is not None else 0,
                'cell_type_label_file': self.cell_type_label_file
            }
        
        return info
    
    def analyze_ci_elec_correlation(self, save_path: Optional[str] = None,
                                   notebook: bool = False, overwrite: bool = False,
                                   font_size: Optional[str] = None):
        """
        Analyze correlation between calcium imaging signals and electrophysiology signals.
        
        This is an example of a method that uses both analyzers for joint analysis.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the analysis plot
        notebook : bool
            Whether to display in notebook
        overwrite : bool
            Whether to overwrite existing file
        font_size : str, optional
            Font size for plot elements
            
        Returns:
        --------
        dict
            Analysis results
        """
        if not (self.ci_loaded and self.elec_loaded):
            print("Both CI and Elec analyzers must be loaded for correlation analysis")
            return {}
        
        from bokeh.plotting import figure
        from bokeh.layouts import column
        from scipy.stats import pearsonr
        import numpy as np
        
        # Get calcium activity (average across all neurons)
        if hasattr(self.ci, 'C_zsc') and self.ci.C_zsc is not None:
            ca_signal = np.mean(self.ci.C_zsc, axis=0)
            ca_time = self.ci.t
        else:
            print("No calcium signal available")
            return {}
        
        # Get electrophysiology signal
        if hasattr(self.elec, 'signal') and self.elec.signal is not None:
            elec_signal = self.elec.signal
            elec_time = self.elec.elc_t
        else:
            print("No electrophysiology signal available")
            return {}
        
        # Resample to common timebase (use CI timebase)
        elec_resampled = np.interp(ca_time, elec_time, elec_signal)
        
        # Calculate correlation
        correlation, p_value = pearsonr(ca_signal, elec_resampled)
        
        # Create plot
        p1 = figure(width=800, height=300, title="Average Calcium Activity",
                   x_axis_label="Time (s)", y_axis_label="Z-score",
                   tools="pan,wheel_zoom,box_zoom,reset,save")
        p1.line(ca_time, ca_signal, line_width=2, color='blue', alpha=0.7)
        
        p2 = figure(width=800, height=300, title="Electrophysiology Signal",
                   x_axis_label="Time (s)", y_axis_label="Amplitude",
                   x_range=p1.x_range,
                   tools="pan,wheel_zoom,box_zoom,reset,save")
        p2.line(ca_time, elec_resampled, line_width=2, color='red', alpha=0.7)
        
        # Correlation plot
        p3 = figure(width=800, height=400, title=f"CI-Elec Correlation (r={correlation:.3f}, p={p_value:.3e})",
                   x_axis_label="Average Calcium Signal", y_axis_label="Electrophysiology Signal",
                   tools="pan,wheel_zoom,box_zoom,reset,save")
        p3.scatter(ca_signal, elec_resampled, size=3, alpha=0.6, color='purple')
        
        layout = column(p1, p2, p3)
        
        # Save/show plot
        if self.ci:
            self.ci.output_bokeh_plot(layout, save_path=save_path, 
                                    title="CI-Elec Correlation Analysis",
                                    notebook=notebook, overwrite=overwrite, font_size=font_size)
        
        results = {
            'correlation': correlation,
            'p_value': p_value,
            'ca_signal_mean': np.mean(ca_signal),
            'elec_signal_mean': np.mean(elec_resampled),
            'plot': layout
        }
        
        print(f"CI-Elec correlation: r = {correlation:.4f}, p = {p_value:.2e}")
        
        return results
    
    def plot_combined_overview(self, time_start: float = 0, time_end: Optional[float] = None,
                              save_path: Optional[str] = None, notebook: bool = False, 
                              overwrite: bool = False, font_size: Optional[str] = None):
        """
        Create a combined overview plot showing both CI and Elec data with behavioral information.
        
        Parameters:
        -----------
        time_start : float
            Start time in seconds
        time_end : float, optional
            End time in seconds (if None, uses session duration)
        save_path : str, optional
            Path to save the plot
        notebook : bool
            Whether to display in notebook
        overwrite : bool
            Whether to overwrite existing file
        font_size : str, optional
            Font size for plot elements
            
        Returns:
        --------
        bokeh.layouts.layout
            Combined plot layout
        """
        if time_end is None:
            time_end = min(60, self.session_duration)  # Default to first 60 seconds
        
        from bokeh.plotting import figure
        from bokeh.layouts import column
        import numpy as np
        
        plots = []
        
        # Plot calcium imaging data if available
        if self.ci_loaded and self.ci:
            if hasattr(self.ci, 'C_zsc') and self.ci.C_zsc is not None:
                p_ci = figure(width=1000, height=300, title="Calcium Imaging (Average)",
                             x_axis_label="Time (s)", y_axis_label="Z-score",
                             x_range=(time_start, time_end),
                             tools="pan,wheel_zoom,box_zoom,reset,save")
                
                # Plot average calcium signal
                ca_avg = np.mean(self.ci.C_zsc, axis=0)
                time_mask = (self.ci.t >= time_start) & (self.ci.t <= time_end)
                p_ci.line(self.ci.t[time_mask], ca_avg[time_mask], 
                         line_width=2, color='blue', alpha=0.8, legend_label="Avg Ca2+")
                
                p_ci.legend.click_policy = "hide"
                plots.append(p_ci)
        
        # Plot electrophysiology data if available
        if self.elec_loaded and self.elec:
            if hasattr(self.elec, 'signal') and self.elec.signal is not None:
                p_elec = figure(width=1000, height=300, title="Electrophysiology",
                               x_axis_label="Time (s)", y_axis_label="Amplitude",
                               x_range=(time_start, time_end) if plots else None,
                               tools="pan,wheel_zoom,box_zoom,reset,save")
                
                time_mask = (self.elec.elc_t >= time_start) & (self.elec.elc_t <= time_end)
                p_elec.line(self.elec.elc_t[time_mask], self.elec.signal[time_mask], 
                           line_width=1, color='red', alpha=0.8, legend_label="LFP")
                
                p_elec.legend.click_policy = "hide"
                plots.append(p_elec)
        
        # Plot behavioral data if available (from either analyzer)
        behavior_analyzer = self.ci if self.ci_loaded else self.elec
        if behavior_analyzer and hasattr(behavior_analyzer, 'smoothed_velocity'):
            p_behav = figure(width=1000, height=300, title="Behavior",
                            x_axis_label="Time (s)", y_axis_label="Velocity",
                            x_range=plots[0].x_range if plots else (time_start, time_end),
                            tools="pan,wheel_zoom,box_zoom,reset,save")
            
            time_mask = (behavior_analyzer.t >= time_start) & (behavior_analyzer.t <= time_end)
            p_behav.line(behavior_analyzer.t[time_mask], 
                        behavior_analyzer.smoothed_velocity[time_mask], 
                        line_width=2, color='green', alpha=0.8, legend_label="Velocity")
            
            if hasattr(behavior_analyzer, 'lick') and behavior_analyzer.lick is not None:
                p_behav.line(behavior_analyzer.t[time_mask], 
                            behavior_analyzer.lick[time_mask], 
                            line_width=2, color='orange', alpha=0.8, legend_label="Lick")
            
            p_behav.legend.click_policy = "hide"
            plots.append(p_behav)
        
        if not plots:
            print("No data available for plotting")
            return None
        
        layout = column(plots)
        
        # Save/show plot using available analyzer
        output_analyzer = self.ci if self.ci_loaded else self.elec
        if output_analyzer:
            output_analyzer.output_bokeh_plot(layout, save_path=save_path,
                                             title="Combined CI-Elec Overview",
                                             notebook=notebook, overwrite=overwrite, 
                                             font_size=font_size)
        
        return layout
    
    def analyze_striatal_network_coordination(self, save_path: Optional[str] = None,
                                             notebook: bool = False, overwrite: bool = False,
                                             font_size: Optional[str] = None):
        """
        Unified analysis of striatal network coordination: D1-D2-CHI-LFP interactions.
        
        This method provides a comprehensive analysis of the striatal neural network by examining
        the coordinated activation patterns between D1 MSNs, D2 MSNs, cholinergic interneurons (CHI),
        and local field potentials (LFP) as an integrated system.
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the analysis plot
        notebook : bool
            Whether to display in notebook
        overwrite : bool
            Whether to overwrite existing file
        font_size : str, optional
            Font size for plot elements
            
        Returns:
        --------
        dict
            Comprehensive analysis results including:
            - Cross-correlation matrix (D1-D2-CHI-LFP)
            - Network coordination metrics
            - CHI modulation effects on MSNs
            - Temporal dynamics and phase relationships
            - Coordinated activation events
        """
        if not (self.celltype_loaded and self.elec_loaded):
            print("Both CellType and Elec analyzers must be loaded for striatal network analysis")
            return {}
        
        from bokeh.plotting import figure
        from bokeh.layouts import column, row, gridplot
        from bokeh.models import HoverTool, ColorBar, LinearColorMapper
        from bokeh.transform import transform
        from scipy.stats import pearsonr
        import numpy as np
        import pandas as pd
        
        # Get D1, D2, CHI signals
        if (hasattr(self.celltype, 'C_zsc') and self.celltype.C_zsc is not None and
                hasattr(self.celltype, 'd1_indices') and hasattr(self.celltype, 'd2_indices') and
                hasattr(self.celltype, 'chi_indices')):
                
            d1_signal = np.mean(self.celltype.C_zsc[self.celltype.d1_indices], axis=0)
            d2_signal = np.mean(self.celltype.C_zsc[self.celltype.d2_indices], axis=0)
            chi_signal = np.mean(self.celltype.C_zsc[self.celltype.chi_indices], axis=0)
            ca_time = self.celltype.t
        else:
            print("No D1/D2/CHI cell type signals available")
            return {}
        
        # Get electrophysiology signal
        if hasattr(self.elec, 'signal') and self.elec.signal is not None:
            elec_signal = self.elec.signal
            elec_time = self.elec.elc_t
            elec_resampled = np.interp(ca_time, elec_time, elec_signal)
        else:
            print("No electrophysiology signal available")
            return {}
        
        # Create signal matrix for analysis
        signals = {
            'D1': d1_signal,
            'D2': d2_signal, 
            'CHI': chi_signal,
            'LFP': elec_resampled
        }
        signal_names = list(signals.keys())
        
        # Calculate cross-correlation matrix
        corr_matrix = np.zeros((4, 4))
        p_values = np.zeros((4, 4))
        
        for i, name1 in enumerate(signal_names):
            for j, name2 in enumerate(signal_names):
                if i <= j:
                    corr, p_val = pearsonr(signals[name1], signals[name2])
                    corr_matrix[i, j] = corr
                    corr_matrix[j, i] = corr  # Symmetric matrix
                    p_values[i, j] = p_val
                    p_values[j, i] = p_val
        
        # Calculate network coordination metrics
        # Overall network coherence (average of all pairwise correlations excluding diagonal)
        network_coherence = np.mean(corr_matrix[np.triu_indices_from(corr_matrix, k=1)])
        
        # MSN coordination (D1-D2 correlation)
        msn_coordination = corr_matrix[0, 1]  # D1-D2
        
        # CHI modulation strength (CHI correlation with D1 and D2)
        chi_modulation_d1 = abs(corr_matrix[2, 0])  # CHI-D1
        chi_modulation_d2 = abs(corr_matrix[2, 1])  # CHI-D2
        chi_modulation_strength = np.mean([chi_modulation_d1, chi_modulation_d2])
        
        # LFP coupling strength (LFP correlation with all cell types)
        lfp_coupling = np.mean([abs(corr_matrix[3, 0]), abs(corr_matrix[3, 1]), abs(corr_matrix[3, 2])])
        
        # Detect CHI pause events and their effects
        chi_threshold = np.mean(chi_signal) - 0.5 * np.std(chi_signal)
        chi_pauses = chi_signal < chi_threshold
        
        # Find pause periods
        pause_starts = []
        pause_ends = []
        in_pause = False
        
        for i, is_pause in enumerate(chi_pauses):
            if is_pause and not in_pause:
                pause_starts.append(i)
                in_pause = True
            elif not is_pause and in_pause:
                pause_ends.append(i)
                in_pause = False
        
        if in_pause:
            pause_ends.append(len(chi_pauses))
        
        # Filter for significant pauses (>400ms)
        min_pause_samples = int(0.4 * self.celltype.ci_rate)
        significant_pauses = [(start, end) for start, end in zip(pause_starts, pause_ends) 
                            if end - start >= min_pause_samples]
        
        # Calculate MSN activity changes during CHI pauses
        d1_pause_effect = 0
        d2_pause_effect = 0
        
        if significant_pauses:
            d1_baseline = []
            d2_baseline = []
            d1_during_pause = []
            d2_during_pause = []
            
            for start, end in significant_pauses:
                # Baseline activity (before pause)
                baseline_start = max(0, start - (end - start))
                d1_baseline.extend(d1_signal[baseline_start:start])
                d2_baseline.extend(d2_signal[baseline_start:start])
                
                # Activity during pause
                d1_during_pause.extend(d1_signal[start:end])
                d2_during_pause.extend(d2_signal[start:end])
            
            if d1_baseline and d1_during_pause:
                d1_pause_effect = ((np.mean(d1_baseline) - np.mean(d1_during_pause)) / 
                                 np.mean(d1_baseline) * 100)
                d2_pause_effect = ((np.mean(d2_baseline) - np.mean(d2_during_pause)) / 
                                 np.mean(d2_baseline) * 100)
        
        # Detect coordinated activation events (all signals above threshold simultaneously)
        activation_thresholds = {name: np.mean(signal) + 0.5 * np.std(signal) 
                               for name, signal in signals.items()}
        
        coordinated_events = []
        min_event_duration = int(0.2 * self.celltype.ci_rate)  # 200ms minimum
        
        # Find periods where all signals are simultaneously activated
        all_active = np.all([signals[name] > activation_thresholds[name] for name in signal_names], axis=0)
        
        event_starts = []
        event_ends = []
        in_event = False
        
        for i, is_active in enumerate(all_active):
            if is_active and not in_event:
                event_starts.append(i)
                in_event = True
            elif not is_active and in_event:
                event_ends.append(i)
                in_event = False
        
        if in_event:
            event_ends.append(len(all_active))
        
        coordinated_events = [(start, end) for start, end in zip(event_starts, event_ends)
                            if end - start >= min_event_duration]
        
        # Create visualizations
        
        # 1. Time series plot of all signals
        p1 = figure(width=1200, height=400, title="Striatal Network Activity",
                   x_axis_label="Time (s)", y_axis_label="Z-score / Amplitude",
                   tools="pan,wheel_zoom,box_zoom,reset,save")
        
        colors = {'D1': 'blue', 'D2': 'red', 'CHI': 'orange', 'LFP': 'green'}
        for name, signal in signals.items():
            # Normalize LFP for better visualization
            if name == 'LFP':
                signal_norm = (signal - np.mean(signal)) / np.std(signal)
            else:
                signal_norm = signal
            
            p1.line(ca_time, signal_norm, line_width=2, color=colors[name], 
                   alpha=0.8, legend_label=name)
        
        # Highlight CHI pause periods
        for start, end in significant_pauses:
            start_time = ca_time[start]
            end_time = ca_time[end] if end < len(ca_time) else ca_time[-1]
            p1.add_layout(p1.rect(x=(start_time + end_time)/2, y=0, 
                                width=end_time-start_time, height=6, 
                                alpha=0.2, color='gray', line_color=None))
        
        # Highlight coordinated events
        for start, end in coordinated_events:
            start_time = ca_time[start]
            end_time = ca_time[end] if end < len(ca_time) else ca_time[-1]
            p1.add_layout(p1.rect(x=(start_time + end_time)/2, y=3, 
                                width=end_time-start_time, height=1, 
                                alpha=0.4, color='purple', line_color=None))
        
        p1.legend.click_policy = "hide"
        
        # 2. Correlation matrix heatmap
        p2 = figure(width=400, height=400, title="Network Cross-Correlation Matrix",
                   x_range=signal_names, y_range=list(reversed(signal_names)),
                   toolbar_location=None, tools="")
        
        # Create heatmap data
        x_coords = []
        y_coords = []
        correlations = []
        
        for i, name1 in enumerate(signal_names):
            for j, name2 in enumerate(signal_names):
                x_coords.append(name1)
                y_coords.append(signal_names[3-j])  # Reverse y-axis
                correlations.append(corr_matrix[3-j, i])
        
        # Color mapping
        mapper = LinearColorMapper(palette="RdBu11", low=-1, high=1)
        
        p2.rect(x=x_coords, y=y_coords, width=1, height=1,
               fill_color=transform('correlation', mapper),
               line_color="white", line_width=2)
        
        # Add correlation values as text
        for i, (x, y, corr) in enumerate(zip(x_coords, y_coords, correlations)):
            p2.text(x=[x], y=[y], text=[f"{corr:.3f}"], 
                   text_align="center", text_baseline="middle",
                   text_color="white" if abs(corr) > 0.5 else "black")
        
        # Add colorbar
        color_bar = ColorBar(color_mapper=mapper, width=8, location=(0,0))
        p2.add_layout(color_bar, 'right')
        
        # 3. Network metrics summary
        p3 = figure(width=600, height=300, title="Network Coordination Metrics",
                   x_range=['Network\nCoherence', 'MSN\nCoordination', 'CHI\nModulation', 'LFP\nCoupling'],
                   y_range=(0, 1),
                   tools="")
        
        metrics = [network_coherence, abs(msn_coordination), chi_modulation_strength, lfp_coupling]
        metric_colors = ['purple', 'darkblue', 'orange', 'green']
        
        p3.vbar(x=['Network\nCoherence', 'MSN\nCoordination', 'CHI\nModulation', 'LFP\nCoupling'],
               top=metrics, width=0.6, color=metric_colors, alpha=0.8)
        
        # Add value labels on bars
        for i, (x, y) in enumerate(zip(p3.x_range.factors, metrics)):
            p3.text(x=[x], y=[y + 0.02], text=[f"{y:.3f}"], 
                   text_align="center", text_baseline="bottom")
        
        p3.yaxis.axis_label = "Coordination Strength"
        
        # 4. CHI modulation effects
        p4 = figure(width=400, height=300, title="CHI Pause Effects on MSNs",
                   x_range=['D1 Reduction', 'D2 Reduction'],
                   y_range=(0, max(60, max(d1_pause_effect, d2_pause_effect) * 1.2)),
                   tools="")
        
        p4.vbar(x=['D1 Reduction', 'D2 Reduction'],
               top=[d1_pause_effect, d2_pause_effect], 
               width=0.6, color=['blue', 'red'], alpha=0.8)
        
        # Add value labels
        p4.text(x=['D1 Reduction', 'D2 Reduction'], 
               y=[d1_pause_effect + 2, d2_pause_effect + 2],
               text=[f"{d1_pause_effect:.1f}%", f"{d2_pause_effect:.1f}%"],
               text_align="center", text_baseline="bottom")
        
        p4.yaxis.axis_label = "Activity Reduction (%)"
        
        # Layout arrangement
        top_row = row(p1)
        middle_row = row(p2, column(p3, p4))
        layout = column(top_row, middle_row)
        
        # Save/show plot
        if self.celltype:
            self.celltype.output_bokeh_plot(layout, save_path=save_path,
                                          title="Striatal Network Coordination Analysis",
                                          notebook=notebook, overwrite=overwrite, font_size=font_size)
        
        # Compile comprehensive results
        results = {
            # Network-level metrics
            'network_coherence': network_coherence,
            'msn_coordination': msn_coordination,
            'chi_modulation_strength': chi_modulation_strength,
            'lfp_coupling_strength': lfp_coupling,
            
            # Correlation matrix
            'correlation_matrix': corr_matrix,
            'p_value_matrix': p_values,
            'signal_names': signal_names,
            
            # Individual correlations
            'd1_d2_correlation': corr_matrix[0, 1],
            'd1_chi_correlation': corr_matrix[0, 2],
            'd2_chi_correlation': corr_matrix[1, 2],
            'd1_lfp_correlation': corr_matrix[0, 3],
            'd2_lfp_correlation': corr_matrix[1, 3],
            'chi_lfp_correlation': corr_matrix[2, 3],
            
            # CHI modulation effects
            'd1_pause_effect_percent': d1_pause_effect,
            'd2_pause_effect_percent': d2_pause_effect,
            'num_chi_pauses': len(significant_pauses),
            'total_pause_duration_sec': sum([(end-start)/self.celltype.ci_rate for start, end in significant_pauses]),
            
            # Coordinated events
            'num_coordinated_events': len(coordinated_events),
            'total_coordination_duration_sec': sum([(end-start)/self.celltype.ci_rate for start, end in coordinated_events]),
            'coordination_rate_per_min': len(coordinated_events) / (len(ca_time) / self.celltype.ci_rate / 60),
            
            # Signal statistics
            'signal_means': {name: np.mean(signal) for name, signal in signals.items()},
            'signal_stds': {name: np.std(signal) for name, signal in signals.items()},
            
            'plot': layout
        }
        
        # Print comprehensive summary
        print(f"🧠 Striatal Network Coordination Analysis:")
        print(f"  📊 Network coherence: {network_coherence:.4f}")
        print(f"  🔗 MSN coordination (D1-D2): r = {msn_coordination:.4f}")
        print(f"  🎯 CHI modulation strength: {chi_modulation_strength:.4f}")
        print(f"  ⚡ LFP coupling strength: {lfp_coupling:.4f}")
        print(f"")
        print(f"  🔄 Individual correlations:")
        print(f"    D1-D2: r = {corr_matrix[0,1]:.4f}")
        print(f"    D1-CHI: r = {corr_matrix[0,2]:.4f}")
        print(f"    D2-CHI: r = {corr_matrix[1,2]:.4f}")
        print(f"    CHI-LFP: r = {corr_matrix[2,3]:.4f}")
        print(f"")
        print(f"  🚫 CHI pause effects:")
        print(f"    D1 activity reduction: {d1_pause_effect:.1f}%")
        print(f"    D2 activity reduction: {d2_pause_effect:.1f}%")
        print(f"    Number of CHI pauses (>400ms): {len(significant_pauses)}")
        print(f"")
        print(f"  🎊 Coordinated activation events:")
        print(f"    Number of events: {len(coordinated_events)}")
        print(f"    Coordination rate: {len(coordinated_events) / (len(ca_time) / self.celltype.ci_rate / 60):.2f} events/min")
        
        return results
    
    def analyze_coordination(self, analysis_type: str = 'auto', **kwargs):
        """
        Smart coordination analysis that selects the best method based on available data.
        
        Parameters:
        -----------
        analysis_type : str
            Type of analysis ('auto', 'basic', 'striatal', 'full')
            - 'auto': Automatically select best available analysis
            - 'basic': Basic CI-Elec correlation (requires ci + elec)
            - 'striatal': Unified striatal network analysis (requires celltype + elec)
            - 'full': All available analyses
        **kwargs
            Additional arguments passed to specific analysis methods
            
        Returns:
        --------
        dict
            Analysis results from the selected method(s)
        """
        results = {'analysis_type_used': [], 'results': {}}
        
        if analysis_type == 'auto':
            # Automatically select best available analysis
            if self.celltype_loaded and self.elec_loaded:
                print("🧠 Running unified striatal network analysis (CellType + Elec available)")
                results['results']['striatal_network'] = self.analyze_striatal_network_coordination(**kwargs)
                results['analysis_type_used'].append('striatal_network')
            elif self.ci_loaded and self.elec_loaded:
                print("Running basic CI-Elec correlation analysis (CI + Elec available)")
                results['results']['ci_elec'] = self.analyze_ci_elec_correlation(**kwargs)
                results['analysis_type_used'].append('ci_elec')
            else:
                print("Insufficient data for coordination analysis. Need at least CI+Elec or CellType+Elec")
                return results
                
        elif analysis_type == 'basic':
            if self.ci_loaded and self.elec_loaded:
                results['results']['ci_elec'] = self.analyze_ci_elec_correlation(**kwargs)
                results['analysis_type_used'].append('ci_elec')
            else:
                print("Basic analysis requires CI + Elec analyzers")
                
        elif analysis_type == 'striatal':
            if self.celltype_loaded and self.elec_loaded:
                results['results']['striatal_network'] = self.analyze_striatal_network_coordination(**kwargs)
                results['analysis_type_used'].append('striatal_network')
            else:
                print("Striatal network analysis requires CellType + Elec analyzers")
                
        elif analysis_type == 'full':
            # Run all possible analyses
            if self.ci_loaded and self.elec_loaded:
                results['results']['ci_elec'] = self.analyze_ci_elec_correlation(**kwargs)
                results['analysis_type_used'].append('ci_elec')
                
            if self.celltype_loaded and self.elec_loaded:
                results['results']['striatal_network'] = self.analyze_striatal_network_coordination(**kwargs)
                results['analysis_type_used'].append('striatal_network')
        
        return results
    
    def plot_comprehensive_overview(self, time_start: float = 0, time_end: Optional[float] = None,
                                   save_path: Optional[str] = None, notebook: bool = False, 
                                   overwrite: bool = False, font_size: Optional[str] = None):
        """
        Create the most comprehensive overview plot based on available data.
        
        This method creates different overview plots depending on what analyzers are loaded:
        - CI + Elec: Shows average Ca + LFP + behavior
        - CellType + Elec: Shows D1/D2/CHI + LFP + behavior  
        - All three: Shows everything with maximum detail
        
        Parameters:
        -----------
        time_start : float
            Start time in seconds
        time_end : float, optional
            End time in seconds
        save_path : str, optional
            Path to save the plot
        notebook : bool
            Whether to display in notebook
        overwrite : bool
            Whether to overwrite existing file
        font_size : str, optional
            Font size for plot elements
            
        Returns:
        --------
        bokeh.layouts.layout
            Comprehensive overview plot layout
        """
        if time_end is None:
            time_end = min(60, self.session_duration)
        
        try:
            from bokeh.plotting import figure
            from bokeh.layouts import column
            import numpy as np
            
            plots = []
            
            # Plot cell-type-specific calcium data if available (preferred)
            if self.celltype_loaded and self.celltype:
                if (hasattr(self.celltype, 'C_zsc') and self.celltype.C_zsc is not None and
                    hasattr(self.celltype, 'd1_indices') and hasattr(self.celltype, 'd2_indices') and
                    hasattr(self.celltype, 'chi_indices')):
                    
                    p_celltype = figure(width=1000, height=350, title="Cell-Type-Specific Calcium Activity",
                                       x_axis_label="Time (s)", y_axis_label="Z-score",
                                       x_range=(time_start, time_end),
                                       tools="pan,wheel_zoom,box_zoom,reset,save")
                    
                    # Plot D1, D2, CHI signals
                    d1_signal = np.mean(self.celltype.C_zsc[self.celltype.d1_indices], axis=0)
                    d2_signal = np.mean(self.celltype.C_zsc[self.celltype.d2_indices], axis=0)
                    chi_signal = np.mean(self.celltype.C_zsc[self.celltype.chi_indices], axis=0)
                    
                    time_mask = (self.celltype.t >= time_start) & (self.celltype.t <= time_end)
                    p_celltype.line(self.celltype.t[time_mask], d1_signal[time_mask], 
                                   line_width=2, color='blue', alpha=0.8, legend_label="D1 MSNs")
                    p_celltype.line(self.celltype.t[time_mask], d2_signal[time_mask], 
                                   line_width=2, color='red', alpha=0.8, legend_label="D2 MSNs")
                    p_celltype.line(self.celltype.t[time_mask], chi_signal[time_mask], 
                                   line_width=2, color='orange', alpha=0.8, legend_label="CHI")
                    
                    p_celltype.legend.click_policy = "hide"
                    plots.append(p_celltype)
                    
            # Plot basic calcium data if CellType not available but CI is
            elif self.ci_loaded and self.ci and not self.celltype_loaded:
                if hasattr(self.ci, 'C_zsc') and self.ci.C_zsc is not None:
                    p_ci = figure(width=1000, height=300, title="Average Calcium Activity",
                                 x_axis_label="Time (s)", y_axis_label="Z-score",
                                 x_range=(time_start, time_end),
                                 tools="pan,wheel_zoom,box_zoom,reset,save")
                    
                    ca_avg = np.mean(self.ci.C_zsc, axis=0)
                    time_mask = (self.ci.t >= time_start) & (self.ci.t <= time_end)
                    p_ci.line(self.ci.t[time_mask], ca_avg[time_mask], 
                             line_width=2, color='blue', alpha=0.8, legend_label="Avg Ca2+")
                    
                    p_ci.legend.click_policy = "hide"
                    plots.append(p_ci)
            
            # Plot electrophysiology data if available
            if self.elec_loaded and self.elec:
                if hasattr(self.elec, 'signal') and self.elec.signal is not None:
                    p_elec = figure(width=1000, height=300, title="Electrophysiology (LFP)",
                                   x_axis_label="Time (s)", y_axis_label="Amplitude",
                                   x_range=plots[0].x_range if plots else (time_start, time_end),
                                   tools="pan,wheel_zoom,box_zoom,reset,save")
                    
                    time_mask = (self.elec.elc_t >= time_start) & (self.elec.elc_t <= time_end)
                    p_elec.line(self.elec.elc_t[time_mask], self.elec.signal[time_mask], 
                               line_width=1, color='green', alpha=0.8, legend_label="LFP")
                    
                    p_elec.legend.click_policy = "hide"
                    plots.append(p_elec)
            
            # Plot behavioral data (prefer CellType, then CI, then Elec)
            behavior_analyzer = None
            if self.celltype_loaded and self.celltype:
                behavior_analyzer = self.celltype
            elif self.ci_loaded and self.ci:
                behavior_analyzer = self.ci
            elif self.elec_loaded and self.elec:
                behavior_analyzer = self.elec
                
            if behavior_analyzer and hasattr(behavior_analyzer, 'smoothed_velocity'):
                p_behav = figure(width=1000, height=300, title="Behavioral Data",
                                x_axis_label="Time (s)", y_axis_label="Velocity/Lick",
                                x_range=plots[0].x_range if plots else (time_start, time_end),
                                tools="pan,wheel_zoom,box_zoom,reset,save")
                
                time_mask = (behavior_analyzer.t >= time_start) & (behavior_analyzer.t <= time_end)
                p_behav.line(behavior_analyzer.t[time_mask], 
                            behavior_analyzer.smoothed_velocity[time_mask], 
                            line_width=2, color='purple', alpha=0.8, legend_label="Velocity")
                
                if hasattr(behavior_analyzer, 'lick') and behavior_analyzer.lick is not None:
                    p_behav.line(behavior_analyzer.t[time_mask], 
                                behavior_analyzer.lick[time_mask], 
                                line_width=2, color='cyan', alpha=0.8, legend_label="Lick")
                
                p_behav.legend.click_policy = "hide"
                plots.append(p_behav)
            
            if not plots:
                print("No data available for comprehensive overview")
                return None
            
            layout = column(plots)
            
            # Determine title based on what's included
            if self.celltype_loaded and self.elec_loaded:
                title = "Comprehensive Overview: Cell-Type + Electrophysiology + Behavior"
            elif self.ci_loaded and self.elec_loaded:
                title = "Overview: Calcium Imaging + Electrophysiology + Behavior"
            else:
                title = "Data Overview"
            
            # Save/show plot using best available analyzer
            output_analyzer = self.celltype if self.celltype_loaded else (self.ci if self.ci_loaded else self.elec)
            if output_analyzer:
                output_analyzer.output_bokeh_plot(layout, save_path=save_path,
                                                 title=title,
                                                 notebook=notebook, overwrite=overwrite, 
                                                 font_size=font_size)
            
            return layout
            
        except Exception as e:
            print(f"Error creating comprehensive overview plot: {str(e)}")
            return None
    
    def __repr__(self):
        """String representation of CompositeTank."""
        status = []
        if self.ci_loaded:
            status.append("CI✓")
        if self.elec_loaded:
            status.append("Elec✓")
        if self.celltype_loaded:
            status.append("CellType✓")
        
        return f"CompositeTank(session='{self.session_name}', loaded=[{', '.join(status)}])"


if __name__ == "__main__":
    # Example usage
    pass
