import numpy as np
import pandas as pd
from bokeh.plotting import figure, output_file, save, show
from bokeh.layouts import column, row, gridplot
from bokeh.models import HoverTool, ColumnDataSource, Div, Legend, LegendItem
from bokeh.palettes import Category10, Set3
from bokeh.io import output_notebook
import os
from typing import List, Dict, Any, Optional, Union
import json
from collections import defaultdict
import warnings
from scipy import stats
from scipy.stats import ttest_ind, mannwhitneyu, kruskal
import matplotlib.pyplot as plt
import seaborn as sns

from civis.src.CellTypeTank import CellTypeTank
from civis.src.CITank import CITank
from civis.src.ElecTank import ElecTank
from civis.src.VirmenTank import VirmenTank


class MultiSessionAnalyzer:
    """
    Multi-session neural data analyzer that can handle multiple sessions
    and perform cross-session analysis of neural activity.
    """
    
    def __init__(self, sessions_config: List[Dict[str, Any]], analyzer_type: str = 'CellTypeTank'):
        """
        Initialize MultiSessionAnalyzer with multiple sessions.
        
        Parameters:
        -----------
        sessions_config : List[Dict[str, Any]]
            List of configuration dictionaries for each session.
            Each dict should contain parameters needed for the analyzer class.
        analyzer_type : str
            Type of analyzer to use ('CellTypeTank', 'CITank', 'ElecTank', 'VirmenTank')
        """
        self.sessions_config = sessions_config
        self.analyzer_type = analyzer_type
        self.sessions = {}
        self.session_names = []
        self.loaded_sessions = []
        
        # Initialize analyzers for each session
        self._load_sessions()
        
        # Cross-session analysis results storage
        self.cross_session_results = {}
        
    def _load_sessions(self):
        """Load all sessions based on configuration."""
        analyzer_class = self._get_analyzer_class()
        
        for i, config in enumerate(self.sessions_config):
            session_name = config.get('session_name', f'Session_{i+1}')
            self.session_names.append(session_name)
            
            try:
                # Validate configuration before creating analyzer
                validated_config = self._validate_session_config(config)
                
                # Create analyzer instance with validated config
                analyzer = analyzer_class(**validated_config)
                self.sessions[session_name] = analyzer
                self.loaded_sessions.append(session_name)
                print(f"Successfully loaded {session_name}")
            except Exception as e:
                print(f"Failed to load {session_name}: {str(e)}")
                # Store failed session info for debugging
                if not hasattr(self, 'failed_sessions'):
                    self.failed_sessions = {}
                self.failed_sessions[session_name] = str(e)
                
    def _get_analyzer_class(self):
        """Get the appropriate analyzer class based on type."""
        analyzer_classes = {
            'CellTypeTank': CellTypeTank,
            'CITank': CITank,
            'ElecTank': ElecTank,
            'VirmenTank': VirmenTank
        }
        
        if self.analyzer_type not in analyzer_classes:
            raise ValueError(f"Unknown analyzer type: {self.analyzer_type}")
            
        return analyzer_classes[self.analyzer_type]
    
    def _validate_session_config(self, config):
        """
        Validate and process session configuration.
        
        Parameters:
        -----------
        config : dict
            Session configuration dictionary
            
        Returns:
        --------
        dict
            Validated and processed configuration
        """
        validated_config = config.copy()
        
        # Ensure session_name is provided
        if 'session_name' not in validated_config:
            raise ValueError("session_name is required in session configuration")
        
        session_name = validated_config['session_name']
        
        # Check if files exist when paths are provided (optional check)
        file_paths = ['ci_path', 'virmen_path', 'gcamp_path', 'tdt_org_path', 'tdt_adjusted_path', 'cell_type_label_file', 'elec_path']
        
        for path_key in file_paths:
            if path_key in validated_config and validated_config[path_key] is not None:
                file_path = validated_config[path_key]
                if not os.path.exists(file_path):
                    print(f"Warning: File not found for {session_name}: {file_path}")
                    # Don't raise error, let the Tank class handle it
        
        # Set default values based on analyzer type
        if self.analyzer_type == 'VirmenTank':
            default_values = {
                'vm_rate': 20,
                'session_duration': 30 * 60,
                'maze_type': None
            }
            # Remove ci_rate if present (VirmenTank doesn't use it)
            if 'ci_rate' in validated_config:
                del validated_config['ci_rate']
                
        elif self.analyzer_type == 'CITank':
            default_values = {
                'ci_rate': 20,
                'vm_rate': 20,
                'session_duration': 30 * 60,
                'maze_type': None
            }
            
        elif self.analyzer_type == 'ElecTank':
            default_values = {
                'vm_rate': 20,
                'resample_fs': 200,
                'session_duration': 30 * 60,
                'maze_type': None,
                'notch_fs': [60],
                'notch_Q': 30
            }
            # Remove ci_rate if present (ElecTank doesn't use it)
            if 'ci_rate' in validated_config:
                del validated_config['ci_rate']
                
        elif self.analyzer_type == 'CellTypeTank':
            default_values = {
                'ci_rate': 20,
                'vm_rate': 20,
                'session_duration': 30 * 60,
                'maze_type': None
            }
        else:
            # Fallback defaults
            default_values = {
                'vm_rate': 20,
                'session_duration': 30 * 60,
                'maze_type': None
            }
        
        # Apply default values for missing parameters
        for key, default_value in default_values.items():
            if key not in validated_config:
                validated_config[key] = default_value
        
        return validated_config
    
    def get_session_info(self) -> pd.DataFrame:
        """Get information about all loaded sessions."""
        info_data = []
        
        for session_name in self.loaded_sessions:
            session = self.sessions[session_name]
            info = {
                'Session Name': session_name,
                'Session Duration (min)': session.session_duration / 60,
                'Analyzer Type': self.analyzer_type,
                'Status': 'Loaded'
            }
            
            # Add specific information based on analyzer type
            if hasattr(session, 'C_raw') and session.C_raw is not None:
                info['Number of Neurons'] = session.C_raw.shape[0]
                info['Recording Length (samples)'] = session.C_raw.shape[1]
            elif hasattr(session, 'C') and session.C is not None:
                info['Number of Neurons'] = session.C.shape[0]
                info['Recording Length (samples)'] = session.C.shape[1]
            else:
                info['Number of Neurons'] = 0
                info['Recording Length (samples)'] = 0
                
            if hasattr(session, 'd1_indices') and hasattr(session, 'd2_indices') and hasattr(session, 'chi_indices'):
                info['D1 Neurons'] = len(session.d1_indices) if session.d1_indices is not None else 0
                info['D2 Neurons'] = len(session.d2_indices) if session.d2_indices is not None else 0
                info['CHI Neurons'] = len(session.chi_indices) if session.chi_indices is not None else 0
                    
            info_data.append(info)
        
        # Add failed sessions info
        if hasattr(self, 'failed_sessions'):
            for session_name, error in self.failed_sessions.items():
                info = {
                    'Session Name': session_name,
                    'Session Duration (min)': 'N/A',
                    'Analyzer Type': self.analyzer_type,
                    'Status': f'Failed: {error[:50]}...' if len(error) > 50 else f'Failed: {error}',
                    'Number of Neurons': 0,
                    'Recording Length (samples)': 0
                }
                if self.analyzer_type == 'CellTypeTank':
                    info.update({'D1 Neurons': 0, 'D2 Neurons': 0, 'CHI Neurons': 0})
                info_data.append(info)
            
        return pd.DataFrame(info_data)
    
    def get_failed_sessions_info(self) -> Dict[str, str]:
        """Get detailed information about failed sessions."""
        if hasattr(self, 'failed_sessions'):
            return self.failed_sessions.copy()
        return {}
    
    def compare_neuron_counts_across_sessions(self, save_path: Optional[str] = None, 
                                            notebook: bool = False, overwrite: bool = False,
                                            font_size: Optional[str] = None):
        """Compare neuron counts across sessions."""
        if self.analyzer_type != 'CellTypeTank':
            print("Neuron count comparison is only available for CellTypeTank analyzer")
            return
            
        # Collect data
        session_data = []
        for session_name in self.loaded_sessions:
            session = self.sessions[session_name]
            if hasattr(session, 'd1_indices') and hasattr(session, 'd2_indices') and hasattr(session, 'chi_indices'):
                d1_count = len(session.d1_indices) if session.d1_indices is not None else 0
                d2_count = len(session.d2_indices) if session.d2_indices is not None else 0
                chi_count = len(session.chi_indices) if session.chi_indices is not None else 0
                total_count = d1_count + d2_count + chi_count
                
                session_data.append({
                    'Session': session_name,
                    'D1': d1_count,
                    'D2': d2_count,
                    'CHI': chi_count,
                    'Total': total_count
                })
        
        if not session_data:
            print("No cell type data available for comparison")
            return
            
        # Create visualization
        p = figure(x_range=[d['Session'] for d in session_data],
                  width=800, height=400,
                  title="Neuron Counts Across Sessions",
                  )
        
        # Prepare data for stacked bar chart
        sessions = [d['Session'] for d in session_data]
        d1_counts = np.array([d['D1'] for d in session_data])
        d2_counts = np.array([d['D2'] for d in session_data])
        chi_counts = np.array([d['CHI'] for d in session_data])
        
        # Create stacked bars
        d1_bars = p.vbar(x=sessions, top=d1_counts, width=0.8, color='navy', alpha=0.7)
        d2_bars = p.vbar(x=sessions, top=d2_counts+d1_counts, width=0.8, color='red', 
                        alpha=0.7, bottom=d1_counts)
        chi_bars = p.vbar(x=sessions, top=d1_counts+d2_counts+chi_counts, width=0.8, color='green', 
                         alpha=0.7, bottom=d1_counts+d2_counts)
        
        # Create legend items and place legend outside the plot
        legend_items = [
            LegendItem(label="D1", renderers=[d1_bars]),
            LegendItem(label="D2", renderers=[d2_bars]),
            LegendItem(label="CHI", renderers=[chi_bars])
        ]
        
        # Create legend and place it outside the plot (to the right)
        legend = Legend(items=legend_items, location="center")
        p.add_layout(legend, 'right')
        
        p.xaxis.axis_label = "Sessions"
        p.yaxis.axis_label = "Number of Neurons"
        p.xaxis.major_label_orientation = 45
        p.legend.click_policy = "hide"
        
        # Save/show plot using existing method
        if self.loaded_sessions:
            self.sessions[self.loaded_sessions[0]].output_bokeh_plot(
                p, save_path=save_path, title="Neuron Counts Comparison", 
                notebook=notebook, overwrite=overwrite, font_size=font_size
            )
        
        return pd.DataFrame(session_data)
    
    def analyze_cross_session_activity_patterns(self, signal_type: str = 'zsc',
                                              save_path: Optional[str] = None,
                                              notebook: bool = False, overwrite: bool = False,
                                              font_size: Optional[str] = None):
        """Analyze activity patterns across sessions."""
        if self.analyzer_type != 'CellTypeTank':
            print("Cross-session activity analysis is only available for CellTypeTank analyzer")
            return
            
        # Collect activity statistics for each session
        session_stats = {}
        
        for session_name in self.loaded_sessions:
            session = self.sessions[session_name]
            if not (hasattr(session, 'd1_indices') and hasattr(session, 'd2_indices') and hasattr(session, 'chi_indices')):
                continue
                
            stats_data = {}
            
            # Get appropriate signal
            if signal_type == 'zsc' and hasattr(session, 'C_zsc'):
                signal = session.C_zsc
            elif signal_type == 'dff' and hasattr(session, 'C_raw_deltaF_over_F'):
                signal = session.C_raw_deltaF_over_F
            elif hasattr(session, 'C_raw'):
                signal = session.C_raw
            else:
                continue
                
            if signal is None:
                continue
                
            # Process each cell type
            cell_types_info = {
                'D1': session.d1_indices,
                'D2': session.d2_indices,
                'CHI': session.chi_indices
            }
            
            for cell_type, indices in cell_types_info.items():
                if indices is not None and len(indices) > 0:
                    cell_signals = signal[indices, :]
                    
                    # Find peaks for spike rate calculation
                    spike_rate = 0
                    peak_attr = f'{cell_type.lower()}_peak_indices'
                    if hasattr(session, peak_attr):
                        peak_indices = getattr(session, peak_attr)
                        if peak_indices is not None:
                            # peak_indices is a list of arrays, count total peaks
                            total_peaks = sum(len(peaks) for peaks in peak_indices)
                            spike_rate = total_peaks / (session.session_duration / 60)
                    
                    stats_data[cell_type] = {
                        'mean_activity': np.mean(cell_signals),
                        'std_activity': np.std(cell_signals),
                        'max_activity': np.max(cell_signals),
                        'min_activity': np.min(cell_signals),
                        'median_activity': np.median(cell_signals),
                        'spike_rate': spike_rate
                    }
            
            session_stats[session_name] = stats_data
        
        # Create comparison plots
        plots = []
        
        # Mean activity comparison
        if session_stats:
            cell_types = ['D1', 'D2', 'CHI']
            colors = {'D1': 'blue', 'D2': 'red', 'CHI': 'green'}    
            
            p1 = figure(x_range=list(session_stats.keys()),
                       width=800, height=300,
                       title=f"Mean {signal_type.upper()} Activity Across Sessions",
                       )
            
            for cell_type in cell_types:
                means = []
                sessions = []
                for session_name, stats in session_stats.items():
                    if cell_type in stats:
                        means.append(stats[cell_type]['mean_activity'])
                        sessions.append(session_name)
                
                if means:
                    p1.line(sessions, means, legend_label=cell_type, 
                           line_width=2, color=colors[cell_type])
                    p1.scatter(sessions, means, size=8, color=colors[cell_type])
            
            p1.xaxis.axis_label = "Sessions"
            p1.yaxis.axis_label = f"Mean {signal_type.upper()} Activity"
            p1.legend.location = "top_left"
            p1.xaxis.major_label_orientation = 45
            p1.legend.click_policy = "hide"
            plots.append(p1)
            
            # Spike rate comparison
            p2 = figure(x_range=list(session_stats.keys()),
                       width=800, height=300,
                       title="Spike Rate Across Sessions",
                       )
            
            for cell_type in cell_types:
                rates = []
                sessions = []
                for session_name, stats in session_stats.items():
                    if cell_type in stats:
                        rates.append(stats[cell_type]['spike_rate'])
                        sessions.append(session_name)
                
                if rates:
                    p2.line(sessions, rates, legend_label=cell_type, 
                           line_width=2, color=colors[cell_type])
                    p2.scatter(sessions, rates, size=8, color=colors[cell_type])
            
            p2.xaxis.axis_label = "Sessions"
            p2.yaxis.axis_label = "Spike Rate (spikes/min)"
            p2.legend.location = "top_left"
            p2.xaxis.major_label_orientation = 45
            p2.legend.click_policy = "hide"
            plots.append(p2)
        
        # Combine plots
        if plots:
            combined_plot = column(plots)
            
            if self.loaded_sessions:
                self.sessions[self.loaded_sessions[0]].output_bokeh_plot(
                    combined_plot, save_path=save_path, 
                    title="Cross-Session Activity Analysis", 
                    notebook=notebook, overwrite=overwrite, font_size=font_size
                )
        
        return session_stats
    
    def compare_movement_patterns_across_sessions(self, save_path: Optional[str] = None,
                                                notebook: bool = False, overwrite: bool = False,
                                                font_size: Optional[str] = None):
        """Compare movement patterns across sessions."""
        # Collect movement data
        movement_data = {}
        
        for session_name in self.loaded_sessions:
            session = self.sessions[session_name]
            
            data = {}
            if hasattr(session, 'velocity') and session.velocity is not None:
                data['mean_velocity'] = np.mean(session.velocity)
                data['max_velocity'] = np.max(session.velocity)
                data['velocity_std'] = np.std(session.velocity)
                
            if hasattr(session, 'movement_onset_indices'):
                data['movement_onsets'] = len(session.movement_onset_indices) if session.movement_onset_indices is not None and len(session.movement_onset_indices) > 0 else 0
                data['movement_rate'] = data['movement_onsets'] / (session.session_duration / 60)
                
            if hasattr(session, 'lick_indices'):
                data['lick_count'] = len(session.lick_indices) if session.lick_indices is not None and len(session.lick_indices) > 0 else 0
                data['lick_rate'] = data['lick_count'] / (session.session_duration / 60)
                
            movement_data[session_name] = data
        
        # Create visualization
        plots = []
        
        if movement_data:
            # Velocity comparison
            if all('mean_velocity' in data for data in movement_data.values()):
                sessions = list(movement_data.keys())
                mean_vels = [movement_data[s]['mean_velocity'] for s in sessions]
                max_vels = [movement_data[s]['max_velocity'] for s in sessions]
                
                p1 = figure(x_range=sessions, width=800, height=300,
                           title="Velocity Comparison Across Sessions",
                           toolbar_location=None)
                
                p1.vbar(x=sessions, top=mean_vels, width=0.4, 
                       color='blue', alpha=0.7, legend_label='Mean Velocity')
                p1.line(sessions, max_vels, line_width=2, color='red', 
                        legend_label='Max Velocity')
                p1.scatter(sessions, max_vels, size=8, color='red')
                
                p1.xaxis.axis_label = "Sessions"
                p1.yaxis.axis_label = "Velocity"
                p1.legend.location = "top_left"
                p1.xaxis.major_label_orientation = 45
                plots.append(p1)
            
            # Movement and lick rates
            if all('movement_rate' in data for data in movement_data.values()):
                sessions = list(movement_data.keys())
                mov_rates = [movement_data[s]['movement_rate'] for s in sessions]
                
                p2 = figure(x_range=sessions, width=800, height=300,
                           title="Movement Rate Across Sessions",
                           toolbar_location=None)
                
                p2.vbar(x=sessions, top=mov_rates, width=0.6, 
                       color='orange', alpha=0.7)
                
                p2.xaxis.axis_label = "Sessions"
                p2.yaxis.axis_label = "Movement Onsets per Minute"
                p2.xaxis.major_label_orientation = 45
                plots.append(p2)
        
        # Combine and save plots
        if plots:
            combined_plot = column(plots)
            
            if self.loaded_sessions:
                self.sessions[self.loaded_sessions[0]].output_bokeh_plot(
                    combined_plot, save_path=save_path, 
                    title="Cross-Session Movement Analysis", 
                    notebook=notebook, overwrite=overwrite, font_size=font_size
                )
        
        return movement_data
    
    def statistical_comparison_across_sessions(self, metric: str = 'mean_activity', 
                                             signal_type: str = 'zsc') -> Dict[str, Any]:
        """Perform statistical comparison across sessions."""
        if self.analyzer_type != 'CellTypeTank':
            print("Statistical comparison is only available for CellTypeTank analyzer")
            return {}
            
        # Collect data for each cell type across sessions
        cell_type_data = {'D1': [], 'D2': [], 'CHI': []}
        session_labels = {'D1': [], 'D2': [], 'CHI': []}
        
        for session_name in self.loaded_sessions:
            session = self.sessions[session_name]
            
            if not (hasattr(session, 'd1_indices') and hasattr(session, 'd2_indices') and hasattr(session, 'chi_indices')):
                continue
                
            # Get appropriate signal
            if signal_type == 'zsc' and hasattr(session, 'C_zsc'):
                signal = session.C_zsc
            elif signal_type == 'dff' and hasattr(session, 'C_raw_deltaF_over_F'):
                signal = session.C_raw_deltaF_over_F
            elif hasattr(session, 'C_raw'):
                signal = session.C_raw
            else:
                continue
                
            if signal is None:
                continue
            
            # Process each cell type
            cell_types_info = {
                'D1': session.d1_indices,
                'D2': session.d2_indices,
                'CHI': session.chi_indices
            }
                
            for cell_type, indices in cell_types_info.items():
                if indices is not None and len(indices) > 0 and cell_type in cell_type_data:
                    cell_signals = signal[indices, :]
                    
                    if metric == 'mean_activity':
                        values = np.mean(cell_signals, axis=1)
                    elif metric == 'max_activity':
                        values = np.max(cell_signals, axis=1)
                    elif metric == 'std_activity':
                        values = np.std(cell_signals, axis=1)
                    else:
                        values = np.mean(cell_signals, axis=1)
                    
                    cell_type_data[cell_type].extend(values)
                    session_labels[cell_type].extend([session_name] * len(values))
        
        # Perform statistical tests
        results = {}
        
        # Test for differences between cell types (pooled across sessions)
        cell_types = ['D1', 'D2', 'CHI']
        available_types = [ct for ct in cell_types if cell_type_data[ct]]
        
        if len(available_types) >= 2:
            # Kruskal-Wallis test (non-parametric ANOVA)
            test_data = [cell_type_data[ct] for ct in available_types]
            try:
                h_stat, p_value = kruskal(*test_data)
                results['cell_type_comparison'] = {
                    'test': 'Kruskal-Wallis',
                    'statistic': h_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05,
                    'cell_types': available_types
                }
            except Exception as e:
                results['cell_type_comparison'] = {'error': str(e)}
        
        # Test for differences between sessions (for each cell type)
        for cell_type in available_types:
            if len(set(session_labels[cell_type])) >= 2:
                # Group data by session
                session_groups = {}
                for value, session in zip(cell_type_data[cell_type], session_labels[cell_type]):
                    if session not in session_groups:
                        session_groups[session] = []
                    session_groups[session].append(value)
                
                # Perform Kruskal-Wallis test across sessions
                try:
                    test_data = list(session_groups.values())
                    h_stat, p_value = kruskal(*test_data)
                    results[f'{cell_type}_session_comparison'] = {
                        'test': 'Kruskal-Wallis',
                        'statistic': h_stat,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'sessions': list(session_groups.keys()),
                        'group_sizes': [len(group) for group in test_data]
                    }
                except Exception as e:
                    results[f'{cell_type}_session_comparison'] = {'error': str(e)}
        
        return results
    
    def generate_comprehensive_report(self, save_dir: Optional[str] = None,     
                                    notebook: bool = False, overwrite: bool = False,
                                    font_size: Optional[str] = None) -> str:
        """Generate a comprehensive analysis report for all sessions."""
        if save_dir is None:
            save_dir = "multi_session_analysis"
        
        os.makedirs(save_dir, exist_ok=True)
        
        report_content = []
        report_content.append("# Multi-Session Neural Data Analysis Report\n")
        report_content.append(f"Analyzer Type: {self.analyzer_type}\n")
        report_content.append(f"Number of Sessions: {len(self.loaded_sessions)}\n")
        report_content.append(f"Sessions: {', '.join(self.loaded_sessions)}\n\n")
        
        # Session information
        session_info = self.get_session_info()
        report_content.append("## Session Information\n")
        report_content.append(session_info.to_string(index=False))
        report_content.append("\n\n")
        
        # Generate analyses and plots
        analyses_performed = []
        
        # 1. Neuron count comparison
        if self.analyzer_type == 'CellTypeTank':
            try:
                neuron_df = self.compare_neuron_counts_across_sessions(
                    save_path=os.path.join(save_dir, "neuron_counts_comparison.html"),
                    notebook=notebook, overwrite=overwrite, font_size=font_size
                )
                if neuron_df is not None:
                    analyses_performed.append("Neuron count comparison")
                    report_content.append("## Neuron Count Comparison\n")
                    report_content.append(neuron_df.to_string(index=False))
                    report_content.append("\n\n")
            except Exception as e:
                report_content.append(f"Error in neuron count comparison: {str(e)}\n\n")
        
        # 2. Activity patterns analysis
        if self.analyzer_type == 'CellTypeTank':
            try:
                activity_stats = self.analyze_cross_session_activity_patterns(
                    save_path=os.path.join(save_dir, "activity_patterns_comparison.html"),
                    notebook=notebook, overwrite=overwrite, font_size=font_size
                )
                if activity_stats:
                    analyses_performed.append("Activity patterns analysis")
                    report_content.append("## Activity Patterns Analysis\n")
                    for session, stats in activity_stats.items():
                        report_content.append(f"### {session}\n")
                        for cell_type, data in stats.items():
                            report_content.append(f"**{cell_type}**: Mean activity = {data['mean_activity']:.4f}, "
                                                f"Spike rate = {data['spike_rate']:.2f} spikes/min\n")
                        report_content.append("\n")
            except Exception as e:
                report_content.append(f"Error in activity patterns analysis: {str(e)}\n\n")
        
        # 3. Movement patterns analysis
        try:
            movement_data = self.compare_movement_patterns_across_sessions(
                save_path=os.path.join(save_dir, "movement_patterns_comparison.html"),
                notebook=notebook, overwrite=overwrite, font_size=font_size
            )
            if movement_data:
                analyses_performed.append("Movement patterns analysis")
                report_content.append("## Movement Patterns Analysis\n")
                for session, data in movement_data.items():
                    report_content.append(f"**{session}**: ")
                    if 'mean_velocity' in data:
                        report_content.append(f"Mean velocity = {data['mean_velocity']:.2f}, ")
                    if 'movement_rate' in data:
                        report_content.append(f"Movement rate = {data['movement_rate']:.2f} onsets/min")
                    report_content.append("\n")
                report_content.append("\n")
        except Exception as e:
            report_content.append(f"Error in movement patterns analysis: {str(e)}\n\n")
        
        # 4. Statistical comparison  
        if self.analyzer_type == 'CellTypeTank':
            try:
                stat_results = self.statistical_comparison_across_sessions()
                if stat_results:
                    analyses_performed.append("Statistical comparison")
                    report_content.append("## Statistical Analysis\n")
                    for test_name, result in stat_results.items():
                        if 'error' not in result:
                            report_content.append(f"**{test_name}**: {result['test']} test, "
                                                f"p-value = {result['p_value']:.6f}, "
                                                f"{'Significant' if result['significant'] else 'Not significant'}\n")
                        else:
                            report_content.append(f"**{test_name}**: Error - {result['error']}\n")
                    report_content.append("\n")
            except Exception as e:
                report_content.append(f"Error in statistical analysis: {str(e)}\n\n")
        
        # Summary
        report_content.append("## Summary\n")
        report_content.append(f"Successfully performed {len(analyses_performed)} analyses:\n")
        for analysis in analyses_performed:
            report_content.append(f"- {analysis}\n")
        
        # Save report
        report_path = os.path.join(save_dir, "analysis_report.md")
        with open(report_path, 'w') as f:
            f.write(''.join(report_content))
        
        print(f"Comprehensive analysis report saved to: {report_path}")
        print(f"Analysis plots saved to: {save_dir}")
        
        return report_path
    
    def get_session(self, session_name: str):
        """Get a specific session analyzer."""
        return self.sessions.get(session_name, None)
    
    def get_all_sessions(self) -> Dict[str, Any]:
        """Get all session analyzers."""
        return self.sessions
    
    def add_session(self, session_config: Dict[str, Any], session_name: Optional[str] = None):
        """Add a new session to the analyzer."""
        if session_name is None:
            session_name = session_config.get('session_name', f'Session_{len(self.sessions)+1}')
        
        analyzer_class = self._get_analyzer_class()
        
        try:
            analyzer = analyzer_class(**session_config)
            self.sessions[session_name] = analyzer
            if session_name not in self.loaded_sessions:
                self.loaded_sessions.append(session_name)
            if session_name not in self.session_names:
                self.session_names.append(session_name)
            print(f"Successfully added {session_name}")
        except Exception as e:
            print(f"Failed to add {session_name}: {str(e)}")
    
    def remove_session(self, session_name: str):
        """Remove a session from the analyzer."""
        if session_name in self.sessions:
            del self.sessions[session_name]
            if session_name in self.loaded_sessions:
                self.loaded_sessions.remove(session_name)
            if session_name in self.session_names:
                self.session_names.remove(session_name)
            print(f"Successfully removed {session_name}")
        else:
            print(f"Session {session_name} not found") 


if __name__ == "__main__":
    sessions_config = [
        {'session_name': '391_06012025'},{'session_name': '391_05152025'}, {'session_name': '391_05062025'}, {'session_name': '392_05082025'}
    ]

    analyzer = MultiSessionAnalyzer(sessions_config, analyzer_type='CellTypeTank')

    neuron_counts_df = analyzer.compare_neuron_counts_across_sessions()