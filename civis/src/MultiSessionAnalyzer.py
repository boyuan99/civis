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
    
    def _merge_session_data(self) -> Dict[str, Any]:
        """
        Merge data from all sessions into a single combined dataset.
        This allows treating all neurons from all sessions as one large population.
        
        Returns:
        --------
        dict
            Dictionary containing merged data from all sessions
        """
        if self.analyzer_type != 'CellTypeTank':
            print("Data merging is currently only supported for CellTypeTank analyzer")
            return {}
            
        if not self.loaded_sessions:
            print("No sessions loaded for merging")
            return {}
        
        merged_data = {
            'sessions_included': self.loaded_sessions.copy(),
            'total_sessions': len(self.loaded_sessions),
            'session_boundaries': {},  # Track which neurons/timepoints belong to which session
            'combined_signals': {},
            'combined_peak_indices': {'d1': [], 'd2': [], 'chi': []},
            'combined_cell_indices': {'d1': [], 'd2': [], 'chi': []},
            'combined_rising_edges': {'d1': [], 'd2': [], 'chi': []},
            'session_offsets': {},  # Time offsets for each session
            'metadata': {'total_neurons': 0, 'total_timepoints': 0},
            'velocity_data': None
        }
        
        current_neuron_offset = 0
        current_time_offset = 0
        all_sessions_signals = []
        
        # Collect and merge data from all sessions
        for session_idx, session_name in enumerate(self.loaded_sessions):
            session = self.sessions[session_name]
            
            # Store session boundaries and offsets
            merged_data['session_boundaries'][session_name] = {
                'neuron_start': current_neuron_offset,
                'time_start': current_time_offset
            }
            merged_data['session_offsets'][session_name] = {
                'neuron_offset': current_neuron_offset,
                'time_offset': current_time_offset
            }
            
            # Get signals (prefer zsc, then denoised, then raw)
            signals = None
            signal_type = None
            if hasattr(session, 'C_zsc') and session.C_zsc is not None:
                signals = session.C_zsc
                signal_type = 'zsc'
            elif hasattr(session, 'C_denoised') and session.C_denoised is not None:
                signals = session.C_denoised
                signal_type = 'denoised'
            elif hasattr(session, 'C_raw') and session.C_raw is not None:
                signals = session.C_raw
                signal_type = 'raw'
            
            if signals is None:
                print(f"Warning: No signals found for session {session_name}")
                continue
            
            all_sessions_signals.append(signals)
            
            # Initialize combined signals if first session
            if 'signal_type' not in merged_data:
                merged_data['signal_type'] = signal_type
            
            # Collect velocity data from first session with available data
            if merged_data['velocity_data'] is None and hasattr(session, 'smoothed_velocity') and session.smoothed_velocity is not None:
                merged_data['velocity_data'] = session.smoothed_velocity
            
            # Process cell type indices and peak indices
            cell_types = ['d1', 'd2', 'chi']
            for cell_type in cell_types:
                indices_attr = f'{cell_type}_indices'
                peaks_attr = f'{cell_type}_peak_indices'
                edges_attr = f'{cell_type}_rising_edges_starts'
                
                if hasattr(session, indices_attr):
                    cell_indices = getattr(session, indices_attr)
                    if cell_indices is not None and len(cell_indices) > 0:
                        # Adjust cell indices by neuron offset
                        adjusted_indices = [idx + current_neuron_offset for idx in cell_indices]
                        merged_data['combined_cell_indices'][cell_type].extend(adjusted_indices)
                        
                        # Process peak indices
                        if hasattr(session, peaks_attr):
                            peak_indices = getattr(session, peaks_attr)
                            if peak_indices is not None:
                                for neuron_idx, neuron_peaks in enumerate(peak_indices):
                                    if neuron_peaks is not None and len(neuron_peaks) > 0:
                                        # Keep original peak times (don't adjust for time offset in merged analysis)
                                        merged_data['combined_peak_indices'][cell_type].append(neuron_peaks)
                                    else:
                                        merged_data['combined_peak_indices'][cell_type].append([])
                        
                        # Process rising edges
                        if hasattr(session, edges_attr):
                            rising_edges = getattr(session, edges_attr)
                            if rising_edges is not None:
                                for neuron_idx, neuron_edges in enumerate(rising_edges):
                                    if neuron_edges is not None and len(neuron_edges) > 0:
                                        # Keep original edge times
                                        merged_data['combined_rising_edges'][cell_type].append(neuron_edges)
                                    else:
                                        merged_data['combined_rising_edges'][cell_type].append([])
            
            # Update session boundaries end points
            n_neurons_this_session = signals.shape[0]
            n_timepoints_this_session = signals.shape[1]
            
            merged_data['session_boundaries'][session_name].update({
                'neuron_end': current_neuron_offset + n_neurons_this_session,
                'time_end': current_time_offset + n_timepoints_this_session,
                'n_neurons': n_neurons_this_session,
                'n_timepoints': n_timepoints_this_session
            })
            
            # Update offsets for next session
            current_neuron_offset += n_neurons_this_session
        
        # Combine all session signals along neuron axis
        if all_sessions_signals:
            merged_data['combined_signals'] = np.concatenate(all_sessions_signals, axis=0)
        
        # Update metadata
        if merged_data['combined_signals'] is not None:
            merged_data['metadata'] = {
                'total_neurons': merged_data['combined_signals'].shape[0],
                'total_timepoints': merged_data['combined_signals'].shape[1],
                'signal_type': merged_data.get('signal_type', 'unknown'),
                'd1_neurons': len(merged_data['combined_cell_indices']['d1']),
                'd2_neurons': len(merged_data['combined_cell_indices']['d2']),
                'chi_neurons': len(merged_data['combined_cell_indices']['chi'])
            }
        
        return merged_data
    
    def _create_merged_analyzer(self, merged_data):
        """
        Create a properly initialized CellTypeTank instance with merged data.
        
        Parameters:
        -----------
        merged_data : dict
            Dictionary containing merged session data
            
        Returns:
        --------
        CellTypeTank
            Initialized analyzer with merged data
        """
        if not self.loaded_sessions:
            return None
        
        # Use first session as template
        template_session = self.sessions[self.loaded_sessions[0]]
        
        # Create new instance properly
        temp_analyzer = CellTypeTank.__new__(CellTypeTank)
        
        # Copy essential attributes from template
        essential_attrs = [
            'session_name', 'ci_rate', 'vm_rate', 'session_duration', 'config',
            'output_bokeh_plot', 'maze_type'
        ]
        
        for attr in essential_attrs:
            if hasattr(template_session, attr):
                setattr(temp_analyzer, attr, getattr(template_session, attr))
        
        # Set merged session name
        temp_analyzer.session_name = f"Merged_{len(self.loaded_sessions)}_Sessions"
        
        # Set signal data based on signal type
        signal_type = merged_data['signal_type']
        signals = merged_data['combined_signals']
        
        # Set all signal types (initialize with merged data)
        if signal_type == 'zsc':
            temp_analyzer.C_zsc = signals
            temp_analyzer.C_raw = signals  # Fallback
            temp_analyzer.C_denoised = signals  # Fallback
        elif signal_type == 'denoised':
            temp_analyzer.C_denoised = signals
            temp_analyzer.C_raw = signals  # Fallback
            temp_analyzer.C_zsc = signals  # Fallback
        else:
            temp_analyzer.C_raw = signals
            temp_analyzer.C_denoised = signals  # Fallback
            temp_analyzer.C_zsc = signals  # Fallback
        
        # Set additional signal attributes
        temp_analyzer.C_deconvolved = signals
        temp_analyzer.C_baseline = signals
        temp_analyzer.C_reraw = signals
        
        # Set cell type indices
        temp_analyzer.d1_indices = np.array(merged_data['combined_cell_indices']['d1'])
        temp_analyzer.d2_indices = np.array(merged_data['combined_cell_indices']['d2'])
        temp_analyzer.chi_indices = np.array(merged_data['combined_cell_indices']['chi'])
        
        # Set peak indices
        temp_analyzer.d1_peak_indices = merged_data['combined_peak_indices']['d1']
        temp_analyzer.d2_peak_indices = merged_data['combined_peak_indices']['d2']
        temp_analyzer.chi_peak_indices = merged_data['combined_peak_indices']['chi']
        
        # Set rising edges
        temp_analyzer.d1_rising_edges_starts = merged_data['combined_rising_edges']['d1']
        temp_analyzer.d2_rising_edges_starts = merged_data['combined_rising_edges']['d2']
        temp_analyzer.chi_rising_edges_starts = merged_data['combined_rising_edges']['chi']
        
        # Set cell-type-specific signals
        if len(temp_analyzer.d1_indices) > 0:
            temp_analyzer.d1_raw = signals[temp_analyzer.d1_indices]
            temp_analyzer.d1_denoised = signals[temp_analyzer.d1_indices]
            temp_analyzer.d1_zsc = signals[temp_analyzer.d1_indices]
            temp_analyzer.d1_deconvolved = signals[temp_analyzer.d1_indices]
            temp_analyzer.d1_baseline = signals[temp_analyzer.d1_indices]
            temp_analyzer.d1_reraw = signals[temp_analyzer.d1_indices]
        
        if len(temp_analyzer.d2_indices) > 0:
            temp_analyzer.d2_raw = signals[temp_analyzer.d2_indices]
            temp_analyzer.d2_denoised = signals[temp_analyzer.d2_indices]
            temp_analyzer.d2_zsc = signals[temp_analyzer.d2_indices]
            temp_analyzer.d2_deconvolved = signals[temp_analyzer.d2_indices]
            temp_analyzer.d2_baseline = signals[temp_analyzer.d2_indices]
            temp_analyzer.d2_reraw = signals[temp_analyzer.d2_indices]
        
        if len(temp_analyzer.chi_indices) > 0:
            temp_analyzer.chi_raw = signals[temp_analyzer.chi_indices]
            temp_analyzer.chi_denoised = signals[temp_analyzer.chi_indices]
            temp_analyzer.chi_zsc = signals[temp_analyzer.chi_indices]
            temp_analyzer.chi_deconvolved = signals[temp_analyzer.chi_indices]
            temp_analyzer.chi_baseline = signals[temp_analyzer.chi_indices]
            temp_analyzer.chi_reraw = signals[temp_analyzer.chi_indices]
        
        # Set velocity data if available
        if merged_data['velocity_data'] is not None:
            temp_analyzer.smoothed_velocity = merged_data['velocity_data']
        else:
            # Create dummy velocity data if needed
            temp_analyzer.smoothed_velocity = np.zeros(signals.shape[1])
        
        # Set other required attributes with defaults
        temp_analyzer.A = np.zeros((signals.shape[0], 100, 100))  # Dummy spatial components
        temp_analyzer.centroids = np.zeros((signals.shape[0], 2))  # Dummy centroids
        temp_analyzer.Coor = [np.array([0, 0]) for _ in range(signals.shape[0])]  # Dummy coordinates
        
        # Set peak indices attribute used by some methods
        temp_analyzer.peak_indices = [[] for _ in range(signals.shape[0])]
        temp_analyzer.rising_edges_starts = [[] for _ in range(signals.shape[0])]
        
        # Fill peak indices with proper data
        neuron_idx = 0
        for cell_type in ['d1', 'd2', 'chi']:
            indices = getattr(temp_analyzer, f'{cell_type}_indices')
            peaks = getattr(temp_analyzer, f'{cell_type}_peak_indices')
            edges = getattr(temp_analyzer, f'{cell_type}_rising_edges_starts')
            
            for i, global_idx in enumerate(indices):
                if i < len(peaks):
                    temp_analyzer.peak_indices[global_idx] = peaks[i]
                if i < len(edges):
                    temp_analyzer.rising_edges_starts[global_idx] = edges[i]
        
        return temp_analyzer
    
    def merged_neuron_centered_analysis(self, center_cell_type: str = 'D1', 
                                       signal_type: str = 'auto', activity_window: float = 1.0,
                                       save_path: Optional[str] = None, title: Optional[str] = None,
                                       notebook: bool = False, overwrite: bool = False, 
                                       font_size: Optional[str] = None):
        """
        Perform neuron-centered analysis on merged data from all sessions.
        This treats all neurons from all sessions as one large population.
        
        Parameters:
        -----------
        center_cell_type : str
            The cell type to center analysis on ('D1', 'D2', or 'CHI')
        signal_type : str
            Signal type to use ('auto', 'zsc', 'denoised', 'raw')
        activity_window : float
            Time window in seconds to look for activities around peaks
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
        
        Returns:
        --------
        tuple
            (bokeh.plotting.figure, dict) - The created figure and summary statistics
        """
        if self.analyzer_type != 'CellTypeTank':
            print("Merged neuron-centered analysis is only available for CellTypeTank analyzer")
            return None, {}
        
        # Get merged data
        merged_data = self._merge_session_data()
        if not merged_data or 'combined_signals' not in merged_data:
            print("Failed to merge session data")
            return None, {}
        
        # Create properly initialized analyzer with merged data
        temp_analyzer = self._create_merged_analyzer(merged_data)
        if temp_analyzer is None:
            print("Failed to create merged analyzer")
            return None, {}
        
        # Determine signal type to use
        signal_type_to_use = merged_data['signal_type'] if signal_type == 'auto' else signal_type
        
        # Set default title
        if title is None:
            title = f"Merged Neuron-Centered Analysis: {center_cell_type} from {len(self.loaded_sessions)} Sessions"
        
        # Perform the analysis
        try:
            plot, stats = temp_analyzer.plot_neuron_centered_activity(
                center_cell_type=center_cell_type,
                signal_type=signal_type_to_use,
                activity_window=activity_window,
                save_path=save_path,
                title=title,
                notebook=notebook,
                overwrite=overwrite,
                font_size=font_size
            )
            
            # Add merger information to stats
            stats['merger_info'] = {
                'sessions_merged': merged_data['sessions_included'],
                'total_sessions': merged_data['total_sessions'],
                'total_neurons_merged': merged_data['metadata']['total_neurons'],
                'signal_type_used': signal_type_to_use
            }
            
            print(f"Merged analysis completed with {len(merged_data['sessions_included'])} sessions")
            print(f"Total neurons analyzed: {merged_data['metadata']['total_neurons']}")
            
            return plot, stats
            
        except Exception as e:
            print(f"Error in merged neuron-centered analysis: {str(e)}")
            return None, {}
    
    def merged_population_analysis(self, center_cell_type: str = 'D1', 
                                  signal_type: str = 'auto', time_window: float = 3.0,
                                  save_path: Optional[str] = None, title: Optional[str] = None,
                                  notebook: bool = False, overwrite: bool = False, 
                                  font_size: Optional[str] = None,
                                  baseline_correct: bool = False):
        """
        Perform population-centered analysis on merged data from all sessions.
        This averages ALL neurons of each type from all sessions.
        
        Parameters:
        -----------
        center_cell_type : str
            The cell type to center analysis on ('D1', 'D2', or 'CHI')
        signal_type : str
            Signal type to use ('auto', 'zsc', 'denoised', 'raw')
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
        
        Returns:
        --------
        tuple
            (bokeh.plotting.figure, dict) - The created figure and summary statistics
        """
        if self.analyzer_type != 'CellTypeTank':
            print("Merged population analysis is only available for CellTypeTank analyzer")
            return None, {}
        
        # Get merged data
        merged_data = self._merge_session_data()
        if not merged_data or 'combined_signals' not in merged_data:
            print("Failed to merge session data")
            return None, {}
        
        # Create properly initialized analyzer with merged data
        temp_analyzer = self._create_merged_analyzer(merged_data)
        if temp_analyzer is None:
            print("Failed to create merged analyzer")
            return None, {}
        
        # Determine signal type to use
        signal_type_to_use = merged_data['signal_type'] if signal_type == 'auto' else signal_type
        
        # Set default title
        if title is None:
            title = f"Merged Population Analysis: {center_cell_type} from {len(self.loaded_sessions)} Sessions"
        
        # Perform the analysis
        try:
            plot, stats = temp_analyzer.plot_population_centered_activity(
                center_cell_type=center_cell_type,
                signal_type=signal_type_to_use,
                time_window=time_window,
                save_path=save_path,
                title=title,
                notebook=notebook,
                overwrite=overwrite,
                font_size=font_size,
                baseline_correct=baseline_correct
            )
            
            # Add merger information to stats
            stats['merger_info'] = {
                'sessions_merged': merged_data['sessions_included'],
                'total_sessions': merged_data['total_sessions'],
                'total_neurons_merged': merged_data['metadata']['total_neurons'],
                'signal_type_used': signal_type_to_use
            }
            
            print(f"Merged population analysis completed with {len(merged_data['sessions_included'])} sessions")
            print(f"Total neurons analyzed: {merged_data['metadata']['total_neurons']}")
            
            return plot, stats
            
        except Exception as e:
            print(f"Error in merged population analysis: {str(e)}")
            return None, {}
    
    def merged_all_neuron_centered_analyses(self, signal_type: str = 'auto', activity_window: float = 1.0,
                                           save_path: Optional[str] = None, notebook: bool = False, 
                                           overwrite: bool = False, font_size: Optional[str] = None):
        """
        Perform all neuron-centered analyses (D1, D2, CHI) on merged data from all sessions.
        
        Parameters:
        -----------
        signal_type : str
            Signal type to use ('auto', 'zsc', 'denoised', 'raw')
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
        if self.analyzer_type != 'CellTypeTank':
            print("Merged neuron-centered analyses are only available for CellTypeTank analyzer")
            return {}
        
        # Get merged data
        merged_data = self._merge_session_data()
        if not merged_data or 'combined_signals' not in merged_data:
            print("Failed to merge session data")
            return {}
        
        # Create properly initialized analyzer with merged data
        temp_analyzer = self._create_merged_analyzer(merged_data)
        if temp_analyzer is None:
            print("Failed to create merged analyzer")
            return {}
        
        # Determine signal type to use
        signal_type_to_use = merged_data['signal_type'] if signal_type == 'auto' else signal_type
        
        # Perform the analysis
        try:
            results = temp_analyzer.plot_all_neuron_centered_analyses(
                signal_type=signal_type_to_use,
                activity_window=activity_window,
                save_path=save_path,
                notebook=notebook,
                overwrite=overwrite,
                font_size=font_size
            )
            
            # Add merger information to all results
            merger_info = {
                'sessions_merged': merged_data['sessions_included'],
                'total_sessions': merged_data['total_sessions'],
                'total_neurons_merged': merged_data['metadata']['total_neurons'],
                'signal_type_used': signal_type_to_use
            }
            
            for key in results:
                if isinstance(results[key], dict) and 'stats' in results[key]:
                    results[key]['stats']['merger_info'] = merger_info
            
            print(f"Merged all neuron-centered analyses completed with {len(merged_data['sessions_included'])} sessions")
            print(f"Total neurons analyzed: {merged_data['metadata']['total_neurons']}")
            
            return results
            
        except Exception as e:
            print(f"Error in merged all neuron-centered analyses: {str(e)}")
            return {}
    
    def merged_spike_velocity_analysis(self, d1_peak_indices=None, d2_peak_indices=None, 
                                      chi_peak_indices=None, save_path: Optional[str] = None,
                                      title: Optional[str] = None, notebook: bool = False, 
                                      overwrite: bool = False, font_size: Optional[str] = None):
        """
        Perform spike-velocity analysis on merged data from all sessions.
        
        Parameters:
        -----------
        d1_peak_indices : list, optional
            D1 peak indices. If None, uses merged peak indices
        d2_peak_indices : list, optional
            D2 peak indices. If None, uses merged peak indices
        chi_peak_indices : list, optional
            CHI peak indices. If None, uses merged peak indices
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
        
        Returns:
        --------
        list
            List of plots created
        """
        if self.analyzer_type != 'CellTypeTank':
            print("Merged spike-velocity analysis is only available for CellTypeTank analyzer")
            return []
        
        # Get merged data
        merged_data = self._merge_session_data()
        if not merged_data or 'combined_signals' not in merged_data:
            print("Failed to merge session data")
            return []
        
        # Create properly initialized analyzer with merged data
        temp_analyzer = self._create_merged_analyzer(merged_data)
        if temp_analyzer is None:
            print("Failed to create merged analyzer")
            return []
        
        # Use provided or merged peak indices
        if d1_peak_indices is None:
            d1_peak_indices = merged_data['combined_peak_indices']['d1']
        if d2_peak_indices is None:
            d2_peak_indices = merged_data['combined_peak_indices']['d2']
        if chi_peak_indices is None:
            chi_peak_indices = merged_data['combined_peak_indices']['chi']
        
        # Set default title
        if title is None:
            title = f"Merged Spike-Velocity Analysis from {len(self.loaded_sessions)} Sessions"
        
        # Perform the analysis
        try:
            plots = temp_analyzer.compare_neuron_peaks_with_velocity(
                d1_peak_indices=d1_peak_indices,
                d2_peak_indices=d2_peak_indices,
                chi_peak_indices=chi_peak_indices,
                save_path=save_path,
                title=title,
                notebook=notebook,
                overwrite=overwrite,
                font_size=font_size
            )
            
            print(f"Merged spike-velocity analysis completed with {len(merged_data['sessions_included'])} sessions")
            return plots
            
        except Exception as e:
            print(f"Error in merged spike-velocity analysis: {str(e)}")
            return []
    
    def merged_connectivity_analysis(self, save_path: Optional[str] = None, 
                                   notebook: bool = False, overwrite: bool = False, 
                                   font_size: Optional[str] = None):
        """
        Perform connectivity analysis on merged data from all sessions.
        
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
        dict
            Dictionary containing connectivity analysis results
        """
        if self.analyzer_type != 'CellTypeTank':
            print("Merged connectivity analysis is only available for CellTypeTank analyzer")
            return {}
        
        # Get merged data
        merged_data = self._merge_session_data()
        if not merged_data or 'combined_signals' not in merged_data:
            print("Failed to merge session data")
            return {}
        
        # Create properly initialized analyzer with merged data
        temp_analyzer = self._create_merged_analyzer(merged_data)
        if temp_analyzer is None:
            print("Failed to create merged analyzer")
            return {}
        
        # Perform the analysis
        try:
            # Create connectivity analyzer
            connectivity_analyzer = temp_analyzer.create_connectivity_analyzer()
            
            # Analyze connectivity
            results = connectivity_analyzer.analyze_and_plot_connectivity(
                timing_save_path=save_path,
                summary_save_path=save_path.replace('.html', '_summary.html') if save_path else None,
                notebook=notebook,
                overwrite=overwrite,
                font_size=font_size
            )
            
            # Add merger information
            results['merger_info'] = {
                'sessions_merged': merged_data['sessions_included'],
                'total_sessions': merged_data['total_sessions'],
                'total_neurons_merged': merged_data['metadata']['total_neurons']
            }
            
            print(f"Merged connectivity analysis completed with {len(merged_data['sessions_included'])} sessions")
            return results
            
        except Exception as e:
            print(f"Error in merged connectivity analysis: {str(e)}")
            return {}
    
    def compare_connectivity_across_sessions(self, save_path: Optional[str] = None, 
                                           notebook: bool = False, overwrite: bool = False, 
                                           font_size: Optional[str] = None):
        """
        Compare connectivity patterns across individual sessions.
        
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
        dict
            Dictionary containing connectivity comparison results
        """
        if self.analyzer_type != 'CellTypeTank':
            print("Connectivity comparison is only available for CellTypeTank analyzer")
            return {}
        
        # Analyze connectivity for each session
        session_connectivity = {}
        
        for session_name in self.loaded_sessions:
            session = self.sessions[session_name]
            
            try:
                # Create connectivity analyzer for this session
                connectivity_analyzer = session.create_connectivity_analyzer()
                results = connectivity_analyzer.analyze_and_plot_connectivity(
                    save_path=None,  # Don't save individual plots
                    notebook=False,
                    overwrite=overwrite,
                    create_plots=False  # Only get data, no plots
                )
                
                session_connectivity[session_name] = results
                
            except Exception as e:
                print(f"Failed connectivity analysis for {session_name}: {str(e)}")
                session_connectivity[session_name] = {'error': str(e)}
        
        # Compare results
        comparison_results = {
            'individual_sessions': session_connectivity,
            'comparison_summary': {}
        }
        
        # Extract key metrics for comparison
        valid_sessions = {k: v for k, v in session_connectivity.items() if 'error' not in v}
        
        if len(valid_sessions) >= 2:
            # Compare peak timing relationships
            timing_comparison = {}
            for session_name, results in valid_sessions.items():
                if 'peak_timing_relationships' in results:
                    timing_data = results['peak_timing_relationships']
                    timing_comparison[session_name] = {
                        'total_peaks_analyzed': timing_data.get('total_peaks_analyzed', 0),
                        'sequential_patterns': timing_data.get('sequential_activation_patterns', {}),
                        'interval_stats': timing_data.get('interval_statistics', {})
                    }
            
            comparison_results['comparison_summary']['peak_timing'] = timing_comparison
            
            print(f"Connectivity comparison completed for {len(valid_sessions)} sessions")
        else:
            print("Not enough valid sessions for connectivity comparison")
        
        return comparison_results
    
    def compare_spike_patterns_across_sessions(self, signal_type: str = 'zsc',
                                             save_path: Optional[str] = None, 
                                             notebook: bool = False, overwrite: bool = False, 
                                             font_size: Optional[str] = None):
        """
        Compare spike patterns across sessions.
        
        Parameters:
        -----------
        signal_type : str
            Signal type to use ('zsc', 'denoised', 'raw')
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
        dict
            Dictionary containing spike pattern comparison results
        """
        if self.analyzer_type != 'CellTypeTank':
            print("Spike pattern comparison is only available for CellTypeTank analyzer")
            return {}
        
        # Collect spike statistics for each session
        spike_patterns = {}
        
        for session_name in self.loaded_sessions:
            session = self.sessions[session_name]
            
            patterns = {
                'session_name': session_name,
                'cell_counts': {},
                'spike_rates': {},
                'peak_statistics': {}
            }
            
            # Get cell counts
            if hasattr(session, 'd1_indices'):
                patterns['cell_counts']['d1'] = len(session.d1_indices) if session.d1_indices else 0
            if hasattr(session, 'd2_indices'):
                patterns['cell_counts']['d2'] = len(session.d2_indices) if session.d2_indices else 0
            if hasattr(session, 'chi_indices'):
                patterns['cell_counts']['chi'] = len(session.chi_indices) if session.chi_indices else 0
            
            # Calculate spike rates
            duration_min = session.session_duration / 60 if hasattr(session, 'session_duration') else 30
            
            for cell_type in ['d1', 'd2', 'chi']:
                peak_attr = f'{cell_type}_peak_indices'
                if hasattr(session, peak_attr):
                    peak_indices = getattr(session, peak_attr)
                    if peak_indices is not None and len(peak_indices) > 0:
                        total_spikes = sum(len(peaks) for peaks in peak_indices)
                        patterns['spike_rates'][cell_type] = total_spikes / duration_min
                    else:
                        patterns['spike_rates'][cell_type] = 0
            
            spike_patterns[session_name] = patterns
        
        # Create comparison visualization
        from bokeh.plotting import figure
        from bokeh.layouts import column, row
        from bokeh.models import ColumnDataSource
        
        plots = []
        
        # Spike rate comparison
        if spike_patterns:
            sessions = list(spike_patterns.keys())
            cell_types = ['d1', 'd2', 'chi']
            colors = {'d1': 'blue', 'd2': 'red', 'chi': 'green'}
            
            p1 = figure(x_range=sessions, width=800, height=400,
                       title="Spike Rates Across Sessions",
                       x_axis_label="Sessions", y_axis_label="Spikes per minute")
            
            for cell_type in cell_types:
                rates = [spike_patterns[session][f'spike_rates'].get(cell_type, 0) for session in sessions]
                p1.line(sessions, rates, legend_label=cell_type.upper(), 
                       line_width=2, color=colors[cell_type])
                p1.scatter(sessions, rates, size=8, color=colors[cell_type])
            
            p1.legend.location = "top_left"
            p1.xaxis.major_label_orientation = 45
            plots.append(p1)
            
            # Cell count comparison
            p2 = figure(x_range=sessions, width=800, height=400,
                       title="Cell Counts Across Sessions",
                       x_axis_label="Sessions", y_axis_label="Number of Cells")
            
            for cell_type in cell_types:
                counts = [spike_patterns[session]['cell_counts'].get(cell_type, 0) for session in sessions]
                p2.vbar(x=sessions, top=counts, width=0.2, 
                       color=colors[cell_type], alpha=0.7, legend_label=cell_type.upper())
            
            p2.legend.location = "top_left"
            p2.xaxis.major_label_orientation = 45
            plots.append(p2)
        
        # Save/show plots
        if plots and self.loaded_sessions:
            combined_plot = column(plots)
            self.sessions[self.loaded_sessions[0]].output_bokeh_plot(
                combined_plot, save_path=save_path, 
                title="Cross-Session Spike Pattern Comparison", 
                notebook=notebook, overwrite=overwrite, font_size=font_size
            )
        
        return {
            'spike_patterns': spike_patterns,
            'plots': plots if plots else None
        }
    
    def statistical_comparison_of_merged_vs_sessions(self, metric: str = 'mean_activity', 
                                                   signal_type: str = 'zsc') -> Dict[str, Any]:
        """
        Statistical comparison between merged analysis and individual sessions.
        
        Parameters:
        -----------
        metric : str
            Metric to compare ('mean_activity', 'max_activity', 'std_activity', 'spike_rate')
        signal_type : str
            Signal type to use ('zsc', 'denoised', 'raw')
        
        Returns:
        --------
        dict
            Dictionary containing statistical comparison results
        """
        if self.analyzer_type != 'CellTypeTank':
            print("Statistical comparison is only available for CellTypeTank analyzer")
            return {}
        
        # Get data from individual sessions
        individual_session_data = {'d1': [], 'd2': [], 'chi': []}
        session_labels = {'d1': [], 'd2': [], 'chi': []}
        
        for session_name in self.loaded_sessions:
            session = self.sessions[session_name]
            
            if not (hasattr(session, 'd1_indices') and hasattr(session, 'd2_indices') and hasattr(session, 'chi_indices')):
                continue
            
            # Get appropriate signal
            if signal_type == 'zsc' and hasattr(session, 'C_zsc'):
                signal = session.C_zsc
            elif signal_type == 'denoised' and hasattr(session, 'C_denoised'):
                signal = session.C_denoised
            elif hasattr(session, 'C_raw'):
                signal = session.C_raw
            else:
                continue
            
            if signal is None:
                continue
            
            # Process each cell type
            cell_types_info = {
                'd1': session.d1_indices,
                'd2': session.d2_indices,
                'chi': session.chi_indices
            }
            
            for cell_type, indices in cell_types_info.items():
                if indices is not None and len(indices) > 0:
                    cell_signals = signal[indices, :]
                    
                    if metric == 'mean_activity':
                        values = np.mean(cell_signals, axis=1)
                    elif metric == 'max_activity':
                        values = np.max(cell_signals, axis=1)
                    elif metric == 'std_activity':
                        values = np.std(cell_signals, axis=1)
                    elif metric == 'spike_rate':
                        # Calculate spike rate using peak indices
                        values = []
                        peak_attr = f'{cell_type}_peak_indices'
                        if hasattr(session, peak_attr):
                            peak_indices = getattr(session, peak_attr)
                            if peak_indices is not None and len(peak_indices) > 0:
                                duration_min = session.session_duration / 60
                                for peaks in peak_indices:
                                    spike_rate = len(peaks) / duration_min if len(peaks) > 0 else 0
                                    values.append(spike_rate)
                        values = np.array(values) if len(values) > 0 else np.array([0])
                    else:
                        values = np.mean(cell_signals, axis=1)
                    
                    individual_session_data[cell_type].extend(values)
                    session_labels[cell_type].extend([session_name] * len(values))
        
        # Get merged data
        merged_data = self._merge_session_data()
        if not merged_data or 'combined_signals' not in merged_data:
            print("Failed to merge session data")
            return {}
        
        merged_values = {'d1': [], 'd2': [], 'chi': []}
        
        # Calculate values for merged data
        signal = merged_data['combined_signals']
        cell_types_info = {
            'd1': merged_data['combined_cell_indices']['d1'],
            'd2': merged_data['combined_cell_indices']['d2'],
            'chi': merged_data['combined_cell_indices']['chi']
        }
        
        for cell_type, indices in cell_types_info.items():
            if indices is not None and len(indices) > 0:
                cell_signals = signal[indices, :]
                
                if metric == 'mean_activity':
                    values = np.mean(cell_signals, axis=1)
                elif metric == 'max_activity':
                    values = np.max(cell_signals, axis=1)
                elif metric == 'std_activity':
                    values = np.std(cell_signals, axis=1)
                elif metric == 'spike_rate':
                    # Calculate spike rate using merged peak indices
                    values = []
                    peak_indices = merged_data['combined_peak_indices'][cell_type]
                    if peak_indices is not None and len(peak_indices) > 0:
                        # Use average session duration
                        avg_duration_min = np.mean([
                            self.sessions[session_name].session_duration / 60 
                            for session_name in self.loaded_sessions
                        ])
                        for peaks in peak_indices:
                            spike_rate = len(peaks) / avg_duration_min if len(peaks) > 0 else 0
                            values.append(spike_rate)
                    values = np.array(values) if len(values) > 0 else np.array([0])
                else:
                    values = np.mean(cell_signals, axis=1)
                
                merged_values[cell_type] = values
            else:
                # Ensure empty array for cell types with no data
                merged_values[cell_type] = np.array([])
        
        # Perform statistical comparisons
        results = {
            'metric_analyzed': metric,
            'signal_type': signal_type,
            'individual_sessions_stats': {},
            'merged_data_stats': {},
            'comparisons': {}
        }
        
        for cell_type in ['d1', 'd2', 'chi']:
            individual_data = individual_session_data[cell_type]
            merged_data_values = merged_values[cell_type]
            
            if (len(individual_data) > 0 and 
                hasattr(merged_data_values, '__len__') and len(merged_data_values) > 0):
                # Individual sessions statistics
                results['individual_sessions_stats'][cell_type] = {
                    'mean': np.mean(individual_data),
                    'std': np.std(individual_data),
                    'n_samples': len(individual_data),
                    'median': np.median(individual_data)
                }
                
                # Merged data statistics
                results['merged_data_stats'][cell_type] = {
                    'mean': np.mean(merged_data_values),
                    'std': np.std(merged_data_values),
                    'n_samples': len(merged_data_values),
                    'median': np.median(merged_data_values)
                }
                
                # Statistical comparison
                try:
                    # Mann-Whitney U test (non-parametric)
                    statistic, p_value = mannwhitneyu(individual_data, merged_data_values, alternative='two-sided')
                    results['comparisons'][cell_type] = {
                        'test': 'Mann-Whitney U',
                        'statistic': statistic,
                        'p_value': p_value,
                        'significant': p_value < 0.05,
                        'effect_size': abs(np.mean(individual_data) - np.mean(merged_data_values)) / np.sqrt((np.var(individual_data) + np.var(merged_data_values)) / 2)
                    }
                except Exception as e:
                    results['comparisons'][cell_type] = {'error': str(e)}
        
        return results
    
    def comprehensive_merged_analysis(self, signal_type: str = 'auto', save_dir: Optional[str] = None,
                                    notebook: bool = False, overwrite: bool = False, 
                                    font_size: Optional[str] = None):
        """
        Perform comprehensive analysis on merged data from all sessions.
        This runs multiple analysis types on the combined dataset.
        
        Parameters:
        -----------
        signal_type : str
            Signal type to use ('auto', 'zsc', 'denoised', 'raw')
        save_dir : str, optional
            Directory to save all analysis outputs
        notebook : bool
            Whether to display in notebook
        overwrite : bool
            Whether to overwrite existing files
        font_size : str, optional
            Font size for plot text elements
        
        Returns:
        --------
        dict
            Dictionary containing all analysis results
        """
        if self.analyzer_type != 'CellTypeTank':
            print("Comprehensive merged analysis is only available for CellTypeTank analyzer")
            return {}
        
        if save_dir is None:
            save_dir = "comprehensive_merged_analysis"
        
        os.makedirs(save_dir, exist_ok=True)
        
        print("Starting comprehensive merged analysis...")
        print(f"Analyzing {len(self.loaded_sessions)} sessions combined")
        
        results = {
            'sessions_included': self.loaded_sessions.copy(),
            'save_directory': save_dir,
            'analyses_performed': []
        }
        
        # 1. Merged neuron-centered analyses
        print("\n1. Running merged neuron-centered analyses...")
        neuron_results = self.merged_all_neuron_centered_analyses(
            signal_type=signal_type,
            save_path=os.path.join(save_dir, "merged_neuron_centered_analyses.html"),
            notebook=notebook,
            overwrite=overwrite,
            font_size=font_size
        )
        results['neuron_centered_analyses'] = neuron_results
        results['analyses_performed'].append('Neuron-centered analyses')
        
        # 2. Merged population analyses
        print("\n2. Running merged population analyses...")
        try:
            # Run all three population analyses
            population_results = {}
            for center_type in ['D1', 'D2', 'CHI']:
                plot, stats = self.merged_population_analysis(
                    center_cell_type=center_type,
                    signal_type=signal_type,
                    save_path=os.path.join(save_dir, f"merged_population_{center_type.lower()}.html"),
                    notebook=notebook,
                    overwrite=overwrite,
                    font_size=font_size,
                    baseline_correct=True
                )
                population_results[f'{center_type.lower()}_centered'] = {'plot': plot, 'stats': stats}
            
            results['population_analyses'] = population_results
            results['analyses_performed'].append('Population analyses')
        except Exception as e:
            print(f"Error in population analyses: {str(e)}")
            results['population_analyses'] = {'error': str(e)}
        
        # 3. Merged spike-velocity analysis
        print("\n3. Running merged spike-velocity analysis...")
        try:
            spike_velocity_plots = self.merged_spike_velocity_analysis(
                save_path=os.path.join(save_dir, "merged_spike_velocity.html"),
                notebook=notebook,
                overwrite=overwrite,
                font_size=font_size
            )
            results['spike_velocity_analysis'] = spike_velocity_plots
            results['analyses_performed'].append('Spike-velocity analysis')
        except Exception as e:
            print(f"Error in spike-velocity analysis: {str(e)}")
            results['spike_velocity_analysis'] = {'error': str(e)}
        
        # 4. Merged connectivity analysis
        print("\n4. Running merged connectivity analysis...")
        connectivity_results = self.merged_connectivity_analysis(
            save_path=os.path.join(save_dir, "merged_connectivity.html"),
            notebook=notebook,
            overwrite=overwrite,
            font_size=font_size
        )
        results['connectivity_analysis'] = connectivity_results
        results['analyses_performed'].append('Connectivity analysis')
        
        # 5. Statistical comparisons
        print("\n5. Running statistical comparisons...")
        stat_results = {}
        for metric in ['mean_activity', 'spike_rate']:
            stat_results[metric] = self.statistical_comparison_of_merged_vs_sessions(
                metric=metric, signal_type=signal_type if signal_type != 'auto' else 'zsc'
            )
        results['statistical_comparisons'] = stat_results
        results['analyses_performed'].append('Statistical comparisons')
        
        # 6. Generate summary report
        print("\n6. Generating comprehensive report...")
        try:
            report_content = []
            report_content.append("# Comprehensive Merged Multi-Session Analysis Report\n")
            report_content.append(f"**Analysis Date**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            report_content.append(f"**Sessions Analyzed**: {len(self.loaded_sessions)}\n")
            report_content.append(f"**Sessions**: {', '.join(self.loaded_sessions)}\n")
            report_content.append(f"**Signal Type Used**: {signal_type}\n\n")
            
            # Get merged data info
            merged_data = self._merge_session_data()
            if merged_data:
                report_content.append("## Merged Dataset Summary\n")
                report_content.append(f"- Total neurons: {merged_data['metadata']['total_neurons']}\n")
                report_content.append(f"- D1 neurons: {merged_data['metadata']['d1_neurons']}\n")
                report_content.append(f"- D2 neurons: {merged_data['metadata']['d2_neurons']}\n")
                report_content.append(f"- CHI neurons: {merged_data['metadata']['chi_neurons']}\n\n")
            
            report_content.append("## Analyses Performed\n")
            for analysis in results['analyses_performed']:
                report_content.append(f"- [OK] {analysis}\n")
            
            report_content.append("\n## Analysis Files Generated\n")
            report_content.append("- `merged_neuron_centered_analyses.html`: Neuron-centered activity analysis\n")
            report_content.append("- `merged_population_d1.html`: D1-centered population analysis\n")
            report_content.append("- `merged_population_d2.html`: D2-centered population analysis\n")
            report_content.append("- `merged_population_chi.html`: CHI-centered population analysis\n")
            report_content.append("- `merged_spike_velocity.html`: Spike-velocity relationship analysis\n")
            report_content.append("- `merged_connectivity.html`: Neural connectivity analysis\n")
            
            # Save report
            report_path = os.path.join(save_dir, "comprehensive_analysis_report.md")
            with open(report_path, 'w') as f:
                f.write(''.join(report_content))
            
            results['report_path'] = report_path
            
        except Exception as e:
            print(f"Error generating report: {str(e)}")
            results['report_generation'] = {'error': str(e)}
        
        print(f"\n[SUCCESS] Comprehensive merged analysis completed!")
        print(f"[FILES] Results saved to: {save_dir}")
        print(f"[STATS] Analyses performed: {len(results['analyses_performed'])}")
        
        return results


if __name__ == "__main__":
    sessions_config = [
        {'session_name': '391_06012025'},{'session_name': '391_05152025'}, {'session_name': '391_05062025'}, {'session_name': '392_05082025'}
    ]

    analyzer = MultiSessionAnalyzer(sessions_config, analyzer_type='CellTypeTank')

    neuron_counts_df = analyzer.compare_neuron_counts_across_sessions()