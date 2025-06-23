# MultiSessionAnalyzer Documentation

## Overview

The `MultiSessionAnalyzer` is a powerful tool for analyzing neural data across multiple experimental sessions. It supports both traditional cross-session comparisons and **new merged analysis capabilities** that treat all sessions as one large dataset for increased statistical power.

## Key Features

### 1. **Traditional Cross-Session Analysis**
- Compare neuron counts across sessions
- Analyze activity patterns between sessions
- Movement pattern comparisons
- Statistical testing across sessions

### 2. **🆕 NEW: Merged Analysis** (Recommended)
- **Increased Statistical Power**: Combine all neurons from all sessions
- **All CellTypeTank Analyses Available**: Apply any CellTypeTank analysis to merged data
- **Better Signal Detection**: Noise averaging across sessions
- **Population-Level Insights**: Understand neural patterns at scale

### 3. **Advanced Comparisons**
- Connectivity pattern analysis across sessions
- Spike pattern comparisons
- Statistical validation of merged vs individual analysis

### 4. **Comprehensive Analysis**
- One-click analysis running all methods
- Automatic report generation
- Complete statistical summaries

## Installation

```python
from civis.src.MultiSessionAnalyzer import MultiSessionAnalyzer
```

## Basic Usage

### Session Configuration

```python
sessions_config = [
    {
        'session_name': 'Session_391_Day1',
        'ci_path': '/path/to/calcium_imaging.mat',
        'virmen_path': '/path/to/virmen_data.mat',
        'cell_type_label_file': '/path/to/cell_labels.json',
        'ci_rate': 20,
        'vm_rate': 20,
        'session_duration': 30 * 60,  # 30 minutes
        'maze_type': 'linear_track',  # optional
    },
    {
        'session_name': 'Session_391_Day2',
        'ci_path': '/path/to/session2_calcium.mat',
        'virmen_path': '/path/to/session2_virmen.mat',
        'cell_type_label_file': '/path/to/session2_labels.json',
        'ci_rate': 20,
        'vm_rate': 20,
        'session_duration': 30 * 60,
    },
    # Add more sessions...
]

# Initialize analyzer
analyzer = MultiSessionAnalyzer(sessions_config, analyzer_type='CellTypeTank')
```

### Quick Start: Comprehensive Analysis

For a complete analysis of all sessions combined:

```python
# One-click comprehensive analysis
results = analyzer.comprehensive_merged_analysis(
    signal_type='auto',
    save_dir="my_analysis",
    overwrite=True
)
```

This generates:
- Merged neuron-centered analyses
- Population analyses for all cell types
- Spike-velocity relationships
- Connectivity analysis
- Statistical comparisons
- Comprehensive report

## API Reference

### Core Data Management

#### `__init__(sessions_config, analyzer_type='CellTypeTank')`
Initialize MultiSessionAnalyzer with multiple sessions.

```python
analyzer = MultiSessionAnalyzer(
    sessions_config=sessions_config,
    analyzer_type='CellTypeTank'  # or 'CITank', 'ElecTank', 'VirmenTank'
)
```

**Parameters:**
- `sessions_config`: List of session configuration dictionaries
- `analyzer_type`: Type of analyzer ('CellTypeTank', 'CITank', 'ElecTank', 'VirmenTank')

#### `get_session_info()`
Get comprehensive information about all loaded sessions.

```python
info_df = analyzer.get_session_info()
print(info_df)
```

**Returns:** DataFrame with session details including:
- Session name and status
- Number of neurons and recording length
- Session duration and analyzer type

#### `get_failed_sessions_info()`
Get information about sessions that failed to load.

```python
failed_info = analyzer.get_failed_sessions_info()
for session, error in failed_info.items():
    print(f"{session}: {error}")
```

#### `get_session(session_name)`
Get a specific session analyzer object.

```python
session = analyzer.get_session('Session_391_Day1')
# Now you can use any CellTypeTank method on this session
plot = session.plot_d1_centered_analysis()
```

#### `get_all_sessions()`
Get all loaded session objects.

```python
all_sessions = analyzer.get_all_sessions()
for name, session in all_sessions.items():
    print(f"{name}: {len(session.d1_indices)} D1 neurons")
```

#### `add_session(session_config, session_name=None)`
Add a new session to the analyzer.

```python
new_session_config = {
    'session_name': 'Session_392_Day1',
    'ci_path': '/path/to/new_session.mat',
    'cell_type_label_file': '/path/to/new_labels.json',
    # ... other parameters
}
analyzer.add_session(new_session_config)
```

#### `remove_session(session_name)`
Remove a session from the analyzer.

```python
analyzer.remove_session('Session_391_Day1')
```

#### `_merge_session_data()`
Internal method that combines data from all sessions into one dataset.

```python
merged_data = analyzer._merge_session_data()
print(f"Total neurons: {merged_data['metadata']['total_neurons']}")
print(f"D1 neurons: {merged_data['metadata']['d1_neurons']}")
print(f"Sessions included: {merged_data['sessions_included']}")
```

**Returns:** Dictionary containing:
- `combined_signals`: Merged neural signals
- `combined_cell_indices`: Cell type indices for merged data
- `combined_peak_indices`: Peak indices for merged data
- `session_boundaries`: Information about session boundaries
- `metadata`: Summary statistics

### 🆕 Merged Analysis Methods

These methods treat all sessions as one large population, providing increased statistical power.

#### `merged_neuron_centered_analysis(center_cell_type, signal_type, ...)`
Analyze neural activity centered on specific cell type spikes using merged data.

```python
plot, stats = analyzer.merged_neuron_centered_analysis(
    center_cell_type='D1',
    signal_type='auto',
    activity_window=1.0,
    save_path="merged_d1_analysis.html",
    title="D1-Centered Analysis - All Sessions",
    notebook=False,
    overwrite=True,
    font_size='12pt'
)

# Access statistics
print(f"Total sessions merged: {stats['merger_info']['total_sessions']}")
print(f"Total neurons analyzed: {stats['merger_info']['total_neurons_merged']}")
```

**Parameters:**
- `center_cell_type`: 'D1', 'D2', or 'CHI'
- `signal_type`: 'auto', 'zsc', 'denoised', or 'raw'
- `activity_window`: Time window for detecting co-activity (seconds)
- `save_path`: Output HTML file path
- `title`: Custom plot title
- `notebook`: Whether displaying in Jupyter notebook
- `overwrite`: Whether to overwrite existing files
- `font_size`: Font size for plot elements

**Returns:**
- `plot`: Bokeh plot object
- `stats`: Dictionary with analysis statistics and merger info

#### `merged_population_analysis(center_cell_type, ...)`
Population-level analysis averaging ALL neurons of each type.

```python
plot, stats = analyzer.merged_population_analysis(
    center_cell_type='D1',
    signal_type='auto',
    time_window=3.0,
    baseline_correct=True,
    save_path="merged_population.html",
    title="D1 Population Analysis - Merged Sessions"
)

# Check results
print(f"Center peaks analyzed: {stats.get('center_peaks_analyzed', 'N/A')}")
print(f"Valid peaks: {stats.get('valid_peaks', 'N/A')}")
```

**Parameters:**
- `center_cell_type`: 'D1', 'D2', or 'CHI'
- `signal_type`: 'auto', 'zsc', 'denoised', or 'raw'
- `time_window`: Time window before/after peaks (seconds)
- `baseline_correct`: Whether to apply baseline correction
- `save_path`: Output HTML file path
- Additional parameters: `title`, `notebook`, `overwrite`, `font_size`

#### `merged_all_neuron_centered_analyses(...)`
Run neuron-centered analysis for all cell types (D1, D2, CHI) on merged data.

```python
results = analyzer.merged_all_neuron_centered_analyses(
    signal_type='auto',
    activity_window=1.0,
    save_path="all_merged_analyses.html",
    notebook=False,
    overwrite=True
)

# Access individual results
d1_results = results.get('d1_centered', {})
d2_results = results.get('d2_centered', {})
chi_results = results.get('chi_centered', {})

print(f"D1 analysis completed: {'plot' in d1_results}")
print(f"D2 analysis completed: {'plot' in d2_results}")
print(f"CHI analysis completed: {'plot' in chi_results}")
```

**Returns:** Dictionary with results for each cell type:
- `d1_centered`: D1-centered analysis results
- `d2_centered`: D2-centered analysis results  
- `chi_centered`: CHI-centered analysis results

#### `merged_spike_velocity_analysis(...)`
Analyze spike-velocity relationships using merged data.

```python
plots = analyzer.merged_spike_velocity_analysis(
    d1_peak_indices=None,  # Use merged peak indices
    d2_peak_indices=None,
    chi_peak_indices=None,
    save_path="merged_spike_velocity.html",
    title="Spike-Velocity Analysis - All Sessions"
)

print(f"Generated {len(plots)} plots")
```

**Parameters:**
- `d1_peak_indices`: Custom D1 peak indices (optional)
- `d2_peak_indices`: Custom D2 peak indices (optional)
- `chi_peak_indices`: Custom CHI peak indices (optional)
- `save_path`: Output HTML file path
- `title`: Custom plot title
- Additional parameters: `notebook`, `overwrite`, `font_size`

**Returns:** List of generated plots

#### `merged_connectivity_analysis(...)`
Perform comprehensive connectivity analysis on merged neural data.

```python
results = analyzer.merged_connectivity_analysis(
    save_path="merged_connectivity.html",
    notebook=False,
    overwrite=True
)

# Access connectivity results
print("Connectivity Analysis Results:")
print(f"- Conditional probabilities: {len(results.get('conditional_probs', {}))}")
print(f"- Cross-correlations: {len(results.get('cross_correlations', {}))}")
print(f"- Mutual information: {len(results.get('mutual_information', {}))}")
print(f"- Network nodes: {results.get('network', {}).get('nodes', 0)}")

# Access merger information
merger_info = results.get('merger_info', {})
print(f"Sessions merged: {merger_info.get('sessions_merged', [])}")
```

**Returns:** Dictionary containing:
- `conditional_probs`: Conditional activation probabilities
- `cross_correlations`: Time-lagged cross-correlations
- `coactivation_patterns`: Co-activation pattern analysis
- `mutual_information`: Mutual information between cell types
- `network`: Connectivity network graph
- `merger_info`: Information about merged sessions

### Cross-Session Comparison Methods

#### `compare_connectivity_across_sessions(...)`
Compare connectivity patterns between individual sessions.

```python
comparison = analyzer.compare_connectivity_across_sessions(
    save_path="connectivity_comparison.html",
    notebook=False,
    overwrite=True
)

# Access comparison results
individual_results = comparison['individual_sessions']
comparison_summary = comparison['comparison_summary']

for session_name, results in individual_results.items():
    if 'error' not in results:
        print(f"{session_name}: Connectivity analysis successful")
        timing_data = results.get('peak_timing_relationships', {})
        print(f"  - Total peaks analyzed: {timing_data.get('total_peaks_analyzed', 0)}")
    else:
        print(f"{session_name}: Analysis failed - {results['error']}")
```

**Returns:** Dictionary containing:
- `individual_sessions`: Connectivity results for each session
- `comparison_summary`: Cross-session comparison metrics

#### `compare_spike_patterns_across_sessions(...)`
Compare spike patterns and rates across sessions.

```python
results = analyzer.compare_spike_patterns_across_sessions(
    signal_type='zsc',
    save_path="spike_patterns.html"
)

# Access spike pattern data
spike_patterns = results['spike_patterns']
for session_name, patterns in spike_patterns.items():
    spike_rates = patterns['spike_rates']
    cell_counts = patterns['cell_counts']
    
    print(f"\n{session_name}:")
    print(f"  D1: {spike_rates.get('d1', 0):.2f} spikes/min ({cell_counts.get('d1', 0)} cells)")
    print(f"  D2: {spike_rates.get('d2', 0):.2f} spikes/min ({cell_counts.get('d2', 0)} cells)")
    print(f"  CHI: {spike_rates.get('chi', 0):.2f} spikes/min ({cell_counts.get('chi', 0)} cells)")
```

**Returns:** Dictionary containing:
- `spike_patterns`: Spike rate and count data for each session
- `plots`: Generated comparison plots

#### `statistical_comparison_of_merged_vs_sessions(metric, signal_type)`
Statistical comparison between merged and individual session analyses.

```python
# Compare mean activity
stats = analyzer.statistical_comparison_of_merged_vs_sessions(
    metric='mean_activity',
    signal_type='zsc'
)

# Compare spike rates
spike_stats = analyzer.statistical_comparison_of_merged_vs_sessions(
    metric='spike_rate',
    signal_type='zsc'
)

# Analyze results
for cell_type in ['d1', 'd2', 'chi']:
    if cell_type in stats['comparisons']:
        comparison = stats['comparisons'][cell_type]
        if 'error' not in comparison:
            print(f"\n{cell_type.upper()} Statistical Comparison:")
            print(f"  Test: {comparison['test']}")
            print(f"  p-value: {comparison['p_value']:.6f}")
            print(f"  Significant: {comparison['significant']}")
            print(f"  Effect size: {comparison['effect_size']:.3f}")
        else:
            print(f"{cell_type.upper()}: Analysis failed - {comparison['error']}")
```

**Available metrics:**
- `'mean_activity'`: Average neural activity
- `'max_activity'`: Peak activity levels
- `'std_activity'`: Activity variability
- `'spike_rate'`: Firing rate (spikes per minute)

**Returns:** Dictionary containing:
- `individual_sessions_stats`: Statistics from individual sessions
- `merged_data_stats`: Statistics from merged analysis
- `comparisons`: Statistical test results for each cell type

### Traditional Cross-Session Methods

#### `compare_neuron_counts_across_sessions(...)`
Compare cell type counts between sessions.

```python
df = analyzer.compare_neuron_counts_across_sessions(
    save_path="neuron_counts.html",
    notebook=False,
    overwrite=True,
    font_size='14pt'
)

# Display results
print("Neuron Counts Across Sessions:")
print(df)

# Calculate summary statistics
print(f"\nSummary Statistics:")
print(f"Average D1 neurons: {df['D1_Count'].mean():.1f} ± {df['D1_Count'].std():.1f}")
print(f"Average D2 neurons: {df['D2_Count'].mean():.1f} ± {df['D2_Count'].std():.1f}")
print(f"Average CHI neurons: {df['CHI_Count'].mean():.1f} ± {df['CHI_Count'].std():.1f}")
```

**Returns:** DataFrame with columns:
- `Session`: Session name
- `D1_Count`: Number of D1 neurons
- `D2_Count`: Number of D2 neurons
- `CHI_Count`: Number of CHI neurons
- `Total_Count`: Total number of neurons

#### `analyze_cross_session_activity_patterns(...)`
Analyze activity patterns across sessions.

```python
stats = analyzer.analyze_cross_session_activity_patterns(
    signal_type='zsc',
    save_path="activity_patterns.html",
    notebook=False,
    overwrite=True
)

# Access session-specific statistics
for session_name, session_stats in stats.items():
    if isinstance(session_stats, dict) and 'error' not in session_stats:
        print(f"\n{session_name}:")
        print(f"  D1 mean activity: {session_stats.get('d1_mean_activity', 'N/A'):.3f}")
        print(f"  D2 mean activity: {session_stats.get('d2_mean_activity', 'N/A'):.3f}")
        print(f"  CHI mean activity: {session_stats.get('chi_mean_activity', 'N/A'):.3f}")
        print(f"  Total neurons: {session_stats.get('total_neurons', 'N/A')}")
```

**Returns:** Dictionary with activity statistics for each session

#### `compare_movement_patterns_across_sessions(...)`
Compare movement and behavioral patterns.

```python
data = analyzer.compare_movement_patterns_across_sessions(
    save_path="movement_patterns.html",
    notebook=False,
    overwrite=True
)

# Access movement data
for session_name, movement_data in data.items():
    if isinstance(movement_data, dict):
        print(f"\n{session_name} Movement Patterns:")
        print(f"  Movement onsets: {movement_data.get('movement_onsets', 'N/A')}")
        print(f"  Lick count: {movement_data.get('lick_count', 'N/A')}")
        print(f"  Session duration: {movement_data.get('session_duration', 'N/A'):.1f} min")
```

**Returns:** Dictionary with movement pattern data for each session

#### `statistical_comparison_across_sessions(metric, signal_type)`
Statistical tests between sessions.

```python
results = analyzer.statistical_comparison_across_sessions(
    metric='mean_activity',
    signal_type='zsc'
)

# Access statistical results
print("Cross-Session Statistical Comparison:")
print(f"Metric analyzed: {results['metric']}")
print(f"Signal type: {results['signal_type']}")

# Cell type comparisons
for cell_type in ['d1', 'd2', 'chi']:
    if cell_type in results:
        comparison = results[cell_type]
        print(f"\n{cell_type.upper()}:")
        print(f"  Test statistic: {comparison.get('statistic', 'N/A')}")
        print(f"  p-value: {comparison.get('p_value', 'N/A')}")
        print(f"  Significant difference: {comparison.get('significant', 'N/A')}")
```

**Returns:** Dictionary with statistical comparison results

### Comprehensive Analysis

#### `comprehensive_merged_analysis(...)`
🆕 **One-click comprehensive analysis** (Recommended)

```python
results = analyzer.comprehensive_merged_analysis(
    signal_type='auto',
    save_dir="comprehensive_analysis",
    notebook=False,
    overwrite=True,
    font_size='12pt'
)

# Access all results
print("Comprehensive Analysis Results:")
print(f"Sessions included: {results['sessions_included']}")
print(f"Save directory: {results['save_directory']}")
print(f"Analyses performed: {results['analyses_performed']}")

# Access specific analysis results
neuron_results = results.get('neuron_centered_analyses', {})
population_results = results.get('population_analyses', {})
connectivity_results = results.get('connectivity_analysis', {})
statistical_results = results.get('statistical_comparisons', {})

# Check for errors
for analysis_type, result in results.items():
    if isinstance(result, dict) and 'error' in result:
        print(f"Error in {analysis_type}: {result['error']}")
```

This runs:
1. All merged neuron-centered analyses
2. All population analyses (D1, D2, CHI)
3. Spike-velocity analysis
4. Connectivity analysis
5. Statistical comparisons (mean_activity, spike_rate)
6. Generates comprehensive report

**Returns:** Dictionary containing:
- `sessions_included`: List of analyzed sessions
- `save_directory`: Output directory path
- `analyses_performed`: List of completed analyses
- `neuron_centered_analyses`: Neuron-centered analysis results
- `population_analyses`: Population analysis results
- `spike_velocity_analysis`: Spike-velocity analysis results
- `connectivity_analysis`: Connectivity analysis results
- `statistical_comparisons`: Statistical comparison results
- `report_path`: Path to generated report

#### `generate_comprehensive_report(...)`
Generate traditional multi-session report.

```python
report_path = analyzer.generate_comprehensive_report(
    save_dir="traditional_analysis",
    notebook=False,
    overwrite=True
)

print(f"Traditional analysis report saved to: {report_path}")
```

**Returns:** Path to generated report file

## Advanced Usage Examples

### Custom Analysis Workflow

```python
# 1. Initialize and check sessions
analyzer = MultiSessionAnalyzer(sessions_config)
print(f"Loaded {len(analyzer.loaded_sessions)} sessions successfully")

# Check for failed sessions
failed = analyzer.get_failed_sessions_info()
if failed:
    print("Warning: Some sessions failed to load:", failed)

# 2. Quick overview
session_info = analyzer.get_session_info()
print(session_info)

# 3. Merged analysis for specific cell type
d1_plot, d1_stats = analyzer.merged_neuron_centered_analysis(
    center_cell_type='D1',
    signal_type='zsc',
    save_path="d1_merged_analysis.html"
)

# 4. Population analysis with baseline correction
pop_plot, pop_stats = analyzer.merged_population_analysis(
    center_cell_type='D1',
    baseline_correct=True,
    save_path="d1_population_analysis.html"
)

# 5. Connectivity analysis
connectivity = analyzer.merged_connectivity_analysis(
    save_path="connectivity_analysis.html"
)

# 6. Statistical validation
validation = analyzer.statistical_comparison_of_merged_vs_sessions(
    metric='mean_activity',
    signal_type='zsc'
)

# 7. Cross-session comparison
cross_session = analyzer.compare_spike_patterns_across_sessions(
    save_path="cross_session_patterns.html"
)
```

### Batch Processing Multiple Experiments

```python
# Process multiple experiments
experiments = {
    'Experiment_A': [session_configs_A],
    'Experiment_B': [session_configs_B],
    'Experiment_C': [session_configs_C]
}

all_results = {}

for exp_name, sessions in experiments.items():
    print(f"Processing {exp_name}...")
    
    analyzer = MultiSessionAnalyzer(sessions)
    
    # Run comprehensive analysis for each experiment
    results = analyzer.comprehensive_merged_analysis(
        save_dir=f"results_{exp_name}",
        overwrite=True
    )
    
    all_results[exp_name] = results
    print(f"Completed {exp_name}: {len(results['analyses_performed'])} analyses")

# Compare results across experiments
for exp_name, results in all_results.items():
    stats = results.get('statistical_comparisons', {})
    mean_activity = stats.get('mean_activity', {})
    
    print(f"\n{exp_name} Summary:")
    for cell_type in ['d1', 'd2', 'chi']:
        merged_stats = mean_activity.get('merged_data_stats', {}).get(cell_type, {})
        mean_val = merged_stats.get('mean', 'N/A')
        n_samples = merged_stats.get('n_samples', 'N/A')
        print(f"  {cell_type.upper()}: {mean_val:.3f} (n={n_samples})")
```

### Working with Individual Sessions

```python
# Access individual session objects
session_1 = analyzer.get_session('Session_391_Day1')
session_2 = analyzer.get_session('Session_391_Day2')

# Run CellTypeTank analyses on individual sessions
if session_1:
    # Any CellTypeTank method can be used
    d1_plot = session_1.plot_d1_centered_analysis(save_path="session1_d1.html")
    connectivity = session_1.create_connectivity_analyzer()
    conn_results = connectivity.analyze_and_plot_connectivity()

# Compare specific metrics between sessions
all_sessions = analyzer.get_all_sessions()
d1_counts = {}
for name, session in all_sessions.items():
    d1_counts[name] = len(session.d1_indices)

print("D1 neuron counts by session:")
for name, count in d1_counts.items():
    print(f"  {name}: {count} D1 neurons")
```

## Recommended Workflow

### For Maximum Statistical Power (Recommended):

```python
# 1. Quick comprehensive overview
results = analyzer.comprehensive_merged_analysis(
    signal_type='auto',
    save_dir="comprehensive_analysis"
)

# 2. Detailed investigation of specific aspects
d1_plot, d1_stats = analyzer.merged_neuron_centered_analysis(
    center_cell_type='D1',
    signal_type='zsc'
)

connectivity = analyzer.merged_connectivity_analysis()

# 3. Validation with statistical comparison
validation = analyzer.statistical_comparison_of_merged_vs_sessions('mean_activity')

# 4. Cross-session validation
cross_session = analyzer.compare_connectivity_across_sessions()
```

### For Session-by-Session Comparison:

```python
# Traditional cross-session analysis
neuron_counts = analyzer.compare_neuron_counts_across_sessions()
activity_patterns = analyzer.analyze_cross_session_activity_patterns()
movement_patterns = analyzer.compare_movement_patterns_across_sessions()
statistical_comparison = analyzer.statistical_comparison_across_sessions('mean_activity')
report = analyzer.generate_comprehensive_report()
```

### For Hypothesis Testing:

```python
# Test specific hypotheses about cell type interactions
d1_analysis = analyzer.merged_neuron_centered_analysis(center_cell_type='D1')
d2_analysis = analyzer.merged_neuron_centered_analysis(center_cell_type='D2')
chi_analysis = analyzer.merged_neuron_centered_analysis(center_cell_type='CHI')

# Test connectivity hypotheses
connectivity = analyzer.merged_connectivity_analysis()
conditional_probs = connectivity['conditional_probs']

# Statistical validation of differences
mean_activity_stats = analyzer.statistical_comparison_of_merged_vs_sessions('mean_activity')
spike_rate_stats = analyzer.statistical_comparison_of_merged_vs_sessions('spike_rate')
```

## Output Files

### Merged Analysis Outputs
- `merged_neuron_centered_analyses.html`: Combined neuron-centered analysis
- `merged_population_d1.html`: D1-centered population analysis
- `merged_population_d2.html`: D2-centered population analysis  
- `merged_population_chi.html`: CHI-centered population analysis
- `merged_spike_velocity.html`: Spike-velocity relationships
- `merged_connectivity.html`: Neural connectivity analysis
- `merged_connectivity_summary.html`: Connectivity summary plots
- `comprehensive_analysis_report.md`: Summary report

### Traditional Analysis Outputs
- `neuron_counts_comparison.html`: Cell count comparisons
- `activity_patterns_comparison.html`: Activity pattern analysis
- `movement_patterns_comparison.html`: Movement comparisons
- `spike_patterns_comparison.html`: Spike pattern comparisons
- `connectivity_comparison.html`: Cross-session connectivity comparison
- `analysis_report.md`: Traditional report

### Statistical Output Files
- Analysis results include detailed statistical summaries
- All plots are interactive Bokeh HTML files
- Reports are in Markdown format for easy viewing

## Key Advantages of Merged Analysis

### 🔬 **Scientific Benefits**
- **Increased Statistical Power**: More neurons = better statistics
- **Noise Reduction**: Averaging across sessions reduces noise
- **Population Insights**: Understand neural patterns at population level
- **Better Connectivity Detection**: More data for connectivity analysis
- **Effect Size Detection**: Better power to detect small effects
- **Robust Patterns**: Identify consistent patterns across sessions

### 🛠️ **Technical Benefits**
- **All CellTypeTank Methods Available**: Use any existing analysis
- **Automatic Data Management**: Handles time alignment and indexing
- **Unified Analysis**: Consistent analysis across all sessions
- **Comprehensive Reports**: Detailed statistical summaries
- **Memory Efficient**: Optimized data structures for large datasets
- **Parallel Processing**: Vectorized operations for speed

### 📊 **Analytical Benefits**
- **Statistical Validation**: Compare merged vs individual results
- **Comprehensive Coverage**: All analysis types in one workflow
- **Cross-Session Validation**: Verify findings across sessions
- **Hypothesis Testing**: Robust framework for testing hypotheses
- **Publication Ready**: High-quality plots and statistical reports

## Signal Types

The analyzer supports multiple signal types with automatic selection:

- `'auto'`: Automatically select best available signal (zsc > denoised > raw)
- `'zsc'`: Z-score normalized signals (recommended for population analysis)
- `'denoised'`: Denoised calcium signals (good for individual neuron analysis)
- `'raw'`: Raw calcium imaging signals (baseline analysis)

```python
# Signal type usage examples
results_auto = analyzer.merged_neuron_centered_analysis(signal_type='auto')
results_zsc = analyzer.merged_neuron_centered_analysis(signal_type='zsc')
results_denoised = analyzer.merged_neuron_centered_analysis(signal_type='denoised')
```

## Error Handling

The analyzer includes robust error handling:

```python
# Check for failed sessions
failed = analyzer.get_failed_sessions_info()
if failed:
    print("Failed sessions:", failed)
    for session, error in failed.items():
        print(f"  {session}: {error}")

# Check session loading status
info = analyzer.get_session_info()
print(f"Successfully loaded: {len(analyzer.loaded_sessions)} sessions")

# All analysis methods include error handling
results = analyzer.comprehensive_merged_analysis()
for analysis_type, result in results.items():
    if isinstance(result, dict) and 'error' in result:
        print(f"Error in {analysis_type}: {result['error']}")
```

## Performance Considerations

### Memory Usage
- Merged analysis uses more memory (all sessions loaded simultaneously)
- Consider available RAM when loading many sessions
- Use `signal_type='auto'` to optimize memory usage
- Monitor memory usage for large datasets

```python
# Check memory usage
import psutil
process = psutil.Process()
memory_before = process.memory_info().rss / 1024 / 1024  # MB

analyzer = MultiSessionAnalyzer(sessions_config)
merged_data = analyzer._merge_session_data()

memory_after = process.memory_info().rss / 1024 / 1024  # MB
print(f"Memory usage: {memory_after - memory_before:.1f} MB")
```

### Processing Time
- Merged analysis is faster per neuron (vectorized operations)
- Initial data merging takes time proportional to number of sessions
- Use `comprehensive_merged_analysis()` for efficiency
- Connectivity analysis scales with number of neurons squared

```python
import time

# Time the analysis
start_time = time.time()
results = analyzer.comprehensive_merged_analysis()
end_time = time.time()

print(f"Analysis completed in {end_time - start_time:.1f} seconds")
print(f"Analyzed {len(results['sessions_included'])} sessions")
```

### Optimization Tips
1. **Use 'auto' signal type** for best performance/quality balance
2. **Run comprehensive analysis** instead of individual methods
3. **Process in batches** for very large datasets
4. **Save intermediate results** for iterative analysis
5. **Use appropriate time windows** to balance detail vs. speed

## Examples

See `civis/examples/complete_multi_session_analysis_guide.py` for a complete example demonstrating all features.

### Quick Start Example

```python
from civis.src.MultiSessionAnalyzer import MultiSessionAnalyzer

# Configuration
sessions_config = [
    {'session_name': 'Day1', 'ci_path': 'day1.mat', 'cell_type_label_file': 'day1_labels.json'},
    {'session_name': 'Day2', 'ci_path': 'day2.mat', 'cell_type_label_file': 'day2_labels.json'},
    {'session_name': 'Day3', 'ci_path': 'day3.mat', 'cell_type_label_file': 'day3_labels.json'},
]

# Analysis
analyzer = MultiSessionAnalyzer(sessions_config)
results = analyzer.comprehensive_merged_analysis(save_dir="my_results")

print(f"Analysis complete! Results saved to: {results['save_directory']}")
```

## Troubleshooting

### Common Issues

1. **No sessions loaded**
   ```python
   # Check file paths in session configuration
   failed = analyzer.get_failed_sessions_info()
   print("Failed sessions:", failed)
   ```

2. **Memory errors**
   ```python
   # Reduce number of sessions or use smaller signal types
   # Monitor memory usage and process in batches if needed
   ```

3. **Missing analyses**
   ```python
   # Ensure analyzer_type='CellTypeTank' for full functionality
   print(f"Analyzer type: {analyzer.analyzer_type}")
   ```

4. **File path issues**
   ```python
   # Check that all file paths exist
   import os
   for config in sessions_config:
       for key, path in config.items():
           if 'path' in key and path:
               print(f"{key}: {os.path.exists(path)}")
   ```

5. **Signal type issues**
   ```python
   # Check available signal types for each session
   for session_name in analyzer.loaded_sessions:
       session = analyzer.get_session(session_name)
       print(f"{session_name}:")
       print(f"  Has C_zsc: {hasattr(session, 'C_zsc') and session.C_zsc is not None}")
       print(f"  Has C_denoised: {hasattr(session, 'C_denoised') and session.C_denoised is not None}")
       print(f"  Has C_raw: {hasattr(session, 'C_raw') and session.C_raw is not None}")
   ```

### Debug Information

```python
# Comprehensive debugging
print("=== MultiSessionAnalyzer Debug Info ===")
print(f"Loaded sessions: {analyzer.loaded_sessions}")
print(f"Failed sessions: {analyzer.get_failed_sessions_info()}")
print(f"Analyzer type: {analyzer.analyzer_type}")

# Check merged data
try:
    merged_data = analyzer._merge_session_data()
    print(f"Merged data available: {bool(merged_data)}")
    if merged_data:
        print(f"Total neurons: {merged_data['metadata']['total_neurons']}")
        print(f"D1 neurons: {merged_data['metadata']['d1_neurons']}")
        print(f"D2 neurons: {merged_data['metadata']['d2_neurons']}")
        print(f"CHI neurons: {merged_data['metadata']['chi_neurons']}")
        print(f"Signal type: {merged_data['signal_type']}")
except Exception as e:
    print(f"Error merging data: {e}")

# Check individual sessions
for session_name in analyzer.loaded_sessions:
    session = analyzer.get_session(session_name)
    if session:
        print(f"\n{session_name}:")
        print(f"  D1 neurons: {len(getattr(session, 'd1_indices', []))}")
        print(f"  D2 neurons: {len(getattr(session, 'd2_indices', []))}")
        print(f"  CHI neurons: {len(getattr(session, 'chi_indices', []))}")
        if hasattr(session, 'C_raw') and session.C_raw is not None:
            print(f"  Signal shape: {session.C_raw.shape}")
```

### Performance Debugging

```python
import time
import psutil

def profile_analysis():
    process = psutil.Process()
    
    # Memory before
    mem_before = process.memory_info().rss / 1024 / 1024
    time_start = time.time()
    
    # Run analysis
    results = analyzer.comprehensive_merged_analysis()
    
    # Memory after
    mem_after = process.memory_info().rss / 1024 / 1024
    time_end = time.time()
    
    print(f"Analysis Performance:")
    print(f"  Time: {time_end - time_start:.1f} seconds")
    print(f"  Memory used: {mem_after - mem_before:.1f} MB")
    print(f"  Sessions analyzed: {len(results.get('sessions_included', []))}")
    print(f"  Analyses completed: {len(results.get('analyses_performed', []))}")
    
    return results

results = profile_analysis()
```

## Version History

- **v2.1**: Enhanced documentation and error handling
- **v2.0**: Added merged analysis capabilities
- **v1.0**: Original cross-session analysis

## Contributing

To contribute to MultiSessionAnalyzer:

1. Follow the existing code style
2. Add comprehensive docstrings
3. Include error handling
4. Add usage examples
5. Update documentation

## Citation

If you use MultiSessionAnalyzer in your research, please cite:

```
MultiSessionAnalyzer: A comprehensive tool for multi-session neural data analysis
[Your citation information here]
```

---

For more examples and advanced usage, see the example scripts in the `examples/` directory. 