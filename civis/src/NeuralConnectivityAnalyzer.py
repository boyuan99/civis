import os
import numpy as np

class NeuralConnectivityAnalyzer:
    """
    Independent class for analyzing functional connectivity between neural types.
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
            
            # Initialize probability matrix
            prob_matrix = np.zeros((n_neurons, n_neurons))
            
            # Calculate conditional probabilities for all neuron pairs (including self-activation)
            for i in range(n_neurons):
                for j in range(n_neurons):
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
            window_results = {}
            
            # Calculate conditional probabilities for all cell type pairs (including self-activation)
            for source_type in ['D1', 'D2', 'CHI']:
                for target_type in ['D1', 'D2', 'CHI']:
                    prob = calc_cell_type_conditional_prob(
                        cell_type_signals[source_type], 
                        cell_type_signals[target_type], 
                        window
                    )
                    window_results[f'{source_type}_to_{target_type}'] = prob
            
            results[f'window_{window}'] = window_results
        
        return results
    
    def _calculate_proper_baseline_neuron_proportions(self, window_size=5):
        """
        Calculate baseline neuron activation proportions using forward-looking time windows
        This matches the conditional probability calculation logic exactly
        
        Parameters:
        -----------
        window_size : int
            Size of forward-looking time window after each activation event
        
        Returns:
        --------
        dict : Baseline activation proportions for each cell type
        """
        baseline_rates = {}
        
        for cell_type in ['D1', 'D2', 'CHI']:
            signals = self.binary_signals[cell_type]
            
            # Find SPECIFIC activation events for this cell type (same as used in conditional probability)
            # We need to use the same activation detection logic as the conditional probability calculation
            
            # Get activation events from binary signals (these are based on detected peaks)
            # Find when the population first becomes active (rising edges of population activity)
            population_activity = np.sum(signals, axis=0)  # Sum across all neurons of this type
            
            # Find rising edges: transition from 0 to >0 activity
            population_binary = (population_activity > 0).astype(int)
            rising_edges = np.where(np.diff(population_binary) == 1)[0] + 1  # +1 because diff shifts by 1
            
            # Also include the first timepoint if it starts with activity
            if population_binary[0] == 1:
                rising_edges = np.concatenate([[0], rising_edges])
            
            active_timepoints = rising_edges
            
            if len(active_timepoints) == 0:
                # No activity found for this cell type
                baseline_rates[cell_type] = 0.0
                print(f"Warning: No activity found for {cell_type} neurons")
            else:
                # Calculate average proportion using time windows around active timepoints
                n_neurons = signals.shape[0]
                proportions = []
                
                for center_t in active_timepoints:
                    # Create forward-looking window EXACTLY like conditional probability calculation
                    start = center_t + 1  # Start AFTER the activation event
                    end = min(signals.shape[1], center_t + window_size + 1)
                    
                    # Skip if window is invalid
                    if start >= signals.shape[1] or end <= start:
                        continue
                    
                    # Count activations in the window
                    window_activations = np.sum(signals[:, start:end])
                    
                    # Calculate proportion (normalize by window size and neuron count)
                    window_length = end - start
                    proportion = window_activations / (n_neurons * window_length) if (n_neurons * window_length) > 0 else 0
                    proportions.append(proportion)
                
                baseline_rates[cell_type] = np.mean(proportions)
                
                print(f"{cell_type} baseline (forward window={window_size}): {baseline_rates[cell_type]:.4f} "
                      f"(from {len(proportions)} activation events, {n_neurons} neurons)")
        
        return baseline_rates
    
    def _calculate_pre_activation_baseline_neuron_proportions(self, window_size=5):
        """
        Calculate baseline neuron activation proportions using PRE-activation time windows
        This represents the traditional neuroscience approach: comparing post-activation to pre-activation
        
        Parameters:
        -----------
        window_size : int
            Size of backward-looking time window before each activation event
        
        Returns:
        --------
        dict : Baseline activation proportions for each cell type
        """
        baseline_rates = {}
        
        for cell_type in ['D1', 'D2', 'CHI']:
            signals = self.binary_signals[cell_type]
            
            # Find SPECIFIC activation events for this cell type (same logic as forward method)
            population_activity = np.sum(signals, axis=0)  # Sum across all neurons of this type
            
            # Find rising edges: transition from 0 to >0 activity
            population_binary = (population_activity > 0).astype(int)
            rising_edges = np.where(np.diff(population_binary) == 1)[0] + 1  # +1 because diff shifts by 1
            
            # Also include the first timepoint if it starts with activity
            if population_binary[0] == 1:
                rising_edges = np.concatenate([[0], rising_edges])
            
            active_timepoints = rising_edges
            
            if len(active_timepoints) == 0:
                # No activity found for this cell type
                baseline_rates[cell_type] = 0.0
                print(f"Warning: No activity found for {cell_type} neurons")
            else:
                # Calculate average proportion using PRE-activation time windows
                n_neurons = signals.shape[0]
                proportions = []
                
                for center_t in active_timepoints:
                    # Create backward-looking window BEFORE the activation event
                    start = max(0, center_t - window_size)
                    end = center_t  # Up to but not including the activation timepoint
                    
                    # Skip if window is invalid
                    if end <= start:
                        continue
                    
                    # Count activations in the PRE-activation window
                    window_activations = np.sum(signals[:, start:end])
                    
                    # Calculate proportion (normalize by window size and neuron count)
                    window_length = end - start
                    proportion = window_activations / (n_neurons * window_length) if (n_neurons * window_length) > 0 else 0
                    proportions.append(proportion)
                
                baseline_rates[cell_type] = np.mean(proportions)
                
                print(f"{cell_type} baseline (pre-activation window={window_size}): {baseline_rates[cell_type]:.4f} "
                      f"(from {len(proportions)} pre-activation windows, {n_neurons} neurons)")
        
        return baseline_rates
    
    def _calculate_proper_baseline_neuron_proportions_v2(self, window_size=5):
        """
        Alternative baseline calculation: random time windows
        This provides a true 'background' activation rate
        
        Parameters:
        -----------
        window_size : int
            Size of time window to use for baseline calculation
        
        Returns:
        --------
        dict : Baseline activation proportions for each cell type
        """
        baseline_rates = {}
        n_samples = 1000  # Number of random samples
        
        for cell_type in ['D1', 'D2', 'CHI']:
            signals = self.binary_signals[cell_type]
            n_neurons = signals.shape[0]
            signal_length = signals.shape[1]
            
            if signal_length < window_size:
                baseline_rates[cell_type] = 0.0
                continue
            
            # Sample random time windows
            proportions = []
            for _ in range(n_samples):
                # Random starting point
                start = np.random.randint(0, signal_length - window_size)
                end = start + window_size
                
                # Count activations in random window
                window_activations = np.sum(signals[:, start:end])
                
                # Calculate proportion per time step (to match conditional prob)
                proportion = window_activations / (n_neurons * window_size) if (n_neurons * window_size) > 0 else 0
                proportions.append(proportion)
            
            baseline_rates[cell_type] = np.mean(proportions)
            
            print(f"{cell_type} random baseline: {baseline_rates[cell_type]:.4f} "
                  f"(from {n_samples} random windows, {n_neurons} neurons)")
        
        return baseline_rates
    
    def _calculate_proper_baseline_neuron_proportions_original(self):
        """
        Original baseline calculation: single timepoint activation
        Kept for comparison and backward compatibility
        
        Returns:
        --------
        dict : Baseline activation proportions for each cell type
        """
        baseline_rates = {}
        
        for cell_type in ['D1', 'D2', 'CHI']:
            signals = self.binary_signals[cell_type]
            
            # Find timepoints where this cell type has any activity
            population_activity = np.sum(signals, axis=0)  # Sum across all neurons of this type
            active_timepoints = population_activity > 0     # Boolean mask for active timepoints
            
            if np.sum(active_timepoints) == 0:
                # No activity found for this cell type
                baseline_rates[cell_type] = 0.0
                print(f"Warning: No activity found for {cell_type} neurons")
            else:
                # Calculate average proportion of neurons active during active timepoints
                n_neurons = signals.shape[0]
                proportions = []
                
                for t in np.where(active_timepoints)[0]:
                    active_neurons_at_t = np.sum(signals[:, t])
                    proportion = active_neurons_at_t / n_neurons if n_neurons > 0 else 0
                    proportions.append(proportion)
                
                baseline_rates[cell_type] = np.mean(proportions)
                
                print(f"{cell_type} baseline (instantaneous): {baseline_rates[cell_type]:.4f} "
                      f"(from {len(proportions)} active timepoints, {n_neurons} neurons)")
        
        return baseline_rates
    
    def _calculate_legacy_baseline_rates(self):
        """
        Calculate baseline rates using the original method (time proportion)
        Kept for backward compatibility and comparison
        
        Returns:
        --------
        dict : Baseline time proportions for each cell type
        """
        baseline_rates = {}
        
        # Get binary signals for each cell type
        d1_binary = self.binary_signals['D1']
        d2_binary = self.binary_signals['D2']
        chi_binary = self.binary_signals['CHI']
        
        # Calculate baseline activation probability for each cell type
        # (proportion of time when at least one neuron of that type is active)
        d1_population_active = np.sum(d1_binary, axis=0) > 0
        d2_population_active = np.sum(d2_binary, axis=0) > 0
        chi_population_active = np.sum(chi_binary, axis=0) > 0
        
        baseline_rates['D1'] = np.mean(d1_population_active)
        baseline_rates['D2'] = np.mean(d2_population_active)
        baseline_rates['CHI'] = np.mean(chi_population_active)
        
        return baseline_rates
    
    def _aggregate_by_cell_type(self, prob_matrix, cell_types):
        """Aggregate individual neuron probabilities by cell type"""
        aggregated = {}
        
        # Create indices for each cell type
        cell_type_indices = {}
        for i, cell_type in enumerate(cell_types):
            if cell_type not in cell_type_indices:
                cell_type_indices[cell_type] = []
            cell_type_indices[cell_type].append(i)
        
        # Calculate mean probabilities between cell types (including self-activation)
        for source_type in ['D1', 'D2', 'CHI']:
            for target_type in ['D1', 'D2', 'CHI']:
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
            
            # Calculate conditional probabilities for all pairs (including self-activation)
            signals = {'D1': d1_active, 'D2': d2_active, 'CHI': chi_active}
            
            for source_type in ['D1', 'D2', 'CHI']:
                for target_type in ['D1', 'D2', 'CHI']:
                    prob = calc_conditional_prob(signals[source_type], signals[target_type], window)
                    window_results[f'{source_type}_to_{target_type}'] = prob
            
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
                pattern_names = [' -> '.join(pattern) for pattern, _ in top_patterns]
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
    

    def create_conditional_probability_heatmap(self, save_path=None, title="Conditional Probability Heatmap", 
                                        notebook=False, overwrite=False, font_size=None, 
                                        normalization_method='neuron_proportion',
                                        baseline_method='pre_activation',
                                        window_size=10,
                                        baseline_window_size=10):
        """
        Create conditional probability heatmap matrix visualization with multiple normalization options
        
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
        normalization_method : str, optional
            Normalization method for conditional probabilities:
            - 'neuron_proportion': Normalize by baseline neuron activation proportions (recommended)
            - 'activation_frequency': Normalize by activation time proportions (legacy method)
            - 'none': No normalization, show raw conditional probabilities
        baseline_method : str, optional
            Method for calculating baseline when using neuron_proportion normalization:
            - 'windowed': Use forward-looking windows after target activation events (default)
            - 'pre_activation': Use time windows BEFORE target activation events (traditional approach)
            - 'instantaneous': Use single timepoint activation (original method)
            - 'random': Use random time windows for true background rate
        window_size : int, optional
            Size of time window for conditional probability calculations (default=5)
            Must match available conditional probability data
        baseline_window_size : int, optional
            Size of time window for baseline calculations (default=5)
            Both conditional probability and baseline use forward-looking windows after activation events
        
        Returns:
        --------
        bokeh.plotting.figure or None
            The heatmap plot or None if no data available
        """
        if not hasattr(self, 'conditional_probs'):
            return None
            
        from bokeh.plotting import figure
        from bokeh.models import ColumnDataSource, HoverTool, ColorBar, LinearColorMapper
        from bokeh.transform import linear_cmap
        from bokeh.palettes import RdYlBu11
        import pandas as pd
        import numpy as np
        
        # Check if conditional probabilities data exists
        if not hasattr(self, 'conditional_probs') or self.conditional_probs is None:
            print("Warning: No conditional probabilities data found. Please run 'run_conditional_probabilities_analysis()' first.")
            return None
        
        # Calculate baseline rates based on selected normalization method
        baseline_rates = {}
        
        if normalization_method == 'neuron_proportion':
            print(f"Calculating baseline neuron activation proportions using {baseline_method} method...")
            
            if baseline_method == 'windowed':
                baseline_rates = self._calculate_proper_baseline_neuron_proportions(
                    window_size=baseline_window_size
                )
                print(f"Baseline calculated using forward windows (size={baseline_window_size}).")
            elif baseline_method == 'pre_activation':
                baseline_rates = self._calculate_pre_activation_baseline_neuron_proportions(
                    window_size=baseline_window_size
                )
                print(f"Baseline calculated using pre-activation windows (size={baseline_window_size}).")
            elif baseline_method == 'instantaneous':
                baseline_rates = self._calculate_proper_baseline_neuron_proportions_original()
                print(f"Baseline calculated using instantaneous active timepoints.")
            elif baseline_method == 'random':
                baseline_rates = self._calculate_proper_baseline_neuron_proportions_v2(window_size=baseline_window_size)
                print(f"Baseline calculated using random time windows (size={baseline_window_size}).")
            else:
                raise ValueError(f"Invalid baseline_method: {baseline_method}. "
                               f"Must be 'windowed', 'pre_activation', 'instantaneous', or 'random'")
            
        elif normalization_method == 'activation_frequency':
            print("Calculating baseline activation frequencies for normalization (legacy method)...")
            baseline_rates = self._calculate_legacy_baseline_rates()
            print(f"Baseline activation frequencies:")
            for cell_type, rate in baseline_rates.items():
                print(f"  {cell_type}: {rate:.4f}")
                
        elif normalization_method == 'none':
            print("No normalization will be applied.")
            baseline_rates = None
            
        else:
            raise ValueError(f"Invalid normalization_method: {normalization_method}. "
                           f"Must be 'neuron_proportion', 'activation_frequency', or 'none'")
        
        # Use data that matches the specified window size, or closest available
        target_window_key = f'window_{window_size}'
        if target_window_key in self.conditional_probs:
            window_key = target_window_key
            probs = self.conditional_probs[window_key]
            print(f"Using conditional probabilities with window_size={window_size}")
        else:
            # Fall back to the first available window if exact match not found
            window_key = list(self.conditional_probs.keys())[0]
            probs = self.conditional_probs[window_key]
            actual_window = window_key.replace('window_', '')
            print(f"Warning: Requested window_size={window_size} not found. Using window_size={actual_window}")
            print(f"Available windows: {list(self.conditional_probs.keys())}")
            print(f"To use window_size={window_size}, please run conditional probabilities analysis with time_windows=[{window_size}, ...]")
        
        # Create 3x3 matrices - one for raw and one for normalized
        cell_types = ['CHI', 'D1', 'D2']
        matrix_data = []
        
        # Also store the raw probability matrix for comparison
        raw_matrix = np.zeros((3, 3))
        normalized_matrix = np.zeros((3, 3))
        
        for i, source in enumerate(cell_types):
            for j, target in enumerate(cell_types):
                key = f'{source}_to_{target}'
                prob_value = probs.get(key, 0)
                
                # Handle both numeric and non-numeric values
                if isinstance(prob_value, (int, float, np.integer, np.floating)):
                    raw_prob = prob_value
                else:
                    raw_prob = 0
                
                # Store raw probability
                raw_matrix[i, j] = raw_prob
                
                # Calculate normalized probability based on method
                if baseline_rates is not None:
                    # Normalize: P(Target|Source) / P(Target_baseline)
                    if baseline_rates[target] > 0:
                        normalized_prob = raw_prob / baseline_rates[target]
                    else:
                        normalized_prob = 0
                    
                    # Store normalized value
                    normalized_matrix[i, j] = normalized_prob
                    
                    # Format text based on normalization
                    if normalized_prob > 0:
                        # Show percent change instead of fold change
                        percent_change = (normalized_prob - 1) * 100
                        prob_formatted = f'{percent_change:+.1f}%\n({raw_prob:.3f})'
                    else:
                        prob_formatted = '0'
                    
                    # Store additional info for hover
                    fold_change = normalized_prob
                    percent_change = (normalized_prob - 1) * 100 if normalized_prob > 0 else -100

                    display_prob = percent_change

                else:
                    # No normalization - use raw probability
                    normalized_prob = raw_prob
                    display_prob = raw_prob
                    prob_formatted = f'{raw_prob:.3f}' if raw_prob > 0 else '0'
                    fold_change = 1
                    percent_change = 0
                
                matrix_data.append({
                    'source': source,
                    'target': target,
                    'raw_probability': raw_prob,
                    'normalized_probability': normalized_prob,
                    'display_value': display_prob,
                    'probability_text': prob_formatted,
                    'x': j,      # Column index - Target
                    'y': 2-i,    # Row index - Source (flipped: CHI=2, D1=1, D2=0)
                    'color_value': display_prob,
                    'baseline_rate': baseline_rates.get(target, 0) if baseline_rates else 0,
                    'fold_change': fold_change,
                    'percent_change': percent_change,
                    'normalization_method': normalization_method
                })
        
        df = pd.DataFrame(matrix_data)
        
        # Create HoverTool with detailed information based on normalization method
        if baseline_rates is not None:
            if normalization_method == 'neuron_proportion':
                hover = HoverTool(tooltips=[
                    ('From', '@source'),
                    ('To', '@target'),
                    ('Raw Probability', '@raw_probability{0.000}'),
                    ('Target Baseline Proportion', '@baseline_rate{0.000}'),
                    ('Normalized (Fold Change)', '@normalized_probability{0.00}'),
                    ('Percent Change', '@percent_change{+0.0}%'),
                    ('Method', 'Neuron Proportion Normalization')
                ])
            else:  # activation_frequency
                hover = HoverTool(tooltips=[
                    ('From', '@source'),
                    ('To', '@target'),
                    ('Raw Probability', '@raw_probability{0.000}'),
                    ('Target Baseline Frequency', '@baseline_rate{0.000}'),
                    ('Normalized (Fold Change)', '@normalized_probability{0.00}'),
                    ('Percent Change', '@percent_change{+0.0}%'),
                    ('Method', 'Activation Frequency Normalization (Legacy)')
                ])
        else:
            hover = HoverTool(tooltips=[
                ('From', '@source'),
                ('To', '@target'),
                ('Probability', '@raw_probability{0.000}'),
                ('Method', 'No Normalization')
            ])
        
        # Adjust title and labels based on normalization method
        if normalization_method == 'neuron_proportion':
            plot_title = "Normalized Conditional Activation Matrix\nP(Target|Source) / P(Target neuron proportion)"
            color_label = "Percent Change Over Baseline"
        elif normalization_method == 'activation_frequency':
            plot_title = "Normalized Conditional Activation Matrix (Legacy)\nP(Target|Source) / P(Target time frequency)"
            color_label = "Percent Change Over Baseline"
        else:  # 'none'
            plot_title = "Conditional Activation Probability Matrix\nP(Target|Source)"
            color_label = "Probability"
        
        p = figure(width=500, height=450,
                title=plot_title,
                x_range=[-0.5, 2.5],
                y_range=[-0.5, 2.5],
                tools=[hover, 'pan', 'wheel_zoom', 'box_zoom', 'reset', 'save'])
        
        source_data = ColumnDataSource(df)
        

        # Set color scale based on normalization method
        max_val = df['display_value'].max()
        min_val = df['display_value'].min()
        
        from bokeh.palettes import RdBu11
        palette = RdBu11
        
        if baseline_rates is not None:
            # For normalized data, center color scale around 0% change
            # Use symmetric range for better visualization
            abs_max = max(abs(max_val), abs(min_val))
            color_range = min(abs_max, 200)  # Cap at ±200% for extreme values
            color_mapper = LinearColorMapper(
                palette=palette,
                low=-color_range,
                high=color_range
            )
        else:
            # For raw probabilities, use 0 to max range
            color_mapper = LinearColorMapper(
                palette=palette,
                low=0,
                high=max_val if max_val > 0 else 1
            )
        
        # Draw heatmap squares
        rect_glyph = p.rect(x='x', y='y', width=0.9, height=0.9, source=source_data,
                        fill_color={'field': 'color_value', 'transform': color_mapper},
                        line_color='white', line_width=3)
        
        # Add probability value text
        p.text(x='x', y='y', text='probability_text', source=source_data,
            text_align='center', text_baseline='middle', text_font_size='14pt',
            text_color='black', text_font_style='bold')
        
        # Add color bar
        color_bar = ColorBar(color_mapper=color_mapper, width=8, location=(0,0),
                            title=color_label)
        p.add_layout(color_bar, 'right')
        
        # Set axis labels
        p.xaxis.ticker = [0, 1, 2]
        p.yaxis.ticker = [0, 1, 2]
        p.xaxis.major_label_overrides = {0: "CHI", 1: "D1", 2: "D2"}
        p.yaxis.major_label_overrides = {0: "D2", 1: "D1", 2: "CHI"}
        
        # Correct axis labels
        p.xaxis.axis_label = "Target (Activated Neuron Type)"
        p.yaxis.axis_label = "Source (Activating Neuron Type)"
        
        # Add axis label styling
        p.xaxis.axis_label_text_font_size = "12pt"
        p.yaxis.axis_label_text_font_size = "12pt"
        p.xaxis.major_label_text_font_size = "12pt"
        p.yaxis.major_label_text_font_size = "12pt"
        
        # Print summary statistics
        if baseline_rates is not None:
            print(f"\nNormalized Conditional Probability Summary ({normalization_method}):")
            print("=" * 60)
            for i, source in enumerate(cell_types):
                for j, target in enumerate(cell_types):
                    norm_val = normalized_matrix[i, j]
                    raw_val = raw_matrix[i, j]
                    percent_change = (norm_val - 1) * 100 if norm_val > 0 else -100
                    
                    if norm_val > 1.5:
                        print(f"{source} -> {target}: {norm_val:.2f}x baseline ({percent_change:+.1f}%, raw={raw_val:.3f}) - STRONG ENHANCEMENT")
                    elif norm_val > 1.2:
                        print(f"{source} -> {target}: {norm_val:.2f}x baseline ({percent_change:+.1f}%, raw={raw_val:.3f}) - Moderate enhancement")
                    elif norm_val < 0.8:
                        print(f"{source} -> {target}: {norm_val:.2f}x baseline ({percent_change:+.1f}%, raw={raw_val:.3f}) - Suppression")
                    else:
                        print(f"{source} -> {target}: {norm_val:.2f}x baseline ({percent_change:+.1f}%, raw={raw_val:.3f}) - Normal")
        else:
            print(f"\nRaw Conditional Probability Summary:")
            print("=" * 40)
            for i, source in enumerate(cell_types):
                for j, target in enumerate(cell_types):
                    raw_val = raw_matrix[i, j]
                    print(f"{source} -> {target}: {raw_val:.3f}")
        
        # Set default save path if none provided
        if save_path is None:
            if normalization_method == 'neuron_proportion':
                suffix = f'_normalized_{baseline_method}_bw{baseline_window_size}_cw{window_size}'
            elif normalization_method == 'activation_frequency':
                suffix = '_normalized_legacy'
            else:
                suffix = f'_raw_w{window_size}'
            save_path = os.path.join(self.tank.config["SummaryPlotsPath"], 
                                self.tank.session_name, 
                                f'conditional_probability_heatmap{suffix}.html')
        
        # Use tank's output_bokeh_plot method if save_path is provided
        if save_path:
            self.tank.output_bokeh_plot(p, save_path=save_path, title=title, 
                                    notebook=notebook, overwrite=overwrite, font_size=font_size)
        
        return p
    
    def test_normalization_methods(self, window_size=5, baseline_window_size=5):
        """
        Test function to compare different normalization methods with consistent forward windows
        This helps verify the implementation correctness and window consistency
        """
        if not hasattr(self, 'conditional_probs') or self.conditional_probs is None:
            print("Error: No conditional probabilities data found. Please run analysis first.")
            return None
            
        print("Testing different normalization methods with forward windows...")
        print("=" * 60)
        
        # Test all combinations
        methods = ['neuron_proportion', 'activation_frequency', 'none']
        baseline_methods = ['windowed', 'pre_activation', 'random', 'instantaneous']
        
        results = {}
        
        for method in methods:
            print(f"\n=== Testing {method} method ===")
            results[method] = {}
            
            if method == 'neuron_proportion':
                for baseline_method in baseline_methods:
                    print(f"\n--- Baseline method: {baseline_method} ---")
                    results[method][baseline_method] = {}
                    
                    if baseline_method == 'windowed':
                        print(f"  Forward window size: {baseline_window_size}")
                        try:
                            baseline_rates = self._calculate_proper_baseline_neuron_proportions(
                                window_size=baseline_window_size
                            )
                            results[method][baseline_method] = {
                                'baseline_rates': baseline_rates,
                                'method_valid': True,
                                'window_size': baseline_window_size
                            }
                            
                            print(f"    Success! Baseline rates:")
                            for cell_type, rate in baseline_rates.items():
                                print(f"      {cell_type}: {rate:.4f}")
                                
                        except Exception as e:
                            print(f"    Error: {str(e)}")
                            results[method][baseline_method] = {
                                'baseline_rates': None,
                                'method_valid': False,
                                'error': str(e)
                            }
                    elif baseline_method == 'pre_activation':
                        print(f"  Pre-activation window size: {baseline_window_size}")
                        try:
                            baseline_rates = self._calculate_pre_activation_baseline_neuron_proportions(
                                window_size=baseline_window_size
                            )
                            results[method][baseline_method] = {
                                'baseline_rates': baseline_rates,
                                'method_valid': True,
                                'window_size': baseline_window_size
                            }
                            
                            print(f"    Success! Baseline rates:")
                            for cell_type, rate in baseline_rates.items():
                                print(f"      {cell_type}: {rate:.4f}")
                                
                        except Exception as e:
                            print(f"    Error: {str(e)}")
                            results[method][baseline_method] = {
                                'baseline_rates': None,
                                'method_valid': False,
                                'error': str(e)
                            }
                    else:
                        try:
                            if baseline_method == 'random':
                                baseline_rates = self._calculate_proper_baseline_neuron_proportions_v2(window_size=baseline_window_size)
                            else:  # instantaneous
                                baseline_rates = self._calculate_proper_baseline_neuron_proportions_original()
                                
                            results[method][baseline_method]['default'] = {
                                'baseline_rates': baseline_rates,
                                'method_valid': True
                            }
                            
                            print(f"    Success! Baseline rates:")
                            for cell_type, rate in baseline_rates.items():
                                print(f"      {cell_type}: {rate:.4f}")
                                
                        except Exception as e:
                            print(f"    Error: {str(e)}")
                            results[method][baseline_method]['default'] = {
                                'baseline_rates': None,
                                'method_valid': False,
                                'error': str(e)
                            }
                            
            elif method == 'activation_frequency':
                try:
                    baseline_rates = self._calculate_legacy_baseline_rates()
                    results[method]['legacy'] = {
                        'baseline_rates': baseline_rates,
                        'method_valid': True
                    }
                    
                    print(f"  Success! Baseline rates:")
                    for cell_type, rate in baseline_rates.items():
                        print(f"    {cell_type}: {rate:.4f}")
                        
                except Exception as e:
                    print(f"  Error: {str(e)}")
                    results[method]['legacy'] = {
                        'baseline_rates': None,
                        'method_valid': False,
                        'error': str(e)
                    }
            else:  # none
                print("  No normalization applied.")
                results[method]['none'] = {
                    'baseline_rates': None,
                    'method_valid': True
                }
        
        # Summary of window consistency
        print(f"\n=== Window Consistency Analysis ===")
        if f'window_{window_size}' in self.conditional_probs:
            print(f"✅ Conditional probabilities available for window_size={window_size}")
        else:
            available = [k.replace('window_', '') for k in self.conditional_probs.keys()]
            print(f"⚠️  Conditional probabilities NOT available for window_size={window_size}")
            print(f"   Available windows: {available}")
        
        return results

    def create_network_strength_diagram(self, save_path=None, title="Network Connection Strength Diagram", 
                                        notebook=False, overwrite=False, font_size=None):
        """
        Create network connection strength diagram visualization with bidirectional arrows
        
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
        bokeh.plotting.figure or None
            The network diagram plot or None if no data available
        """
        # Check if conditional probabilities data exists
        if not hasattr(self, 'conditional_probs') or self.conditional_probs is None:
            print("Warning: No conditional probabilities data found. Please run 'run_conditional_probabilities_analysis()' first.")
            return None
            
        from bokeh.plotting import figure
        from bokeh.models import Arrow, OpenHead, NormalHead
        import numpy as np
        
        window_key = list(self.conditional_probs.keys())[0]
        probs = self.conditional_probs[window_key]
        
        # Extract only numeric probabilities for cell type connections
        numeric_probs = {}
        for connection, prob in probs.items():
            if (connection.endswith('_to_D1') or connection.endswith('_to_D2') or 
                connection.endswith('_to_CHI')) and isinstance(prob, (int, float, np.integer, np.floating)):
                numeric_probs[connection] = prob
        
        if not numeric_probs:
            return None
        
        p = figure(width=800, height=600,
                    title="Neural Network Connection Strength Diagram (Bidirectional)",
                    tools=['pan', 'wheel_zoom', 'box_zoom', 'reset'])
        
        # Node positions - triangular layout highlighting hierarchical relationships
        positions = {
            'CHI': (0.5, 0.8),    # Top - regulator
            'D1': (0.2, 0.3),     # Bottom left - relay
            'D2': (0.8, 0.3)      # Bottom right - executor
        }
        
        # Draw nodes
        node_colors = {'CHI': 'green', 'D1': 'navy', 'D2': 'crimson'}
        node_sizes = {'CHI': 80, 'D1': 80, 'D2': 80}
        
        # Create all possible bidirectional connections
        cell_types = ['CHI', 'D1', 'D2']
        bidirectional_connections = {}
        
        # Initialize bidirectional connections (including self-activation)
        for source in cell_types:
            for target in cell_types:
                forward_key = f"{source}_to_{target}"
                backward_key = f"{target}_to_{source}"
                
                # Get probabilities for both directions
                forward_prob = numeric_probs.get(forward_key, 0.0)
                backward_prob = numeric_probs.get(backward_key, 0.0)
                
                bidirectional_connections[(source, target)] = {
                    'forward': forward_prob,
                    'backward': backward_prob
                }
        
        # Draw bidirectional arrows
        max_prob = max(numeric_probs.values()) if numeric_probs.values() else 1
        
        # Offset for bidirectional arrows to avoid overlap
        offset_distance = 0.02
        
        for (source, target), probs_dict in bidirectional_connections.items():
            x1, y1 = positions[source]
            x2, y2 = positions[target]
            
            # Handle self-connections (loops)
            if source == target:
                # Draw self-loop
                if probs_dict['forward'] > 0.05:  # Show even weak connections
                    # Line width proportional to probability
                    line_width = max(1, probs_dict['forward'] / max_prob * 12)
                    alpha = 0.3 + (probs_dict['forward'] / max_prob) * 0.7
                    
                    # Use source node color for self-loop
                    source_color = node_colors[source]
                    
                    # Create loop coordinates (small circle above the node)
                    loop_radius = 0.05
                    loop_x = x1
                    loop_y = y1 + loop_radius
                    
                    # Draw loop as a small circle
                    from bokeh.models import Ellipse
                    loop = Ellipse(x=loop_x, y=loop_y, width=loop_radius*2, height=loop_radius*2,
                                    line_color=source_color, line_width=line_width, fill_alpha=0,
                                    line_alpha=alpha)
                    p.add_glyph(loop)
                    
                    # Add probability label for self-loop
                    p.text([loop_x], [loop_y + loop_radius*0.5], [f'{probs_dict["forward"]:.2f}'], 
                            text_align='center', text_baseline='middle',
                            text_font_size='9pt', text_color=source_color,
                            background_fill_color='white', background_fill_alpha=0.8)
                continue
            
            # Calculate offset for bidirectional arrows
            dx = x2 - x1
            dy = y2 - y1
            length = np.sqrt(dx**2 + dy**2)
            
            if length > 0:
                # Normalize and create perpendicular offset
                dx_norm = dx / length
                dy_norm = dy / length
                perp_dx = -dy_norm * offset_distance
                perp_dy = dx_norm * offset_distance
                
                # Forward arrow (source to target)
                if probs_dict['forward'] > 0.05:  # Show even weak connections
                    # Line width proportional to probability
                    line_width = max(1, probs_dict['forward'] / max_prob * 12)
                    alpha = 0.3 + (probs_dict['forward'] / max_prob) * 0.7
                    
                    # Use source node color for forward arrow
                    source_color = node_colors[source]
                    
                    # Draw forward arrow with offset
                    p.line([x1 + perp_dx, x2 + perp_dx], [y1 + perp_dy, y2 + perp_dy], 
                            line_width=line_width, color=source_color, alpha=alpha)
                    
                    # Add arrow head
                    arrow = Arrow(end=OpenHead(size=8, line_color=source_color),
                                x_start=x1 + perp_dx, y_start=y1 + perp_dy,
                                x_end=x2 + perp_dx, y_end=y2 + perp_dy)
                    p.add_layout(arrow)
                    
                    # Add probability label for forward direction
                    mid_x, mid_y = (x1 + x2) / 2 + perp_dx, (y1 + y2) / 2 + perp_dy
                    p.text([mid_x], [mid_y], [f'{probs_dict["forward"]:.2f}'], 
                            text_align='center', text_baseline='middle',
                            text_font_size='9pt', text_color=source_color,
                            background_fill_color='white', background_fill_alpha=0.8)
                
                # Backward arrow (target to source)
                if probs_dict['backward'] > 0.05:  # Show even weak connections
                    # Line width proportional to probability
                    line_width = max(1, probs_dict['backward'] / max_prob * 12)
                    alpha = 0.3 + (probs_dict['backward'] / max_prob) * 0.7
                    
                    # Use target node color for backward arrow (since target is now the source)
                    target_color = node_colors[target]
                    
                    # Draw backward arrow with opposite offset
                    p.line([x2 - perp_dx, x1 - perp_dx], [y2 - perp_dy, y1 - perp_dy], 
                            line_width=line_width, color=target_color, alpha=alpha)
                    
                    # Add arrow head
                    arrow = Arrow(end=OpenHead(size=8, line_color=target_color),
                                x_start=x2 - perp_dx, y_start=y2 - perp_dy,
                                x_end=x1 - perp_dx, y_end=y1 - perp_dy)
                    p.add_layout(arrow)
                    
                    # Add probability label for backward direction
                    mid_x, mid_y = (x1 + x2) / 2 - perp_dx, (y1 + y2) / 2 - perp_dy
                    p.text([mid_x], [mid_y], [f'{probs_dict["backward"]:.2f}'], 
                            text_align='center', text_baseline='middle',
                            text_font_size='9pt', text_color=target_color,
                            background_fill_color='white', background_fill_alpha=0.8)
        
        # Draw nodes on top of arrows
        for node, (x, y) in positions.items():
            p.scatter([x], [y], size=node_sizes[node], color=node_colors[node], 
                    alpha=0.9, line_width=3, line_color='white')
            p.text([x], [y], [node], text_align='center', text_baseline='middle',
                    text_font_size='14pt', text_color='black', text_font_style='bold')
        
        # Add legend
        legend_items = [
            ("CHI Connections", "green"),
            ("D1 Connections", "navy"),
            ("D2 Connections", "crimson"),
            ("Self-activation loops", "gray")
        ]
        
        for i, (label, color) in enumerate(legend_items):
            p.text([0.05], [0.95 - i*0.05], [label], 
                    text_align='left', text_baseline='top',
                    text_font_size='12pt', text_color=color,
                    text_font_style='bold')
        
        p.axis.visible = False
        p.grid.visible = False
        p.x_range.start, p.x_range.end = -0.1, 1.1
        p.y_range.start, p.y_range.end = 0, 1
        
        # Set default save path if none provided
        if save_path is None:
            save_path = os.path.join(self.tank.config["SummaryPlotsPath"], 
                                    self.tank.session_name, 
                                    'network_strength_diagram_bidirectional.html')
        
        # Use tank's output_bokeh_plot method if save_path is provided
        if save_path:
            self.tank.output_bokeh_plot(p, save_path=save_path, title=title, 
                                        notebook=notebook, overwrite=overwrite, font_size=font_size)
        
        return p
    
    def create_activation_cascade_plot(self, save_path=None, title="Activation Cascade Timeline", 
                                    notebook=False, overwrite=False, font_size=None):
        """
        Create activation cascade timeline plot visualization
        
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
        bokeh.plotting.figure or None
            The cascade plot or None if no data available
        """
        # Check if peak timing relationships data exists
        if not hasattr(self, 'peak_timing_relationships') or self.peak_timing_relationships is None:
            print("Warning: No peak timing relationships data found. Please run 'run_peak_timing_analysis()' first.")
            return None
            
        from bokeh.plotting import figure
        import numpy as np
        
        proximities = self.peak_timing_relationships['peak_proximities']
        
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
        
        # Set default save path if none provided
        if save_path is None:
            save_path = os.path.join(self.tank.config["SummaryPlotsPath"], 
                                    self.tank.session_name, 
                                    'activation_cascade_plot.html')
        
        # Use tank's output_bokeh_plot method if save_path is provided
        if save_path:
            self.tank.output_bokeh_plot(p, save_path=save_path, title=title, 
                                        notebook=notebook, overwrite=overwrite, font_size=font_size)
        
        return p
    
    def create_coactivation_pie_chart(self, save_path=None, title="Co-activation Pattern Distribution", 
                                    notebook=False, overwrite=False, font_size=None):
        """
        Create co-activation pattern pie chart visualization
        
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
        bokeh.plotting.figure or None
            The pie chart plot or None if no data available
        """
        # Check if coactivation patterns data exists
        if not hasattr(self, 'coactivation_patterns') or self.coactivation_patterns is None:
            print("Warning: No coactivation patterns data found. Please run 'run_coactivation_patterns_analysis()' first.")
            return None
            
        from bokeh.plotting import figure
        import numpy as np
        
        patterns = self.coactivation_patterns
        
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
                    x_range=(-1.2, 1.2), y_range=(-1.2, 1.2),
                    title="Co-activation Pattern Distribution",
                    tools=['pan', 'wheel_zoom', 'box_zoom', 'reset'])
        
        # Draw pie chart
        start_angle = 0
        for i, (angle, color, name, prop) in enumerate(zip(angles, colors, pattern_names, proportions)):
            end_angle = start_angle + angle
            
            p.wedge(x=0, y=0, radius=0.8, start_angle=start_angle, end_angle=end_angle,
                    color=color, legend_label=f'{name}: {prop:.1%}')
            
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
        
        # Set default save path if none provided
        if save_path is None:
            save_path = os.path.join(self.tank.config["SummaryPlotsPath"], 
                                    self.tank.session_name, 
                                    'coactivation_pie_chart.html')
        
        # Use tank's output_bokeh_plot method if save_path is provided
        if save_path:
            self.tank.output_bokeh_plot(p, save_path=save_path, title=title, 
                                        notebook=notebook, overwrite=overwrite, font_size=font_size)
        
        return p
    
    def create_information_flow_diagram(self, save_path=None, title="Information Flow Strength Analysis", 
                                        notebook=False, overwrite=False, font_size=None):
        """
        Create information flow diagram visualization
        
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
        bokeh.plotting.figure or None
            The information flow plot or None if no data available
        """
        # Check if required data exists
        if not hasattr(self, 'conditional_probs') or self.conditional_probs is None:
            print("Warning: No conditional probabilities data found. Please run 'run_conditional_probabilities_analysis()' first.")
            return None
        if not hasattr(self, 'mutual_information') or self.mutual_information is None:
            print("Warning: No mutual information data found. Please run 'run_mutual_information_analysis()' first.")
            return None
            
        from bokeh.plotting import figure
        from bokeh.models import ColumnDataSource, HoverTool
        import pandas as pd
        import numpy as np
        
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
        window_key = list(self.conditional_probs.keys())[0]
        cond_probs = self.conditional_probs[window_key]
        mutual_info = self.mutual_information
        
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
        
        # Set default save path if none provided
        if save_path is None:
            save_path = os.path.join(self.tank.config["SummaryPlotsPath"], 
                                    self.tank.session_name, 
                                    'information_flow_diagram.html')
        
        # Use tank's output_bokeh_plot method if save_path is provided
        if save_path:
            self.tank.output_bokeh_plot(p, save_path=save_path, title=title, 
                                        notebook=notebook, overwrite=overwrite, font_size=font_size)
        
        return p
    
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
        from bokeh.layouts import column, row
        
        # Create all plots using the independent methods
        p1 = self.create_conditional_probability_heatmap()
        p2 = self.create_network_strength_diagram()
        p3 = self.create_activation_cascade_plot()
        p4 = self.create_coactivation_pie_chart()
        p5 = self.create_information_flow_diagram()
        
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
        
        # Set default save path if none provided
        if save_path is None:
            save_path = os.path.join(self.tank.config["SummaryPlotsPath"], 
                                    self.tank.session_name, 
                                    'Connectivity_Analysis_Summary.html')
        
        # Use tank's output_bokeh_plot method
        self.tank.output_bokeh_plot(layout, save_path=save_path, title=title, 
                                    notebook=notebook, overwrite=overwrite, font_size=font_size)
        
        return layout
        
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

