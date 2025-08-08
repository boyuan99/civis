from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, TapTool, HoverTool, Div, TextInput, Button, Select, CDSView, GroupFilter, \
    Spacer, Slider, ColorBar, LinearColorMapper
from bokeh.layouts import row, column
import numpy as np
import json
import pickle
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
servers_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(servers_dir)
sys.path.append(project_root)

def connection_bkapp_v2(doc):
    from civis.src.CellTypeTank import CellTypeTank
    
    # Store loaded data globally to avoid reloading on slider changes
    loaded_data = {'cell_tank': None, 'connectivity_analyzer': None, 'session_name': None}
    
    def load_data(session_name, time_window, threshold):
        """Load data from file if not already loaded or if session changed"""
        try:
            # Only reload if session name changed or data not loaded
            if loaded_data['session_name'] != session_name or loaded_data['cell_tank'] is None:
                config_path = os.path.join(project_root, 'config.json')
                with open(config_path, 'r') as file:
                    config = json.load(file)
                
                print(f"Loading neuron data {session_name}...")
                cell_tank = CellTypeTank(session_name)
                print(f"Successfully loaded: {session_name}")

                # Create connectivity analyzer
                print("Creating connectivity analyzer...")
                connectivity_analyzer = cell_tank.create_connectivity_analyzer()
                
                # Run binary signals analysis
                print("Running binary signals analysis...")
                connectivity_analyzer.run_binary_signals_analysis(window_size=5)
                
                # Store loaded data
                loaded_data['cell_tank'] = cell_tank
                loaded_data['connectivity_analyzer'] = connectivity_analyzer
                loaded_data['session_name'] = session_name
            
            # Get stored data
            cell_tank = loaded_data['cell_tank']
            connectivity_analyzer = loaded_data['connectivity_analyzer']
            
            # Run conditional probabilities analysis with the specified time window
            print(f"Running conditional probabilities analysis with time window {time_window}...")
            connectivity_analyzer.run_conditional_probabilities_analysis(
                time_windows=[time_window], 
                method='individual'
            )
            
            # Get analysis results
            results = connectivity_analyzer.get_analysis_results()
            
            return cell_tank, connectivity_analyzer, results, time_window, threshold
            
        except FileNotFoundError:
            print('loading failed')
            return f"File Not Found."
        except Exception as e:
            return f"An unexpected error occurred: {str(e)}"

    # Create initial empty plot
    TOOLS = "pan, wheel_zoom, zoom_in, zoom_out, box_zoom, reset, save"
    p = figure(width=800, height=800, tools=TOOLS,
               active_scroll="wheel_zoom", title="Neural Connectivity Based on Activity Probability")
    
    # Adds empty image
    p.image(image=[], x=0, y=0, palette="Greys256")
    
    # Create color mapper for connection strength with more colors and range 0-0.5
    color_mapper = LinearColorMapper(palette=["lightblue", "blue", "cyan", "green", "yellow", "orange", "red"], low=0, high=0.5)

    source = ColumnDataSource(data=dict(x=[], y=[], ids=[], categories=[], colors=[]))
    lines_source = ColumnDataSource(data=dict(xs=[], ys=[], colors=[], strengths=[]))

    # Initialize widgets
    neuron_path_input = TextInput(value="", title="Session Name:")
    time_window_slider = Slider(start=5, end=100, value=20, step=5, 
                               title="Time Window (samples):", width=300)
    threshold_slider = Slider(start=0.0, end=1.0, value=0.1, step=0.05,
                             title="Connection Threshold:", width=300)
    load_button = Button(label="Load Data", button_type="success")
    details_div = Div(width=300, height=800, sizing_mode="fixed", 
                     text="Details will appear here after loading")
    
    category_select = Select(title="Neuron Category:", value="All", options=["All"])

    view = CDSView(filter=GroupFilter(column_name='categories', group='All'))
    
    # Store current visualization data
    viz_data = {
        'prob_matrix': None,
        'neuron_labels': None,
        'neuron_positions': {},
        'neuron_colors': {},
        'neuron_categories': {},
        'cell_types': None,
        'time_window': None,
        'threshold': None
    }
    
    def update_data(load_new=False):
        """Update visualization with current parameters"""
        session_name = neuron_path_input.value
        time_window = time_window_slider.value
        threshold = threshold_slider.value
        
        # Don't proceed if no session name
        if not session_name:
            details_div.text = "Please enter a session name"
            return
        
        # Only load data if needed (new session or forced reload)
        if load_new or loaded_data['session_name'] != session_name:
            data = load_data(session_name, time_window, threshold)
        elif loaded_data['cell_tank'] is not None:
            # Recompute with new time window if data already loaded
            data = load_data(session_name, time_window, threshold)
        else:
            details_div.text = "Please load data first"
            return
            
        if isinstance(data, str):
            details_div.text = data
            return
        
        cell_tank, connectivity_analyzer, results, time_window, threshold = data
        
        # Get conditional probabilities
        conditional_probs = results['conditional_probs']
        if not conditional_probs:
            details_div.text = "No conditional probability data available"
            return
            
        # Get the probability matrix for the specified time window
        window_key = f'window_{time_window}'
        if window_key not in conditional_probs:
            details_div.text = f"No data for time window {time_window}"
            return
            
        prob_data = conditional_probs[window_key]
        prob_matrix = prob_data['probability_matrix']
        neuron_labels = prob_data['neuron_labels']
        cell_types = prob_data['cell_types']
        
        # Store visualization data
        viz_data['prob_matrix'] = prob_matrix
        viz_data['neuron_labels'] = neuron_labels
        viz_data['cell_types'] = cell_types
        viz_data['time_window'] = time_window
        viz_data['threshold'] = threshold
        
        print(f"Probability matrix shape: {prob_matrix.shape}")
        print(f"Number of neuron labels: {len(neuron_labels)}")
        print(f"Number of cell types: {len(cell_types)}")
        
        # Get neuron positions from cell_tank
        height, width = cell_tank.Cn.shape
        centroids_flipped = np.copy(cell_tank.centroids)
        centroids_flipped[:, 1] = height - cell_tank.centroids[:, 1]
        
        # Create color mapping for cell types
        cell_type_colors = {
            'D1': '#F44336',  # Red
            'D2': '#2196F3',  # Blue  
            'CHI': '#4CAF50'   # Green
        }
        
        # Map neuron labels to positions and colors
        neuron_positions = {}
        neuron_colors = {}
        neuron_categories = {}
        
        print(f"Mapping {len(neuron_labels)} neurons to positions...")
        print(f"Cell tank has {len(cell_tank.ids)} IDs, centroids shape: {centroids_flipped.shape}")
        
        # Get the actual neuron indices for each cell type from cell_tank
        d1_indices = cell_tank.d1_indices
        d2_indices = cell_tank.d2_indices
        chi_indices = cell_tank.chi_indices
        
        print(f"D1 indices: {len(d1_indices)}, D2 indices: {len(d2_indices)}, CHI indices: {len(chi_indices)}")
        
        for i, label in enumerate(neuron_labels):
            cell_type = cell_types[i]
            
            # Extract neuron index from label (e.g., "D1_5" -> 5)
            neuron_idx = int(label.split('_')[1])
            
            # Map to actual neuron position based on cell type
            if cell_type == 'D1' and neuron_idx < len(d1_indices):
                actual_idx = d1_indices[neuron_idx]
            elif cell_type == 'D2' and neuron_idx < len(d2_indices):
                actual_idx = d2_indices[neuron_idx]
            elif cell_type == 'CHI' and neuron_idx < len(chi_indices):
                actual_idx = chi_indices[neuron_idx]
            else:
                print(f"Warning: Invalid mapping for {label} (cell_type: {cell_type}, neuron_idx: {neuron_idx})")
                continue
            
            # Convert to integer and check bounds
            actual_idx = int(actual_idx)
            if 0 <= actual_idx < len(centroids_flipped):
                neuron_positions[label] = centroids_flipped[actual_idx]
                neuron_colors[label] = cell_type_colors[cell_type]
                neuron_categories[label] = cell_type
            else:
                print(f"Warning: actual_idx {actual_idx} out of bounds for {label}")
        
        # Store neuron data for reuse
        viz_data['neuron_positions'] = neuron_positions
        viz_data['neuron_colors'] = neuron_colors
        viz_data['neuron_categories'] = neuron_categories
        
        print(f"Successfully mapped {len(neuron_positions)} neurons")
        
        # Update category select options
        unique_categories = list(set(neuron_categories.values()))
        category_select.options = ["All"] + sorted(unique_categories)
        
        # Prepare source data
        x_coords = []
        y_coords = []
        ids = []
        colors = []
        categories = []
        
        for label, pos in neuron_positions.items():
            x_coords.append(pos[0])
            y_coords.append(pos[1])
            ids.append(label)
            colors.append(neuron_colors[label])
            categories.append(neuron_categories[label])
        
        source.data = dict(
            x=x_coords,
            y=y_coords,
            ids=ids,
            colors=colors,
            categories=categories
        )
        
        # Update plot ranges
        p.x_range.start = 0
        p.x_range.end = width
        p.y_range.start = 0
        p.y_range.end = height
        
        # Update image
        p.image(image=[np.flipud(cell_tank.Cn)], x=0, y=0, dw=width, dh=height, palette="Greys256")
        
        # Add centroids with selection highlighting
        circle_renderer = p.scatter(x='x', y='y', source=source, size=10, fill_color='colors',
                                    line_color='colors', alpha=0.7,
                                    nonselection_alpha=0.7)
        
        # Create selection glyph with larger size and brighter colors
        selection_glyph = circle_renderer.glyph.clone()
        selection_glyph.size = 15  # Larger size for selected neuron
        selection_glyph.fill_alpha = 1.0  # Full opacity for selected neuron
        selection_glyph.line_alpha = 1.0  # Full opacity for line
        
        # Create non-selection glyph with normal appearance
        nonselection_glyph = circle_renderer.glyph.clone()
        nonselection_glyph.size = 10  # Normal size
        nonselection_glyph.fill_alpha = 0.7  # Normal opacity
        nonselection_glyph.line_alpha = 0.7  # Normal opacity
        
        circle_renderer.selection_glyph = selection_glyph
        circle_renderer.nonselection_glyph = nonselection_glyph
        
        # Add lines (for connections) with thicker lines
        p.multi_line(xs="xs", ys="ys", color="colors", line_width="strengths", 
                    alpha=0.8, source=lines_source, line_cap="round", line_join="round")
        
        # HoverTool with more information
        hover = HoverTool(renderers=[circle_renderer], 
                         tooltips=[
                             ("Neuron ID", "@ids"), 
                             ("Cell Type", "@categories"),
                             ("Position", "(@x{0.0}, @y{0.0})")
                         ])
        p.add_tools(hover)
        
        # TapTool
        tap_tool = TapTool(renderers=[circle_renderer])
        p.add_tools(tap_tool)
        
        # Add color bar for connection strength (only if not already added)
        # Check if color bar already exists in the plot
        has_colorbar = any(isinstance(renderer, ColorBar) for renderer in p.right)
        if not has_colorbar:
            color_bar = ColorBar(color_mapper=color_mapper, 
                               label_standoff=12, 
                               border_line_color=None, 
                               location=(0, 0),
                               title="Connection Strength",
                               title_text_font_size="12pt",
                               title_standoff=20)
            p.add_layout(color_bar, 'right')
        
        # Update lines if a neuron is already selected
        if source.selected.indices:
            update_lines_from_selection()
    
    def update_lines_from_selection():
        """Update connection lines based on current selection and threshold"""
        selected_indices = source.selected.indices
        threshold = threshold_slider.value
        
        if not selected_indices or viz_data['prob_matrix'] is None:
            lines_source.data = {'xs': [], 'ys': [], 'colors': [], 'strengths': []}
            if not selected_indices:
                details_div.text = "No neuron selected"
            return
        
        selected_index = selected_indices[0]
        selected_label = source.data['ids'][selected_index]
        
        # Find the index of selected neuron in probability matrix
        if selected_label in viz_data['neuron_labels']:
            selected_prob_idx = viz_data['neuron_labels'].index(selected_label)
            
            new_xs, new_ys, new_colors, new_strengths = [], [], [], []
            connection_details = []
            
            for i, prob in enumerate(viz_data['prob_matrix'][selected_prob_idx]):
                if i != selected_prob_idx and prob >= threshold:  # Only show connections above threshold
                    target_label = viz_data['neuron_labels'][i]
                    
                    if target_label in viz_data['neuron_positions']:
                        x0, y0 = viz_data['neuron_positions'][selected_label]
                        x1, y1 = viz_data['neuron_positions'][target_label]
                        
                        new_xs.append([x0, x1])
                        new_ys.append([y0, y1])
                        
                        # Color based on connection strength using color mapper (range 0-0.5)
                        # Clamp probability to 0.5 max for better visualization
                        clamped_prob = min(prob, 0.5)
                        color_idx = int(clamped_prob / 0.5 * (len(color_mapper.palette) - 1))
                        color = color_mapper.palette[color_idx]
                        
                        new_colors.append(color)
                        new_strengths.append(max(3, clamped_prob * 16))  # Scale line width for 0-0.5 range
                        
                        connection_details.append(f"{target_label}: {prob:.3f}")
            
            lines_source.data = {
                'xs': new_xs, 
                'ys': new_ys, 
                'colors': new_colors, 
                'strengths': new_strengths
            }
            
            details = f"Selected Neuron: {selected_label}<br>" \
                     f"Category: {source.data['categories'][selected_index]}<br>" \
                     f"Time Window: {viz_data['time_window']} samples<br>" \
                     f"Threshold: {threshold}<br>" \
                     f"Connections: {len(connection_details)}<br>" \
                     f"Connection Details:<br>" + "<br>".join(connection_details)
            details_div.text = details
    
    def update_display_based_on_category(attr, old, new):
        selected_category = category_select.value
        if selected_category == "All":
            source.data['all'] = ['all'] * len(source.data['x'])
            view.filter = GroupFilter(column_name='all', group='all')
        else:
            view.filter = GroupFilter(column_name='categories', group=selected_category)
    
    def selection_change_callback(attr, old, new):
        """Callback for when neuron selection changes"""
        update_lines_from_selection()
    
    def threshold_change_callback(attr, old, new):
        """Callback for when threshold slider changes"""
        update_lines_from_selection()
    
    def time_window_change_callback(attr, old, new):
        """Callback for when time window slider changes"""
        if loaded_data['cell_tank'] is not None:
            update_data(load_new=False)
    
    # Connect callbacks
    category_select.on_change('value', update_display_based_on_category)
    source.selected.on_change('indices', selection_change_callback)
    threshold_slider.on_change('value', threshold_change_callback)
    time_window_slider.on_change('value', time_window_change_callback)
    
    # Format graph and load button
    load_button.on_click(lambda: update_data(load_new=True))
    
    controls = column(
        row(neuron_path_input, column(Spacer(height=20), load_button)),
        time_window_slider,
        threshold_slider
    )
    layout = row(p, column(controls, column(category_select, details_div)))
    doc.add_root(layout)

connection_bkapp_v2(curdoc())