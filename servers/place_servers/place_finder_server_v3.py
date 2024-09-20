import numpy as np
import json
import os
import sys
import pickle

from bokeh.plotting import figure, curdoc
from bokeh.models import (ColumnDataSource, Slider, Button, BoxAnnotation, Span, TextInput, TabPanel, Tabs,
                          Spacer, LinearColorMapper, ColorBar)
from bokeh.palettes import Turbo256
from bokeh.layouts import column, row

current_dir = os.path.dirname(os.path.abspath(__file__))
servers_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(servers_dir)
sys.path.append(project_root)


def get_data(ciTank, peak_indices_par, neuron_id, normalization_method='percentile'):
    from scipy.ndimage import gaussian_filter
    import numpy as np

    # Get x and y coordinates for all timepoints
    x_all = ciTank.virmen_data['x']
    y_all = ciTank.virmen_data['y']

    # Create occupancy map
    occupancy, xedges, yedges = np.histogram2d(x_all, y_all, bins=(67, 140), range=[[-67.5, 67.5], [-100, 180]])
    occupancy = occupancy.T  # Transpose to match the orientation of our heatmap

    # Get x and y coordinates for spike events
    x_spikes = ciTank.virmen_data.iloc[peak_indices_par[neuron_id]]['x']
    y_spikes = ciTank.virmen_data.iloc[peak_indices_par[neuron_id]]['y']

    # Create spike count map
    spike_counts, _, _ = np.histogram2d(x_spikes, y_spikes, bins=(67, 140), range=[[-67.5, 67.5], [-100, 180]])
    spike_counts = spike_counts.T  # Transpose to match the orientation of our heatmap

    # Calculate firing rate map (spikes per second)
    time_spent = occupancy / ciTank.vm_rate  # Convert frame counts to time (assuming vm_rate is in Hz)
    firing_rate = np.divide(spike_counts, time_spent, where=time_spent!=0)  # Avoid division by zero
    firing_rate[time_spent == 0] = 0  # Set rate to 0 where no time was spent

    # Apply Gaussian smoothing
    smoothed_rate = gaussian_filter(firing_rate, sigma=1)

    if normalization_method == 'percentile':
        # Percentile-based normalization
        p_low, p_high = np.percentile(smoothed_rate[smoothed_rate > 0], [2, 98])
        normalized_rate = np.clip(smoothed_rate, p_low, p_high)
        normalized_rate = (normalized_rate - p_low) / (p_high - p_low)
    elif normalization_method == 'log':
        # Logarithmic scaling
        epsilon = 1e-10  # Small constant to avoid log(0)
        log_rate = np.log(smoothed_rate + epsilon)
        normalized_rate = (log_rate - np.min(log_rate)) / (np.max(log_rate) - np.min(log_rate))
    else:
        # Simple min-max normalization
        normalized_rate = (smoothed_rate - np.min(smoothed_rate)) / (np.max(smoothed_rate) - np.min(smoothed_rate))

    return normalized_rate


def place_cell_vis_bkapp_v3(doc):
    from src import CITank

    global session_name, peak_indices, ci, remain_trial_indices, trials, data, \
        peak_trial_source, peak_point_source, remain_trial_source, heatmap_source

    # First tab plot (new version)
    plot_t1 = figure(width=550, height=800, y_range=(-100, 180), title="Mouse Movement Trajectory")

    # ColumnDataSources for different parts of the plot_t1
    peak_trial_source = ColumnDataSource(data={'x': [], 'y': []})
    peak_point_source = ColumnDataSource(data={'px': [], 'py': []})
    remain_trial_source = ColumnDataSource(data={'rx': [], 'ry': []})

    # Drawing the lines and points
    plot_t1.multi_line(xs='x', ys='y', source=peak_trial_source, line_width=2, color='red', alpha=0.5)
    plot_t1.circle(x='px', y='py', source=peak_point_source, size=7, color='red', alpha=1)
    plot_t1.multi_line(xs='rx', ys='ry', source=remain_trial_source, line_width=2, color='blue', alpha=0.3)

    # Adding maze outline
    xpts = np.array([-67.5, -50, 0, 50, 67.5, 10, 10, -10, -10, -10, -67.5])
    ypts = np.array([107.5, 125, 75, 125, 107.5, 50, -65, -65, -50, 50, 107.5])
    source_maze = ColumnDataSource(dict(xs=[xpts], ys=[ypts]))
    plot_t1.multi_line(xs="xs", ys="ys", source=source_maze, line_color="#8073ac", line_width=2)

    image_tab1 = TabPanel(child=plot_t1, title="Trajectory")

    # Second tab plot (heatmap)
    plot_t2 = figure(width=550, height=800, y_range=(-100, 180), x_range=(-67.5, 67.5), title="Firing Places")
    color_mapper = LinearColorMapper(palette=Turbo256[16:], low=0, high=1)
    heatmap_source = ColumnDataSource(data={'image': []})
    plot_t2.image(image='image', x=-67.5, y=-100, dw=135, dh=280, color_mapper=color_mapper, source=heatmap_source)
    color_bar = ColorBar(color_mapper=color_mapper, width=8, location=(0, 0))
    plot_t2.add_layout(color_bar, 'right')
    image_tab2 = TabPanel(child=plot_t2, title="Heatmap")

    # Widgets
    neuron_id_slider = Slider(start=0, end=100, value=0, step=1, width=600, title="Neuron ID", disabled=True)
    session_input = TextInput(value='', title="Session Name:", width=400)
    load_button = Button(label="Load Data", button_type="success")
    previous_button = Button(label="Previous", width=100, disabled=True)
    next_button = Button(label="Next", width=100, disabled=True)
    neuron_index_input = TextInput(value=str(neuron_id_slider.value), title="Neuron Index:", disabled=True)

    def load_data():
        global session_name, peak_indices, ci, remain_trial_indices, trial_indices, trials, data, \
            peak_trial_source, peak_point_source, remain_trial_source

        session_name = session_input.value
        config_path = os.path.join(project_root, 'config.json')
        with open(config_path, 'r') as file:
            config = json.load(file)

        # load in the DataTank
        neuron_path = os.path.join(config['ProcessedFilePath'], session_name, f'{session_name}_v7.mat')
        virmen_path = os.path.join(config['VirmenFilePath'], f'{session_name}.txt')
        print("Loading neuron data " + session_name + "...")
        ci = CITank(neuron_path, virmen_path, height=4)
        print("Successfully loaded: " + neuron_path)

        neuron_id_slider.disabled = False
        previous_button.disabled = False
        next_button.disabled = False
        neuron_index_input.disabled = False

        # load in the peak indices
        peak_indices_path = config['ProcessedFilePath'] + session_name + "/" + session_name + "_peak_indices.pkl"
        if os.path.exists(peak_indices_path):
            print("Found peak indices file, loading...")
            with open(peak_indices_path, 'rb') as f:
                peak_indices = pickle.load(f)
            print("Loaded peak indices!")
        else:
            print("Peak indices file not found, calculating...")
            peak_indices = ci.find_peaks_in_traces(notebook=False)
            with open(peak_indices_path, 'wb') as f:
                pickle.dump(peak_indices, f)
            print("Saved peak indices!")

        print("Waiting for finalizing visualization...")
        neuron_id_slider.end = ci.neuron_num - 1
        neuron_id_slider.value = 0
        neuron_index_input.value = "0"

        trials = ci.virmen_trials
        data = ci.virmen_data

        trial_bounds = ci.compute_trial_bounds()
        trial_indices = []
        for indices in peak_indices:
            trial_index = ci.find_trial_for_indices(trial_bounds, indices)
            trial_indices.append(trial_index)

        remain_trial_indices = []
        for i in range(ci.neuron_num):
            remain_trials_index = [item for item in np.array(range(0, len(trials))) if
                                   item not in list(trial_indices[i].values())]
            remain_trial_indices.append(remain_trials_index)

        update_plot(None, None, None)

        print("Visualization loaded!")
        print("=================================================")

    load_button.on_click(load_data)

    def update_plot(attr, old, new):
        global session_name, peak_indices, ci, remain_trial_indices, trial_indices, trials, data, \
            peak_trial_source, peak_point_source, remain_trial_source, heatmap_source

        picked_neuron = neuron_id_slider.value
        neuron_index_input.value = str(picked_neuron)
        trial_lines = {'x': [], 'y': []}
        peak_points = {'px': [], 'py': []}
        remain_trial_lines = {'rx': [], 'ry': []}

        # Populate data for peak trials and points
        for i in range(len(trial_indices[picked_neuron])):
            trial_lines['x'].append(
                trials[list(trial_indices[picked_neuron].values())[i]]['x'])  # Ensure conversion to list if necessary
            trial_lines['y'].append(trials[list(trial_indices[picked_neuron].values())[i]]['y'])
            peak_points['px'].append(data.iloc[peak_indices[picked_neuron][i]]['x'])
            peak_points['py'].append(data.iloc[peak_indices[picked_neuron][i]]['y'])

        # Populate data for remaining trials
        for i in remain_trial_indices[picked_neuron]:
            remain_trial_lines['rx'].append(trials[i]['x'])
            remain_trial_lines['ry'].append(trials[i]['y'])

        heatmap_source.data['image'] = [get_data(ci, peak_indices, picked_neuron)]

        peak_trial_source.data = trial_lines
        peak_point_source.data = peak_points
        remain_trial_source.data = remain_trial_lines

    neuron_id_slider.on_change('value', update_plot)

    def previous_trial():
        if neuron_id_slider.value > neuron_id_slider.start:
            neuron_id_slider.value -= 1
            neuron_index_input.value = str(neuron_id_slider.value)

    def next_trial():
        if neuron_id_slider.value < neuron_id_slider.end:
            neuron_id_slider.value += 1
            neuron_index_input.value = str(neuron_id_slider.value)

    previous_button.on_click(previous_trial)
    next_button.on_click(next_trial)

    def update_index(attr, old, new):
        global ci
        try:
            new_index = int(new)
            # Ensure the new index is within the valid range
            if np.min(ci.ids) <= new_index <= np.max(ci.ids):
                neuron_id_slider.value = new_index
            else:
                print("Input index out of range")
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

    # Attach the callback function to the TextInput widget
    neuron_index_input.on_change('value', update_index)

    file_input_row = row(session_input, column(Spacer(height=20), load_button))
    trial_navigation_row = row(previous_button, next_button)
    tool_widgets = column(file_input_row, Spacer(height=30), neuron_index_input, neuron_id_slider, trial_navigation_row)
    images = Tabs(tabs=[image_tab1, image_tab2])
    layout = row(images, Spacer(width=30), tool_widgets)
    doc.add_root(layout)


# Uncomment the following line to run the Bokeh server application
place_cell_vis_bkapp_v3(curdoc())
