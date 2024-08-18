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

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)


def get_data(ciTank, peak_indices_par, neuron_id):
    from scipy.ndimage import gaussian_filter

    x = ciTank.virmen_data.iloc[peak_indices_par[neuron_id]]['x']
    y = ciTank.virmen_data.iloc[peak_indices_par[neuron_id]]['y']
    hist, xedges, yedges = np.histogram2d(x, y, bins=(20, 100), range=[[-11, 11], [-51, 51]])
    hist_img = np.transpose(hist)
    smoothed_image = gaussian_filter(hist_img, sigma=1)
    return smoothed_image


def place_cell_vis_bkapp_v2(doc):
    from src import CITank

    global session_name, peak_indices, ci, remain_trial_indices, trials, data, \
        peak_trial_source, peak_point_source, remain_trial_source, heatmap_source

    plot_t1 = figure(width=300, height=800, y_range=[-60, 60], x_range=[-10, 10], title="Firing Places")

    # ColumnDataSources for different parts of the plot_t1
    peak_trial_source = ColumnDataSource(data={'x': [], 'y': []})
    peak_point_source = ColumnDataSource(data={'px': [], 'py': []})
    remain_trial_source = ColumnDataSource(data={'rx': [], 'ry': []})
    heatmap_source = ColumnDataSource(data={'image': []})

    # Drawing the lines and points
    plot_t1.multi_line(xs='x', ys='y', source=peak_trial_source, line_width=2, color='red', alpha=0.5)
    plot_t1.circle(x='px', y='py', source=peak_point_source, size=7, color='red', alpha=1)
    plot_t1.multi_line(xs='rx', ys='ry', source=remain_trial_source, line_width=2, color='blue', alpha=0.3)

    # Annotations and spans
    high_box = BoxAnnotation(bottom=50, fill_alpha=0.5, fill_color='blue')
    plot_t1.add_layout(high_box)
    low_box = BoxAnnotation(top=-50, fill_alpha=0.5, fill_color='blue')
    plot_t1.add_layout(low_box)
    vline0 = Span(location=-9, dimension='height', line_color='black', line_width=2)
    plot_t1.add_layout(vline0)
    vline1 = Span(location=9, dimension='height', line_color='black', line_width=2)
    plot_t1.add_layout(vline1)
    image_tab1 = TabPanel(child=plot_t1, title="Trajectory")

    plot_t2 = figure(width=300, height=800, y_range=[-51, 51], x_range=[-11, 11],  title="Firing Places")
    color_mapper = LinearColorMapper(palette=Turbo256[16:], low=0, high=1)
    plot_t2.image(image='image', x=-11, y=-51, dw=22, dh=102, color_mapper=color_mapper, source=heatmap_source)
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
        with open('config.json', 'r') as file:
            config = json.load(file)

        # load in the DataTank
        neuron_path = config['ProcessedFilePath'] + session_name + '/' + session_name + '_v7.mat'
        virmen_path = config['VirmenFilePath'] + session_name + ".txt"
        print("Loading neuron data " + session_name + "...")
        ci = CITank(neuron_path, virmen_path, maze_type="straight50", height=4)
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

        trial_bounds = ci.compute_trial_bounds(ci.virmen_data, threshold=[50, -50])
        trial_indices = []
        for indices in peak_indices:
            trial_index = ci.find_trial_for_indices(trial_bounds, indices)
            trial_indices.append(trial_index)

        remain_trial_indices = []
        for i in range(ci.neuron_num):
            remain_trials_index = [item for item in np.array(range(0, len(trials))) if
                                   item not in list(trial_indices[i].values())]
            remain_trial_indices.append(remain_trials_index)

        trial_lines = {'x': [], 'y': []}
        peak_points = {'px': [], 'py': []}
        remain_trial_lines = {'rx': [], 'ry': []}
        # Populate data for peak trials and points
        for i in range(len(trial_indices[0])):
            trial_lines['x'].append(
                trials[list(trial_indices[0].values())[i]]['x'])  # Ensure conversion to list if necessary
            trial_lines['y'].append(trials[list(trial_indices[0].values())[i]]['y'])
            peak_points['px'].append(data.iloc[peak_indices[0][i]]['x'])
            peak_points['py'].append(data.iloc[peak_indices[0][i]]['y'])

        # Populate data for remaining trials
        for i in remain_trial_indices[0]:
            remain_trial_lines['rx'].append(trials[i]['x'])
            remain_trial_lines['ry'].append(trials[i]['y'])

        peak_trial_source.data = trial_lines
        peak_point_source.data = peak_points
        remain_trial_source.data = remain_trial_lines

        heatmap_source.data['image'] = [get_data(ci, peak_indices, 0)]

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

# place_cell_vis_bkapp_v1(curdoc())
