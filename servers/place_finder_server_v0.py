import numpy as np
import json
import os
import sys
import pickle

from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, Slider, Button, BoxAnnotation, Span, TextInput, Spacer
from bokeh.layouts import column, row

current_dir = os.path.dirname(__file__)
parent_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.insert(0, parent_dir)


def place_cell_vis(doc):
    from src import StraightMazeTank

    global source, ci, session_name, peak_indices

    plot = figure(width=300, height=800, y_range=[-30, 30], x_range=[-10, 10],
                  title="Mouse Movement Trajectory")


    source = ColumnDataSource(data=dict(x=[], y=[]))
    plot.circle(x="x", y="y", source=source, size=5, line_color=None)  # , fill_alpha='alpha')

    # Annotations
    high_box = BoxAnnotation(bottom=25, fill_alpha=0.5, fill_color='blue')
    plot.add_layout(high_box)
    low_box = BoxAnnotation(top=-25, fill_alpha=0.5, fill_color='blue')
    plot.add_layout(low_box)
    vline0 = Span(location=-9, dimension='height', line_color='black', line_width=2)
    plot.add_layout(vline0)
    vline1 = Span(location=9, dimension='height', line_color='black', line_width=2)
    plot.add_layout(vline1)

    # Widgets
    neuron_id_slider = Slider(start=0, end=100, value=0, step=1, width=600, title="Neuron ID", disabled=True)
    session_input = TextInput(value='', title="Session Name:", width=400)
    load_button = Button(label="Load Data", button_type="success")
    previous_button = Button(label="Previous", width=100, disabled=True)
    next_button = Button(label="Next", width=100, disabled=True)
    neuron_index_input = TextInput(value=str(neuron_id_slider.value), title="Neuron Index:", disabled=True)

    def load_data():
        global session_name, x_pos_all, y_pos_all, peak_indices, source, ci

        session_name = session_input.value
        with open('config.json', 'r') as file:
            config = json.load(file)

        # load in the DataTank
        neuron_path = config['ProcessedFilePath'] + session_name + '/' + session_name + '_v7.mat'
        ci = StraightMazeTank(neuron_path, threshold=[25, -25], height=4)
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

        neuron_id_slider.end = ci.neuron_num - 1
        neuron_id_slider.value = 0
        neuron_index_input.value = "0"

        [x_pos_all, y_pos_all] = find_places(ci, peak_indices)

        new_data = {"x": x_pos_all[0],
                    "y": y_pos_all[0]}

        source.data = new_data


    load_button.on_click(load_data)

    def find_places(ci, peak_indices):
        x_pos_all = []
        y_pos_all = []
        x_pos = np.array(ci.virmen_data["x"])[:ci.ci_rate * ci.session_duration]
        y_pos = np.array(ci.virmen_data["y"])[:ci.ci_rate * ci.session_duration]

        for i in range(ci.neuron_num):
            x_pos_all.append(x_pos[peak_indices[i]])
            y_pos_all.append(y_pos[peak_indices[i]])

        return x_pos_all, y_pos_all


    def update_plot(attr, old, new):
        global peak_indices, source, x_pos_all, y_pos_all
        selected_neuron = neuron_id_slider.value
        neuron_index_input.value = str(selected_neuron)

        new_data = {"x": x_pos_all[selected_neuron],
                    "y": y_pos_all[selected_neuron]}

        source.data = new_data


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
    layout = row(plot, Spacer(width=30), tool_widgets)
    doc.add_root(layout)


# place_cell_vis(curdoc())