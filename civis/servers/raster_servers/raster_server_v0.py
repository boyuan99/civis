from bokeh.models import ColumnDataSource, TextInput, Button, BoxSelectTool, Spacer, Div
from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.events import SelectionGeometry, Reset
import pickle
import numpy as np
import json
import os


def raster_bkapp_v0(doc):
    from civis.src.CITank import CITank
    def load_data(session_name):
        with open('config.json', 'r') as file:
            config = json.load(file)

        neuron_path = config['ProcessedFilePath'] + session_name + '/' + session_name + '_v7.mat'
        peak_indices_path = config['ProcessedFilePath'] + session_name + "/" + session_name + "_peak_indices.pkl"
        virmen_path = config['VirmenFilePath'] + session_name + ".txt"

        ci = CITank(neuron_path, virmen_path)
        print("Successfully loaded: " + neuron_path)

        # load in the peak indices
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

        spike_times = peak_indices
        spike_stats = ci.get_spike_statistics(peak_indices)

        data = {'x_starts': [], 'y_starts': [], 'x_ends': [], 'y_ends': []}
        for neuron_idx, spikes in enumerate(spike_times):
            for spike_time in spikes:
                data['x_starts'].append(spike_time / ci.ci_rate)
                data['y_starts'].append(neuron_idx)
                data['x_ends'].append(spike_time / ci.ci_rate)
                data['y_ends'].append(neuron_idx + 0.7)

        return data, ci, spike_stats

    # Create initial empty plot
    raster_source = ColumnDataSource({'x_starts': [], 'y_starts': [], 'x_ends': [], 'y_ends': []})
    selected_raster_source = ColumnDataSource({'x_starts': [], 'y_starts': [], 'x_ends': [], 'y_ends': []})
    line_source = ColumnDataSource({'x': [], 'velocity': [], 'lick': [], 'pstcr': [], 'spike_stats': []})
    selected_line_source = ColumnDataSource({'x': [], 'velocity': [], 'lick': [], 'pstcr': [], 'spike_stats': []})

    # Create shared x-range for all plots
    shared_x_range = figure().x_range

    # Raster plot

    p = figure(width=1000, height=1000, title="Raster Plot", x_axis_label='Time (s)', y_axis_label='Neuron',
               x_range=shared_x_range, active_scroll='wheel_zoom', min_border_left=100)
    p.segment(x0='x_starts', y0='y_starts', x1='x_ends', y1='y_ends', source=raster_source, color="black", alpha=1,
              line_width=2)
    p.segment(x0='x_starts', y0='y_starts', x1='x_ends', y1='y_ends', source=selected_raster_source, color="red",
              alpha=1, line_width=2)

    # Signal plot
    v = figure(width=1000, height=200, x_range=shared_x_range, active_scroll='wheel_zoom',
               tools=['pan', 'wheel_zoom', 'reset'], min_border_left=100)
    v.line('x', 'velocity', source=line_source, color='SteelBlue', legend_label='velocity', alpha=1)
    v.line('x', 'lick', source=line_source, color='SandyBrown', legend_label='lick', alpha=1)
    v.line('x', 'pstcr', source=line_source, color='Crimson', legend_label='pstcr', alpha=1)
    v.line('x', 'velocity', source=selected_line_source, color='navy', legend_label='velocity', alpha=1)
    v.line('x', 'lick', source=selected_line_source, color='gold', legend_label='lick', alpha=1)
    v.line('x', 'pstcr', source=selected_line_source, color='DarkRed', legend_label='pstcr', alpha=1)
    v.legend.click_policy = 'hide'

    # Spike stats plot
    s = figure(width=1000, height=200, x_range=shared_x_range, active_scroll='wheel_zoom', min_border_left=100)
    s.line('x', 'spike_stats', source=line_source, color='Chocolate', alpha=1, legend_label='spikes count')
    s.line('x', 'spike_stats', source=selected_line_source, color='Sienna', alpha=1, legend_label='spikes count')
    s.legend.click_policy = 'hide'

    box_select_p = BoxSelectTool(dimensions="width")
    p.add_tools(box_select_p)
    # p.toolbar.active_drag = box_select_p
    v.add_tools(box_select_p)
    # v.toolbar.active_drag = box_select_p

    # Text input widget for session name
    session_input = TextInput(value="", title="Session Name:")

    # Button to load data
    load_button = Button(label="Load Data", button_type="success")

    # Div to show the picked neurons
    neurons_div = Div(width=1000, height=400, text="Neurons will be shown after you choose an interval.")

    def update_data():
        print("Loading Data...")
        session_name = session_input.value
        data, ci, spike_stats = load_data(session_name)
        raster_source.data = data
        p.yaxis.ticker = np.arange(0, ci.neuron_num)
        p.yaxis.major_label_overrides = {i: f"Neuron {i}" for i in range(ci.neuron_num)}
        line_source.data = dict(x=ci.t,
                                velocity=ci.normalize_signal(ci.velocity),
                                lick=ci.lick,
                                pstcr=ci.pstcr,
                                spike_stats=spike_stats)

    load_button.on_click(update_data)

    def selection_handler(event):
        geometry = event.geometry
        x0, x1 = geometry['x0'], geometry['x1']

        # Filter the data based on the selection
        filtered_data = {'x_starts': [], 'y_starts': [], 'x_ends': [], 'y_ends': []}
        for x_start, y_start, x_end, y_end in zip(raster_source.data['x_starts'],
                                                  raster_source.data['y_starts'],
                                                  raster_source.data['x_ends'],
                                                  raster_source.data['y_ends']):
            if x0 <= x_start <= x1:
                filtered_data['x_starts'].append(x_start)
                filtered_data['y_starts'].append(y_start)
                filtered_data['x_ends'].append(x_end)
                filtered_data['y_ends'].append(y_end)

        selected_raster_source.data = filtered_data

        filtered_data_x = []
        filtered_data_velocity = []
        filtered_data_lick = []
        filtered_data_pstcr = []
        filtered_data_spikes = []

        for i, x_val in enumerate(line_source.data['x']):
            if x0 <= x_val <= x1:
                filtered_data_x.append(x_val)
                filtered_data_velocity.append(line_source.data['velocity'][i])
                filtered_data_lick.append(line_source.data['lick'][i])
                filtered_data_pstcr.append(line_source.data['pstcr'][i])
                filtered_data_spikes.append(line_source.data['spike_stats'][i])

        # Update the line_source with the filtered data
        selected_line_source.data = {
            'x': filtered_data_x,
            'velocity': filtered_data_velocity,
            'lick': filtered_data_lick,
            'pstcr': filtered_data_pstcr,
            'spike_stats': filtered_data_spikes,
        }

        # Additional code to display selected neurons and spike counts
        selected_neurons = set(selected_raster_source.data['y_starts'])
        neuron_spike_counts = {neuron: 0 for neuron in selected_neurons}
        for neuron in selected_raster_source.data['y_starts']:
            neuron_spike_counts[neuron] += 1

        total_spikes = sum(neuron_spike_counts.values())

        # Update the neurons_div to display total spike count and neurons in rows of three
        neurons_info = "<br>".join(
            [f"Neuron {int(neuron)}: {neuron_spike_counts[neuron]} spikes" for neuron in sorted(neuron_spike_counts)])
        formatted_neurons_info = "<br>".join(
            [", ".join(neurons_info.split("<br>")[i:i + 3]) for i in range(0, len(neurons_info.split("<br>")), 3)])
        neurons_div.text = f"Total Spike Count: {total_spikes}<br>{formatted_neurons_info}"

    p.on_event(SelectionGeometry, selection_handler)
    v.on_event(SelectionGeometry, selection_handler)
    s.on_event(SelectionGeometry, selection_handler)

    def clear_selected_sources(event):
        selected_raster_source.data = {'x_starts': [], 'y_starts': [], 'x_ends': [], 'y_ends': []}
        selected_line_source.data = {'x': [], 'velocity': [], 'lick': [], 'pstcr': [], 'spike_stats': []}
        neurons_div.text = "Neurons will be shown after you choose an interval."

    p.on_event(Reset, clear_selected_sources)
    v.on_event(Reset, clear_selected_sources)
    s.on_event(Reset, clear_selected_sources)

    # Layout
    blank_left = Spacer(width=30)
    layout = row(blank_left, column(row(session_input, column(Spacer(height=20), load_button)), row(p, neurons_div), v, s))
    doc.add_root(layout)
