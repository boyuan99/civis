from bokeh.models import ColumnDataSource, TextInput, Button, BoxSelectTool, Spacer
from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.events import SelectionGeometry, Reset
import pickle
import numpy as np
import json
from src import StraightMazeTank


def raster_bkapp(doc):
    def load_data(session_name):

        with open('config.json', 'r') as file:
            config = json.load(file)

        neuron_path = config['ProcessedFilePath'] + session_name + '/' + session_name + '_v7.mat'
        peak_indices_path = config['ProcessedFilePath'] + session_name + "/" + session_name + "_peak_indices.pkl"

        ci = StraightMazeTank(neuron_path)
        print("Successful loaded: " + neuron_path)

        with open(peak_indices_path, 'rb') as f:
            peak_indices = pickle.load(f)

        spike_times = peak_indices

        data = {'x_starts': [], 'y_starts': [], 'x_ends': [], 'y_ends': []}
        for neuron_idx, spikes in enumerate(spike_times):
            for spike_time in spikes:
                data['x_starts'].append(spike_time / ci.ci_rate)
                data['y_starts'].append(neuron_idx + 1)
                data['x_ends'].append(spike_time / ci.ci_rate)
                data['y_ends'].append(neuron_idx + 1 + 0.7)

        return data, ci

    # Create initial empty plot
    raster_source = ColumnDataSource({'x_starts': [], 'y_starts': [], 'x_ends': [], 'y_ends': []})
    selected_raster_source = ColumnDataSource({'x_starts': [], 'y_starts': [], 'x_ends': [], 'y_ends': []})
    line_source = ColumnDataSource({'x': [], 'velocity': [], 'lick': []})
    selected_line_source = ColumnDataSource({'x': [], 'velocity': [], 'lick': []})

    # Raster plot
    p = figure(width=1000, height=1000, title="Raster Plot", x_axis_label='Time (s)', y_axis_label='Neuron',
               active_scroll='wheel_zoom')
    p.segment(x0='x_starts', y0='y_starts', x1='x_ends', y1='y_ends', source=raster_source, color="black", alpha=1,
              line_width=2)
    p.segment(x0='x_starts', y0='y_starts', x1='x_ends', y1='y_ends', source=selected_raster_source, color="red",
              alpha=1, line_width=2)

    # Signal plot
    v = figure(width=1000, height=200, x_range=p.x_range, active_scroll='wheel_zoom',
               tools=['pan', 'wheel_zoom', 'reset'])
    v.line('x', 'velocity', source=line_source, color='SteelBlue', legend_label='velocity', alpha=0.1)
    v.line('x', 'lick', source=line_source, color='SandyBrown', legend_label='lick', alpha=0.3)
    v.line('x', 'velocity', source=selected_line_source, color='SteelBlue', legend_label='velocity', alpha=1)
    v.line('x', 'lick', source=selected_line_source, color='SandyBrown', legend_label='lick', alpha=1)
    v.legend.click_policy = 'hide'

    box_select_p = BoxSelectTool(dimensions="width")
    p.add_tools(box_select_p)
    p.toolbar.active_drag = box_select_p
    v.add_tools(box_select_p)
    v.toolbar.active_drag = box_select_p

    # Text input widget for session name
    session_input = TextInput(value="", title="Session Name:")

    # Button to load data
    load_button = Button(label="Load Data", button_type="success")

    def update_data():
        print("Loading Data...")
        session_name = session_input.value
        data, ci = load_data(session_name)
        raster_source.data = data
        p.yaxis.ticker = np.arange(1, ci.neuron_num + 1)
        p.yaxis.major_label_overrides = {i + 1: f"Neuron {i + 1}" for i in range(ci.neuron_num)}
        line_source.data = dict(x=ci.t,
                                velocity=ci.normalize_signal(ci.velocity),
                                lick=ci.lick)

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

        for i, x_val in enumerate(line_source.data['x']):
            if x0 <= x_val <= x1:
                filtered_data_x.append(x_val)
                filtered_data_velocity.append(line_source.data['velocity'][i])
                filtered_data_lick.append(line_source.data['lick'][i])

        # Update the line_source with the filtered data
        selected_line_source.data = {
            'x': filtered_data_x,
            'velocity': filtered_data_velocity,
            'lick': filtered_data_lick,
        }

    p.on_event(SelectionGeometry, selection_handler)
    v.on_event(SelectionGeometry, selection_handler)

    def clear_selected_sources(event):
        selected_raster_source.data = {'x_starts': [], 'y_starts': [], 'x_ends': [], 'y_ends': []}
        selected_line_source.data = {'x': [], 'velocity': [], 'lick': []}

    p.on_event(Reset, clear_selected_sources)
    v.on_event(Reset, clear_selected_sources)

    # Layout
    blank_left = Spacer(width=30)
    layout = row(blank_left, column(row(session_input, load_button), p, v))
    doc.add_root(layout)
