from bokeh.models import ColumnDataSource, TextInput, Button, BoxSelectTool, Spacer, Arrow, VeeHead, RangeSlider
from bokeh.plotting import figure
from bokeh.layouts import column, row
from bokeh.events import SelectionGeometry, Reset
import pickle
import numpy as np
import json
import pandas as pd
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
servers_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(servers_dir)
sys.path.append(project_root)


def raster_bkapp_v1(doc):
    from civis.src.CITank import CITank

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
    v.add_tools(box_select_p)

    # Text input widget for session name
    session_input = TextInput(value="", title="Session Name:")

    # Button to load data
    load_button = Button(label="Load Data", button_type="success")

    # Trajectory part
    global virmen_source, virmen_data, range_slider

    virmen_data = pd.DataFrame()
    virmen_source = ColumnDataSource({'x': [], 'y': [], 'face_angle': []})
    plot = figure(width=550, height=800, y_range=(-100,180),title="Mouse Movement Trajectory")
    plot.line('x', 'y', source=virmen_source, line_width=2)

    arrow = Arrow(end=VeeHead(fill_color="orange", size=10, line_width=1),
                  x_start=0, y_start=0,
                  x_end=0, y_end=1, line_color="orange")
    plot.add_layout(arrow)

    xpts = np.array([-67.5, -50, 0, 50, 67.5, 10, 10, -10, -10, -10, -67.5])
    ypts = np.array([107.5, 125, 75, 125, 107.5, 50, -65, -65, -50, 50, 107.5])

    source_maze = ColumnDataSource(dict(
        xs=[xpts],
        ys=[ypts],
    ))

    plot.multi_line(xs="xs", ys="ys", source=source_maze, line_color="#8073ac", line_width=2)
    plot.patch([-67.5, -50, -70, -87.5], [107.5, 125, 145, 127.5], alpha=0.5)
    plot.patch([67.5, 50, 70, 87.5], [107.5, 125, 145, 127.5], alpha=0.5)

    # Widgets
    range_slider = RangeSlider(start=0, end=100, value=(0, 100), step=1, width=600, title="Progress")
    range_slider.disabled = True

    def load_data(session_name):
        global virmen_source, virmen_data, range_slider, ci

        config_path = os.path.join(project_root, 'config.json')
        with open(config_path, 'r') as file:
            config = json.load(file)

        neuron_path = config['ProcessedFilePath'] + session_name + '/' + session_name + '_v7.mat'
        peak_indices_path = config['ProcessedFilePath'] + session_name + "/" + session_name + "_peak_indices.pkl"
        virmen_path = config['VirmenFilePath'] + session_name + ".txt"

        ci = CITank(session_name)
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

        virmen_data = ci.virmen_data[:ci.session_duration * ci.ci_rate]
        if not virmen_data.empty:
            # Enable the widgets now that data is loaded
            range_slider.disabled = False

            new_data = {'x': virmen_data['x'].tolist(),
                        'y': virmen_data['y'].tolist(),
                        'face_angle': virmen_data['face_angle'].tolist()}
            virmen_source.data = new_data

            range_slider.end = ci.t[-1]
            range_slider.value = (0, 0)

        return data, ci, spike_stats

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

    def update_plot(attr, old, new):
        global virmen_data, ci

        # Get the selected time range
        selected_start_time, selected_end_time = range_slider.value

        # Filter the virmen_data based on the selected time range
        data_slice = virmen_data[(ci.t >= selected_start_time) & (ci.t <= selected_end_time)]

        # Update the virmen_source with the selected data
        virmen_source.data = {'x': data_slice['x'],
                              'y': data_slice['y'],
                              'face_angle': data_slice['face_angle']}

        if not data_slice.empty:
            last_x = data_slice['x'].iloc[-1]
            last_y = data_slice['y'].iloc[-1]
            angle_rad = data_slice['face_angle'].iloc[-1] + np.pi / 2  # Adjust angle to make arrow face up
            arrow_length = 1  # Adjust as necessary for your visualization

            arrow.x_start = last_x
            arrow.y_start = last_y
            arrow.x_end = last_x + arrow_length * np.cos(angle_rad)
            arrow.y_end = last_y + arrow_length * np.sin(angle_rad)
        else:
            arrow.x_start = 0
            arrow.y_start = 0
            arrow.x_end = 0
            arrow.y_end = 1

    range_slider.on_change('value', update_plot)

    def selection_handler(event):
        geometry = event.geometry
        x0, x1 = geometry['x0'], geometry['x1']

        # Update the RangeSlider values based on the selection
        range_slider.value = (x0, x1)

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

        selected_line_source.data = {
            'x': filtered_data_x,
            'velocity': filtered_data_velocity,
            'lick': filtered_data_lick,
            'pstcr': filtered_data_pstcr,
            'spike_stats': filtered_data_spikes,
        }

    p.on_event(SelectionGeometry, selection_handler)
    v.on_event(SelectionGeometry, selection_handler)
    s.on_event(SelectionGeometry, selection_handler)

    def clear_selected_sources(event):
        selected_raster_source.data = {'x_starts': [], 'y_starts': [], 'x_ends': [], 'y_ends': []}
        selected_line_source.data = {'x': [], 'velocity': [], 'lick': [], 'pstcr': [], 'spike_stats': []}

    p.on_event(Reset, clear_selected_sources)
    v.on_event(Reset, clear_selected_sources)
    s.on_event(Reset, clear_selected_sources)

    # Layout
    layout = row(Spacer(width=30), column(row(session_input, column(Spacer(height=20), load_button)), p, v, s, ), Spacer(width=30), column(Spacer(height=70), plot, range_slider))
    doc.add_root(layout)


# raster_bkapp_v1(curdoc())
