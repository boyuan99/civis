from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, RangeSlider, Button, Arrow, VeeHead, TextInput, Spacer
from bokeh.layouts import column, row
import numpy as np
import json
import pandas as pd


def read_and_process_data_v2(file_path, usecols=[0, 1, 2], threshold=[175.0, -175.0]):
    data = pd.read_csv(file_path, sep=r'\s+|,', engine='python', header=None,
                       usecols=usecols, names=['x', 'y', 'face_angle'])

    # Identifying trials
    trials = []
    start = 0
    for i in range(len(data)):
        if abs(data.iloc[i]['y']) + abs(data.iloc[i]['x']) >= threshold[0]:
            trials.append(data[start:i + 1].to_dict(orient='list'))
            start = i + 1

    return [trials, data]

global virmen_source, trials, virmen_data, range_slider

trials = []
virmen_data = pd.DataFrame()
virmen_source = ColumnDataSource({'x': [], 'y': [], 'face_angle': []})
plot = figure(width=550, height=800,
              title="Mouse Movement Trajectory")
plot.line('x', 'y', source=virmen_source, line_width=2)

arrow = Arrow(end=VeeHead(fill_color="orange", size=10, line_width=1),
              x_start=0, y_start=0,
              x_end=0, y_end=1, line_color="orange")
plot.add_layout(arrow)

xpts = np.array([-67.5, -50, 0, 50, 67.5, 10, 10, 67.5, 50, 0, -50, -67.5, -10, -10, -67.5])
ypts = np.array([107.5, 125, 75, 125, 107.5, 50, -50, -107.5, -125, -75, -125, -107.5, -50, 50, 107.5])

source_maze = ColumnDataSource(dict(
        xs=[xpts],
        ys=[ypts],
    ),
)

plot.multi_line(xs="xs", ys="ys", source=source_maze, line_color="#8073ac", line_width=2)
plot.patch([67.5, 50, 70, 87.5], [-107.5, -125, -145, -127.5], alpha=0.5)
plot.patch([-67.5, -50, -70, -87.5], [107.5, 125, 145, 127.5], alpha=0.5)
plot.patch([-67.5, -50, -70, -87.5], [-107.5, -125, -145, -127.5], alpha=0.5)
plot.patch([67.5, 50, 70, 87.5], [107.5, 125, 145, 127.5], alpha=0.5)

# Widgets
range_slider = RangeSlider(start=0, end=100, value=(0, 100), step=1, width=600, title="Progress")
filename_input = TextInput(value='', title="File Path:", width=400)
load_button = Button(label="Load Data", button_type="success")

range_slider.disabled = True

def load_data():
    global virmen_source, trials, virmen_data, range_slider

    with open('config.json', 'r') as file:
        config = json.load(file)

    session_name = filename_input.value
    file = config['VirmenFilePath'] + session_name

    [trials, virmen_data] = read_and_process_data_v2(file, usecols=[0, 1, 2], threshold=[175.0, -175.0])
    virmen_data = virmen_data[:36000]

    if not virmen_data.empty:
        # Enable the widgets now that data is loaded
        range_slider.disabled = False

        new_data = {'x': virmen_data['x'].tolist(),
                    'y': virmen_data['y'].tolist(),
                    'face_angle': virmen_data['face_angle'].tolist()}
        virmen_source.data = new_data

        range_slider.end = len(virmen_data) - 1
        range_slider.value = (0, len(virmen_data) - 1)

load_button.on_click(load_data)

def update_plot(attr, old, new):
    global virmen_data

    start, end = range_slider.value
    data_slice = virmen_data.iloc[int(start):int(end)]

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

    new_data = {'x': data_slice['x'].tolist(),
                'y': data_slice['y'].tolist(),
                'face_angle': data_slice['face_angle'].tolist()}
    virmen_source.data = new_data

range_slider.on_change('value', update_plot)


file_input_row = row(filename_input, load_button)
tool_widgets = column(file_input_row, range_slider)
layout = row(plot, Spacer(width=30), tool_widgets)
curdoc().add_root(layout)
