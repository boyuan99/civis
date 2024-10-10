from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, Slider, Button, Arrow, VeeHead, BoxAnnotation, Span, TextInput, Spacer
from bokeh.layouts import column, row
import numpy as np
import pandas as pd

def read_and_process_data(file_path, usecols=[0, 1, 2], threshold=[70.0, -70.0]):
    data = pd.read_csv(file_path, sep=r'\s+|,', engine='python', header=None,
                       usecols=usecols, names=['x', 'y', 'face_angle'])

    # Identifying trials
    trials = []
    start = 0
    for i in range(len(data)):
        if data.iloc[i]['y'] >= threshold[0] or data.iloc[i]['y'] <= threshold[1]:
            trials.append(data[start:i + 1].to_dict(orient='list'))
            start = i + 1

    return trials

def trajectory_bkapp_v0(doc):
    global source, trials

    trials = []
    source = ColumnDataSource({'x': [], 'y': [], 'face_angle': []})
    plot = figure(width=300, height=800, y_range=[-80, 80], x_range=[-10, 10],
                  title="Mouse Movement Trajectory")
    plot.line('x', 'y', source=source, line_width=2)

    arrow = Arrow(end=VeeHead(fill_color="orange", size=10, line_width=1),
                  x_start=0, y_start=0,
                  x_end=0, y_end=1, line_color="orange")
    plot.add_layout(arrow)

    # Annotations
    high_box = BoxAnnotation(bottom=70, fill_alpha=0.5, fill_color='blue')
    plot.add_layout(high_box)
    low_box = BoxAnnotation(top=-70, fill_alpha=0.5, fill_color='blue')
    plot.add_layout(low_box)
    vline0 = Span(location=-9, dimension='height', line_color='black', line_width=2)
    plot.add_layout(vline0)
    vline1 = Span(location=9, dimension='height', line_color='black', line_width=2)
    plot.add_layout(vline1)

    # Widgets
    play_button = Button(label="► Play", width=60)
    trial_slider = Slider(start=0, end=10, value=0, step=1, width=600, title="Trial")
    progress_slider = Slider(start=0, end=100, value=0, step=1, width=600, title="Progress")
    filename_input = TextInput(value='', title="File Path:", width=400)
    load_button = Button(label="Load Data", button_type="success")

    trial_slider.disabled = True
    progress_slider.disabled = True
    play_button.disabled = True

    def load_data():
        global source, trials

        file = filename_input.value
        trials = read_and_process_data(file)

        if trials:
            # Enable the widgets now that data is loaded
            trial_slider.disabled = False
            progress_slider.disabled = False
            play_button.disabled = False

            initial_trial = trials[0]
            new_data = {'x': [initial_trial['x'][0]],
                        'y': [initial_trial['y'][0]],
                        'face_angle': [initial_trial['face_angle'][0]]}
            source.data = new_data

            trial_slider.end = len(trials) - 1
            trial_slider.value = 0

            progress_slider.end = 100
            progress_slider.value = 0

    load_button.on_click(load_data)


    def update_plot(attr, old, new):
        trial_index = trial_slider.value
        progress = progress_slider.value / 100
        trial_data = trials[trial_index]
        max_index = int(len(trial_data['x']) * progress)

        if max_index > 0:
            last_x = trial_data['x'][max_index - 1]
            last_y = trial_data['y'][max_index - 1]
            angle_rad = trial_data['face_angle'][max_index - 1] + np.pi/2  # Adjust angle to make arrow face up
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


        new_data = {'x': trial_data['x'][:max_index],
                    'y': trial_data['y'][:max_index],
                    'face_angle': trial_data['face_angle'][:max_index]}
        source.data = new_data


    trial_slider.on_change('value', update_plot)
    progress_slider.on_change('value', update_plot)


    # Initialize play state
    global is_playing, play_interval_id
    is_playing = False
    play_interval_id = None


    def update_progress():
        current_value = progress_slider.value
        if current_value < progress_slider.end:
            progress_slider.value = current_value + 1
        else:
            toggle_play()


    def toggle_play():
        global is_playing, play_interval_id
        if not is_playing:
            # Check if the progress slider is at its maximum value and reset if so
            if progress_slider.value == progress_slider.end:
                progress_slider.value = 0  # Reset progress to start

            play_button.label = "❚❚ Pause"
            is_playing = True
            # Schedule the periodic callback with an interval of 100ms (0.1 seconds)
            play_interval_id = doc.add_periodic_callback(update_progress, 100)
        else:
            play_button.label = "► Play"
            is_playing = False
            # Remove the periodic callback to stop updates
            if play_interval_id:
                doc.remove_periodic_callback(play_interval_id)


    # Bind the toggle function to the play button
    play_button.on_click(toggle_play)

    file_input_row = row(filename_input, column(Spacer(height=20), load_button))
    tool_widgets = column(file_input_row, trial_slider, progress_slider, play_button)
    layout = row(plot, Spacer(width=30), tool_widgets)
    doc.add_root(layout)


trajectory_bkapp_v0(curdoc())
