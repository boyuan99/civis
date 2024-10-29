import numpy as np
import json
import os
import sys

from bokeh.plotting import figure, curdoc

from bokeh.models import (ColumnDataSource, Slider, Button, Arrow, VeeHead, Div, TextInput, Spacer,
                          LinearColorMapper, ColorBar, BasicTicker, Select, TabPanel, Tabs)
from bokeh.layouts import column, row
from bokeh.palettes import Blues8

current_dir = os.path.dirname(os.path.abspath(__file__))
servers_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(servers_dir)
sys.path.append(project_root)


def trajectory_bkapp_v6(doc):
    from civis.src.VirmenTank import VirmenTank
    global source, trials, confusion_matrix_source, correct_array

    trials = []
    source = ColumnDataSource({'x': [], 'y': [], 'face_angle': []})
    plot = figure(width=550, height=800, y_range=(-100, 180),
                  title="Mouse Movement Trajectory")
    plot.line('x', 'y', source=source, line_width=2)

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
    play_button = Button(label="► Play", width=60)
    trial_slider = Slider(start=0, end=10, value=0, step=1, width=600, title="Trial")
    progress_slider = Slider(start=0, end=100, value=0, step=1, width=600, title="Progress")
    filename_input = TextInput(value='', title="File Path:", width=400)
    load_button = Button(label="Load Data", button_type="success")
    previous_button = Button(label="Previous", width=100)
    next_button = Button(label="Next", width=100)
    starts_div = Div(text="Start Time: ", width=400)
    correctness_div = Div(text="Trial Correctness: ", width=400)
    error_div = Div(text="")

    # New dropdown menus for correct and incorrect trials
    correct_trials_dropdown = Select(title="Correct Trials:", value="", options=[], width=200)
    incorrect_trials_dropdown = Select(title="Incorrect Trials:", value="", options=[], width=200)

    trial_slider.disabled = True
    progress_slider.disabled = True
    play_button.disabled = True
    previous_button.disabled = True
    next_button.disabled = True
    correct_trials_dropdown.disabled = True
    incorrect_trials_dropdown.disabled = True

    # Initialize confusion matrix plot with default data
    default_matrix = np.array([[-1, -1], [-1, -1]])
    x_labels = ['Left', 'Right']
    y_labels = ['Left', 'Right']
    confusion_matrix_source = ColumnDataSource(data={
        'x': [x for x in x_labels for _ in y_labels],
        'y': y_labels * len(x_labels),
        'value': default_matrix.flatten().tolist(),
    })
    # Initialize accuracy vs time plot and velocty and lick plot with default data
    accuracy_source = ColumnDataSource(data={'x': [], 'y': []})
    velocity_source = ColumnDataSource(data={'x': [], 'y': []})
    pstcr_source = ColumnDataSource(data={'x': [], 'y': []})
    lick_source = ColumnDataSource(data={'x': [], 'y': []})
    current_velocity_source = ColumnDataSource(data={'x': [], 'y': []})

    color_mapper = LinearColorMapper(palette=Blues8[::-1], low=-1, high=1)

    confusion_matrix_plot = figure(title="Confusion Matrix of Animal Turns",
                                   x_range=x_labels, y_range=list(reversed(y_labels)),
                                   y_axis_label="Actual Turn", x_axis_label="Should Turn",
                                   width=400, height=350, toolbar_location=None, tools="")

    confusion_matrix_plot.rect(x='x', y='y', width=1, height=1, source=confusion_matrix_source,
                               line_color=None, fill_color={'field': 'value', 'transform': color_mapper})

    color_bar = ColorBar(color_mapper=color_mapper, ticker=BasicTicker(desired_num_ticks=len(Blues8)),
                         label_standoff=6, border_line_color=None, location=(0, 0))

    confusion_matrix_plot.add_layout(color_bar, 'right')

    confusion_matrix_plot.text(x='x', y='y', text='value', source=confusion_matrix_source,
                               text_align="center", text_baseline="middle")

    confusion_matrix_tab = TabPanel(child=confusion_matrix_plot, title="Confusion Matrix")
    
    # Creates empty velocity and lick plot
    velocity_and_lick_plot = figure(width=800, height=300,
                                    active_drag='pan', active_scroll='wheel_zoom', title="Velocity")
    velocity_and_lick_plot.line(x='x', y='y', source=velocity_source, line_color='navy', legend_label='velocity',
                                line_width=2)
    velocity_and_lick_plot.line(x='x', y='y', source=current_velocity_source, line_color='red',
                                legend_label='current_velocity', line_width=2)
    velocity_and_lick_plot.line(x='x', y='y', source=pstcr_source,
                                line_color='orange', legend_label='pstcr', line_width=2, alpha=0.7)
    velocity_and_lick_plot.multi_line(xs='x', ys='y', source=lick_source,
                                      line_color='green', legend_label='lick', )
    velocity_and_lick_plot.legend.click_policy = "hide"
    # add line called pstcr

    velocity_and_lick_tab = TabPanel(child=velocity_and_lick_plot, title="Velocity")

    # Make accuracy vs trial graph

    accuracy_plot = figure(title="Accuracy Plot", x_axis_label='Trial', y_axis_label='Accuracy(%)',
                           width=800, height=300, y_range=(-0.1, 1))
    accuracy_plot.line(x='x', y='y', source=accuracy_source)

    accuracy_tab = TabPanel(child=accuracy_plot, title="Accuracy")

    def load_data():
        global source, trials, starts, confusion_matrix_source, correct_array, accuracy_trials, vm_rate, pstcr

        try:
            config_path = os.path.join(project_root, 'config.json')
            with open(config_path, 'r') as file:
                config = json.load(file)

            session_name = filename_input.value
            if ".txt" in session_name:
                file = os.path.join(config['VirmenFilePath'], session_name)
            else:
                file = os.path.join(config['VirmenFilePath'], f'{session_name}.txt')

            # Check if the maze type is correct
            if VirmenTank.determine_maze_type(file).lower() != "turnv1":
                raise ValueError(
                    f"Invalid maze type. Expected 'TurnV1', but got '{VirmenTank.determine_maze_type(file)}'")
            else:
                vm = VirmenTank(session_name, height=35)

            trials = vm.virmen_trials

            data_array = np.array(vm.virmen_data)
            start_indicies = vm.trials_start_indices

            trial_length = sum(len(d['x']) for d in trials)
            start_indicies = np.append(start_indicies, trial_length)
            pstcr = {}

            for i in range(len(vm.virmen_trials)):
                dx = np.diff(data_array[start_indicies[i]: start_indicies[i+1], 0])
                dy = np.diff(data_array[start_indicies[i]: start_indicies[i+1], 1])
                pstcr[i] = np.sqrt(dx ** 2 + dy ** 2)

            vm_rate = vm.vm_rate
            starts = [x / vm_rate for x in vm.trials_start_indices]
            correct_array = vm.extend_data.correct_array

            print("Successfully loaded: " + file)
            error_div.text = ""  # Clear any previous error messages

            if trials:
                # Enable the widgets now that data is loaded
                trial_slider.disabled = False
                progress_slider.disabled = False
                play_button.disabled = False
                previous_button.disabled = False
                next_button.disabled = False
                correct_trials_dropdown.disabled = False
                incorrect_trials_dropdown.disabled = False

                initial_trial = trials[0]
                new_data = {'x': [initial_trial['x'][0]],
                            'y': [initial_trial['y'][0]],
                            'face_angle': [initial_trial['face_angle'][0]]}
                source.data = new_data

                accuracy_trials = vm.extend_data.current_accuracy()
                accuracy_source.data = {'x': list(accuracy_trials.keys()), 'y': list(accuracy_trials.values())}

                trial_slider.end = len(trials) - 1
                trial_slider.value = 0

                progress_slider.end = 100
                progress_slider.value = 0

                starts_div.text = f"Start Time: {starts[0]}"

                if correct_array[0]:
                    correctness_div.text = f"Trial Correctness: Correct"
                else:
                    correctness_div.text = f"Trial Correctness: Wrong"

                # Update confusion matrix plot
                new_confusion_matrix = vm.extend_data.confusion_matrix
                confusion_matrix_source.data.update({
                    'value': new_confusion_matrix.flatten().tolist()
                })
                color_mapper.low = new_confusion_matrix.min()
                color_mapper.high = new_confusion_matrix.max()

                # Update dropdown menus
                correct_trials = [f"Trial {i}" for i, correct in enumerate(correct_array) if correct]
                incorrect_trials = [f"Trial {i}" for i, correct in enumerate(correct_array) if not correct]

                correct_trials_dropdown.options = correct_trials
                incorrect_trials_dropdown.options = incorrect_trials

                if correct_trials:
                    correct_trials_dropdown.value = correct_trials[0]
                if incorrect_trials:
                    incorrect_trials_dropdown.value = incorrect_trials[0]

        except ValueError as e:
            error_div.text = f"Error: {str(e)}"
            # Disable widgets if data loading fails
            trial_slider.disabled = True
            progress_slider.disabled = True
            play_button.disabled = True
            previous_button.disabled = True
            next_button.disabled = True
            correct_trials_dropdown.disabled = True
            incorrect_trials_dropdown.disabled = True

        except Exception as e:
            error_div.text = f"An unexpected error occurred: {str(e)}"
            print(e)
            # Disable widgets if data loading fails
            trial_slider.disabled = True
            progress_slider.disabled = True
            play_button.disabled = True
            previous_button.disabled = True
            next_button.disabled = True
            correct_trials_dropdown.disabled = True
            incorrect_trials_dropdown.disabled = True

    load_button.on_click(load_data)

    def update_plot(attr, old, new):
        trial_index = trial_slider.value
        progress = progress_slider.value / 100
        trial_data = trials[trial_index]
        max_index = int(len(trial_data['x']) * progress)

        if max_index > 0:
            last_x = trial_data['x'][max_index - 1]
            last_y = trial_data['y'][max_index - 1]
            angle_rad = trial_data['face_angle'][max_index - 1] + np.pi / 2  # Adjust angle to make arrow face up
            arrow_length = 5

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
        starts_div.text = f"Start Time: {starts[trial_index]}"

        # edits data source for velocity and lick graph using (1/vm_rate) to convert indicies to seconds
        dx = np.array(trial_data['dx'])
        dy = np.array(trial_data['dy'])
        velocity = np.sqrt(dx ** 2 + dy ** 2)

        pstcr_array = np.array(pstcr[trial_index])
        pstcr_array = np.append(pstcr_array, 0 )

        velocity_source.data = {'x': (1 / vm_rate) * np.arange(len(trial_data['x'])), 'y': velocity}
        pstcr_source.data = {'x': (1 / vm_rate) * np.arange(len(trial_data['x'])),
                             'y': pstcr_array * vm_rate}
        current_velocity_source.data = {"x": [(1 / vm_rate) * max_index, (1 / vm_rate) * max_index],
                                        "y": [0, velocity.max()]}
        x_pos_lick = []
        y_pos_lick = []

        for indices in trial_data["lick"]:
            x_pos_lick.append([indices * (1 / vm_rate), indices * (1 / vm_rate)])
            y_pos_lick.append([0, velocity.max()])

        lick_source.data = {"x": x_pos_lick, "y": y_pos_lick}

        if correct_array[trial_index]:
            correctness_div.text = f"Trial Correctness: Correct"
        else:
            correctness_div.text = f"Trial Correctness: Wrong"

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
            # Schedule the periodic callback with an interval of 50ms (0.05 seconds)
            play_interval_id = doc.add_periodic_callback(update_progress, 50)
        else:
            play_button.label = "► Play"
            is_playing = False
            # Remove the periodic callback to stop updates
            if play_interval_id:
                doc.remove_periodic_callback(play_interval_id)

    # Bind the toggle function to the play button
    play_button.on_click(toggle_play)

    def previous_trial():
        if trial_slider.value > trial_slider.start:
            trial_slider.value -= 1

    def next_trial():
        if trial_slider.value < trial_slider.end:
            trial_slider.value += 1

    previous_button.on_click(previous_trial)
    next_button.on_click(next_trial)

    def update_trial_from_dropdown(attr, old, new):
        if new:
            trial_index = int(new.split()[1])
            trial_slider.value = trial_index

    correct_trials_dropdown.on_change('value', update_trial_from_dropdown)
    incorrect_trials_dropdown.on_change('value', update_trial_from_dropdown)

    tab_buttons = Tabs(tabs=[confusion_matrix_tab, accuracy_tab, velocity_and_lick_tab])

    file_input_row = row(filename_input, column(Spacer(height=20), load_button))
    trial_navigation_row = row(previous_button, next_button)
    dropdown_row = row(correct_trials_dropdown, incorrect_trials_dropdown)
    tool_widgets = column(file_input_row, trial_slider, progress_slider, play_button,
                          trial_navigation_row, starts_div, correctness_div, dropdown_row, error_div)
    layout = row(plot, Spacer(width=30), column(tool_widgets, tab_buttons))
    doc.add_root(layout)


# Uncomment the following line to run the Bokeh server application
trajectory_bkapp_v6(curdoc())
