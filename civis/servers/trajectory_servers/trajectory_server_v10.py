import numpy as np
import json
import os
import sys

from bokeh.plotting import figure, curdoc
from bokeh.models import ColumnDataSource, Slider, Button, Arrow, VeeHead, Div, TextInput, Spacer
from bokeh.layouts import column, row

current_dir = os.path.dirname(os.path.abspath(__file__))
servers_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(servers_dir)
sys.path.append(project_root)


def trajectory_bkapp_v10(doc):
    """
    Trajectory visualization for wide70 maze type.
    Based on v9 but without fall tracking (wide70 has walls, no falls).
    """
    from civis.src.VirmenTank import VirmenTank
    global source, trials, vm_rate, pstcr

    trials = []
    source = ColumnDataSource({'x': [], 'y': [], 'face_angle': []})
    # Initialize velocity sources
    velocity_source = ColumnDataSource(data={'x': [], 'y': []})
    current_velocity_source = ColumnDataSource(data={'x': [], 'y': []})
    pstcr_source = ColumnDataSource(data={'x': [], 'y': []})
    lick_source = ColumnDataSource(data={'x': [], 'y': []})
    plot = figure(width=400, height=800, y_range=[-80, 80], x_range=[-20, 20],
                  title="Mouse Movement Trajectory (Wide70)")
    plot.line('x', 'y', source=source, line_width=2)

    arrow = Arrow(end=VeeHead(fill_color="orange", size=10, line_width=1),
                  x_start=0, y_start=0,
                  x_end=0, y_end=1, line_color="orange")
    plot.add_layout(arrow)

    # Maze boundaries for wide70 (same as straight70v3)
    xpts = np.array([-20, -20, 20, 20])
    ypts = np.array([-70, 70, 70, -70])

    source_maze = ColumnDataSource(dict(
        xs=[xpts],
        ys=[ypts],
    ))

    plot.multi_line(xs="xs", ys="ys", source=source_maze, line_color="#8073ac", line_width=2)
    plot.patch([-20, -20, 20, 20], [70, 80, 80, 70], alpha=0.5)
    plot.patch([-20, -20, 20, 20], [-70, -80, -80, -70], alpha=0.5)

    # Widgets
    play_button = Button(label="► Play", width=60)
    trial_slider = Slider(start=0, end=10, value=0, step=1, width=600, title="Trial")
    progress_slider = Slider(start=0, end=100, value=0, step=1, width=600, title="Progress")
    filename_input = TextInput(value='', title="File Path:", width=400)
    load_button = Button(label="Load Data", button_type="success")
    previous_button = Button(label="Previous", width=100)
    next_button = Button(label="Next", width=100)
    starts_div = Div(text="Start Time: ", width=400)
    error_div = Div(text="")

    trial_slider.disabled = True
    progress_slider.disabled = True
    play_button.disabled = True
    previous_button.disabled = True
    next_button.disabled = True

    def load_data():
        global source, trials, starts, vm_rate, pstcr, vm

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
            maze_type = VirmenTank.determine_maze_type(file).lower()
            if maze_type != "wide70":
                raise ValueError(
                    f"Invalid maze type. Expected 'wide70', but got '{maze_type}'")

            vm = VirmenTank(file)

            # Use regular trials (wide70 has no extend_data since walls prevent falls)
            trials = vm.virmen_trials
            starts = [idx / vm.vm_rate for idx in vm.trials_start_indices]

            # Get the data array and indices
            data_array = np.array(vm.virmen_data)
            start_indices = vm.trials_start_indices
            end_indices = vm.trials_end_indices

            print(f"Successfully loaded: {file}")
            print(f"Trials found: {len(trials)}")
            error_div.text = ""  # Clear any previous error messages

            if trials:
                # Enable the widgets now that data is loaded
                trial_slider.disabled = False
                progress_slider.disabled = False
                play_button.disabled = False
                previous_button.disabled = False
                next_button.disabled = False

                initial_trial = trials[0]
                new_data = {'x': [initial_trial['x'][0]],
                            'y': [initial_trial['y'][0]],
                            'face_angle': [initial_trial['face_angle'][0]]}
                source.data = new_data

                trial_slider.end = len(trials) - 1
                trial_slider.value = 0

                progress_slider.end = 100
                progress_slider.value = 0

                starts_div.text = f"Start Time: {starts[0]:.2f}s"

            vm_rate = vm.vm_rate
            pstcr = {}

            for i in range(len(trials)):
                if i < len(start_indices) and i < len(end_indices):
                    start_idx = start_indices[i]
                    end_idx = end_indices[i]
                    dx = np.diff(data_array[start_idx:end_idx + 1, 0])
                    dx = np.append(dx, dx[-1] if len(dx) > 0 else 0)
                    dy = np.diff(data_array[start_idx:end_idx + 1, 1])
                    dy = np.append(dy, dy[-1] if len(dy) > 0 else 0)
                    dx = dx.astype(np.float32)
                    dy = dy.astype(np.float32)
                    pstcr[i] = np.sqrt(dx ** 2 + dy ** 2)

        except ValueError as e:
            error_div.text = f"Error: {str(e)}"
            print(f"ValueError: {str(e)}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            trial_slider.disabled = True
            progress_slider.disabled = True
            play_button.disabled = True
            previous_button.disabled = True
            next_button.disabled = True
        except Exception as e:
            error_div.text = f"An unexpected error occurred: {str(e)}"
            print(f"Unexpected error: {str(e)}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            trial_slider.disabled = True
            progress_slider.disabled = True
            play_button.disabled = True
            previous_button.disabled = True
            next_button.disabled = True

    load_button.on_click(load_data)

    def update_plot(attr, old, new):
        global vm
        trial_index = trial_slider.value
        progress = progress_slider.value / 100
        trial_data = trials[trial_index]
        max_index = int(len(trial_data['x']) * progress)

        if max_index > 0:
            last_x = trial_data['x'][max_index - 1]
            last_y = trial_data['y'][max_index - 1]
            angle_rad = trial_data['face_angle'][max_index - 1] + np.pi / 2  # Adjust angle to make arrow face up
            arrow_length = 2

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

        # Display trial information
        trial_duration = len(trial_data['x'])
        starts_div.text = f"Start Time: {starts[trial_index]:.2f}s | Duration: {trial_duration} frames"

        # Update velocity plot data
        dx = np.array(trial_data['dx'])
        dy = np.array(trial_data['dy'])
        velocity = np.sqrt(dx ** 2 + dy ** 2)

        if trial_index in pstcr:
            pstcr_array = np.array(pstcr[trial_index])
        else:
            pstcr_array = np.zeros_like(velocity)

        velocity_source.data = {'x': (1/vm_rate) * np.arange(len(trial_data['x'])), 'y': velocity}
        pstcr_source.data = {'x': (1/vm_rate) * np.arange(len(trial_data['x'])),
                           'y': pstcr_array * vm_rate}
        current_velocity_source.data = {"x": [(1/vm_rate) * max_index, (1/vm_rate) * max_index],
                                     "y": [0, velocity.max()]}

        x_pos_lick = []
        y_pos_lick = []

        for indices in trial_data["lick"]:
            x_pos_lick.append([indices * (1/vm_rate), indices * (1/vm_rate)])
            y_pos_lick.append([0, velocity.max()])

        lick_source.data = {"x": x_pos_lick, "y": y_pos_lick}

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

    # Add velocity plot
    velocity_and_lick_plot = figure(width=800, height=300,
                                  active_drag='pan', active_scroll='wheel_zoom', title="Velocity")
    velocity_and_lick_plot.line(x='x', y='y', source=velocity_source, line_color='navy', legend_label='velocity',
                              line_width=2)
    velocity_and_lick_plot.line(x='x', y='y', source=current_velocity_source, line_color='red',
                              legend_label='current_velocity', line_width=2)
    velocity_and_lick_plot.line(x='x', y='y', source=pstcr_source,
                              line_color='orange', legend_label='pstcr', line_width=2, alpha=0.7)
    velocity_and_lick_plot.multi_line(xs='x', ys='y', source=lick_source,
                                    line_color='green', legend_label='lick')
    velocity_and_lick_plot.legend.click_policy = "hide"

    file_input_row = row(filename_input, column(Spacer(height=20), load_button))
    trial_navigation_row = row(previous_button, next_button)
    tool_widgets = column(file_input_row, trial_slider, progress_slider, play_button,
                          trial_navigation_row, starts_div, error_div)
    layout = row(plot, Spacer(width=30), column(tool_widgets, velocity_and_lick_plot))
    doc.add_root(layout)


# Uncomment the following line to run the Bokeh server application
trajectory_bkapp_v10(curdoc())
