import numpy as np
import pandas as pd
from bokeh.plotting import figure
from bokeh.models import HoverTool, ColumnDataSource, Slider, Button, TextInput, Spacer, Div, Select, TapTool, Tabs, \
    TabPanel, ImageRGBA, Toggle, Patches, GlyphRenderer
from bokeh.layouts import column, row
import json
import h5py
import os
import sys
from pathlib import Path
from tifffile import imread

# Set up paths similar to place finder server
current_dir = os.path.dirname(os.path.abspath(__file__))
servers_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(servers_dir)
sys.path.append(project_root)

CONFIG_PATH = os.path.join(project_root, 'config.json')


def load_data(filename):
    """
    Load all data
    :param filename: .mat file containing config, spatial, Cn, Coor, ids, etc...
    :return: essential variables for plotting
    """
    with h5py.File(filename, 'r') as file:
        data = file['data']
        Cn = np.transpose(data['Cn'][()])
        ids = data['ids'][()] - 1
        Coor_cell_array = data['Coor']
        Coor = []
        C_raw = np.transpose(data['C_raw'][()])
        C = np.transpose(data['C'][()])
        C_denoised = np.transpose(data['C_denoised'][()])
        C_deconvolved = np.transpose(data['C_deconvolved'][()])
        C_reraw = np.transpose(data['C_reraw'][()])
        centroids = np.transpose(data['centroids'][()])
        virmenPath = data['virmenPath'][()].tobytes().decode('utf-16le')

        for i in range(Coor_cell_array.shape[1]):
            ref = Coor_cell_array[0, i]  # Get the reference
            coor_data = file[ref]  # Use the reference to access the data
            coor_matrix = np.array(coor_data)  # Convert to a NumPy array

            Coor.append(coor_matrix)

    return C, C_raw, Cn, ids, Coor, centroids, virmenPath, C_denoised, C_deconvolved, C_reraw


def load_tiff_image(image_path, color="red"):
    """
    Load and process a TIFF image into RGBA format
    :param image_path: Path to the TIFF file
    :param color: "red", "green", or "blue" to specify the color channel
    """
    try:
        # Load the image
        gray_image = imread(image_path)
        
        # Normalize to 0-1 range
        normalized_image = gray_image.astype(float) / gray_image.max()
        
        # Create empty uint32 array
        img = np.empty(gray_image.shape, dtype=np.uint32)
        
        # Create view as uint8 to set RGBA values
        view = img.view(dtype=np.uint8).reshape((*gray_image.shape, 4))
        
        # Set all channels to 0 initially
        view[:, :, :] = 0
        
        # Set the appropriate color channel
        if color == "red":
            view[:, :, 0] = (np.flipud(normalized_image) * 255).astype(np.uint8)  # Red channel
        elif color == "green":
            view[:, :, 1] = (np.flipud(normalized_image) * 255).astype(np.uint8)  # Green channel
        elif color == "blue":
            view[:, :, 2] = (np.flipud(normalized_image) * 255).astype(np.uint8)  # Blue channel
        
        view[:, :, 3] = 255  # Alpha channel (fully opaque)
        
        return img, gray_image.shape
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None


def labeler_bkapp_v2(doc):
    global C, C_raw, ids, labels, image_source, session_name, C_denoised, C_deconvolved, C_reraw, Coor_original, Cn_shape
    filename = ''
    labels = np.zeros((3, 3), dtype=bool)
    labels[:, 2] = True
    Coor_original = None  # Store original coordinate data
    Cn_shape = None  # Store image shape

    """
    ========================================================================================================================
            Bokeh Setup
    ========================================================================================================================
    """

    custom_bright_colors = [
        '#e6194B',  # Bright Red
        '#3cb44b',  # Bright Green
        '#ffe119',  # Bright Yellow
        '#4363d8',  # Bright Blue
        '#f58231',  # Bright Orange
        '#911eb4',  # Bright Purple
        '#42d4f4',  # Bright Cyan
        '#f032e6',  # Bright Magenta
        '#bfef45',  # Bright Lime
        '#fabed4',  # Light Pink
        '#469990',  # Teal
        '#dcbeff',  # Lavender
        '#9A6324',  # Brown
        '#fffac8',  # Beige
        '#800000',  # Maroon
        '#aaffc3',  # Mint
        '#808000',  # Olive
        '#ffd8b1',  # Coral
        '#000075',  # Navy
    ]

    # Our main spatial data source now includes fill_alpha and line_alpha so we can show/hide subsets.
    spatial_source = ColumnDataSource(data={
        'xs': [],
        'ys': [],
        'id': [],
        'colors': [],
        'fill_alpha': [],
        'line_alpha': []
    })

    temporal_source = ColumnDataSource(data={'x': [],
                                             'y_lowpass': [],
                                             'y_raw': [],
                                             'y_denoised': [],
                                             'y_deconvolved': [],
                                             'y_reraw': []
                                             })

    image_source = ColumnDataSource(data={'image': []})

    TOOLS = "crosshair,pan,wheel_zoom,zoom_in,zoom_out,box_zoom,undo,redo,reset,save,box_select,poly_select,lasso_select,examine,help"

    # spatial image
    spatial = figure(title="Neuronal Segmentation", width=800, height=800,
                     active_scroll='wheel_zoom', tools=TOOLS)
    # Hide the grid
    spatial.grid.visible = False
    spatial.xgrid.visible = False
    spatial.ygrid.visible = False
    spatial.image_rgba(image='image', x=0, y=0, dw=1, dh=1, source=image_source)

    contour_renderer = spatial.patches(xs='xs',
                                       ys='ys',
                                       source=spatial_source,
                                       fill_alpha='fill_alpha',
                                       line_alpha='line_alpha',
                                       fill_color='colors',
                                       line_color='colors',
                                       line_width=2)

    hover = HoverTool(tooltips=[("ID", "@id")], renderers=[contour_renderer])
    spatial.add_tools(hover)
    taptool = TapTool(mode='replace', renderers=[contour_renderer])
    spatial.add_tools(taptool)

    # temporal1 transient
    temporal1 = figure(title="Temporal Activity", width=800, height=400, active_scroll="wheel_zoom")
    temporal1.line('x', 'y_lowpass', source=temporal_source, line_width=2, color="red", legend_label="Lowpass")
    temporal1.line('x', 'y_raw', source=temporal_source, line_width=2, color="blue", legend_label="Raw")
    temporal1.legend.click_policy = "hide"
    temporal_tab1 = TabPanel(child=temporal1, title="Raw")

    # temporal2 transient
    temporal2 = figure(title="Temporal Activity", width=800, height=400, x_range=temporal1.x_range,
                       active_scroll="wheel_zoom")
    temporal2.line('x', 'y_deconvolved', source=temporal_source, line_width=2, color="green", alpha=0.7,
                   legend_label="Deconvolved")
    temporal2.line('x', 'y_denoised', source=temporal_source, line_width=2, color="red", alpha=0.7,
                   legend_label="Denoised")
    temporal2.line('x', 'y_reraw', source=temporal_source, line_width=2, color='blue', alpha=0.7, legend_label="Reraw")
    temporal2.legend.click_policy = "hide"
    temporal_tab2 = TabPanel(child=temporal2, title="Rebased")

    # Callback function to print the ID of the selected neuron
    def update_temporal(attr, old, new):
        # 'new' is directly the list of selected indices
        if new:
            selected_index = new[0]  # Get the index of the first selected neuron
            neuron_id = int(spatial_source.data['id'][selected_index][0])
            new_y_lowpass = C[neuron_id]
            new_y_raw = C_raw[neuron_id]
            new_y_denoised = C_denoised[neuron_id]
            new_y_deconvolved = C_deconvolved[neuron_id]
            new_y_reraw = C_reraw[neuron_id]
            new_x = np.arange(0, len(C[neuron_id]) / 20, 0.05)

            temporal_source.data = {
                'x': new_x,
                'y_lowpass': new_y_lowpass,
                'y_raw': new_y_raw,
                'y_denoised': new_y_denoised,
                'y_deconvolved': new_y_deconvolved,
                'y_reraw': new_y_reraw
            }

            temporal1.title.text = f"Temporal Activity: Neuron {neuron_id}"
            temporal2.title.text = f"Temporal Activity: Neuron {neuron_id}"
            neuron_id_slider.value = selected_index

    spatial_source.selected.on_change('indices', update_temporal)

    """
    ========================================================================================================================
            Add Slider to Bokeh layout
    ========================================================================================================================
    """
    neuron_id_slider = Slider(start=0, end=100, value=0, step=1, width=600, title="Neuron ID", disabled=True)

    def update_from_slider(attr, old, new):
        neuron_id = new  # Adjust for zero indexing; slider value starts from 1

        # Update the temporal1 plot
        new_y_lowpass = C[neuron_id]
        new_y_raw = C_raw[neuron_id]
        new_y_denoised = C_denoised[neuron_id]
        new_y_deconvolved = C_deconvolved[neuron_id]
        new_y_reraw = C_reraw[neuron_id]

        new_x = np.arange(0, len(C[neuron_id]) / 20, 0.05)
        temporal_source.data = {
            'x': new_x,
            'y_lowpass': new_y_lowpass,
            'y_raw': new_y_raw,
            'y_denoised': new_y_denoised,
            'y_deconvolved': new_y_deconvolved,
            'y_reraw': new_y_reraw
        }
        temporal1.title.text = f"Temporal Activity: Neuron {neuron_id}"  # Adjust text to match ID
        temporal2.title.text = f"Temporal Activity: Neuron {neuron_id}"  # Adjust text to match ID

        # Highlight the selected neuron in the spatial plot
        spatial_source.selected.indices = [neuron_id]
        neuron_index_input.value = str(neuron_id)
        update_button_styles()

    neuron_id_slider.on_change('value', update_from_slider)

    """
    ========================================================================================================================
            Add Buttons to Bokeh layout
    ========================================================================================================================
    """
    # Create "Previous" and "Next" buttons
    previous_button = Button(label="<<Previous", button_type="primary", disabled=True)
    next_button = Button(label="Next>>", button_type="primary", disabled=True)

    # Callback function for the "Previous" button
    def go_previous():
        current_value = neuron_id_slider.value
        if current_value > np.min(ids):
            neuron_id_slider.value = current_value - 1
            neuron_index_input.value = str(current_value - 1)

    # Callback function for the "Next" button
    def go_next():
        current_value = neuron_id_slider.value
        if current_value < np.max(ids):
            neuron_id_slider.value = current_value + 1
            neuron_index_input.value = str(current_value + 1)

    # Attach the callback functions to the buttons
    previous_button.on_click(go_previous)
    next_button.on_click(go_next)

    """
    ========================================================================================================================
            Add TextInput to Bokeh layout
    ========================================================================================================================
    """
    # Create a TextInput widget for neuron index input
    neuron_index_input = TextInput(value=str(neuron_id_slider.value), title="Neuron Index:", disabled=True)

    # Callback function for the TextInput to update the neuron index
    def update_index(attr, old, new):
        try:
            new_index = int(new)
            # Ensure the new index is within the valid range
            if np.min(ids) <= new_index <= np.max(ids):
                neuron_id_slider.value = new_index
            else:
                print("Input index out of range")
        except ValueError:
            print("Invalid input. Please enter a valid integer.")

    # Attach the callback function to the TextInput widget
    neuron_index_input.on_change('value', update_index)

    """
    ========================================================================================================================
            Labeling Widgets
    ========================================================================================================================
    """
    # Create buttons for labeling
    d1_button = Button(label="D1", button_type="default", disabled=True)
    d2_button = Button(label="D2", button_type="default", disabled=True)
    cholinergic_button = Button(label="Cholinergic", button_type="default", disabled=True)
    unknown_button = Button(label="Unknown", button_type="default", disabled=True)
    discard_button = Button(label="Discard", button_type="default", disabled=True)

    # Create buttons for saving and loading labels and a TextInput for the file path
    save_labels_button = Button(label="Save Labels", button_type="success", disabled=True)
    load_labels_button = Button(label="Load Labels", button_type="primary", disabled=True)
    file_path_input = TextInput(value="labels.json", title="File Path:", width=400, disabled=True)

    # Initialize labels dictionary (string keys for each shape index)
    labels = {}

    """
    ========================================================================================================================
             Toggles for D1, D2, Unknown, Discard
    ========================================================================================================================
    """
    toggle_d1 = Toggle(label="Show D1", button_type="success", active=True)
    toggle_d2 = Toggle(label="Show D2", button_type="success", active=True)
    toggle_cholinergic = Toggle(label="Show Cholinergic", button_type="success", active=True)
    toggle_unknown = Toggle(label="Show Unknown", button_type="success", active=True)
    toggle_discard = Toggle(label="Show Discard", button_type="success", active=True)

    # Toggle for coordinate transformation (XY swap only; Y-flip is always applied)
    toggle_transform = Toggle(label="Swap X-Y Coordinates", button_type="primary", active=False, disabled=True)

    def apply_coordinate_transform(Coor_data, height, apply_xy_swap=True):
        """
        Apply coordinate transformation on mask coordinates
        :param Coor_data: List of coordinate arrays for each neuron
        :param height: Image height for y-axis flipping
        :param apply_xy_swap: If True, swap x and y coordinates (diagonal transform);
                              Y-axis flip is ALWAYS applied regardless (Bokeh requirement)
        :return: Transformed x and y position lists
        """
        x_positions_all = []
        y_positions_all = []

        for i in range(len(Coor_data)):
            if apply_xy_swap:
                # Apply coordinate swap (diagonal transformation)
                x_positions = Coor_data[i][:, 1]  # Swap: use column 1 for x
                y_positions = Coor_data[i][:, 0]  # Swap: use column 0 for y
            else:
                # Use original column order (no swap)
                x_positions = Coor_data[i][:, 0]  # Use column 0 for x
                y_positions = Coor_data[i][:, 1]  # Use column 1 for y

            x_positions_all.append(x_positions)
            y_positions_all.append(y_positions)

        # ALWAYS flip y-coordinates vertically (Bokeh coordinate system requirement)
        y_positions_all = [[height - y for y in y_list] for y_list in y_positions_all]

        return x_positions_all, y_positions_all

    def update_toggle_colors():
        """Update toggle button colors and labels based on their state"""
        for toggle, label_base in [(toggle_d1, "D1"), (toggle_d2, "D2"), 
                                 (toggle_cholinergic, "Cholinergic"), (toggle_unknown, "Unknown"), (toggle_discard, "Discard")]:
            toggle.button_type = "danger" if toggle.active else "success"
            toggle.label = f"Hide {label_base}" if toggle.active else f"Show {label_base}"

    def toggle_callback(toggle):
        """Generic callback for all toggles"""
        update_toggle_colors()
        update_mask_visibility()

    # Update toggle callbacks
    toggle_d1.on_change('active', lambda attr, old, new: toggle_callback(toggle_d1))
    toggle_d2.on_change('active', lambda attr, old, new: toggle_callback(toggle_d2))
    toggle_cholinergic.on_change('active', lambda attr, old, new: toggle_callback(toggle_cholinergic))
    toggle_unknown.on_change('active', lambda attr, old, new: toggle_callback(toggle_unknown))
    toggle_discard.on_change('active', lambda attr, old, new: toggle_callback(toggle_discard))

    def toggle_transform_callback(_attr, _old, new):
        """Callback for XY coordinate swap toggle"""
        if Coor_original is None or Cn_shape is None:
            return  # Data not loaded yet

        # Apply or remove XY coordinate swap (Y-flip always applied)
        # Note: Button logic is inverted - active=True means DO swap
        height, _width = Cn_shape
        x_positions_all, y_positions_all = apply_coordinate_transform(Coor_original, height, new)

        # Update spatial_source with transformed coordinates
        spatial_source.data['xs'] = x_positions_all
        spatial_source.data['ys'] = y_positions_all

        # Update button appearance
        toggle_transform.button_type = "success" if new else "primary"
        toggle_transform.label = "X-Y Swapped" if new else "Original X-Y Order"

    toggle_transform.on_change('active', toggle_transform_callback)

    def update_mask_visibility():
        """Update visibility of masks based on their labels and toggle states"""
        new_fill_alpha = []
        new_line_alpha = []
        
        # Get current lists of neurons from dropdowns (excluding the default option)
        d1_neurons = [int(x) for x in d1_neurons_select.options if x != "Click to see options..."]
        d2_neurons = [int(x) for x in d2_neurons_select.options if x != "Click to see options..."]
        cholinergic_neurons = [int(x) for x in cholinergic_neurons_select.options if x != "Click to see options..."]
        unknown_neurons = [int(x) for x in unknown_neurons_select.options if x != "Click to see options..."]
        discard_neurons = [int(x) for x in discard_neurons_select.options if x != "Click to see options..."]
        
        for i in range(len(spatial_source.data['xs'])):
            # Check which list the neuron belongs to
            if i in d1_neurons:
                visible = toggle_d1.active
            elif i in d2_neurons:
                visible = toggle_d2.active
            elif i in discard_neurons:
                visible = toggle_discard.active
            elif i in unknown_neurons:
                visible = toggle_unknown.active
            elif i in cholinergic_neurons:
                visible = toggle_cholinergic.active
            else:
                visible = toggle_unknown.active  # Default to unknown if not found
            
            new_fill_alpha.append(0.2 if visible else 0)
            new_line_alpha.append(1 if visible else 0)
        
        # Update the source data with new alpha values
        spatial_source.data.update({
            'fill_alpha': new_fill_alpha,
            'line_alpha': new_line_alpha
        })

    # Make sure to call update_mask_visibility after loading data and after updating labels
    def update_labels(new_label):
        selected_neuron = str(neuron_id_slider.value)
        labels[selected_neuron] = new_label
        update_button_styles()
        update_labeled_count()
        update_dropdowns()
        update_mask_visibility()

    def save_labels(event):
        """Save the current labels to a JSON file"""
        try:
            session_name = sessionname_input.value
            if not session_name:
                status_div.text = "<span style='color: red;'>Error: Session name is empty!</span>"
                return

            with open(CONFIG_PATH, 'r') as file:
                config = json.load(file)

            # Construct save path
            base_path = config['ProcessedFilePath']
            save_dir = os.path.join(base_path, session_name, f'{session_name}_neuron_labels')
            os.makedirs(save_dir, exist_ok=True)

            # Base filename without extension
            base_filename = f'{session_name}_neuron_labels'

            # Find the next available number
            counter = 0
            while True:
                # For the first file, don't add a number
                if counter == 0:
                    filename = f'{base_filename}.json'
                else:
                    filename = f'{base_filename}({counter}).json'

                save_path = os.path.join(save_dir, filename)
                if not os.path.exists(save_path):
                    break
                counter += 1

            # Update the label path input box with the new filename
            file_path_input.value = filename

            # Check if labels dictionary is not empty
            if not labels:
                status_div.text = "<span style='color: red;'>Error: No labels to save!</span>"
                return

            # Save the labels
            with open(save_path, 'w') as f:
                json.dump(labels, f)

            # Verify the file was created
            if os.path.exists(save_path):
                file_size = os.path.getsize(save_path)
                status_div.text = (f"<span style='color: green;'>Labels saved successfully to {os.path.basename(save_path)}! "
                                   f"File size: {file_size} bytes</span>")
            else:
                status_div.text = f"<span style='color: red;'>Error: File was not created at {save_path}</span>"

        except FileNotFoundError as e:
            status_div.text = f"<span style='color: red;'>Error: Could not find path: {str(e)}</span>"
        except PermissionError as e:
            status_div.text = f"<span style='color: red;'>Error: Permission denied. Cannot write to: {str(e)}</span>"
        except json.JSONDecodeError as e:
            status_div.text = f"<span style='color: red;'>Error: Invalid JSON in config file: {str(e)}</span>"
        except Exception as e:
            status_div.text = f"<span style='color: red;'>Unexpected error: {str(e)}</span>"

    # Callbacks for label buttons
    d1_button.on_click(lambda: update_labels("D1"))
    d2_button.on_click(lambda: update_labels("D2"))
    cholinergic_button.on_click(lambda: update_labels("cholinergic"))
    unknown_button.on_click(lambda: update_labels("unknown"))
    discard_button.on_click(lambda: update_labels("discard"))
    save_labels_button.on_click(save_labels)

    # Function to update button styles based on current label
    def update_button_styles():
        selected_neuron = str(neuron_id_slider.value)
        current_label = labels.get(selected_neuron, "unknown")

        # Reset all buttons to default
        d1_button.button_type = "default"
        d2_button.button_type = "default"
        cholinergic_button.button_type = "default"
        unknown_button.button_type = "default"
        discard_button.button_type = "default"

        # Highlight the active label
        if current_label == "D1":
            d1_button.button_type = "success"
        elif current_label == "D2":
            d2_button.button_type = "success"
        elif current_label == "cholinergic":
            cholinergic_button.button_type = "success"
        elif current_label == "discard":
            discard_button.button_type = "success"
        else:  # unknown
            unknown_button.button_type = "success"

    """
    ========================================================================================================================
            Div Plain Text Bokeh
    ========================================================================================================================
    """
    labeled_count_div = Div(text="D1: 0 | D2: 0 | Cholinergic: 0 | Unknown: 0 | Discard: 0", width=400, height=30)

    # Function to update the counts displayed in the Div
    def update_labeled_count():
        d1_count = sum(1 for label in labels.values() if label == "D1")
        d2_count = sum(1 for label in labels.values() if label == "D2")
        cholinergic_count = sum(1 for label in labels.values() if label == "cholinergic")
        unknown_count = sum(1 for label in labels.values() if label == "unknown")
        discard_count = sum(1 for label in labels.values() if label == "discard")
        labeled_count_div.text = f"D1: {d1_count} | D2: {d2_count} | Cholinergic: {cholinergic_count} | Unknown: {unknown_count} | Discard: {discard_count}"

    """
    ========================================================================================================================
            Select Menus
    ========================================================================================================================
    """
    d1_neurons_select = Select(title="D1 Neurons", options=[], disabled=True)
    d2_neurons_select = Select(title="D2 Neurons", options=[], disabled=True)
    cholinergic_neurons_select = Select(title="Cholinergic Neurons", options=[], disabled=True)
    unknown_neurons_select = Select(title="Unknown Neurons", options=[], disabled=True)
    discard_neurons_select = Select(title="Discarded Neurons", options=[], disabled=True)

    def update_dropdowns():
        """Update dropdown menus with current labels"""
        # Sort neurons by their ID (converting to int for proper numerical sorting)
        d1_ids = sorted([i for i, label in labels.items() if label == "D1"], key=int)
        d2_ids = sorted([i for i, label in labels.items() if label == "D2"], key=int)
        cholinergic_ids = sorted([i for i, label in labels.items() if label == "cholinergic"], key=int)
        unknown_ids = sorted([i for i, label in labels.items() if label == "unknown"], key=int)
        discard_ids = sorted([i for i, label in labels.items() if label == "discard"], key=int)

        # Update dropdown options
        d1_neurons_select.options = ["Click to see options..."] + d1_ids
        d2_neurons_select.options = ["Click to see options..."] + d2_ids
        cholinergic_neurons_select.options = ["Click to see options..."] + cholinergic_ids
        unknown_neurons_select.options = ["Click to see options..."] + unknown_ids
        discard_neurons_select.options = ["Click to see options..."] + discard_ids

        # Reset selections to default
        d1_neurons_select.value = "Click to see options..."
        d2_neurons_select.value = "Click to see options..."
        cholinergic_neurons_select.value = "Click to see options..."
        unknown_neurons_select.value = "Click to see options..."
        discard_neurons_select.value = "Click to see options..."

    def neuron_selected(attr, old, new):
        if new and new != "Click to see options...":
            neuron_id_slider.value = int(new)

    # Attach the callback to all three select widgets
    d1_neurons_select.on_change('value', neuron_selected)
    d2_neurons_select.on_change('value', neuron_selected)
    cholinergic_neurons_select.on_change('value', neuron_selected)
    unknown_neurons_select.on_change('value', neuron_selected)
    discard_neurons_select.on_change('value', neuron_selected)

    """
    ========================================================================================================================
            Add Widgets to Update Loaded File
    ========================================================================================================================
    """

    # Filename input (supports both session name and file path)
    sessionname_input = TextInput(value=filename, title="Session Name or File Path:", width=400)
    load_data_button = Button(label="Load Data", button_type="success")

    def load_and_update_data(filename):
        """
        Load the data from .mat and update global C, C_raw, etc.
        Then re-populate the spatial_source with new shapes and
        reset labels/data for all neurons to 'unknown' by default.
        """
        global C, C_raw, ids, labels, image_source, C_denoised, C_deconvolved, C_reraw, Coor_original, Cn_shape
        # Enable navigation controls
        neuron_id_slider.disabled = False
        neuron_index_input.disabled = False
        next_button.disabled = False
        previous_button.disabled = False

        # Enable labeling buttons
        d1_button.disabled = False
        d2_button.disabled = False
        cholinergic_button.disabled = False
        unknown_button.disabled = False
        discard_button.disabled = False
        save_labels_button.disabled = False
        load_labels_button.disabled = False  # Enable load labels button

        # Enable select menus
        d1_neurons_select.disabled = False
        d2_neurons_select.disabled = False
        cholinergic_neurons_select.disabled = False
        unknown_neurons_select.disabled = False
        discard_neurons_select.disabled = False

        # Enable file input
        file_path_input.disabled = False

        gcamp_input.disabled = False
        orig_tdt_input.disabled = False
        bfp_input.disabled = False

        gcamp_input.value = f"{session_name}_max.tif"
        orig_tdt_input.value = "m_f_n_001_tdt_max.tif"
        bfp_input.value = "m_f_n_001_bfp_max.tif"

        # Load the data
        [C, C_raw, Cn, ids, Coor, centroids, virmenPath, C_denoised, C_deconvolved, C_reraw] = load_data(filename)

        # Store original coordinate data and image shape for coordinate transformation
        Coor_original = Coor
        Cn_shape = Cn.shape[:2]

        # Enable coordinate transformation toggle
        toggle_transform.disabled = False

        # Initialize labels as a dictionary with all neurons set to "unknown"
        labels = {str(i): "unknown" for i in range(len(C))}

        num_shapes = len(Coor)
        height, width = Cn.shape[:2]

        # Apply coordinate transformation based on toggle state
        x_positions_all, y_positions_all = apply_coordinate_transform(Coor, height, toggle_transform.active)

        colors = [custom_bright_colors[i % len(custom_bright_colors)] for i in range(num_shapes)]

        # Update spatial_source with new data
        spatial_source.data = {
            'xs': x_positions_all,
            'ys': y_positions_all,
            'id': ids,
            'colors': [colors[i % len(colors)] for i in range(num_shapes)],
            'fill_alpha': [0.2]*num_shapes,
            'line_alpha': [1]*num_shapes,
        }

        # Update temporal_source with data from the first neuron
        temporal_source.data = {
            'x': np.arange(0, len(C[0]) / 20, 0.05),
            'y_lowpass': C[0],
            'y_raw': C_raw[0],
            'y_denoised': C_denoised[0],
            'y_deconvolved': C_deconvolved[0],
            'y_reraw': C_reraw[0]
        }

        # Reset or update widgets
        neuron_id_slider.start = np.min(ids)
        neuron_id_slider.end = np.max(ids)
        neuron_id_slider.value = 0
        neuron_index_input.value = '0'
        update_button_styles()
        update_labeled_count()
        update_dropdowns()
        update_mask_visibility()  # Apply toggles if needed

        # Clear any existing image
        image_source.data = {'image': []}

        # Update plot ranges
        spatial.x_range.start = 0
        spatial.x_range.end = width
        spatial.y_range.start = 0
        spatial.y_range.end = height

        # Remove existing patch renderers before adding new ones
        spatial.renderers = [
            r for r in spatial.renderers
            if not (isinstance(r, GlyphRenderer) and isinstance(r.glyph, Patches))
        ]

        # Add the new patches
        contour_renderer = spatial.patches(
            xs='xs',
            ys='ys',
            source=spatial_source,
            fill_alpha='fill_alpha',
            line_alpha='line_alpha',
            fill_color='colors',
            line_color='colors',
            line_width=2,
            level='overlay'
        )

        # Remove old hover and tap tools, add new ones
        spatial.tools = [tool for tool in spatial.tools 
                        if not isinstance(tool, (HoverTool, TapTool))]
        hover = HoverTool(tooltips=[("ID", "@id")], renderers=[contour_renderer])
        spatial.add_tools(hover)
        taptool = TapTool(mode='replace', renderers=[contour_renderer])
        spatial.add_tools(taptool)

        # Update titles
        spatial.title.text = "Neuronal Segmentation"
        temporal1.title.text = "Temporal Activity: Neuron 0"
        temporal2.title.text = "Temporal Activity: Neuron 0"
        file_path_input.value = f"{session_name}_neuron_labels.json"

        # Update document title
        doc.title = f"Data Loaded: {filename}"

        # Add these lines after initializing spatial_source:
        spatial_source.data['fill_alpha'] = [0.2] * len(spatial_source.data['xs'])
        spatial_source.data['line_alpha'] = [1] * len(spatial_source.data['xs'])

    def update_data():
        global session_name
        input_value = sessionname_input.value.strip()

        # Auto-detect if input is a file path or session name
        if "/" in input_value or "\\" in input_value or input_value.endswith('.mat'):
            # Treat as file path
            neuron_path = os.path.normpath(input_value)

            # Check if it's an absolute path
            if not os.path.isabs(neuron_path):
                # If relative path, make it relative to the current working directory
                neuron_path = os.path.abspath(neuron_path)

            # Extract session name from the file path for saving labels
            # Assume the session name is the parent directory name
            session_name = os.path.basename(os.path.dirname(neuron_path))
        else:
            # Treat as session name (original behavior)
            with open(CONFIG_PATH, 'r') as file:
                config = json.load(file)
            session_name = input_value
            neuron_path = os.path.join(config['ProcessedFilePath'], session_name, f'{session_name}_v7.mat')

        # Verify the file exists
        if not os.path.exists(neuron_path):
            status_div.text = f"<span style='color: red;'>Error: File not found at {neuron_path}</span>"
            return

        load_and_update_data(neuron_path)
        print(f"{neuron_path} loaded!")

    load_data_button.on_click(update_data)

    """
    ========================================================================================================================
            Additional Image Loading Functionality
    ========================================================================================================================
    """
    # Create input fields for different image types with default filenames
    gcamp_input = TextInput(value=f"max.tif", title="GCaMP Max Projection", width=400)
    orig_tdt_input = TextInput(value=f"m_f_n_001_tdt_max.tif", title="TDT (D1)", width=400)
    bfp_input = TextInput(value=f"m_f_n_001_bfp_max.tif", title="BFP (Cholinergic)", width=400)

    gcamp_input.disabled = True
    orig_tdt_input.disabled = True
    bfp_input.disabled = True

    # Create load buttons for each image type
    load_gcamp_button = Button(label="Load GCaMP Image", button_type="success")
    load_orig_tdt_button = Button(label="Load TDT Image", button_type="success")
    load_bfp_button = Button(label="Load BFP Image", button_type="success")
    remove_image_button = Button(label="Remove Image", button_type="danger")

    # Status message
    status_div = Div(text="", width=400)

    def load_specific_image(image_type, image_input):
        """
        Load a specific image type (GCaMP, original TDT, or adjusted TDT)
        Supports both absolute paths and relative paths within the session structure
        """
        try:
            # Save current view state
            current_x_range = (spatial.x_range.start, spatial.x_range.end)
            current_y_range = (spatial.y_range.start, spatial.y_range.end)

            input_value = image_input.value
            
            # Check if the input contains path separators (absolute or relative path)
            if "/" in input_value or "\\" in input_value:
                # Treat as absolute or relative path
                image_path = input_value
                # Convert forward slashes to platform-appropriate separators
                image_path = os.path.normpath(image_path)
            else:
                # Treat as filename only - use existing folder structure
                with open(CONFIG_PATH, 'r') as file:
                    config = json.load(file)

                session_name = sessionname_input.value
                base_path = config['ProcessedFilePath']

                # Construct the image path based on type
                if image_type == "gcamp":
                    folder_name = f'{session_name}_tiff_projections'
                    filename = input_value
                    image_path = os.path.join(base_path, session_name, folder_name, filename)
                elif image_type == "orig_tdt":
                    folder_name = f'{session_name}_alignment_check'
                    filename = input_value
                    image_path = os.path.join(base_path, session_name, folder_name, filename)
                elif image_type == "bfp":
                    folder_name = f'{session_name}_alignment_check'
                    filename = input_value
                    image_path = os.path.join(base_path, session_name, folder_name, filename)

            # Set color based on image type
            if image_type == "gcamp":
                color = "green"
            elif image_type == "orig_tdt":
                color = "red"
            elif image_type == "bfp":
                color = "blue"

            # Load and process the image
            img, shape = load_tiff_image(image_path, color)
            if img is not None:
                # Remove only the existing image ren1erers
                spatial.renderers = [r for r in spatial.renderers if not isinstance(r, ImageRGBA)]

                # Add the image with a level between underlay and overlay
                image_renderer = spatial.image_rgba(image='image',
                                                    x=0, y=0,
                                                    dw=shape[1], dh=shape[0],
                                                    source=image_source,
                                                    level='glyph')

                # Update image source with the new image
                image_source.data = {'image': [img]}

                # Only update ranges if they're not already set (i.e., first load)
                if current_x_range[0] is None or current_x_range[1] is None:
                    spatial.x_range.start = 0
                    spatial.x_range.end = shape[1]
                    spatial.y_range.start = 0
                    spatial.y_range.end = shape[0]
                else:
                    # Restore previous view state
                    spatial.x_range.start = current_x_range[0]
                    spatial.x_range.end = current_x_range[1]
                    spatial.y_range.start = current_y_range[0]
                    spatial.y_range.end = current_y_range[1]

                status_div.text = f"<span style='color: green;'>{image_type.upper()} image loaded successfully!</span>"
            else:
                status_div.text = f"<span style='color: red;'>Failed to load {image_type} image!</span>"

        except Exception as e:
            status_div.text = f"<span style='color: red;'>Error: {str(e)}</span>"

    def remove_image():
        """Remove the currently displayed image"""
        image_source.data = {'image': []}
        status_div.text = "<span style='color: blue;'>Image removed</span>"

    # Callbacks for the load buttons
    load_gcamp_button.on_click(lambda: load_specific_image("gcamp", gcamp_input))
    load_orig_tdt_button.on_click(lambda: load_specific_image("orig_tdt", orig_tdt_input))
    load_bfp_button.on_click(lambda: load_specific_image("bfp", bfp_input))
    remove_image_button.on_click(remove_image)

    # Create rows for the new image loading controls
    gcamp_row = row(gcamp_input, column(Spacer(height=20), load_gcamp_button))
    orig_tdt_row = row(orig_tdt_input, column(Spacer(height=20), load_orig_tdt_button))
    bfp_row = row(bfp_input, column(Spacer(height=20), load_bfp_button))
    # Add the new controls to the layout
    image_controls = column(
        gcamp_row,
        orig_tdt_row,
        bfp_row,
        row(remove_image_button)
    )

    """
    ========================================================================================================================
            Loading Labels from File
    ========================================================================================================================
    """
    def load_labels(event):
        """Load labels from a JSON file and update the UI."""
        try:
            session_name = sessionname_input.value
            if not session_name:
                status_div.text = "<span style='color: red;'>Error: Session name is empty!</span>"
                return

            with open(CONFIG_PATH, 'r') as file:
                config = json.load(file)

            # Construct path
            label_filename = file_path_input.value
            base_path = config['ProcessedFilePath']
            labels_path = os.path.join(base_path, session_name, f'{session_name}_neuron_labels', label_filename)

            # Load the labels
            if os.path.exists(labels_path):
                with open(labels_path, 'r') as f:
                    # We replace the global labels dict
                    global labels
                    labels = json.load(f)

                # Update all UI elements
                update_button_styles()
                update_labeled_count()
                update_dropdowns()
                update_mask_visibility()  # reflect loaded labels
                status_div.text = f"<span style='color: green;'>Labels loaded successfully from {label_filename}!</span>"
            else:
                status_div.text = f"<span style='color: red;'>Error: Label file not found at {labels_path}</span>"

        except FileNotFoundError as e:
            status_div.text = f"<span style='color: red;'>Error: Could not find file: {str(e)}</span>"
        except json.JSONDecodeError as e:
            status_div.text = f"<span style='color: red;'>Error: Invalid JSON in labels file: {str(e)}</span>"
        except Exception as e:
            status_div.text = f"<span style='color: red;'>Unexpected error: {str(e)}</span>"

    # Add callback for load button
    load_labels_button.on_click(load_labels)

    """
    ========================================================================================================================
            Setup Bokeh layout
    ========================================================================================================================
    """
    spacer1 = Spacer(height=50)
    spacer2 = Spacer(height=20)
    spacer3 = Spacer(width=20)

    choose_file = row(spacer3, sessionname_input, column(spacer2, load_data_button))

    controls = row(
        spacer3,
        column(
            spacer1,
            row(previous_button, next_button, neuron_index_input),
            neuron_id_slider
        )
    )

    labelling = row(
        spacer3,
        column(
            row(d1_button, d2_button, cholinergic_button, unknown_button, discard_button),
            row(file_path_input, column(spacer2, row(save_labels_button, load_labels_button))),
            labeled_count_div
        )
    )

    menus = row(spacer3, row(d1_neurons_select, d2_neurons_select, cholinergic_neurons_select, unknown_neurons_select, discard_neurons_select))

    temporal = Tabs(tabs=[temporal_tab1, temporal_tab2])

    # Our new row of toggles:
    toggle_row = row(spacer3, toggle_d1, toggle_d2, toggle_cholinergic, toggle_unknown, toggle_discard)

    # Coordinate transformation toggle in a separate row
    transform_row = row(spacer3, toggle_transform)

    layout = row(
        column(
            spatial,
            row(Spacer(width=40), image_controls)
        ),
        column(
            choose_file,
            controls,
            toggle_row,
            transform_row,
            temporal,
            labelling,
            menus,
            status_div
        )
    )

    doc.add_root(layout)
