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

            Coor.append(np.transpose(coor_matrix))

    return C, C_raw, Cn, ids, Coor, centroids, virmenPath, C_denoised, C_deconvolved, C_reraw


def load_tiff_image(image_path, color="red"):
    """
    Load and process a TIFF image into RGBA format
    :param image_path: Path to the TIFF file
    :param color: "red" or "green" to specify the color channel
    """
    try:
        # Load the image
        gray_image = imread(image_path)

        gray_image = np.transpose(gray_image)
        
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
        
        view[:, :, 3] = 255  # Alpha channel (fully opaque)
        
        return img, gray_image.shape
    except Exception as e:
        print(f"Error loading image: {e}")
        return None, None


def labeler_bkapp_v1(doc):
    global C, C_raw, ids, labels, image_source, session_name, C_denoised, C_deconvolved, C_reraw
    filename = ''
    labels = np.zeros((3, 3), dtype=bool)
    labels[:, 2] = True

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

    spatial_source = ColumnDataSource(data={
        'xs': [],
        'ys': [],
        'id': [],
        'colors': [],
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

    contour_renderer = spatial.patches(xs='xs', ys='ys', source=spatial_source,
                                       fill_alpha=0.2, line_alpha=1, fill_color='colors', line_color='colors')

    hover = HoverTool(tooltips=[("ID", "@id")], renderers=[contour_renderer])
    spatial.add_tools(hover)
    taptool = TapTool(mode='replace')
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
    unknown_button = Button(label="Unknown", button_type="default", disabled=True)

    # Create a button for saving labels and a TextInput for the file path
    save_labels_button = Button(label="Save Labels", button_type="success", disabled=True)
    file_path_input = TextInput(value="labels.json", title="File Path:", disabled=True)

    # Initialize labels dictionary
    labels = {}

    # Function to update labels based on button click
    def update_labels(new_label):
        selected_neuron = str(neuron_id_slider.value)  # Convert to string for JSON compatibility
        labels[selected_neuron] = new_label
        update_button_styles()
        update_labeled_count()
        update_dropdowns()

    def save_labels(event):
        """Save the current labels to a JSON file"""
        try:
            session_name = sessionname_input.value
            if not session_name:
                status_div.text = "<span style='color: red;'>Error: Session name is empty!</span>"
                return

            status_div.text = f"<span style='color: blue;'>Debug: Reading config from {CONFIG_PATH}</span>"
            with open(CONFIG_PATH, 'r') as file:
                config = json.load(file)
            
            # Construct save path
            base_path = config['ProcessedFilePath']
            save_dir = os.path.join(base_path, session_name, f'{session_name}_neuron_labels')
            
            status_div.text = f"<span style='color: blue;'>Debug: Attempting to create directory: {save_dir}</span>"
            # Create directory if it doesn't exist
            os.makedirs(save_dir, exist_ok=True)
            
            # Save path for the labels
            save_path = os.path.join(save_dir, f'{session_name}_neuron_labels.json')
            
            status_div.text = f"<span style='color: blue;'>Debug: Attempting to save labels to: {save_path}</span>"
            
            # Check if labels dictionary is not empty
            if not labels:
                status_div.text = "<span style='color: red;'>Error: No labels to save!</span>"
                return
            
            status_div.text = f"<span style='color: blue;'>Debug: Labels content: {str(labels)[:100]}...</span>"
            
            # Save the labels
            with open(save_path, 'w') as f:
                json.dump(labels, f)
            
            # Verify the file was created
            if os.path.exists(save_path):
                file_size = os.path.getsize(save_path)
                status_div.text = (f"<span style='color: green;'>Labels saved successfully to {save_path}! "
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

    # Callbacks for buttons
    d1_button.on_click(lambda: update_labels("D1"))
    d2_button.on_click(lambda: update_labels("D2"))
    unknown_button.on_click(lambda: update_labels("unknown"))
    save_labels_button.on_click(save_labels)

    # Function to update button styles based on current label
    def update_button_styles():
        selected_neuron = str(neuron_id_slider.value)
        current_label = labels.get(selected_neuron, "unknown")
        
        # Reset all buttons to default
        d1_button.button_type = "default"
        d2_button.button_type = "default"
        unknown_button.button_type = "default"
        
        # Highlight the active label
        if current_label == "D1":
            d1_button.button_type = "success"
        elif current_label == "D2":
            d2_button.button_type = "success"
        else:  # unknown
            unknown_button.button_type = "success"

    """
    ========================================================================================================================
            Div Plain Text Bokeh
    ========================================================================================================================
    """
    labeled_count_div = Div(text="D1: 0 | D2: 0 | Unknown: 0", width=400, height=30)

    # Function to update the counts displayed in the Div
    def update_labeled_count():
        d1_count = sum(1 for label in labels.values() if label == "D1")
        d2_count = sum(1 for label in labels.values() if label == "D2")
        unknown_count = sum(1 for label in labels.values() if label == "unknown")
        labeled_count_div.text = f"D1: {d1_count} | D2: {d2_count} | Unknown: {unknown_count}"

    """
    ========================================================================================================================
            Select Menus
    ========================================================================================================================
    """
    d1_neurons_select = Select(title="D1 Neurons", options=[], disabled=True)
    d2_neurons_select = Select(title="D2 Neurons", options=[], disabled=True)
    unknown_neurons_select = Select(title="Unknown Neurons", options=[], disabled=True)

    def update_dropdowns():
        """Update dropdown menus with current labels"""
        # Sort neurons by their ID (converting to int for proper numerical sorting)
        d1_ids = sorted([i for i, label in labels.items() if label == "D1"], key=int)
        d2_ids = sorted([i for i, label in labels.items() if label == "D2"], key=int)
        unknown_ids = sorted([i for i, label in labels.items() if label == "unknown"], key=int)

        # Update dropdown options
        d1_neurons_select.options = ["Click to see options..."] + d1_ids
        d2_neurons_select.options = ["Click to see options..."] + d2_ids
        unknown_neurons_select.options = ["Click to see options..."] + unknown_ids

        # Reset selections to default
        d1_neurons_select.value = "Click to see options..."
        d2_neurons_select.value = "Click to see options..."
        unknown_neurons_select.value = "Click to see options..."

    def neuron_selected(attr, old, new):
        if new and new != "Click to see options...":
            neuron_id_slider.value = int(new)

    # Attach the callback to all three select widgets
    d1_neurons_select.on_change('value', neuron_selected)
    d2_neurons_select.on_change('value', neuron_selected)
    unknown_neurons_select.on_change('value', neuron_selected)

    """
    ========================================================================================================================
            Add Widgets to Update Loaded File
    ========================================================================================================================
    """

    # Filename input
    sessionname_input = TextInput(value=filename, title="SessionName:", width=400)
    load_data_button = Button(label="Load Data", button_type="success")

    def load_and_update_data(filename):
        global C, C_raw, ids, labels, image_source, C_denoised, C_deconvolved, C_reraw
        # Enable navigation controls
        neuron_id_slider.disabled = False
        neuron_index_input.disabled = False
        next_button.disabled = False
        previous_button.disabled = False
        
        # Enable labeling buttons
        d1_button.disabled = False
        d2_button.disabled = False
        unknown_button.disabled = False
        save_labels_button.disabled = False
        
        # Enable select menus
        d1_neurons_select.disabled = False
        d2_neurons_select.disabled = False
        unknown_neurons_select.disabled = False
        
        # Enable file input
        file_path_input.disabled = False

        # Load the data
        [C, C_raw, Cn, ids, Coor, centroids, virmenPath, C_denoised, C_deconvolved, C_reraw] = load_data(filename)
        
        # Initialize labels as a dictionary with all neurons set to "unknown"
        labels = {str(i): "unknown" for i in range(len(C))}
        
        num_shapes = len(Coor)
        height, width = Cn.shape[:2]

        # Prepare data for Bokeh
        x_positions_all = []
        y_positions_all = []

        for i in range(num_shapes):
            x_positions = Coor[i][0]
            y_positions = Coor[i][1]

            # Close the shape by ensuring the first point is repeated at the end
            x_positions_all.append(x_positions)
            y_positions_all.append(y_positions)

        y_positions_all = [[height - y for y in y_list] for y_list in y_positions_all]

        colors = [custom_bright_colors[i % len(custom_bright_colors)] for i in range(num_shapes)]

        # Update spatial_source with new data
        spatial_source.data = {
            'xs': x_positions_all,
            'ys': y_positions_all,
            'id': ids,
            'colors': [colors[i % len(colors)] for i in range(num_shapes)],
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

        # Clear any existing image
        image_source.data = {'image': []}
        
        # Update plot ranges
        spatial.x_range.start = 0
        spatial.x_range.end = width
        spatial.y_range.start = 0
        spatial.y_range.end = height
        
        # Update contour renderer
        contour_renderer = spatial.patches(xs='xs', ys='ys', source=spatial_source,
                                         fill_alpha=0.2, line_alpha=1, fill_color='colors', line_color='colors')

        # Update titles
        spatial.title.text = "Neuronal Segmentation"
        temporal1.title.text = "Temporal Activity: Neuron 0"
        temporal2.title.text = "Temporal Activity: Neuron 0"

        # Update document title
        doc.title = f"Data Loaded: {filename}"

    # Callback function for the "Load Data" button to reload and update the visualization
    def update_data():
        global session_name

        with open(CONFIG_PATH, 'r') as file:
            config = json.load(file)
        session_name = sessionname_input.value
        neuron_path = os.path.join(config['ProcessedFilePath'], session_name, f'{session_name}_v7.mat')
        load_and_update_data(neuron_path)
        print(f"{neuron_path} loaded!")

    load_data_button.on_click(update_data)

    """
    ========================================================================================================================
            Additional Image Loading Functionality
    ========================================================================================================================
    """
    # Create input fields for different image types with default filenames
    gcamp_input = TextInput(value="max.tif", title="GCaMP Max Projection", width=400)
    orig_tdt_input = TextInput(value="tdt_original_16bit.tif", title="Original TDT", width=400)
    adj_tdt_input = TextInput(value="tdt_adjusted_16bit.tif", title="Adjusted TDT", width=400)

    # Create load buttons for each image type
    load_gcamp_button = Button(label="Load GCaMP Image", button_type="success")
    load_orig_tdt_button = Button(label="Load Original TDT", button_type="success")
    load_adj_tdt_button = Button(label="Load Adjusted TDT", button_type="success")
    remove_image_button = Button(label="Remove Image", button_type="danger")

    # Status message
    status_div = Div(text="", width=400)

    def load_specific_image(image_type, image_input):
        """
        Load a specific image type (GCaMP, original TDT, or adjusted TDT)
        """
        try:
            with open(CONFIG_PATH, 'r') as file:
                config = json.load(file)
            
            session_name = sessionname_input.value
            base_path = config['ProcessedFilePath']
            
            # Construct the image path based on type
            if image_type == "gcamp":
                image_path = os.path.join(base_path, session_name, f'{session_name}_tiff_projections', f'{session_name}_max.tif')
                color = "green"
            elif image_type == "orig_tdt":
                image_path = os.path.join(base_path, session_name, f'{session_name}_alignment_check', f'{session_name}_tdt_original_16bit.tif')
                color = "red"
            elif image_type == "adj_tdt":
                image_path = os.path.join(base_path, session_name, f'{session_name}_tdt_adjustment', f'{session_name}_tdt_adjusted_16bit.tif')
                color = "red"
            
            # Load and process the image
            img, shape = load_tiff_image(image_path, color)
            if img is not None:
                # Remove only the existing image renderers
                spatial.renderers = [r for r in spatial.renderers if not isinstance(r, ImageRGBA)]
                
                # Add the image with a level between underlay and overlay
                image_renderer = spatial.image_rgba(image='image', x=0, y=0, 
                                                  dw=shape[1], dh=shape[0], 
                                                  source=image_source, 
                                                  level='glyph')
                
                # Update image source with the new image
                image_source.data = {'image': [img]}
                
                # Update the plot ranges
                spatial.x_range.start = 0
                spatial.x_range.end = shape[1]
                spatial.y_range.start = 0
                spatial.y_range.end = shape[0]
                
                # Make sure patches (masks) are on top
                contour_renderer = spatial.patches(xs='xs', ys='ys', 
                                                source=spatial_source,
                                                fill_alpha=0.2,
                                                line_alpha=1,
                                                fill_color='colors', 
                                                line_color='colors',
                                                level='overlay')
                
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
    load_adj_tdt_button.on_click(lambda: load_specific_image("adj_tdt", adj_tdt_input))
    remove_image_button.on_click(remove_image)

    # Create rows for the new image loading controls
    gcamp_row = row(gcamp_input, column(Spacer(height=20), load_gcamp_button))
    orig_tdt_row = row(orig_tdt_input, column(Spacer(height=20), load_orig_tdt_button))
    adj_tdt_row = row(adj_tdt_input, column(Spacer(height=20), load_adj_tdt_button))

    # Add the new controls to the layout
    image_controls = column(
        gcamp_row,
        orig_tdt_row,
        adj_tdt_row,
        row(remove_image_button),
        status_div
    )

    """
    ========================================================================================================================
            Setup Bokeh layout
    ========================================================================================================================
    """
    spacer1 = Spacer(height=50)
    spacer2 = Spacer(height=20)
    spacer3 = Spacer(width=20)

    choose_file = row(spacer3, sessionname_input, column(spacer2, load_data_button))

    controls = row(spacer3, column(spacer1, row(previous_button, next_button, neuron_index_input), neuron_id_slider))

    labelling = row(spacer3, column(
        row(d1_button, d2_button, unknown_button),
        row(file_path_input, column(spacer2, save_labels_button)),
        labeled_count_div))

    menus = row(spacer3, row(d1_neurons_select, d2_neurons_select, unknown_neurons_select))

    temporal = Tabs(tabs=[temporal_tab1, temporal_tab2])

    # Add this near the other button definitions
    toggle_masks = Toggle(label="Show/Hide Masks", button_type="success", active=True)

    def toggle_masks_visibility(event):
        """Toggle the visibility of all masks"""
        # Get all patch renderers
        patch_renderers = [r for r in spatial.renderers if isinstance(r, GlyphRenderer) 
                          and isinstance(r.glyph, Patches)]
        
        for renderer in patch_renderers:
            renderer.visible = toggle_masks.active
        
        # Update button appearance
        toggle_masks.button_type = "success" if toggle_masks.active else "danger"
        toggle_masks.label = "Hide Masks" if toggle_masks.active else "Show Masks"

    # Add the callback
    toggle_masks.on_click(toggle_masks_visibility)

    layout = row(spatial, column(
        choose_file,
        controls,
        row(spacer3, toggle_masks),
        image_controls,
        temporal,
        labelling,
        menus
    ))

    doc.add_root(layout)

# bp = Blueprint("labeler", __name__, url_prefix='/labeler')
#
#
# @bp.route("/", methods=['GET'])
# def bkapp_page():
#     script = server_document("http://localhost:5006/labeler_bkapp")
#     return render_template("labeler.html", script=script, template="Flask", port=8000)
