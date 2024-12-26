from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, TapTool, HoverTool, Div, TextInput, Button
from bokeh.layouts import row, column
import numpy as np
import h5py


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
        centroids = np.transpose(data['centroids'][()])
        virmenPath = data['virmenPath'][()].tobytes().decode('utf-16le')

        for i in range(Coor_cell_array.shape[1]):
            ref = Coor_cell_array[0, i]  # Get the reference
            coor_data = file[ref]  # Use the reference to access the data
            coor_matrix = np.array(coor_data)  # Convert to a NumPy array

            Coor.append(np.transpose(coor_matrix))

        # C = np.zeros_like(C_raw)
        # for i, C_pick in enumerate(C_raw):
        #     C_base = savgol_filter(C_pick, window_length=2000, polyorder=2, mode='interp')
        #     C[i] = C_pick - C_base

    return C, C_raw, Cn, ids, Coor, centroids, virmenPath


def connection_bkapp_v0(doc):
    global source, lines_source
    source = ColumnDataSource(data=dict(x=[], y=[], ids=[]))
    lines_source = ColumnDataSource(data=dict(xs=[], ys=[], colors=[]))

    # Initialize TextInput widgets for file paths
    neuron_path_input = TextInput(value="", title="Neuron Path:")
    correlation_matrix_path_input = TextInput(value="", title="Correlation Matrix Path:")
    load_button = Button(label="Load Data", button_type="success")
    details_div = Div(width=300, height=800, sizing_mode="fixed", text="Details will appear here after loading")

    # Placeholder for the plot, to be replaced after data is loaded
    placeholder_div = Div(text="Load data to visualize neurons.")

    def load_and_display():
        global source, lines_source
        neuron_path = neuron_path_input.value
        correlation_matrix_path = correlation_matrix_path_input.value

        [C, C_raw, Cn, ids, Coor, centroids, virmen_path] = load_data(neuron_path)
        correlation_matrix = np.load(correlation_matrix_path)
        height, width = Cn.shape

        centroids_flipped = np.copy(centroids)
        centroids_flipped[:, 1] = height - centroids[:, 1]

        # Update source with new data
        source.data = dict(x=centroids[:, 0], y=centroids_flipped[:, 1], ids=[str(id) for id in ids])

        TOOLS = "pan,wheel_zoom,zoom_in,zoom_out,box_zoom,reset"
        p = figure(width=800, height=800, x_range=[0, width], y_range=[0, height], tools=TOOLS,
                   active_scroll="wheel_zoom", title="Neuron Correlation Visualizer")
        # Add image
        p.image(image=[np.flipud(Cn)], x=0, y=0, dw=width, dh=height, palette="Greys256")

        # Add centroids
        circle_renderer = p.scatter(x='x', y='y', source=source, size=10, fill_color='colors',
                                    line_color='colors', alpha=0.7)
        circle_renderer.selection_glyph = circle_renderer.glyph.clone()
        circle_renderer.nonselection_glyph = circle_renderer.glyph.clone()

        # Add lines (for correlations)
        p.multi_line(xs="xs", ys="ys", color="colors", line_width=1, alpha=0.7, source=lines_source)

        # HoverTool that only activates when directly over a glyph
        hover = HoverTool(renderers=[circle_renderer], tooltips=[("ID", "@ids")])
        p.add_tools(hover)
        # Add TapTool to enable selection of neurons
        tap_tool = TapTool(renderers=[circle_renderer])
        p.add_tools(tap_tool)

        def update_lines(attr, old, new):
            selected_indices = source.selected.indices
            if not selected_indices:
                lines_source.data = {'xs': [], 'ys': [], 'colors': []}
                details_div.text = "No neuron selected"
                return

            selected_index = selected_indices[0]
            selected_id = ids[selected_index]
            pos_correlated_ids = []
            neg_correlated_ids = []

            new_xs, new_ys, new_colors = [], [], []
            for i, correlation in enumerate(correlation_matrix[selected_index]):
                if i != selected_index and correlation != 0:
                    x0, y0 = centroids_flipped[selected_index]
                    x1, y1 = centroids_flipped[i]
                    new_xs.append([x0, x1])
                    new_ys.append([y0, y1])
                    color = "red" if correlation > 0 else "blue"
                    new_colors.append(color)
                    if correlation > 0:
                        pos_correlated_ids.append(str(ids[i]))
                    else:
                        neg_correlated_ids.append(str(ids[i]))

            lines_source.data = {'xs': new_xs, 'ys': new_ys, 'colors': new_colors}
            details = f"Selected Neuron: {selected_id}<br>" \
                      f"Positively Correlated Count: {len(pos_correlated_ids)}<br>" \
                      f"Negatively Correlated Count: {len(neg_correlated_ids)}<br>" \
                      f"Positively Correlated: {', '.join(pos_correlated_ids)}<br>" \
                      f"Negatively Correlated: {', '.join(neg_correlated_ids)}"
            details_div.text = details

        source.selected.on_change('indices', update_lines)

        # Replace placeholder with the actual plot and details div
        layout.children[1] = row(p, details_div)

    load_button.on_click(load_and_display)
    # Initial layout with just the inputs, load button, and details div
    layout = column(row(neuron_path_input, correlation_matrix_path_input, load_button), details_div)
    doc.add_root(layout)
