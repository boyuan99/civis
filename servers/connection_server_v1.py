from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, TapTool, HoverTool, Div, TextInput, Button, Select, CDSView, GroupFilter, Spacer
from bokeh.layouts import row, column
import numpy as np
import json
import pickle
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


def connection_bkapp_v1(doc):
    source = ColumnDataSource(data=dict(x=[], y=[], ids=[]))
    lines_source = ColumnDataSource(data=dict(xs=[], ys=[], colors=[]))

    # Initialize TextInput widgets for file paths
    neuron_path_input = TextInput(value="", title="Session Name:")
    load_button = Button(label="Load Data", button_type="success")
    details_div = Div(width=300, height=800, sizing_mode="fixed", text="Details will appear here after loading")

    neuron_categories = {}
    category_select = Select(title="Neuron Category:", value="All",
                             options=["All"] + sorted(list(neuron_categories.keys())))

    view = CDSView(filter=GroupFilter(column_name='categories', group='All'))

    def update_display_based_on_category(attr, old, new):
        selected_category = category_select.value

        if selected_category == "All":
            # Add a dummy column to ColumnDataSource data
            source.data['all'] = ['all'] * len(source.data['x'])

            view.filter = GroupFilter(column_name='all', group='all')
        else:
            view.filter = GroupFilter(column_name='categories', group=selected_category)

    category_select.on_change('value', update_display_based_on_category)

    def load_and_display():

        with open('config.json', 'r') as file:
            config = json.load(file)

        session_name = neuron_path_input.value
        neuron_path = config['ProcessedFilePath'] + session_name + '/' + session_name + '_v7.mat'
        pairwise_path = config['ProcessedFilePath'] + session_name + '/' + session_name + '_cor.npy'
        neuron_categories_path = config['ProcessedFilePath'] + session_name + '/' + session_name + '_neuron_categories.pkl'
        with open(neuron_categories_path, 'rb') as f:
            neuron_categories_all = pickle.load(f)

        exclude_keys = ['velocity', 'lick', 'pstcr']
        neuron_categories = {k: v for k, v in neuron_categories_all.items() if k not in exclude_keys}
        category_select.options = ["All"] + sorted(list(neuron_categories.keys()))


        neuron_colors = [
            "#F44336",  # Red
            "#673AB7",  # Deep Purple
            "#3F51B5",  # Indigo
            "#009688",  # Teal
            "#4CAF50",  # Green
            "#CDDC39",  # Lime
            "#FF9800",  # Orange
            "#795548",  # Brown (might not be as visible on dark, use cautiously)
            "#607D8B",  # Blue Grey
            "#1B5E20",  # Green (dark)
            "#B71C1C",  # Red (dark)
            "#F50057",  # Pink (Accent)
            "#00E5FF",  # Cyan (Accent)
            "#76FF03",  # Lime (Accent)
            "#C6FF00",  # Lime (Accent light)
            "#6200EA",  # Deep Purple (Accent)
            "#304FFE",  # Blue (Accent)
            "#00BFA5",  # Teal (Accent)
            "#FFD600",  # Yellow (Accent)
        ]

        [C, C_raw, Cn, ids, Coor, centroids, virmen_path] = load_data(neuron_path)
        correlation_matrix = np.load(pairwise_path)
        height, width = Cn.shape

        centroids_flipped = np.copy(centroids)
        centroids_flipped[:, 1] = height - centroids[:, 1]

        # Initialize mappings of neuron IDs to colors and categories
        neuron_id_to_color = {}
        neuron_id_to_category = {}  # New dictionary for neuron ID to category mapping

        # Iterate over the categories and assign colors and categories
        for category_index, (category, neuron_ids) in enumerate(neuron_categories.items()):
            color = neuron_colors[category_index % len(neuron_colors)]  # Use modulo for safety
            for neuron_id in neuron_ids:
                neuron_id_to_color[str(neuron_id)] = color
                neuron_id_to_category[str(neuron_id)] = category  # Assign category

        source.data = dict(
            x=centroids[:, 0],
            y=centroids_flipped[:, 1],
            ids=[str(int(id)) for id in ids],
            colors=[neuron_id_to_color.get(str(int(id)), "#FFFFFF") for id in ids],  # Use white as default color
            categories=[neuron_id_to_category.get(str(int(id)), "Unknown") for id in ids]  # Assign categories
        )

        TOOLS = "pan, wheel_zoom, zoom_in, zoom_out, box_zoom, reset, save"
        p = figure(width=800, height=800, x_range=[0, width], y_range=[0, height], tools=TOOLS, active_scroll="wheel_zoom", title="Neuron Correlation Visualizer")
        # Add image
        p.image(image=[np.flipud(Cn)], x=0, y=0, dw=width, dh=height, palette="Greys256")

        # Add centroids
        circle_renderer = p.circle(x='x', y='y', source=source, size=10, color='colors', view=view, alpha=0.7)

        circle_renderer.selection_glyph = circle_renderer.glyph.clone()
        circle_renderer.nonselection_glyph = circle_renderer.glyph.clone()

        # Add lines (for correlations)
        p.multi_line(xs="xs", ys="ys", color="colors", line_width=1, alpha=0.7, source=lines_source)

        # HoverTool that only activates when directly over a glyph
        hover = HoverTool(renderers=[circle_renderer], tooltips=[("ID", "@ids"), ("Category", "@categories")])
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
        layout.children[1] = row(p, column(category_select, details_div))

    load_button.on_click(load_and_display)

    layout = column(row(neuron_path_input, column(Spacer(height=20), load_button)), details_div)
    doc.add_root(layout)
