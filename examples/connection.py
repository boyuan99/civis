from bokeh.io import curdoc
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, TapTool, HoverTool, Div
from bokeh.layouts import row, column
import numpy as np
from servers.utils import load_data

neuron_path = "c364_03072024_v7.mat"
[C, C_raw, Cn, ids, Coor, centroids, virmen_path] = load_data(neuron_path)
neuron_num = C_raw.shape[0]
correlation_matrix = np.load("D:/Calcium Image Processing/ProcessedData/CorrelationPairwise/364_03072024_cor.npy")
height, width = Cn.shape

centroids_flipped = centroids.copy()
centroids_flipped[:, 1] = height - centroids[:, 1]

source = ColumnDataSource(data=dict(
    x=centroids[:, 0],
    y=centroids_flipped[:, 1],
    ids=[str(id) for id in ids],  # Ensure ids are strings
))

lines_source = ColumnDataSource(data=dict(xs=[], ys=[], colors=[]))

TOOLS = "pan,wheel_zoom,zoom_in,zoom_out,box_zoom,reset,tap"

p = figure(width=800, height=800, x_range=[0, width], y_range=[0, height],
           tools=TOOLS, active_scroll="wheel_zoom",
           title="Neuron Correlation Visualizer")

# Add image
p.image(image=[np.flipud(Cn)], x=0, y=0, dw=width, dh=height, palette="Greys256")

# Add centroids
circle_renderer = p.circle('x', 'y', source=source, size=10, color="green", alpha=0.7)
circle_renderer.selection_glyph = circle_renderer.glyph.clone()
circle_renderer.nonselection_glyph = circle_renderer.glyph.clone()

# Add lines (for correlations)
p.multi_line(xs="xs", ys="ys", color="colors", line_width=1, source=lines_source)

details_div = Div(width=300, height=800, sizing_mode="fixed", text="Neuron Details will appear here")

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
              f"Negatively Correlated Count: {len(pos_correlated_ids)}<br>" \
              f"Positively Correlated: {', '.join(pos_correlated_ids)}<br>"\
              f"Negatively Correlated: {', '.join(neg_correlated_ids)}"
    details_div.text = details


source.selected.on_change('indices', update_lines)

# HoverTool that only activates when directly over a glyph
hover = HoverTool(renderers=[circle_renderer], tooltips=[("ID", "@ids")])
p.add_tools(hover)
# Add TapTool to enable selection of neurons
tap_tool = TapTool(renderers=[circle_renderer])
p.add_tools(tap_tool)

layout = row(p, details_div)
curdoc().add_root(layout)