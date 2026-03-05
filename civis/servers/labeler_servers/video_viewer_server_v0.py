import numpy as np
import cv2
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, Slider, Button, TextInput, Spacer, Div, Select, Toggle
from bokeh.layouts import column, row
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
servers_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(servers_dir)
sys.path.append(project_root)


def load_overlay_image(path):
    """Load an overlay image as a float32 grayscale array normalized 0-1."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    normalized = gray.astype(np.float32) / max(gray.max(), 1)
    return normalized


def _build_video_rgb(video_frame_bgr, video_channel_idx):
    """
    Convert BGR frame to a float32 (H, W, 3) RGB image.
    If video_channel_idx is None, return full color.
    Otherwise tint the grayscale into a single channel.
    """
    h, w = video_frame_bgr.shape[:2]
    if video_channel_idx is None:
        return cv2.cvtColor(video_frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    else:
        gray = cv2.cvtColor(video_frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        rgb = np.zeros((h, w, 3), dtype=np.float32)
        rgb[:, :, video_channel_idx] = gray
        return rgb


def _adjust_brightness_contrast(arr, brightness, contrast):
    """Apply brightness/contrast to a float32 array in [0,1] range.
    result = contrast * (arr - 0.5) + 0.5 + brightness
    """
    return np.clip(contrast * (arr - 0.5) + 0.5 + brightness, 0, 1)


def blend_frames(video_frame_bgr, overlay_norm, mode, alpha,
                 video_channel_idx, overlay_channel_idx, video_shape,
                 video_brightness=0.0, video_contrast=1.0,
                 overlay_brightness=0.0, overlay_contrast=1.0):
    """
    Blend video frame and overlay into a single uint32 RGBA array.

    Parameters:
        video_frame_bgr: (H, W, 3) uint8 BGR frame from cv2
        overlay_norm: (H', W') float32 normalized overlay, or None
        mode: "overlay" or "additive"
        alpha: float 0-1, blending strength for overlay mode
        video_channel_idx: None=RGB, 0=R, 1=G, 2=B
        overlay_channel_idx: 0=R, 1=G, 2=B
        video_shape: (H, W) target output shape
        video_brightness: float, brightness offset for video
        video_contrast: float, contrast multiplier for video
        overlay_brightness: float, brightness offset for overlay
        overlay_contrast: float, contrast multiplier for overlay
    Returns:
        uint32 RGBA array (H, W) flipped for Bokeh's bottom-up y-axis
    """
    h, w = video_shape
    video_rgb = _build_video_rgb(video_frame_bgr, video_channel_idx)
    video_rgb = _adjust_brightness_contrast(video_rgb, video_brightness, video_contrast)

    # Resize overlay to match video frame if needed
    if overlay_norm is not None and overlay_norm.shape != (h, w):
        overlay_resized = cv2.resize(overlay_norm, (w, h), interpolation=cv2.INTER_LINEAR)
    elif overlay_norm is not None:
        overlay_resized = overlay_norm.copy()
    else:
        overlay_resized = None

    # Apply brightness/contrast to overlay
    if overlay_resized is not None:
        overlay_resized = _adjust_brightness_contrast(
            overlay_resized, overlay_brightness, overlay_contrast
        )

    img = np.empty((h, w), dtype=np.uint32)
    view = img.view(dtype=np.uint8).reshape((h, w, 4))
    view[:, :, :] = 0
    view[:, :, 3] = 255  # full alpha

    if overlay_resized is None:
        # No overlay -- show video (full color or tinted)
        result = np.clip(video_rgb * 255, 0, 255).astype(np.uint8)
        view[:, :, 0] = result[:, :, 0]
        view[:, :, 1] = result[:, :, 1]
        view[:, :, 2] = result[:, :, 2]
    elif mode == "overlay":
        # Build a colored overlay image (tinted in the selected channel)
        overlay_color = np.zeros((h, w, 3), dtype=np.float32)
        overlay_color[:, :, overlay_channel_idx] = overlay_resized
        # Alpha blend: result = (1 - alpha) * video_rgb + alpha * overlay_color
        blended = (1 - alpha) * video_rgb + alpha * overlay_color
        blended = np.clip(blended * 255, 0, 255).astype(np.uint8)
        view[:, :, 0] = blended[:, :, 0]
        view[:, :, 1] = blended[:, :, 1]
        view[:, :, 2] = blended[:, :, 2]
    elif mode == "additive":
        # Video in one channel, overlay in another
        video_gray = _adjust_brightness_contrast(
            cv2.cvtColor(video_frame_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0,
            video_brightness, video_contrast
        )
        overlay_u8 = (overlay_resized * 255).astype(np.uint8)
        v_ch = video_channel_idx if video_channel_idx is not None else 1
        view[:, :, v_ch] = (video_gray * 255).astype(np.uint8)
        view[:, :, overlay_channel_idx] = np.maximum(
            view[:, :, overlay_channel_idx], overlay_u8
        )

    return np.flipud(img)


def video_viewer_bkapp_v0(doc):
    """
    Video viewer server with image overlay support.
    Loads AVI/MP4 videos, provides frame navigation, and supports
    overlay (alpha blend) and additive (color channel) blend modes.
    """

    PRERENDER_BATCH_SIZE = 50

    # Mutable state container
    state = {
        'cap': None,
        'total_frames': 0,
        'fps': 30.0,
        'frame_width': 0,
        'frame_height': 0,
        'overlay_norm': None,
        'is_playing': False,
        'play_callback_id': None,
        'current_frame_bgr': None,
        # Pre-render cache
        'frame_cache': [],
        'cache_valid': False,
        'cache_display_shape': (0, 0),  # (h, w) of cached frames
        'prerender_callback_id': None,
        'prerender_idx': 0,
        'prerender_params': None,
    }

    # ColumnDataSource for the displayed image
    image_source = ColumnDataSource(data={'image': []})

    # Main figure
    video_fig = figure(
        title="Video Viewer",
        width=800, height=800,
        active_scroll='wheel_zoom',
        tools="pan,wheel_zoom,zoom_in,zoom_out,reset,save",
        match_aspect=True,
    )
    video_fig.xgrid.visible = False
    video_fig.ygrid.visible = False
    video_fig.image_rgba(image='image', x=0, y=0, dw=1, dh=1, source=image_source)

    # ==================== Widgets ====================

    # Video loading
    video_path_input = TextInput(value='', title="Video Path (AVI/MP4):", width=400)
    load_video_button = Button(label="Load Video", button_type="success", width=120)

    # Frame navigation
    frame_slider = Slider(start=0, end=1, value=0, step=1, width=500, title="Frame")
    frame_slider.disabled = True

    # Playback
    play_button = Button(label="Play", width=80)
    play_button.disabled = True
    speed_slider = Slider(start=20, end=200, value=50, step=10, width=200, title="Interval (ms)")
    speed_slider.disabled = True

    # Info displays
    frame_info_div = Div(text="", width=500)
    status_div = Div(text="", width=500)

    # Overlay loading
    overlay_path_input = TextInput(value='', title="Overlay Image Path:", width=400)
    load_overlay_button = Button(label="Load Overlay", button_type="success", width=120)
    remove_overlay_button = Button(label="Remove Overlay", button_type="danger", width=120)

    # Blend controls
    blend_mode_select = Select(
        title="Blend Mode:",
        value="Overlay",
        options=["Overlay", "Additive"],
        width=200,
    )
    blend_mode_select.disabled = True

    alpha_slider = Slider(start=0.0, end=1.0, value=0.5, step=0.05, width=300, title="Alpha")
    alpha_slider.disabled = True

    video_channel_select = Select(
        title="Video Channel:",
        value="RGB",
        options=["RGB", "Red", "Green", "Blue"],
        width=150,
    )
    video_channel_select.disabled = True

    overlay_channel_select = Select(
        title="Overlay Channel:",
        value="Red",
        options=["Red", "Green", "Blue"],
        width=150,
    )
    overlay_channel_select.disabled = True

    # Brightness / Contrast controls
    video_brightness_slider = Slider(
        start=-1.0, end=1.0, value=0.0, step=0.05, width=250,
        title="Video Brightness",
    )
    video_brightness_slider.disabled = True
    video_contrast_slider = Slider(
        start=0.1, end=3.0, value=1.0, step=0.1, width=250,
        title="Video Contrast",
    )
    video_contrast_slider.disabled = True
    overlay_brightness_slider = Slider(
        start=-1.0, end=1.0, value=0.0, step=0.05, width=250,
        title="Overlay Brightness",
    )
    overlay_brightness_slider.disabled = True
    overlay_contrast_slider = Slider(
        start=0.1, end=3.0, value=1.0, step=0.1, width=250,
        title="Overlay Contrast",
    )
    overlay_contrast_slider.disabled = True

    # Transpose controls
    video_transpose_toggle = Toggle(label="Video Transpose", active=False, width=120)
    video_transpose_toggle.disabled = True
    overlay_transpose_toggle = Toggle(label="Overlay Transpose", active=False, width=120)
    overlay_transpose_toggle.disabled = True

    # Pre-render button
    prerender_button = Button(label="Pre-render All", button_type="warning", width=150)
    prerender_button.disabled = True
    prerender_div = Div(text="", width=300)

    # ==================== Core Rendering ====================

    channel_map = {"RGB": None, "Red": 0, "Green": 1, "Blue": 2}

    def _get_blend_params():
        """Gather all current blend parameters from widgets."""
        return dict(
            mode="overlay" if blend_mode_select.value == "Overlay" else "additive",
            alpha=alpha_slider.value,
            video_channel_idx=channel_map.get(video_channel_select.value, 1),
            overlay_channel_idx=channel_map.get(overlay_channel_select.value, 0),
            video_brightness=video_brightness_slider.value,
            video_contrast=video_contrast_slider.value,
            overlay_brightness=overlay_brightness_slider.value,
            overlay_contrast=overlay_contrast_slider.value,
            video_transpose=video_transpose_toggle.active,
            overlay_transpose=overlay_transpose_toggle.active,
        )

    # Track the current display dimensions to avoid unnecessary updates
    _display_dims = {'w': 0, 'h': 0}

    def _update_figure_dims(w, h):
        """Update figure ranges if dimensions changed. Never recreates renderers."""
        if _display_dims['w'] == w and _display_dims['h'] == h:
            return
        _display_dims['w'] = w
        _display_dims['h'] = h
        video_fig.x_range.start = 0
        video_fig.x_range.end = w
        video_fig.y_range.start = 0
        video_fig.y_range.end = h
        # Update the existing glyph's dw/dh instead of destroying and recreating
        glyph = video_fig.renderers[0].glyph
        glyph.dw = w
        glyph.dh = h

    def _compute_blended(frame_bgr, overlay_norm, p):
        """Apply transforms and blend, return (blended_rgba, h, w)."""
        if p['video_transpose']:
            frame_bgr = np.transpose(frame_bgr, (1, 0, 2))
        if overlay_norm is not None and p['overlay_transpose']:
            overlay_norm = overlay_norm.T

        h, w = frame_bgr.shape[:2]
        blended = blend_frames(
            frame_bgr, overlay_norm, p['mode'], p['alpha'],
            p['video_channel_idx'], p['overlay_channel_idx'],
            (h, w),
            p['video_brightness'], p['video_contrast'],
            p['overlay_brightness'], p['overlay_contrast'],
        )
        return blended, h, w

    def render_frame(frame_idx):
        """Display frame at frame_idx. Uses cache if available, otherwise computes live."""
        # Use pre-rendered cache if valid
        if state['cache_valid'] and frame_idx < len(state['frame_cache']):
            cached = state['frame_cache'][frame_idx]
            if cached is not None:
                ch, cw = state['cache_display_shape']
                _update_figure_dims(cw, ch)
                image_source.data = {'image': [cached]}
                frame_info_div.text = (
                    f"Frame: {frame_idx} / {state['total_frames'] - 1} | "
                    f"FPS: {state['fps']:.1f} | "
                    f"Resolution: {state['frame_width']}x{state['frame_height']} (cached)"
                )
                return

        # Live computation fallback
        cap = state['cap']
        if cap is None:
            return

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame_bgr = cap.read()
        if not ret:
            return

        state['current_frame_bgr'] = frame_bgr
        p = _get_blend_params()
        blended, h, w = _compute_blended(frame_bgr, state['overlay_norm'], p)
        _update_figure_dims(w, h)
        image_source.data = {'image': [blended]}

        frame_info_div.text = (
            f"Frame: {frame_idx} / {state['total_frames'] - 1} | "
            f"FPS: {state['fps']:.1f} | "
            f"Resolution: {state['frame_width']}x{state['frame_height']}"
        )

    def re_render_current():
        """Re-blend the current cached frame without seeking the video."""
        if state['current_frame_bgr'] is None:
            return
        # Invalidate pre-render cache since params changed
        _invalidate_cache()

        p = _get_blend_params()
        blended, h, w = _compute_blended(
            state['current_frame_bgr'], state['overlay_norm'], p
        )
        _update_figure_dims(w, h)
        image_source.data = {'image': [blended]}

    def _invalidate_cache():
        """Mark cache as invalid."""
        state['cache_valid'] = False
        state['frame_cache'] = []
        state['cache_display_shape'] = (0, 0)
        prerender_div.text = ""

    # ==================== Video Loading ====================

    def load_video():
        path = video_path_input.value.strip()
        if not path:
            status_div.text = "<span style='color: red;'>Error: Video path is empty</span>"
            return

        path = os.path.normpath(path)

        # Release previous capture if any
        if state['cap'] is not None:
            state['cap'].release()
            state['cap'] = None

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            status_div.text = f"<span style='color: red;'>Error: Cannot open video: {path}</span>"
            return

        state['cap'] = cap
        state['total_frames'] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        state['fps'] = cap.get(cv2.CAP_PROP_FPS) or 30.0
        state['frame_width'] = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        state['frame_height'] = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Enable widgets
        frame_slider.disabled = False
        frame_slider.start = 0
        frame_slider.end = max(state['total_frames'] - 1, 1)
        frame_slider.value = 0
        play_button.disabled = False
        speed_slider.disabled = False
        blend_mode_select.disabled = False
        alpha_slider.disabled = False
        video_channel_select.disabled = False
        overlay_channel_select.disabled = False
        video_brightness_slider.disabled = False
        video_contrast_slider.disabled = False
        overlay_brightness_slider.disabled = False
        overlay_contrast_slider.disabled = False
        video_transpose_toggle.disabled = False
        overlay_transpose_toggle.disabled = False
        prerender_button.disabled = False

        # Invalidate old cache
        _invalidate_cache()

        # Render first frame (this also updates figure dimensions)
        render_frame(0)

        status_div.text = f"<span style='color: green;'>Video loaded: {os.path.basename(path)}</span>"

    load_video_button.on_click(load_video)

    # ==================== Frame Slider ====================

    def on_frame_change(attr, old, new):
        render_frame(int(new))

    frame_slider.on_change('value', on_frame_change)

    # ==================== Playback ====================

    def update_playback():
        current = frame_slider.value
        if current < frame_slider.end:
            frame_slider.value = current + 1
        else:
            toggle_play()

    def toggle_play():
        if not state['is_playing']:
            if frame_slider.value >= frame_slider.end:
                frame_slider.value = 0
            play_button.label = "Pause"
            state['is_playing'] = True
            interval = int(speed_slider.value)
            state['play_callback_id'] = doc.add_periodic_callback(update_playback, interval)
        else:
            play_button.label = "Play"
            state['is_playing'] = False
            if state['play_callback_id']:
                doc.remove_periodic_callback(state['play_callback_id'])
                state['play_callback_id'] = None

    play_button.on_click(toggle_play)

    def on_speed_change(attr, old, new):
        if state['is_playing'] and state['play_callback_id']:
            doc.remove_periodic_callback(state['play_callback_id'])
            state['play_callback_id'] = doc.add_periodic_callback(update_playback, int(new))

    speed_slider.on_change('value', on_speed_change)

    # ==================== Overlay Loading ====================

    def load_overlay():
        path = overlay_path_input.value.strip()
        if not path:
            status_div.text = "<span style='color: red;'>Error: Overlay path is empty</span>"
            return
        try:
            path = os.path.normpath(path)
            state['overlay_norm'] = load_overlay_image(path)
            _invalidate_cache()
            status_div.text = f"<span style='color: green;'>Overlay loaded: {os.path.basename(path)}</span>"
            re_render_current()
        except Exception as e:
            status_div.text = f"<span style='color: red;'>Error loading overlay: {e}</span>"

    load_overlay_button.on_click(load_overlay)

    def remove_overlay():
        state['overlay_norm'] = None
        _invalidate_cache()
        status_div.text = "<span style='color: blue;'>Overlay removed</span>"
        re_render_current()

    remove_overlay_button.on_click(remove_overlay)

    # ==================== Pre-render ====================

    def start_prerender():
        """Start batch pre-rendering all frames with current blend params."""
        cap = state['cap']
        if cap is None:
            return

        # Stop playback if running
        if state['is_playing']:
            toggle_play()

        # Cancel any in-progress pre-render
        if state['prerender_callback_id'] is not None:
            doc.remove_periodic_callback(state['prerender_callback_id'])
            state['prerender_callback_id'] = None

        state['frame_cache'] = [None] * state['total_frames']
        state['cache_valid'] = False
        state['prerender_idx'] = 0
        state['prerender_params'] = _get_blend_params()

        # Prepare overlay (possibly transposed) once
        overlay = state['overlay_norm']
        if overlay is not None and state['prerender_params']['overlay_transpose']:
            state['prerender_overlay'] = overlay.T
        else:
            state['prerender_overlay'] = overlay

        prerender_button.disabled = True
        prerender_div.text = "<b>Pre-rendering: 0%</b>"

        state['prerender_callback_id'] = doc.add_periodic_callback(
            prerender_batch, 10
        )

    def prerender_batch():
        """Process a batch of frames during pre-render."""
        cap = state['cap']
        p = state['prerender_params']
        overlay = state.get('prerender_overlay')
        idx = state['prerender_idx']
        end = min(idx + PRERENDER_BATCH_SIZE, state['total_frames'])

        for i in range(idx, end):
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame_bgr = cap.read()
            if not ret:
                continue

            if p['video_transpose']:
                frame_bgr = np.transpose(frame_bgr, (1, 0, 2))

            h, w = frame_bgr.shape[:2]
            blended = blend_frames(
                frame_bgr, overlay, p['mode'], p['alpha'],
                p['video_channel_idx'], p['overlay_channel_idx'],
                (h, w),
                p['video_brightness'], p['video_contrast'],
                p['overlay_brightness'], p['overlay_contrast'],
            )
            state['frame_cache'][i] = blended
            # Store display shape from the first successful frame
            if state['cache_display_shape'] == (0, 0):
                state['cache_display_shape'] = (h, w)

        state['prerender_idx'] = end
        progress = int(end / state['total_frames'] * 100)
        prerender_div.text = f"<b>Pre-rendering: {progress}%</b>"

        if end >= state['total_frames']:
            # Done
            doc.remove_periodic_callback(state['prerender_callback_id'])
            state['prerender_callback_id'] = None
            state['cache_valid'] = True
            state.pop('prerender_overlay', None)
            state.pop('prerender_params', None)
            prerender_button.disabled = False
            prerender_div.text = (
                f"<span style='color: green;'><b>Cache ready "
                f"({state['total_frames']} frames)</b></span>"
            )
            # Re-display current frame from cache
            render_frame(int(frame_slider.value))

    prerender_button.on_click(start_prerender)

    # ==================== Blend Parameter Callbacks ====================

    for widget in [blend_mode_select, alpha_slider,
                   video_channel_select, overlay_channel_select,
                   video_brightness_slider, video_contrast_slider,
                   overlay_brightness_slider, overlay_contrast_slider]:
        widget.on_change('value', lambda attr, old, new: re_render_current())

    video_transpose_toggle.on_change('active', lambda attr, old, new: re_render_current())
    overlay_transpose_toggle.on_change('active', lambda attr, old, new: re_render_current())

    # ==================== Cleanup ====================

    def cleanup(session_context):
        if state['cap'] is not None:
            state['cap'].release()
            state['cap'] = None

    doc.on_session_destroyed(cleanup)

    # ==================== Layout ====================

    video_load_row = row(
        video_path_input,
        column(Spacer(height=20), load_video_button),
    )

    playback_controls = column(
        frame_slider,
        row(play_button, Spacer(width=20), speed_slider),
        frame_info_div,
    )

    overlay_load_row = row(
        overlay_path_input,
        column(Spacer(height=20), row(load_overlay_button, Spacer(width=10), remove_overlay_button)),
    )

    blend_controls = column(
        Div(text="<b>Blend Settings</b>"),
        blend_mode_select,
        alpha_slider,
        row(video_channel_select, Spacer(width=10), overlay_channel_select),
        Spacer(height=10),
        Div(text="<b>Video Adjustments</b>"),
        video_brightness_slider,
        video_contrast_slider,
        Spacer(height=10),
        Div(text="<b>Overlay Adjustments</b>"),
        overlay_brightness_slider,
        overlay_contrast_slider,
        Spacer(height=10),
        Div(text="<b>Transform</b>"),
        row(video_transpose_toggle, Spacer(width=10), overlay_transpose_toggle),
        Spacer(height=10),
        row(prerender_button, Spacer(width=10), prerender_div),
    )

    controls_panel = column(
        video_load_row,
        Spacer(height=15),
        playback_controls,
        Spacer(height=15),
        overlay_load_row,
        Spacer(height=15),
        blend_controls,
        Spacer(height=15),
        status_div,
    )

    layout = row(video_fig, Spacer(width=20), controls_panel)

    doc.add_root(layout)
    doc.title = "Video Viewer"
