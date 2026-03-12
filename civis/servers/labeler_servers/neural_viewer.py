"""Neural Video Viewer routes and data loading (non-Bokeh)."""

from flask import Blueprint, render_template, send_file, request, jsonify
import numpy as np
import h5py
import os

neural_viewer_bp = Blueprint('neural_viewer', __name__)

_mat_cache = {}  # path -> loaded data dict


def _load_mat_cached(path):
    """Load .mat (HDF5) neuron data with caching."""
    if path in _mat_cache:
        return _mat_cache[path]

    with h5py.File(path, 'r') as f:
        data = f['data']
        C = np.transpose(data['C'][()])
        num_neurons, num_frames = C.shape

        C_raw = np.transpose(data['C_raw'][()]) if 'C_raw' in data else np.zeros((num_neurons, num_frames))
        C_denoised = np.transpose(data['C_denoised'][()]) if 'C_denoised' in data else np.zeros((num_neurons, num_frames))
        C_deconvolved = np.transpose(data['C_deconvolved'][()]) if 'C_deconvolved' in data else np.zeros((num_neurons, num_frames))
        C_reraw = np.transpose(data['C_reraw'][()]) if 'C_reraw' in data else np.zeros((num_neurons, num_frames))
        C_extract = data['temporal_weights'][()].T if 'temporal_weights' in data else np.zeros((num_neurons, num_frames))

        Cn = np.transpose(data['Cn'][()])
        ids = (data['ids'][()] - 1).flatten().tolist()
        centroids = np.transpose(data['centroids'][()]).tolist()

        Coor_cell_array = data['Coor']
        coor_list = []
        for i in range(Coor_cell_array.shape[1]):
            ref = Coor_cell_array[0, i]
            coor_data = np.array(f[ref])
            coor_list.append(coor_data.tolist())

    result = {
        'C': C, 'C_raw': C_raw, 'C_denoised': C_denoised,
        'C_deconvolved': C_deconvolved, 'C_reraw': C_reraw, 'C_extract': C_extract,
        'Cn_shape': list(Cn.shape[:2]),
        'ids': ids, 'centroids': centroids, 'coor': coor_list,
        'num_neurons': num_neurons, 'num_frames': num_frames,
    }
    _mat_cache[path] = result
    return result


@neural_viewer_bp.route('/neural-viewer/v0/')
def neural_viewer_v0():
    return render_template("labeler/neural_video_viewer.html", script="")


@neural_viewer_bp.route('/api/stream-video')
def api_stream_video():
    path = request.args.get('path', '')
    path = os.path.normpath(path)
    if not os.path.isfile(path):
        return jsonify({'error': f'File not found: {path}'}), 404
    return send_file(path, mimetype='video/mp4', conditional=True)


@neural_viewer_bp.route('/api/neuron-data')
def api_neuron_data():
    path = request.args.get('path', '')
    mode = request.args.get('mode', 'metadata')
    path = os.path.normpath(path)

    if not os.path.isfile(path):
        return jsonify({'error': f'File not found: {path}'}), 404

    try:
        mat = _load_mat_cached(path)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    if mode == 'metadata':
        return jsonify({
            'num_neurons': mat['num_neurons'],
            'num_frames': mat['num_frames'],
            'cn_shape': mat['Cn_shape'],
            'ids': mat['ids'],
            'centroids': mat['centroids'],
            'coor': mat['coor'],
            'signal_types': ['C', 'C_raw', 'C_denoised', 'C_deconvolved', 'C_reraw', 'C_extract'],
        })
    elif mode == 'signals':
        idx = int(request.args.get('neuron_idx', 0))
        if idx < 0 or idx >= mat['num_neurons']:
            return jsonify({'error': f'neuron_idx {idx} out of range'}), 400
        return jsonify({
            'neuron_idx': idx,
            'signals': {
                'C': mat['C'][idx].tolist(),
                'C_raw': mat['C_raw'][idx].tolist(),
                'C_denoised': mat['C_denoised'][idx].tolist(),
                'C_deconvolved': mat['C_deconvolved'][idx].tolist(),
                'C_reraw': mat['C_reraw'][idx].tolist(),
                'C_extract': mat['C_extract'][idx].tolist(),
            }
        })
    else:
        return jsonify({'error': f'Unknown mode: {mode}'}), 400
