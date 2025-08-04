from flask import Flask, render_template
import argparse
from threading import Thread
from tornado.ioloop import IOLoop
from bokeh.server.server import Server
from bokeh.embed import server_document
from civis.servers import *

app = Flask(__name__)

# Global variable to store command-line arguments
args = None

@app.route('/')
def home():
    return render_template("index.html")

# Flask routes for embedding Bokeh applications
@app.route('/labeler/v0/')
def labeler_v0_app():
    global args
    script = server_document(f'http://localhost:{args.bokeh_port}/labeler_bkapp_v0')
    return render_template("labeler/labeler_v0.html", script=script, template="Flask", port=args.flask_port)

@app.route('/labeler/v1/')
def labeler_v1_app():
    global args
    script = server_document(f'http://localhost:{args.bokeh_port}/labeler_bkapp_v1')
    return render_template("labeler/labeler_v1.html", script=script, template="Flask", port=args.flask_port)

@app.route('/labeler/v2/')
def labeler_v2_app():
    global args
    script = server_document(f'http://localhost:{args.bokeh_port}/labeler_bkapp_v2')
    return render_template("labeler/labeler_v2.html", script=script, template="Flask", port=args.flask_port)

@app.route('/trajectory/v1/')
def trajectory_v1_app():
    global args
    script = server_document(f'http://localhost:{args.bokeh_port}/trajectory_bkapp_v1')
    return render_template("trajectory/trajectory_v1.html", script=script, template="Flask", port=args.flask_port)

@app.route('/connection/v0/')
def connection_app_v0():
    global args
    script = server_document(f'http://localhost:{args.bokeh_port}/connection_bkapp_v0')
    return render_template("connection/connection_v0.html", script=script, template="Flask", port=args.flask_port)

@app.route('/connection/v1/')
def connection_app_v1():
    global args
    script = server_document(f'http://localhost:{args.bokeh_port}/connection_bkapp_v1')
    return render_template("connection/connection_v1.html", script=script, template="Flask", port=args.flask_port)

@app.route('/connection/v2/')
def connection_app_v2():
    global args
    script = server_document(f'http://localhost:{args.bokeh_port}/connection_bkapp_v2')
    return render_template("connection/connection_v2.html", script=script, template="Flask", port=args.flask_port)

@app.route('/trajectory/v0/')
def trajectory_v0_app():
    global args
    script = server_document(f'http://localhost:{args.bokeh_port}/trajectory_bkapp_v0')
    return render_template("trajectory/trajectory_v0.html", script=script, template="Flask", port=args.flask_port)

@app.route('/raster/v0/')
def raster_app():
    global args
    script = server_document(f'http://localhost:{args.bokeh_port}/raster_bkapp_v0')
    return render_template("raster/raster_v0.html", script=script, template="Flask", port=args.flask_port)

@app.route('/trajectory/v2/')
def trajectory_v2_app():
    global args
    script = server_document(f'http://localhost:{args.bokeh_port}/trajectory_bkapp_v2')
    return render_template("trajectory/trajectory_v2.html", script=script, template="Flask", port=args.flask_port)

@app.route('/trajectory/v3/')
def trajectory_v3_app():
    global args
    script = server_document(f'http://localhost:{args.bokeh_port}/trajectory_bkapp_v3')
    return render_template("trajectory/trajectory_v3.html", script=script, template='Flask', port=args.flask_port)

@app.route('/raster/v1/')
def raster_v1_app():
    global args
    script = server_document(f'http://localhost:{args.bokeh_port}/raster_bkapp_v1')
    return render_template("raster/raster_v1.html", script=script, templates='Flask', port=args.flask_port)

@app.route('/trajectory/v4/')
def trajectory_v4_app():
    global args
    script = server_document(f'http://localhost:{args.bokeh_port}/trajectory_bkapp_v4')
    return render_template("trajectory/trajectory_v4.html", script=script, template='Flask', port=args.flask_port)

@app.route('/trajectory/v5/')
def trajectory_v5_app():
    global args
    script = server_document(f'http://localhost:{args.bokeh_port}/trajectory_bkapp_v5')
    return render_template("trajectory/trajectory_v5.html", script=script, template='Flask', port=args.flask_port)

@app.route('/trajectory/v6/')
def trajectory_v6_app():
    global args
    script = server_document(f'http://localhost:{args.bokeh_port}/trajectory_bkapp_v6')
    return render_template("trajectory/trajectory_v6.html", script=script, template='Flask', port=args.flask_port)

@app.route('/trajectory/v7/')
def trajectory_v7_app():
    global args
    script = server_document(f'http://localhost:{args.bokeh_port}/trajectory_bkapp_v7')
    return render_template("trajectory/trajectory_v7.html", script=script, template='Flask', port=args.flask_port)

@app.route('/trajectory/v8/')
def trajectory_v8_app():
    global args
    script = server_document(f'http://localhost:{args.bokeh_port}/trajectory_bkapp_v8')
    return render_template("trajectory/trajectory_v8.html", script=script, template='Flask', port=args.flask_port)

@app.route('/trajectory/v9/')
def trajectory_v9_app():
    global args
    script = server_document(f'http://localhost:{args.bokeh_port}/trajectory_bkapp_v9')
    return render_template("trajectory/trajectory_v9.html", script=script, template='Flask', port=args.flask_port)

@app.route('/place/v0/')
def place_v0_app():
    global args
    script = server_document(f'http://localhost:{args.bokeh_port}/place_bkapp_v0')
    return render_template("place/place_v0.html", script=script, template='Flask', port=args.flask_port)

@app.route('/place/v1/')
def place_v1_app():
    global args
    script = server_document(f'http://localhost:{args.bokeh_port}/place_bkapp_v1')
    return render_template("place/place_v1.html", script=script, templates='Flask', port=args.flask_port)

@app.route('/place/v2/')
def place_v2_app():
    global args
    script = server_document(f'http://localhost:{args.bokeh_port}/place_bkapp_v2')
    return render_template("place/place_v2.html", script=script, templates='Flask', port=args.flask_port)

@app.route('/place/v3/')
def place_v3_app():
    global args
    script = server_document(f'http://localhost:{args.bokeh_port}/place_bkapp_v3')
    return render_template("place/place_v3.html", script=script, templates='Flask', port=args.flask_port)

@app.route('/place/v4/')
def place_v4_app():
    global args
    script = server_document(f'http://localhost:{args.bokeh_port}/place_bkapp_v4')
    return render_template("place/place_v4.html", script=script, templates='Flask', port=args.flask_port)

def bk_worker(bokeh_port, flask_port):
    # Configure the Bokeh server with applications
    bokeh_apps = {
        '/labeler_bkapp_v0': labeler_bkapp_v0,
        '/labeler_bkapp_v1': labeler_bkapp_v1,
        '/labeler_bkapp_v2': labeler_bkapp_v2,
        '/trajectory_bkapp_v1': trajectory_bkapp_v1,
        '/connection_bkapp_v0': connection_bkapp_v0,
        '/connection_bkapp_v1': connection_bkapp_v1,
        '/connection_bkapp_v2': connection_bkapp_v2,
        '/trajectory_bkapp_v0': trajectory_bkapp_v0,
        '/raster_bkapp_v0': raster_bkapp_v0,
        '/trajectory_bkapp_v2': trajectory_bkapp_v2,
        '/trajectory_bkapp_v3': trajectory_bkapp_v3,
        '/raster_bkapp_v1': raster_bkapp_v1,
        '/trajectory_bkapp_v4': trajectory_bkapp_v4,
        '/place_bkapp_v0': place_cell_vis_bkapp_v0,
        '/place_bkapp_v1': place_cell_vis_bkapp_v1,
        '/trajectory_bkapp_v5': trajectory_bkapp_v5,
        '/trajectory_bkapp_v6': trajectory_bkapp_v6,
        '/trajectory_bkapp_v7': trajectory_bkapp_v7,
        '/trajectory_bkapp_v8': trajectory_bkapp_v8,
        '/trajectory_bkapp_v9': trajectory_bkapp_v9,
        '/place_bkapp_v2': place_cell_vis_bkapp_v2,
        '/place_bkapp_v3': place_cell_vis_bkapp_v3,
        '/place_bkapp_v4': place_cell_vis_bkapp_v4
    }

    server = Server(bokeh_apps,
                    io_loop=IOLoop(),
                    port=bokeh_port,
                    allow_websocket_origin=[f"localhost:{flask_port}", f"127.0.0.1:{flask_port}"])
    server.start()
    server.io_loop.start()

def main():
    global args
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--flask-port', default=8000, type=int,
                        help='Select port for the Flask server to run on')
    parser.add_argument('-b', '--bokeh-port', default=5006, type=int,
                        help='Select port for the Bokeh server to run on')
    args = parser.parse_args()

    # Start the Bokeh server in a separate thread with the specified port
    Thread(target=bk_worker, args=(args.bokeh_port, args.flask_port)).start()

    # Use the specified Flask port when running the app
    app.run(port=args.flask_port, debug=True, use_reloader=False)

if __name__ == "__main__":
    main()
