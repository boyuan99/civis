from flask import Flask, render_template
import argparse
from threading import Thread
from tornado.ioloop import IOLoop
from bokeh.server.server import Server
from bokeh.embed import server_document
from servers import (labeler_bkapp, trajectory_bkapp_v1, connection_bkapp_v1, trajectory_bkapp_v0, raster_bkapp_v0,
                     trajectory_bkapp_v2, trajectory_bkapp_v3, raster_bkapp_v1, trajectory_bkapp_v4)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

# Flask routes for embedding Bokeh applications
@app.route('/labeler/')
def labeler_app():
    script = server_document(f'http://localhost:{args.bokeh_port}/labeler_bkapp')
    return render_template("labeler.html", script=script, template="Flask", port=args.flask_port)

@app.route('/trajectory/v1/')
def trajectory_v1_app():
    script = server_document(f'http://localhost:{args.bokeh_port}/trajectory_bkapp_v1')
    return render_template("trajectory_v1.html", script=script, template="Flask", port=args.flask_port)

@app.route('/connection/v1/')
def connection_app_v1():
    script = server_document(f'http://localhost:{args.bokeh_port}/connection_bkapp_v1')
    return render_template("connection_v1.html", script=script, template="Flask", port=args.flask_port)

@app.route('/trajectory/v0/')
def trajectory_v0_app():
    script = server_document(f'http://localhost:{args.bokeh_port}/trajectory_bkapp_v0')
    return render_template("trajectory_v0.html", script=script, template="Flask", port=args.flask_port)

@app.route('/raster/v0/')
def raster_app():
    script = server_document(f'http://localhost:{args.bokeh_port}/raster_bkapp_v0')
    return render_template("raster_v0.html", script=script, template="Flask", port=args.flask_port)

@app.route('/trajectory/v2/')
def trajectory_v2_app():
    script = server_document(f'http://localhost:{args.bokeh_port}/trajectory_bkapp_v2')
    return render_template("trajectory_v2.html", script=script, template="Flask", port=args.flask_port)

@app.route('/trajectory/v3/')
def trajectory_v3_app():
    script = server_document(f'http://localhost:{args.bokeh_port}/trajectory_bkapp_v3')
    return render_template("trajectory_v3.html", script=script, template='Flask', port=args.flask_port)

@app.route('/raster/v1/')
def raster_v1_app():
    script = server_document(f'http://localhost:{args.bokeh_port}/raster_bkapp_v1')
    return render_template("raster_v1.html", script=script, templates='Flask', port=args.flask_port)

@app.route('/trajectory/v4/')
def trajectory_v4_app():
    script = server_document(f'http://localhost:{args.bokeh_port}/trajectory_bkapp_v4')
    return render_template("trajectory_v4.html", script=script, template='Flask', port=args.flask_port)


def bk_worker(bokeh_port, flask_port):
    # Configure the Bokeh server with applications
    bokeh_apps = {'/labeler_bkapp': labeler_bkapp,
                  '/trajectory_bkapp_v1': trajectory_bkapp_v1,
                  '/connection_bkapp_v1': connection_bkapp_v1,
                  '/trajectory_bkapp_v0': trajectory_bkapp_v0,
                  '/raster_bkapp_v0': raster_bkapp_v0,
                  '/trajectory_bkapp_v2': trajectory_bkapp_v2,
                  '/trajectory_bkapp_v3': trajectory_bkapp_v3,
                  '/raster_bkapp_v1': raster_bkapp_v1,
                  '/trajectory_bkapp_v4': trajectory_bkapp_v4}

    server = Server(bokeh_apps,
                    io_loop=IOLoop(),
                    port=bokeh_port,
                    allow_websocket_origin=[f"localhost:{flask_port}", f"127.0.0.1:{flask_port}"])
    server.start()
    server.io_loop.start()

if __name__ == "__main__":
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

