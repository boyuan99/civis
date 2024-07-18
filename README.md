# CIVisServer

Calcium Imaging data visualization server implemented with [Bokeh](https://bokeh.org/) and [Flask](https://flask.palletsprojects.com/en/3.0.x/)



## Setup

The following command will install the packages according to the configuration file `requirements.txt`.

```bash
pip install -r requirements.txt
```

Use `config.example.json` as a reference to create `config.json` locally, including all of the paths in the example.

## Run

To use the server, run:

```bash
python app.py
```

It will open two ports by default: `8000` for flask `5006` for bokeh. You can specify the ports by:

```bash
python app.py --port 8000 --bokeh 5006
```

