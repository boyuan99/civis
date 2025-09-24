# CIVIS

**CIVIS** is a Calcium Imaging data visualization server built with [Bokeh](https://bokeh.org/) and [Flask](https://flask.palletsprojects.com/en/3.0.x/). It provides an interactive web-based interface for visualizing and analyzing calcium imaging data, facilitating research and data exploration.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Setup](#setup)
- [Running the Server](#running-the-server)

## Features

- **Interactive Visualizations:** Utilize Bokeh for dynamic and responsive data visualizations.
- **Web Interface:** Powered by Flask, offering an accessible web interface for data interaction.
- **Modular Architecture:** Supports multiple visualization modules such as labelers, trajectories, connections, rasters, and place finders.
- **Customizable Configuration:** Easily configure data paths and server settings through a JSON file.
- **Scalable Server:** Capable of handling multiple Bokeh applications concurrently.

## Installation

### Prerequisites

- **Python:** Version 3.9 or higher
- **pip:** Python package installer

### Clone the Repository

```bash
git clone https://github.com/boyuan99/civis.git
cd civis
```
### Create a Conda Environment
```bash
conda env create -f environment.yaml
conda activate civis
```
Note: The environment.yaml file includes the command to install your package in editable mode (`pip install -e .`). This means that after activating the environment, your package is already set up for development.


## Setup

1. **Configuration File:**
   
   Use the provided `civis/config.example.json` as a template to create your local `config.json`. This file includes paths to necessary data directories.

   ```bash
   cp civis/config.example.json civis/config.json
   ```

2. **Edit Configuration:**

   Open `civis/config.json` and update the paths to match your local data directories:

   ```json
   {
     "ProcessedFilePath": "path/to/ProcessedData/",
     "VirmenFilePath": "path/to/VirmenData/",
     "LabelsPath": "path/to/ProcessedData/",
     "ElecPath": "path/to/ElecData/"
   }
   ```

   **Note:** Ensure all paths are absolute or relative to the project root and that they point to valid directories containing your data.

## Running the Server

### Default Ports

By default, running the server will open two ports:

- **Flask Server:** `8000`
- **Bokeh Server:** `5006`

### Start the Server

Execute the following command to start the civis:

```bash
civis
```

### Custom Ports

To specify custom ports for the Flask and Bokeh servers, use the `--flask-port` and `--bokeh-port` flags:

```bash
civis --flask-port 8000 --bokeh-port 5006
```

### Accessing the Application

Once the server is running, open your web browser and navigate to:

```
http://localhost:8000
```

You will see the home page, from which you can access various visualization modules such as Labeler, Trajectory, Connection, Raster, and Place applications.
