# Diffraction Image Filtering App (DIFapp)

Dedicated to filtering diffraction images obtained with techniques such as **RHEED** or **EBSD**. Minimal desktop app to load diffraction images, convert them to HDF5, apply filtering pipelines, preview results, and save outputs.

## Prerequisites

- Python 3.10 or 3.11 (download: https://www.python.org/downloads/release/python-3110/)
- Tk available (Windows bundle it; on Linux install `python3-tk`)

## Installation
Create a virtual environment and install pinned dependencies. **Use Windows Command Prompt (cmd.exe). Do not use PowerShell.**

```ps1
py -m venv .venv

.\.venv\Scripts\activate.bat

python -m pip install --upgrade pip setuptools wheel

pip install -r requirements.txt
```
## Quick start

From the project root:

    python app.py

- Click **Load** to select an image (`.png`, `.jpg`, `.tif`, `.tiff`).
- The app converts to `.h5` internally and displays the image.
- Choose a method, adjust parameters, click **Run**.
- Use the **Export Options** buttons to save the processed image, create side-by-side comparisons, or generate analysis plots with intensity histograms.
- The **Debug** panel under the previews shows logs and internal steps.

The application should start with the following screen:
![main.png](docs/main.png)

Suitable images to test the tool are the ones demonstrated in the publication: [AstroEBSD: exploring new space in pattern indexing with methods launched from an astronomical approach](https://journals.iucr.org/paper?S1600576718010373) by T. B. Britton et al. (2018)
