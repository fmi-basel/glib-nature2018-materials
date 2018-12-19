# glib-nature2018-materials
Image Processing Package for the Paper "Self-organization and symmetry breaking in intestinal organoid development"


## Prerequisites

On Ubuntu, make sure Python 3.6 is installed and pip:

```
sudo apt install python3 python3-pip
```

If your system python defaults to version 2.7, you will have to use
```python3``` and ```pip3``` instead of ```python``` and ```pip``` in
the following steps. Alternatively, you can use Anaconda (as detailled
below) to install and run the code in a separate environment.


On Windows, we recommend using [Anaconda](https://www.anaconda.com/download/#linux).

Use the ```Anaconda prompt``` to create a new environment:

```
conda create -n glib-nature2018 python=3.6
```

and make sure to activate it before running any of the steps below:

```
conda activate glib-nature2018
```

## Installation

Change into the code folder and install the package and its dependencies with ```pip```.

```
cd glib-nature2018-materials
pip install .[cpu]
```

## Usage

The following examples assume you are currently in the root directory
of ```glib-nature2018-materials```. If you installed the package in a
virtual environment, make sure to activate it, e.g. with ```conda
activate glib-nature2018```.

The three workflows can be called as follows.


Generate organoid segmentations from the overview MIP:

```
python run_organoid_mip_segmentation.py
```


Segment organoids in cropped 3D stacks and extract their features:

```
python run_organoid_single_plane_segmentation.py
```


Segment nuclei in cropped 3D stacks, extract features and estimate the cell count per organoid:

```
python run_organoid_single_cell_segmentation.py
```

Note that this script expects that ```run_organoid_single_plane_segmentation.py``` was previously executed.

