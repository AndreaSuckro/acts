# acts
Analyzing CT scans with Machine Learning - a Cognitive Science Master Thesis at the University of Osnabrück.

## Setup
To visualize the data OpenCV has to be installed. The other dependencies can be installed via
`pip` or `conda`.

### Using pip
All dependencies for this project can be installed via pip with `make install` in the projects root directory.

### Using Conda
If you prefer using conda: there is an environment [yml file](https://github.com/AndreaSuckro/acts/tree/master/src/acts-env.yml) available. Just replace the `{PATH-TO-ENV}` with your destination folder and run: `make conda-env`. For mac there are unfortunately some packages not available. It is recommended to use a linux system instead.

### Data
The Lung CT Scans should be downloaded from the [Cancer Imaging Archive](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI). And stored in the `raw` folder under the data directory. The data can be automatically split into test, training and validation set with the `make split` command. The patient data will then be split according to a ratio you define and stored in a path similar to this for example: `data/raw/train/LIDC-IDRI-0666/`.

At the moment this folder contains a few sample data points to get you started. It also contains the trained network that can be analyzed with the scripts in `src/analysis`.

## Learning
First the data needs to be preprocessed. For this use the following command (there are many more commands, but the location of the data directory and the number of patches per patient are the most interesting ones):

`make preprocess ARGS="-d data/ -p 4"`

After this is done, use the learn command to train the network on the data:

`make learn ARGS="-d data/ -l logs/ -e 2000"`

## Visualization
The data can be visualized using the following commands.

`make plot ARGS="-t r -d data/raw/train/ -p 0005"`

Visualizes the patient with number `0005` by playing it's slices after one another.

`make plot ARGS="-t p -d data/ -p 0005"`

Visualizes the data from the preprocessed folder and shows the cubes that are used for learning the network.

## Analysis
The `analysis` folder contains a collection of scripts that measure the performance of the
network or visualize some of it's features. They can be directly called by Python with the
respective parameters and from within the package.
