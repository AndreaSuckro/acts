# acts
Analyzing CT scans with Machine Learning - a Cognitive Science Master Thesis at the University of Osnabrück.

## Setup
To visualize the data OpenCV has to be installed on your machine.

### Using pip
All dependencies for this project can be installed via pip with `make install` in the projects root directory.

### Using Conda
If you prefer using conda: there is also an environment [yml file](https://github.com/AndreaSuckro/acts/tree/master/src/acts-env.yml) available. Just replace the `{PATH-TO-ENV}` with your destination folder and run: `make conda-env`.

### Data
The Lung CT Scans should be downloaded from the [Cancer Imaging Archive](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI).
