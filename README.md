# acts
Analyzing CT scans with Machine Learning - a Cognitive Science Master Thesis at the University of Osnabrück.

## Setup
To visualize the data OpenCV has to be installed on your machine.

### Using pip
All dependencies for this project can be installed via pip with `make install` in the projects root directory.

### Using Conda
If you prefer using conda: there is also an environment [yml file](https://github.com/AndreaSuckro/acts/tree/master/src/acts-env.yml) available. Just replace the `{PATH-TO-ENV}` with your destination folder and run: `make conda-env`.

### Data
The Lung CT Scans should be downloaded from the [Cancer Imaging Archive](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI). And stored in th `raw` folder under the data directory. At the moment the split into test, train and validation set is done manually such that in the end one should have the patient data for example stored in a path similar to this: `data/raw/train/LIDC-IDRI-0666/`.

## Learning
First the data needs to be preprocessed. For this use the following command (there are many more commands, but the location of the data directory and the number of patches per patient are the most interesting ones):

`make preprocess ARGS="-d data/ -p 4"`

After this is done, use the learn command to train the network on the data:

`make learn ARGS="-d data/ -l logs/ -e 2000"`
