# Training Parameters
JOB_ID ?= local

# configure project
#.PHONY: configure
#configure:

# install dependencies for the project with pip
.PHONY: install
install: src/requirements.txt
	pip3 install -r src/requirements.txt -U

# install dependencies for the project with conda environment
.PHONY: conda-env
conda-env: src/acts-env.yml
	conda env create python=3.6 -f src/acts-env.yml

# build documentation from md and tex
.PHONY: doc-pres
doc-pres: doc/presentation/overview.md
	pandoc doc/presentation/overview.md -s -o presentation.pdf

# preprocess data
.PHONY: preprocess
preprocess: src/preprocessing/
	JOB_ID=$JOB_ID python3 src/preprocessing/create_samples.py $(ARGS)

# learn network data
.PHONY: learn
learn: src/learn.py data/processed/train data/processed/test
	JOB_ID=$JOB_ID python3 src/learn.py $(ARGS)

# test learning for code checking
.PHONY: test
test: src/learn.py data/processed/train data/processed/test
	JOB_ID=$JOB_ID python3 src/learn.py -d data/ -l ../tests/ -e 2 -s 1 -b 1 -n ../tests/

# plot sample data
.PHONY: plot
plot: src/show.py src/visualization/data_visualizer.py data/processed/train
	python3 src/show.py $(ARGS)

# open tensorboard
.PHONY: tb
tb:
	tensorboard --logdir=../logs/ --port 6006

# check patient number
.PHONY: check
check: data/raw
	python3 src/tools/patient_numcheck.py

# split patient data
.PHONY: split
split:
	python3 src/tools/split_data.py
