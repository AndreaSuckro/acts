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
	python3 src/preprocessing/create_samples.py $(ARGS)

# learn network data
.PHONY: learn
learn: src/run_learning.py data/processed/train data/processed/test
	python3 src/run_learning.py $(ARGS)

# plot sample data
.PHONY: plot
plot: src/show.py src/visualization/data_visualizer.py data/processed/train
	python3 src/show.py $(ARGS)
