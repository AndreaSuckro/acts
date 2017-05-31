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
