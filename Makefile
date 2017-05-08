# install dependencies for the project
.PHONY: install
install: src/requirements.txt
	pip3 install -r src/requirements.txt -U

# build documentation from md and tex
.PHONY: doc-pres
doc-pres: doc/presentation/overview.md
	pandoc doc/presentation/overview.md -s -o presentation.pdf