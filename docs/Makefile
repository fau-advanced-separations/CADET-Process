# Minimal makefile for Sphinx documentation
#

# You can set these variables from the command line.
SPHINXOPTS    =
SPHINXBUILD   = sphinx-build
SPHINXPROJ    = CADET-Process
SOURCEDIR     = source
BUILDDIR      = build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile html clean

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

html:
	mkdir -p "$(SOURCEDIR)"/examples/parameter_estimation/reference_data
	cp -r ../examples/parameter_estimation/reference_data "$(SOURCEDIR)"/examples/parameter_estimation/
	mkdir -p "$(SOURCEDIR)"/examples/parameter_estimation/reference_simulation
	cp -r ../examples/parameter_estimation/reference_simulation "$(SOURCEDIR)"/examples/parameter_estimation/
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

