ENVDIR=./env
ENV=$(ENVDIR)/bin
REQUIREMENTS_TXT=$(wildcard requirements*.txt)
MARKER=.initialized_for_Makefile

.PHONY: virtualenv
virtualenv: $(ENV)/$(MARKER)

$(ENV)/$(MARKER): $(REQUIREMENTS_TXT) | $(ENV)
	echo $(foreach fpath,$(REQUIREMENTS_TXT),$(fpath))
	$(ENV)/pip install $(foreach fpath,$(REQUIREMENTS_TXT),-r $(fpath))
	touch $(ENV)/$(MARKER)
	

$(ENV):
	virtualenv $(ENVDIR)
