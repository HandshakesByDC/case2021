ENVDIR=./env
ENV=$(ENVDIR)/bin
REQUIREMENTS_TXT=requirements.txt
MARKER=.initialized_for_Makefile

.PHONY: virtualenv
virtualenv: $(ENV)/$(MARKER)

$(ENV)/$(MARKER): $(REQUIREMENTS_TXT) | $(ENV)
	$(ENV)/pip install -r $(REQUIREMENTS_TXT)
	touch $(ENV)/$(MARKER)
	

$(ENV):
	virtualenv $(ENVDIR)
