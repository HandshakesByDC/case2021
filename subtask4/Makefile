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

UNITRANS_DATA_DIR=src/models/UniTrans/data
UNITRANS_TO_MUSE=../MUSE/data
MUSE_DATA_DIR=$(UNITRANS_DATA_DIR)/$(UNITRANS_TO_MUSE)
LANGUAGES=en es pt
SRC_TGTS=en-es en-pt
MUSE_WIKI_EMBEDS=$(foreach lang,$(LANGUAGES),$(MUSE_DATA_DIR)/wiki.$(lang).vec)
MUSE_GT_DICTS=$(foreach srctgt,$(SRC_TGTS),$(MUSE_DATA_DIR)/$(srctgt).5000-6500.txt)
UNITRANS_WIKI_EMBEDS=$(foreach lang,$(LANGUAGES),$(UNITRANS_DATA_DIR)/embedding/wiki.$(lang).vec)
UNITRANS_GT_DICTS=$(foreach srctgt,$(SRC_TGTS),$(UNITRANS_DATA_DIR)/dict/$(srctgt).5000-6500.txt)
UNITRANS_TRAIN_DATA=$(foreach lang,$(LANGUAGES),$(UNITRANS_DATA_DIR)/ner/glocon/$(lang)/train.txt)

.PHONY: UniTrans
UniTrans: $(UNITRANS_WIKI_EMBEDS) $(UNITRANS_GT_DICTS) $(UNITRANS_TRAIN_DATA)

$(UNITRANS_DATA_DIR)/embedding/wiki.%.vec: $(MUSE_DATA_DIR)/wiki.%.vec | $(UNITRANS_DATA_DIR)/embedding
	ln -sfv ../$(patsubst $(UNITRANS_DATA_DIR)/%,%,$<) $@

$(UNITRANS_DATA_DIR)/embedding:
	cd $(UNITRANS_DATA_DIR) && mkdir embedding

$(UNITRANS_DATA_DIR)/dict/%.5000-6500.txt: $(MUSE_DATA_DIR)/%.5000-6500.txt | $(UNITRANS_DATA_DIR)/dict
	ln -sfv ../$(patsubst $(UNITRANS_DATA_DIR)/%,%,$<) $@

$(UNITRANS_DATA_DIR)/dict:
	cd $(UNITRANS_DATA_DIR) && mkdir dict

$(UNITRANS_TRAIN_DATA): $(UNITRANS_DATA_DIR)/ner/glocon/%/train.txt: $(UNITRANS_DATA_DIR)/ner/glocon/%
	ln -sfv ../../../../../../../data/$*-orig.txt $@
	touch data/$*-orig.txt

$(UNITRANS_DATA_DIR)/ner/glocon/%:
	mkdir -p $@

.PHONY: MUSE
MUSE: $(MUSE_DATA_DIR)/crosslingual

$(MUSE_DATA_DIR)/crosslingual:
	echo $(MUSE_DATA_DIR)
	cd $(MUSE_DATA_DIR) && sh get_evaluation.sh

.PRECIOUS: $(MUSE_DATA_DIR)/wiki.%.vec
$(MUSE_DATA_DIR)/wiki.%.vec:
	curl -Lo $@ https://dl.fbaipublicfiles.com/fasttext/vectors-wiki/$(patsubst $(MUSE_DATA_DIR)/%,%,$@)

$(MUSE_DATA_DIR)/%.5000-6500.txt:
	curl -Lo $@ https://dl.fbaipublicfiles.com/arrival/dictionaries/$(patsubst $(MUSE_DATA_DIR)/%,%,$@)
