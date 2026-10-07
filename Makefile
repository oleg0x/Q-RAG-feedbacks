# Shortcuts for everyday commands. Override the interpreter with
# `make <target> PYTHON=/path/to/venv/bin/python`; it needs torch and faiss.
PYTHON ?= python
OFFLINE := HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

.PHONY: help test report index dry experiment list show clean

help:
	@echo "make test                       run pytest"
	@echo "make report                     rebuild runs/INDEX.md and RESULTS.md"
	@echo "make index                      rebuild runs/INDEX.md only (no vLLM, no metrics)"
	@echo "make dry CFG=configs/x.yaml     print the experiment commands"
	@echo "make experiment CFG=configs/x.yaml [TAG=...] [ARGS=...]"
	@echo "make list                       list all registered runs"
	@echo "make show RUN=<run_id>          summary of one run"

test:
	$(OFFLINE) $(PYTHON) -m pytest

report:
	$(PYTHON) exp.py index

index:
	$(PYTHON) src/runs_index.py --only-index

dry:
	@test -n "$(CFG)" || (echo "set CFG=configs/....yaml" && false)
	$(PYTHON) exp.py run --config $(CFG) --dry-run

experiment:
	@test -n "$(CFG)" || (echo "set CFG=configs/....yaml" && false)
	$(PYTHON) exp.py run --config $(CFG) $(if $(TAG),--tag $(TAG),) $(ARGS)

list:
	@$(PYTHON) exp.py list

show:
	@test -n "$(RUN)" || (echo "set RUN=<run_id>" && false)
	@$(PYTHON) exp.py show $(RUN)

# Interpreter caches only; run artifacts are never touched.
clean:
	rm -rf __pycache__ .pytest_cache
