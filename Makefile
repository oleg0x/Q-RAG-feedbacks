# Короткие имена для повседневных команд. Интерпретатор здесь один и тот же
# везде: системный python не годится, в нём нет torch и faiss.
PYTHON := /home/a.anokhin/venvs/gpu/bin/python
OFFLINE := HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

.PHONY: help test report index dry experiment list show clean

help:
	@echo "make test                       pytest, ожидается 153 passed"
	@echo "make report                     пересобрать runs/INDEX.md и RESULTS.md"
	@echo "make index                      только runs/INDEX.md (без vLLM и метрик)"
	@echo "make dry CFG=configs/x.yaml     показать команды эксперимента"
	@echo "make experiment CFG=configs/x.yaml [TAG=...] [ARGS=...]"
	@echo "make list                       все раны реестра"
	@echo "make show RUN=<run_id>          сводка одного рана"

test:
	$(OFFLINE) $(PYTHON) -m pytest

report:
	$(PYTHON) exp.py index

index:
	$(PYTHON) src/runs_index.py --only-index

dry:
	@test -n "$(CFG)" || (echo "укажите CFG=configs/....yaml" && false)
	$(PYTHON) exp.py run --config $(CFG) --dry-run

experiment:
	@test -n "$(CFG)" || (echo "укажите CFG=configs/....yaml" && false)
	$(PYTHON) exp.py run --config $(CFG) $(if $(TAG),--tag $(TAG),) $(ARGS)

list:
	@$(PYTHON) exp.py list

show:
	@test -n "$(RUN)" || (echo "укажите RUN=<run_id>" && false)
	@$(PYTHON) exp.py show $(RUN)

# Кеши интерпретатора; артефакты ранов не трогаются никогда.
clean:
	rm -rf __pycache__ .pytest_cache
