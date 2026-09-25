# Эвал Search-R1 на обученной башне Q-RAG (пак для коллеги)

Всё нужное лежит в этом каталоге (`/dev/shm/qrag-eval/`). Данные в tmpfs:
**после ребута сервера каталог исчезает** — попроси Сашу перезапустить
`prepare.sh`, свои раны ты не потеряешь (они пишутся в твой клон в home).

Что внутри:

| Что | Путь |
|---|---|
| Код (bare-репо, ветка `full-wiki_v2`) | `full-wiki_v2.git` |
| Индекс GTE (шарды, offsets, титулы; faiss-файла нет — прямому поиску не нужен) | `wiki18-gte/` |
| Корпус Wiki-18, 21 015 324 чанка | `wiki_dump.jsonl` |
| Чекпоинт обученной башни + его config.yaml (sha256 `c8b2212d…` сверен) | `checkpoint/` |
| Семь входов Search-R1 | `searchr1/*.jsonl` |
| Python-окружение лаборатории (torch 2.10, hydra, transformers) | `venv/` |
| HF-кэш с GTE-энкодером (грузится offline) | `hf/` |
| Готовые конфиги, по одному на датасет | `configs/sr1_*_best.yaml` |

## 1. Клонировать код и поправить одну константу

```bash
git clone /dev/shm/qrag-eval/full-wiki_v2.git ~/full-wiki
cd ~/full-wiki
sed -i 's|/home/a.anokhin/venvs/gpu/bin/python|/dev/shm/qrag-eval/venv/bin/python|' src/runlib.py
```

Правка `VENV_PYTHON` в `src/runlib.py` — единственное изменение кода.
Во **всех** командах `make` ниже нужен `PYTHON=...`: дефолтный интерпретатор
Makefile лежит в закрытом каталоге Саши.

## 2. Проверить окружение

```bash
make test PYTHON=/dev/shm/qrag-eval/venv/bin/python
```

Ожидается ровно `146 passed, 7 skipped` (проверено на этом сервере 2026-08-14).

## 3. Поднять свой vLLM (ридер и судья)

Один раз (нужно членство в группе `docker`; если его нет — попроси Сашу
создать контейнер этой же командой):

```bash
docker run -d \
  --name qrag-sr1eval-vllm-8011 \
  --gpus device=6 \
  -p 127.0.0.1:8011:8011 \
  --ipc=host \
  --restart unless-stopped \
  -v /home/a.anokhin/.cache/huggingface/hub/models--Qwen--Qwen3-4B:/models/Qwen3-4B:ro \
  vllm/vllm-openai:latest \
  --model /models/Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --served-model-name Qwen3-4B \
  --host 0.0.0.0 \
  --port 8011 \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.30 \
  --max-model-len 32768 \
  --max-num-seqs 128 \
  --enable-prefix-caching \
  --trust-remote-code
```

Проверка: `curl -fsS http://127.0.0.1:8011/v1/models` должен вернуть Qwen3-4B.
Если GPU 6 занята — поменяй `--gpus device=`, порт не трогай: он зашит в конфиги.

## 4. Запустить семь ранов

Retrieve-стадия зашита на **GPU 7** (`cuda_visible_devices: "7"` в конфигах,
матрица занимает ~60 ГиБ на карте). Перед запуском:

```bash
export HF_HOME=/dev/shm/qrag-eval/hf
cd ~/full-wiki
```

Посмотреть команды без запуска:

```bash
make dry PYTHON=/dev/shm/qrag-eval/venv/bin/python CFG=/dev/shm/qrag-eval/configs/sr1_bamboogle_best.yaml
```

Сами раны (последовательно; bamboogle — 125 вопросов, быстрый, начни с него;
popqa — 14 267, самый долгий):

```bash
for ds in bamboogle nq triviaqa popqa hotpotqa 2wiki musique; do
  make experiment PYTHON=/dev/shm/qrag-eval/venv/bin/python \
       CFG=/dev/shm/qrag-eval/configs/sr1_${ds}_best.yaml
done
```

Драйвер сам прогоняет retrieve → judge → score и отказывается затирать
существующий ран — упавший ран можно перезапустить после `rm -rf runs/<run_id>`.

## 5. Результаты

Итог каждого рана — `runs/<run_id>/metrics.json`. Главная метрика —
**`em_alias`** (EM по всем `golden_answers`, как считает Search-R1);
title-метрики у этих датасетов законно `null`. Опубликованные числа этой же
конфигурации (для сверки, из `docs/searchr1.md`):

| nq | triviaqa | popqa | hotpotqa | 2wiki | musique | bamboogle |
|---:|---:|---:|---:|---:|---:|---:|
| 33.4 | 57.3 | 39.8 | 37.7 | 32.3 | 11.6 | 28.8 |

Вопросы — Саше.
