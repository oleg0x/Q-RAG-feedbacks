# Конфиги

Верхний уровень — ровно то, на чём стоят «Главные цифры» README: три
конфига обучения опубликованных башен и восемь eval-конфигов, чьи раны
цитирует главная таблица. Остальные 88 конфигов всех экспериментов журнала
лежат в [configs_history/](configs_history/) и запускаются так же — путь
передаётся `CFG=` целиком.

## Обучение (три файла)

| Файл | Ран | Что это |
|---|---|---|
| [lineA_main.yaml](lineA_main.yaml) | `Aug03_12-44-38_lineA_main` | линия A, HotpotQA + 2Wiki; чекпоинт `model_best` этого рана дал 37.70 / 30.31 / 10.14 EM |
| [rawmix4b.yaml](rawmix4b.yaml) | `Aug08_20-39-33_rawmix4b` | сырая смесь NQ + HotpotQA, награду считает Qwen3-4B |
| [rawmix8b.yaml](rawmix8b.yaml) | `Aug08_20-39-40_rawmix8b` | та же смесь, награду считает Qwen3-8B |

Это байт-копии `config.yaml` из каталогов ранов
([Q-RAG_for_full-wiki/runs/](../Q-RAG_for_full-wiki/runs/)) — снимки, которые
`train_q_rag_search.py` сохраняет при старте (`resolve=False`, поэтому
`${oc.env:...}` в них не разрезолвлены). Они для чтения и сверки: hydra при
запуске собирает конфиг не отсюда, а из
`Q-RAG_for_full-wiki/configs/training_fullwiki_*.yaml`. Как запустить
обучение — [docs/training.md](../docs/training.md).

## Эвал (восемь файлов)

Запускаются драйвером: `make experiment CFG=configs/<имя>.yaml`
(см. [docs/pipeline.md](../docs/pipeline.md)).

| Файл | Ран | Строка главной таблицы |
|---|---|---|
| [search_best_hotpotqa.yaml](search_best_hotpotqa.yaml) | `2026-08-05-lineA-best-hotpotqa` | обученная башня, HotpotQA, 37.70 |
| [search_best_2wiki.yaml](search_best_2wiki.yaml) | `2026-08-05-lineA-best-2wiki` | обученная башня, 2Wiki, 30.31 |
| [search_best_musique.yaml](search_best_musique.yaml) | `2026-08-05-lineA-best-musique` | обученная башня, MuSiQue, 10.14 |
| [search_zeroshot_steps6.yaml](search_zeroshot_steps6.yaml) | `2026-08-02-search-zeroshot-steps6` | zero-shot, HotpotQA, 27.16 |
| [search_zeroshot_2wiki.yaml](search_zeroshot_2wiki.yaml) | `2026-08-05-search-zeroshot-2wiki` | zero-shot, 2Wiki, 15.83 |
| [search_zeroshot_musique.yaml](search_zeroshot_musique.yaml) | `2026-08-05-search-zeroshot-musique` | zero-shot, MuSiQue, 4.96 |
| [no_retrieval.yaml](no_retrieval.yaml) | `2026-07-28-no-retrieval` | ридер без контекста, 15.53 |
| [gte_only_n2_steps6.yaml](gte_only_n2_steps6.yaml) | `2026-08-02-gte-only-n2-steps6` | GTE top-6, квота N=2, 29.29 |

Пути в конфигах машинно-специфичны (см. раздел README «Пути
машинно-специфичны»). Конфиги реранкинга июльской линии в `configs_history/`
дополнительно требуют соседний `../Q-RAG-feedback` как `retrieve.qrag_repo` —
подменять его копией `Q-RAG_for_full-wiki` нельзя, у них разный пулинг.
