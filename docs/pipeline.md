# Пайплайн Full-Wiki

Пошаговый runbook: от эмбеддингов корпуса до итоговой таблицы. Разделы 1–3
выполняются один раз на корпус, разделы 4–7 — на каждый эксперимент.

Команды ниже остаются источником истины по флагам. В повседневной работе
разделы 4, 6 и 7 запускаются не руками, а драйвером `exp.py` — он собирает
ровно эти вызовы и записывает их в `runs/<run_id>/cmd.sh`; см.
[conventions.md](conventions.md).

Входные данные, версии и хеши — [reproducibility.md](reproducibility.md).
Грабли — [gotchas.md](gotchas.md).

## 1. Построение first-stage GTE-индекса

GTE нужен только для генерации top-100 кандидатов. Passage embeddings
L2-нормализуются, поэтому `IndexFlatIP` эквивалентен exact cosine search.

Команда воспроизведения с зафиксированной revision:

```bash
cd /home/a.anokhin/Judge/full-wiki

/home/a.anokhin/venvs/gpu/bin/python src/build_index_wiki_gte.py \
  --corpus /home/a.anokhin/Judge/datasets/data_sources/full-wiki/data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl \
  --output-dir /home/a.anokhin/Judge/datasets/data_sources/full-wiki/wiki18-gte \
  --model Alibaba-NLP/gte-multilingual-base \
  --revision 9bbca17d9273fd0d03d5725c7a4b0f6b45142062 \
  --device cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7 \
  --model-dtype float16 \
  --max-length 256 \
  --batch-size 256 \
  --multi-process-chunk-size 1000 \
  --shard-size 100000 \
  --truncation-samples-per-shard 2048 \
  --cache-dtype float32 \
  --phase all \
  --index-name gte-multilingual-base.flatip.faiss \
  --verify-samples 128 \
  --seed 42 \
  --trust-remote-code
```

Список GPU можно сократить до реально свободных устройств. Он влияет на время
сборки, но не меняет схему артефакта.

Фактический GTE artifact:

| Параметр | Значение |
|---|---|
| Rows | 21 015 324 |
| Dimension | 768 |
| Model inference dtype | float16 |
| Cache dtype | float32 |
| Rows per shard | 100 000 |
| Shards | 211 |
| Normalization | L2 |
| Index | `faiss.IndexFlatIP` |
| Index metric | inner product по L2-нормализованным векторам |
| Index size | 64 559 075 373 bytes |
| ID map | implicit contiguous zero-based row IDs |
| Verification | 128 random vectors, finite values, reconstruction, L2 norms |

Оценка truncation при `max_length=256`: 608 из 432 128 проверенных пассажей,
или 0.1407%; максимальная наблюдавшаяся длина — 829 токенов.

Полный machine-readable источник параметров:

```text
/home/a.anokhin/Judge/datasets/data_sources/full-wiki/wiki18-gte/manifest.json
```

## 2. Построение Q-RAG action-эмбеддингов

Action embeddings извлекаются из `critic.action_embed.*` checkpoint-а. Они не
нормализуются: Q-RAG был обучен на raw dot product. Нельзя L2-нормализовать эти
векторы или заменить pooling без повторного обучения.

Команда, эквивалентная использованной сборке:

```bash
cd /home/a.anokhin/Judge/full-wiki

/home/a.anokhin/venvs/gpu/bin/python src/build_index_wiki_qrag.py \
  --run-dir /home/a.anokhin/Judge/Q-RAG-feedback/runs/Jul21_19-44-36_QRAG_HotPotQA+2WikiMultihopQA \
  --checkpoint best \
  --qrag-repo /home/a.anokhin/Judge/Q-RAG-feedback \
  --corpus /home/a.anokhin/Judge/datasets/data_sources/full-wiki/data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl \
  --output-dir /home/a.anokhin/Judge/datasets/data_sources/full-wiki/wiki18-qrag-jul21-best-raw \
  --text-format raw \
  --device cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:7 \
  --model-dtype float32 \
  --cache-dtype float32 \
  --max-length 256 \
  --batch-size 256 \
  --multi-process-chunk-size 1000 \
  --shard-size 100000 \
  --truncation-samples-per-shard 2048 \
  --phase all \
  --index-name qrag-action-best.raw.flatip.faiss \
  --verify-samples 128 \
  --seed 42
```

Фактически длительная embedding-фаза и короткая index-фаза запускались
раздельно с `--phase embed`, затем `--phase index`. Скрипт restartable: готовые
shards сохраняются, а совместимость повторного запуска проверяется через
`build-state.json`. `--phase all` выполняет тот же пайплайн за один запуск.

Фактический action artifact:

| Параметр | Значение |
|---|---|
| Checkpoint section | critic |
| State prefix | `state_embed.` |
| Action prefix | `action_embed.` |
| Pooling | masked mean |
| Output scale | 0.1 |
| Positions processor | none |
| Model dtype | float32 |
| Cache dtype | float32 |
| Normalization | none |
| Dimension | 768 |
| Rows per shard | 100 000 |
| Shards | 211 |
| Batch size | 256 |
| Multi-process chunk | 1 000 |
| Index | raw `faiss.IndexFlatIP` |
| Index size | 64 559 075 373 bytes |
| Verification | 128 random vectors, finite/nonzero, shard-index equality |

При фактической сборке семь H200 обработали action embeddings примерно за
1 час 49 минут. Запись и проверка FlatIP-индекса заняли около 2.5 минуты.

Полный manifest:

```text
/home/a.anokhin/Judge/datasets/data_sources/full-wiki/
wiki18-qrag-jul21-best-raw/manifest.json
```

Важно: raw action FAISS-файл построен и верифицирован, но двухступенчатый
retrieval не загружает его на hot path. Он читает только нужные action-векторы
из `embedding-shards` по глобальным row ID. FAISS-файл нужен для direct-search
абляции и независимой проверки целостности.

## 3. Построение таблицы byte offsets корпуса

Offsets позволяют по row ID читать нужные строки 14-гигабайтного JSONL без
полного сканирования корпуса на каждом запросе.

```bash
cd /home/a.anokhin/Judge/full-wiki

/home/a.anokhin/venvs/gpu/bin/python src/fullwiki_qrag.py prepare \
  --index-dir /home/a.anokhin/Judge/datasets/data_sources/full-wiki/wiki18-qrag-jul21-best-raw
```

Создаются:

```text
corpus-row-offsets.npy   uint64, shape=(21 015 325,), 168 122 728 bytes
corpus-row-offsets.json  metadata корпуса и последнего offset
```

Таблица содержит на один offset больше, чем строк: последний элемент равен
размеру corpus-файла.

## 3a. Таблица титулов корпуса

Нужна прямому поиску (§4e): квота `N` чанков на титул маскирует строки **до**
`topk`, и титул любой из 21 млн строк приходится знать на каждом шаге. Читать
ради этого сто строк корпуса на шаг нельзя, поэтому титулы раскладываются один
раз в `int32`-таблицу на 84 МБ.

```bash
/home/a.anokhin/venvs/gpu/bin/python src/build_title_table.py \
  --index-dir /home/a.anokhin/Judge/datasets/data_sources/full-wiki/wiki18-gte \
  --output /home/a.anokhin/Judge/datasets/data_sources/full-wiki/wiki18-gte/corpus-title-ids.npy
```

Создаются:

```text
corpus-title-ids.npy              int32, shape=(21 015 324,), 84 061 296 bytes
corpus-title-ids.titles.jsonl.gz  титулы, строка i — это title_id == i
corpus-title-ids.json             идентичность корпуса, число строк и титулов
```

Скрипт сверяет выборку строк с полным JSON-разбором (`--verify-samples`, по
умолчанию 200) и при расхождении не оставляет артефакта. Проверено на первых
100 000 строках корпуса: 29 807 различных титулов, 200 сверенных строк.

## 4. Full-Wiki retrieval для steps 2/4/6

Параметры всех опубликованных retrieval-прогонов:

| Параметр | Значение |
|---|---|
| Dataset | HotpotQA FullWiki dev, 7 405 примеров |
| First stage | `Alibaba-NLP/gte-multilingual-base` |
| Candidate backend | `torch-cuda`, exact float32 matrix |
| Candidate pool | top-100 |
| Q-RAG checkpoint | Jul21 best, critic |
| State source | critic |
| State model dtype | float32 |
| Action dtype | float32 |
| Scoring | raw inner product |
| Mode | fixed |
| Title deduplication | enabled |
| Retrieval batch size | 64 |
| Steps | 2, 4 или 6 |
| GTE query refresh | нет; пул строится один раз |
| Auto-prepare | отключён после явного шага prepare |
| Physical GPU | GPU 1; внутри процесса `cuda:0` |
| Reranker | `qrag` (дефолт) |
| Candidate log | `full` (дефолт) |

Команда для всех трёх вариантов:

```bash
cd /home/a.anokhin/Judge/full-wiki
set -euo pipefail

for qrag_steps in 2 4 6; do
  case "$qrag_steps" in
    2) qrag_output=/home/a.anokhin/Judge/full-wiki/runs/fullwiki_qrag_fixed.jsonl ;;
    4) qrag_output=/home/a.anokhin/Judge/full-wiki/runs/fullwiki_qrag_fixed_steps4.jsonl ;;
    6) qrag_output=/home/a.anokhin/Judge/full-wiki/runs/fullwiki_qrag_fixed_steps6.jsonl ;;
  esac

  CUDA_VISIBLE_DEVICES=1 \
  HF_HUB_OFFLINE=1 \
  TRANSFORMERS_OFFLINE=1 \
  /home/a.anokhin/venvs/gpu/bin/python src/fullwiki_qrag.py retrieve \
    --index-dir /home/a.anokhin/Judge/datasets/data_sources/full-wiki/wiki18-qrag-jul21-best-raw \
    --candidate-index-dir /home/a.anokhin/Judge/datasets/data_sources/full-wiki/wiki18-gte \
    --candidate-backend torch-cuda \
    --qrag-repo /home/a.anokhin/Judge/Q-RAG-feedback \
    --device cuda:0 \
    --model-dtype float32 \
    --state-source critic \
    --input /home/a.anokhin/Judge/datasets/data_sources/hotpotqa/hotpot_dev_fullwiki_v1.json \
    --output "$qrag_output" \
    --batch-size 64 \
    --top-k 100 \
    --steps "$qrag_steps" \
    --mode fixed \
    --dedupe-titles \
    --trust-remote-code \
    --no-auto-prepare
done
```

Если задан `CUDA_VISIBLE_DEVICES=1`, физическая GPU 1 перенумеровывается в
`cuda:0` внутри процесса. Перед запуском нужно проверить свободную память через
`nvidia-smi`; физический номер можно заменить.

Каждая выходная JSONL-запись содержит, среди прочего:

```text
id, question, answer, pred_idx, pred_texts, q_values, retrieval_hops,
candidate_retriever, reranker, retrieval_mode, candidate_pool_size,
gold_titles, retrieved_titles,
title_recall, title_em, title_recall_exact, title_em_exact
```

### Флаги `--reranker` и `--log-candidates`

`--reranker {qrag,none}` (дефолт `qrag`). При `none` Q-RAG-скоринг
пропускается целиком: не грузится ни state-башня, ни action-шарды, а выбор
идёт сверху вниз по ранжированию GTE с тем же исключением уже выбранных
row ID и заголовков. Это baseline, без которого таблица не может ничего
утверждать про вклад Q-RAG. Требует `--candidate-index-dir`.

В режиме `fixed` этот baseline **префикс-согласован**: пул фиксирован, а
ранжирование не зависит от состояния, поэтому выбор на 2 шага — буквально
префикс выбора на 6. Достаточно одного прогона с `--steps 6`, остальные
бюджеты получаются усечением через `build_eval_variants.py`. Для Q-RAG это
неверно: каждый хоп переранжирует пул относительно обновлённого состояния,
и скрипт отказывается усекать такие раны.

`--log-candidates {full,topk,none}` (дефолт `full`) управляет дампом
кандидатов в `retrieval_hops`. Дефолт воспроизводит опубликованные раны
побайтово. `none` уменьшает файлы примерно в семь раз, что важно, потому что
`answer_judge_llms.py` копирует в свой результат все поля входного JSONL:
Q-RAG steps=6 c `full` даёт 467 МБ answer JSON, GTE-6 с `none` — 46 МБ.

### Нормализация титулов

`title_recall` и `title_em` сравнивают титулы нормализованно:
`html.unescape` → NFKC → casefold → `_` в пробел → схлопывание пробелов.
Это необходимо, потому что HotpotQA собран на дампе 2017 года и хранит
титулы вида `Procter &amp; Gamble`, а Wiki-18 — дамп 2018 года с
раскодированной формой. Точное сравнение строк занижает покрытие примерно
на 4.6 п.п.

Поля `title_recall_exact` и `title_em_exact` сохраняют исходное точное
сравнение, поэтому ранее опубликованные числа остаются воспроизводимыми.

### Retrieval-диагностика

| Конфигурация | gold-title recall | gold-title EM | то же, точным сравнением |
|---|---:|---:|---:|
| GTE top-2 | 42.44% | 10.10% | 9.47% |
| GTE top-4 | 49.10% | 19.05% | 17.80% |
| GTE top-6 | 52.33% | 23.57% | 22.15% |
| Q-RAG, 2 шага | 36.60% | 14.98% | 13.87% |
| Q-RAG, 4 шага | 45.02% | 24.15% | 22.39% |
| Q-RAG, 6 шагов | 47.93% | 26.70% | 24.83% |

`title_recall` — доля gold supporting titles среди выбранных заголовков.
`title_em=1`, если найдены все gold titles конкретного примера. Колонка
точного сравнения совпадает с числами, опубликованными до введения
нормализации.

Проверка retrieval-файла:

```bash
jq -s '{
  samples: length,
  steps_min: (map(.pred_texts | length) | min),
  steps_max: (map(.pred_texts | length) | max),
  title_recall: (map(.title_recall) | add / length),
  title_em: (map(.title_em) | add / length),
  title_em_exact: (map(.title_em_exact) | add / length),
  rerankers: (map(.reranker) | unique),
  modes: (map(.retrieval_mode) | unique),
  pool_sizes: (map(.candidate_pool_size) | unique)
}' /path/to/retrieval.jsonl
```

## 4a. Потолок покрытия корпуса

`title_em` не может достичь 1.0 на HotpotQA. Вопросы писались по дампу
Википедии 2017 года, а Wiki-18 — это дамп 2018 года, нарезанный на
100-словные чанки, поэтому часть gold-статей переименована, удалена или не
попала в DPR-срез. Любое retrieval-число интерпретируется только
относительно этого потолка, и потолок надо измерять, а не предполагать.

```bash
cd /home/a.anokhin/Judge/full-wiki

/home/a.anokhin/venvs/gpu/bin/python src/corpus_title_coverage.py \
  --corpus /home/a.anokhin/Judge/datasets/data_sources/full-wiki/data00/jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl \
  --dataset /home/a.anokhin/Judge/datasets/data_sources/hotpotqa/hotpot_dev_fullwiki_v1.json \
  --output runs/hotpotqa_dev_title_coverage.json
```

Сканирование потоковое, только строка титула каждого чанка; 21 015 324
строки обрабатываются примерно за 48 секунд, после чего байтовый парсер
сверяется с полным JSON-декодом на выборке строк.

Результат для HotpotQA dev:

| Матчинг | покрытие титулов | доля примеров, где есть все gold-титулы |
|---|---:|---:|
| точное сравнение | 87.60% | **77.11%** |
| NFKC + casefold + `_` | 89.87% | 80.81% |
| + `html.unescape` | 90.38% | **81.67%** |

В корпусе 3 232 908 уникальных титулов. У 18.33% примеров HotpotQA dev хотя
бы одного gold-титула нет в Wiki-18 ни в какой форме — эти примеры не может
взять никакой ретривер. Из 5 428 промахов `title_em` в ране Q-RAG на шести
шагах 1 357, то есть ровно 25.0%, приходится именно на них.

Отсюда колонка «доля потолка» в итоговой таблице: `title_em / 0.8167`.

## 4b. Baseline-ы: GTE без Q-RAG, no-retrieval и oracle

Три из четырёх baseline-ов не требуют ни GPU, ни повторного ретривала.

Первый — единственный прогон first-stage на шести шагах:

```bash
cd /home/a.anokhin/Judge/full-wiki

CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
/home/a.anokhin/venvs/gpu/bin/python src/fullwiki_qrag.py retrieve \
  --index-dir /home/a.anokhin/Judge/datasets/data_sources/full-wiki/wiki18-qrag-jul21-best-raw \
  --candidate-index-dir /home/a.anokhin/Judge/datasets/data_sources/full-wiki/wiki18-gte \
  --candidate-backend torch-cuda \
  --qrag-repo /home/a.anokhin/Judge/Q-RAG-feedback \
  --device cuda:0 --model-dtype float32 --state-source critic \
  --input /home/a.anokhin/Judge/datasets/data_sources/hotpotqa/hotpot_dev_fullwiki_v1.json \
  --output runs/fullwiki_gte_only_steps6.jsonl \
  --batch-size 64 --top-k 100 --steps 6 --mode fixed --dedupe-titles \
  --reranker none --log-candidates none \
  --trust-remote-code --no-auto-prepare
```

Остальные — производные от него:

```bash
for baseline_steps in 2 4; do
  /home/a.anokhin/venvs/gpu/bin/python src/build_eval_variants.py \
    --context truncate --steps "$baseline_steps" \
    --input runs/fullwiki_gte_only_steps6.jsonl \
    --output "runs/fullwiki_gte_only_steps${baseline_steps}.jsonl"
done

/home/a.anokhin/venvs/gpu/bin/python src/build_eval_variants.py --context none \
  --input runs/fullwiki_gte_only_steps6.jsonl \
  --output runs/fullwiki_no_retrieval.jsonl

/home/a.anokhin/venvs/gpu/bin/python src/build_eval_variants.py --context oracle \
  --oracle-source /home/a.anokhin/Judge/datasets/data_sources/hotpotqa/hotpot_dev_distractor_v1.json \
  --input runs/fullwiki_gte_only_steps6.jsonl \
  --output runs/fullwiki_oracle_sf.jsonl
```

**Источник oracle-контекста — distractor-файл, а не fullwiki.** Поле
`context` в `hotpot_dev_fullwiki_v1.json` содержит не gold-абзацы, а top-10
исходного full-wiki ретривера HotpotQA: у 5 316 из 7 405 примеров там нет ни
одного gold-титула. Поэтому поле `sf_texts`, которое `fullwiki_qrag.py`
восстанавливает из этого контекста, заполнено лишь у 6 162 записей и почти
всегда неполно; для oracle оно непригодно. Файл
`hotpot_dev_distractor_v1.json` содержит те же id в том же порядке и все
gold-абзацы.

Oracle измеряет «идеальный ретривал по HotpotQA», а не по Wiki-18: контекст
собирается из предложений HotpotQA, а не из 100-словных чанков корпуса.

## 4c. Потолок кандидатного пула

Потолок покрытия корпуса (§4a) отвечает на вопрос «что вообще есть в
Wiki-18». Он не отвечает на более узкий вопрос: сколько из этого доходит до
реранкера. Реранкер выбирает только из тех 100 кандидатов, что вернул GTE, и
всё, чего в них нет, для него не существует.

Чтобы измерить этот потолок, нужен ран с сохранённым пулом. Канонические раны
сняты с `--log-candidates none`, поэтому списков кандидатов в них нет:

```bash
cd /home/a.anokhin/Judge/full-wiki

/home/a.anokhin/venvs/gpu/bin/python exp.py run \
  --config configs/configs_history/gte_pool_diag.yaml --only retrieve
```

Ран отличается от `gte_only_steps6` ровно одним полем `log_candidates: full`
и повторяет его выбор чанков в точности — 7 405 из 7 405 совпадений по
`pred_idx`. Цена — 274 МБ `retrieval.jsonl` против 41 МБ. Стадия judge для
него не нужна: ответы уже измерены строкой GTE top-6.

Дальше пул разбирается `candidate_pool.py`:

```bash
IDX=/home/a.anokhin/Judge/datasets/data_sources/full-wiki/wiki18-qrag-jul21-best-raw

/home/a.anokhin/venvs/gpu/bin/python src/candidate_pool.py report \
  --input runs/runs_history/2026-08-02-gte-pool-diag/retrieval.jsonl \
  --index-dir "$IDX" \
  --gold-source /home/a.anokhin/Judge/datasets/data_sources/hotpotqa/hotpot_dev_distractor_v1.json \
  --coverage runs/shared/hotpotqa_dev_title_coverage.json \
  --output runs/shared/hotpotqa_dev_pool_diagnostics.json
```

Пул берётся из `retrieval_hops[0]["first_stage_candidate_idx"]` — сырого
ранжирования GTE. В режиме `fixed` он один и тот же на всех шагах, поэтому
первого хопа достаточно. Row ID резолвятся в титулы через ту же таблицу
байтовых смещений (§3), что и сам ретривал.

Результат на HotpotQA dev, top-100:

| Величина | Значение |
|---|---:|
| title recall@100 | 66.13% |
| title EM@100 | 43.79% |
| доля потолка корпуса (81.67%) | 53.62% |
| примеров без единого gold-титула | 11.53% |
| примеров ровно с одним gold-титулом | 44.67% |
| примеров со всеми gold-титулами | 43.79% |

Ключевая диагностика bridge-хопа — позиции gold-титулов в ранжировании:

| Позиция (0-based) | медиана | p90 |
|---|---:|---:|
| первый найденный gold | 0 | 7 |
| второй найденный gold | 12 | 63 |

**У 50.50% примеров, где первый gold-титул найден, второго в top-100 нет
вообще.** Первую сущность вопроса GTE находит почти всегда и сразу; вторая,
на которую указывает bridge, лежит либо глубоко, либо за пределами пула.
Реранкингом это не лечится — только расширением пула или другим
first-stage.

Второй потолок — гранулярность чанка. Из 9 794 пар «пример, gold-титул», где
титул в пуле есть, gold-предложение целиком лежит в выбранном чанке только у
31.0%; у 40.7% его там нет вовсе, остальное — предложения, разорванные
границей 100-словного чанка. Хотя бы одно gold-предложение теряют 78.3%
примеров, у которых титул найден.

Наконец, `select` собирает вариант с идеальным выбором по титулам:

```bash
/home/a.anokhin/venvs/gpu/bin/python exp.py run \
  --config configs/configs_history/pool_oracle_titles_k6.yaml
```

Это `retrieve.kind: pool` — из логированных 100 кандидатов берутся k чанков,
максимизирующих покрытие gold-титулов, тай-брейк по рангу GTE, дедупликация
титулов соблюдается. Выбор префикс-согласован по бюджету: k=2 является
префиксом k=6.

Поскольку у HotpotQA ровно два gold-титула на пример, `title EM` такого
выбора равен title EM@100 (43.79%) на всех бюджетах — добор четвёртого и
шестого чанка титульных метрик уже не меняет, но меняет контекст ридера.
**Это oracle по титулам, а не по ответу:** он не ограничен сверху ни
гранулярностью чанка, ни покрытием корпуса, поэтому подавать его как «предел
задачи» нельзя. Абсолютный потолок ридера меряет строка
`oracle (gold sentences)`.

### Пул шире top-100

Тот же ран с другим `top_k` отвечает, лечится ли размером пула то, что
половина вторых хопов до него не доходит:

```bash
/home/a.anokhin/venvs/gpu/bin/python exp.py run \
  --config configs/configs_history/gte_pool_diag_k500.yaml --only retrieve
/home/a.anokhin/venvs/gpu/bin/python exp.py run \
  --config configs/configs_history/gte_pool_diag_k1000.yaml --only retrieve
```

У обоих `steps: 1`, а не 6. В режиме `fixed` пул считается один раз и
переиспользуется каждым шагом, а `report` читает только
`retrieval_hops[0]`, поэтому шесть шагов записали бы один и тот же пул шесть
раз: 230 МБ и 448 МБ вместо 1.4 и 2.9 ГБ. Выбор чанков этими ранами не
измеряется, стадии judge и score им не нужны.

`report` по ним — те же флаги, другой `--output`
(`hotpotqa_dev_pool_diagnostics_k500.json`, `..._k1000.json`).

| Величина | top-100 | top-500 | top-1000 |
|---|---:|---:|---:|
| title recall@pool | 66.13% | 74.25% | 77.19% |
| title EM@pool | 43.79% | 56.58% | 61.13% |
| доля потолка корпуса (81.67%) | 53.62% | 69.28% | 74.85% |
| примеров без единого gold-титула | 11.53% | 8.08% | 6.75% |
| второго gold нет при найденном первом | 50.50% | 38.45% | 34.44% |
| медиана позиции второго gold | 12 | 21 | 26 |
| p90 позиции второго gold | 63 | 238 | 399 |

Пул растёт, но не насыщается: расширение в 10 раз даёт +17.3 п.п. title
EM@pool. Цена — глубина: p90 второго gold уходит с 63 на 399, то есть
реранкер над таким пулом обязан уметь вытаскивать нужный чанк с четвёртой
сотни.

### Чанк-осознанный выбор и квота на титул

`select` умеет два независимых послабления:

```bash
/home/a.anokhin/venvs/gpu/bin/python exp.py run \
  --config configs/configs_history/pool_oracle_chunks_k6.yaml
```

`prefer: gold-sentence` (плюс обязательный `gold_source`) меняет то, **какой
чанк gold-титула** берётся: вперёд идёт содержащий gold-предложение —
целиком, затем наполовину, затем никак, — и только при равенстве решает ранг
GTE. Оракул по титулам всегда брал лучший по рангу и тем занижал потолок:
реранкер работает с чанками и мог бы взять другой чанк того же титула.

`max_per_title: N` разрешает взять N чанков одной статьи. Вторые чанки
берутся раундами и только если несут gold-текст, иначе бюджет уходил бы на
уже покрытую статью вместо нового титула. Оба послабления сохраняют
префикс-согласованность: k=2 остаётся префиксом k=6.

## 4d. Refresh: пересчёт запроса первой стадии

В `--mode refresh` GTE запрашивается заново после каждого выбранного чанка
запросом `"вопрос [SEP] чанк1 [SEP] ..."`. Раньше режим был помечен как
экспериментальный и не измерялся на полной выборке.

Мерить его при манифестных 256 токенах бессмысленно: медиана чанка Wiki-18 —
158 токенов, поэтому на шаге 3 из запроса теряется 71 токен, а на шаге 6 —
571 из 827, и шаги 3–6 получают почти одинаковый вход. Флаг
`--first-stage-query-length` поднимает лимит только на стороне запроса;
эмбеддинги корпуса не меняются.

```bash
/home/a.anokhin/venvs/gpu/bin/python exp.py run \
  --config configs/configs_history/gte_refresh_q1024_steps6.yaml
```

**Усекать refresh-ран нельзя** — пул меняется между шагами, префикса не
существует, и `build_eval_variants.py` такой ран не примет. Каждый бюджет —
отдельный ретривал, поэтому конфигов три, а не один.

Контрольный ран `gte_refresh_q256_steps6.yaml` повторяет схему при 256
токенах: разница между ним и `q1024` на шести шагах и есть цена обрезки
запроса.

## 4e. Прямой поиск state-башней (линия A)

Подкоманда `search` — не флаг `retrieve`, а другой ретривер: первой стадии у
него нет вовсе, запросом к индексу служит само состояние, и пул каждого шага
считается заново. Обоснование и решения — в [training.md](training.md),
«Линия A».

```bash
/home/a.anokhin/venvs/gpu/bin/python src/fullwiki_qrag.py search \
  --index-dir /home/a.anokhin/Judge/datasets/data_sources/full-wiki/wiki18-gte \
  --qrag-repo /home/a.anokhin/Judge/full-wiki/Q-RAG_for_full-wiki \
  --device cuda:1 \
  --input /home/a.anokhin/Judge/datasets/data_sources/hotpotqa/hotpot_dev_fullwiki_v1.json \
  --output runs/<run>/retrieval.jsonl \
  --top-k 100 --steps 6 --max-chunks-per-title 2
```

| Флаг | Смысл |
|---|---|
| `--index-dir` | `wiki18-gte`: он служит матрицей действий `M`, а не первой стадией |
| `--qrag-repo` | обязан быть `Q-RAG_for_full-wiki` — см. ниже |
| `--checkpoint` | `model_*.pt` рана линии A; **без него башня стоковая** (zero-shot) |
| `--title-table` | таблица из §3a; по умолчанию берётся из `--index-dir` |
| `--max-chunks-per-title` | квота `N`, умолчание 2 |
| `--log-candidates` | умолчание `none`: шесть шагов на 7 405 вопросов иначе дают 4.4 млн id |

Zero-shot без чекпоинта — это стартовая точка лестницы фазы 2: 33% после
обучения означают разное, если старт был 29% или 24%. Башня собирается той же
архитектурой и тем же путём кодирования, что и обучаемая, поэтому сравниваются
две точки одного пайплайна, а не два разных.

**Про `--qrag-repo`.** В дереве два экземпляра `rl/`: read-only оригинал
считает masked mean с делением на 10, копия — CLS с L2-нормировкой, то есть
ровно то, чем собран `wiki18-gte`. Подключить оригинал молча значит получить
состояние в чужом пространстве. Команда это ловит дважды: по сигнатуре
`BertPredictor` (параметр `normalize` есть только в копии) и, в zero-shot,
сверкой выхода башни со строкой матрицы — `--verify-alignment`, включена по
умолчанию. На срезе корпуса в 100 000 строк косинус равен **0.999999**, и
первые 10 результатов совпадают с ранжированием `SentenceTransformer` GTE
строка в строку.

Усекать такой ран нельзя: каждый шаг переранжирует по обновлённому состоянию,
префикса не существует, и `build_eval_variants.py --context truncate` его
отклонит (`reranker=qrag-state-direct`).

## 4f. Три датасета: HotpotQA, 2Wiki, MuSiQue

Датасет рана называется полем `dataset` в конфиге; известные перечислены в
`runlib.DATASETS` вместе с файлами покрытия. От него зависят две вещи:
потолок в `metrics.json` и таблица RESULTS.md, в которую попадёт строка.
Раны разных датасетов не сравниваются ни таблицей, ни МакНемаром — у них
разные вопросы и разный потолок корпуса.

| Датасет | Ключ | Примеров | Потолок покрытия |
|---|---|---:|---:|
| HotpotQA dev fullwiki | `hotpotqa_dev_fullwiki` | 7 405 | 81.67% |
| 2WikiMultiHopQA dev | `2wiki_dev` | 12 576 | 51.05% |
| MuSiQue-Ans dev | `musique_ans_dev` | 2 417 | 72.11% |

Публичных test-сплитов с ответами нет ни у одного из трёх: `2Wiki test.json`
несёт пустые `answer` и `supporting_facts`, а `musique_*_test.jsonl` вовсе не
содержит поля `answer`. Меряется поэтому dev — то же, что репортят Search-R1
и FlashRAG.

**2Wiki** идёт в пайплайн как есть: его `dev.json` уже в схеме HotpotQA.
**MuSiQue** — нет, у него плоский список `paragraphs` вместо пары
`context`/`supporting_facts`, поэтому вход готовится один раз:

```bash
python src/build_musique_eval.py \
  --input /home/a.anokhin/Judge/datasets/data_sources/musique/musique_ans_v1.0_dev.jsonl \
  --output runs/shared/musique_ans_dev_hotpot_schema.jsonl
```

Пропустить конвертацию нельзя, и падения при этом не будет — будет тихо
неверный результат: без `supporting_facts` список gold-титулов пуст, а
`title_metrics` на пустом голде возвращает `title_recall = title_em = 1.0`.

Берётся `musique_ans`, а не `musique_full`: у full те же 2 417 вопросов
продублированы неотвечаемыми двойниками, а промпты ридера и судьи отказ от
ответа не предусматривают.

**Алиасы ответов.** Официальные эвалы 2Wiki и MuSiQue считают EM максимумом
по списку допустимых ответов, а `answer_judge_llms.py` сравнивает с
единственным `answer`. Основные метрики остаются как есть — иначе строки
станут несравнимы со всеми прошлыми ранами, — а alias-версии считает
`export_eval_json.py` (§7).

## 5. Развёртывание отдельного vLLM

Итоговые answer-метрики получены на собственном loopback-only vLLM, а не на
общем сервере.

Фактическая конфигурация:

| Параметр | Значение |
|---|---|
| Container | `qrag-fullwiki-vllm-8010` |
| Image | `vllm/vllm-openai:latest` |
| Docker image ID | `sha256:8c9aaddfa6011b9651d06834d2fb90bdb9ab6ced4b420ec76925024eb12b22d0` |
| vLLM | 0.15.1 |
| Model | Qwen3-4B |
| Model snapshot | `1cfa9a7208912126459214e8b04321603b3df60c` |
| Dtype | bfloat16 |
| Tensor parallel | 1 |
| Max model length | 32 768 |
| Max sequences | 128 |
| GPU memory utilization | 0.50 |
| Prefix caching | enabled |
| Physical GPU | 4 |
| Endpoint | `http://127.0.0.1:8010/v1` |
| Authentication | отсутствует; порт доступен только на loopback |
| Restart policy | unless-stopped |

Создание контейнера выполняется один раз:

```bash
docker run -d \
  --name qrag-fullwiki-vllm-8010 \
  --gpus device=4 \
  -p 127.0.0.1:8010:8010 \
  --ipc=host \
  --restart unless-stopped \
  -v /home/a.anokhin/.cache/huggingface/hub/models--Qwen--Qwen3-4B:/models/Qwen3-4B:ro \
  vllm/vllm-openai:latest \
  --model /models/Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --served-model-name Qwen3-4B \
  --host 0.0.0.0 \
  --port 8010 \
  --dtype bfloat16 \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.50 \
  --max-model-len 32768 \
  --max-num-seqs 128 \
  --enable-prefix-caching \
  --trust-remote-code
```

Если контейнер уже создан:

```bash
docker start qrag-fullwiki-vllm-8010
curl -fsS http://127.0.0.1:8010/v1/models | jq
```

## 6. Reader и LLM-as-a-Judge

Используется:

```text
/home/a.anokhin/Judge/full-wiki/src/answer_judge.py
```

Это функциональный форк read-only `Q-RAG-feedback/answer_judge_llms.py`: тот
же клиент, те же промпты, то же декодирование, те же аргументы. Добавлены
четыре флага — `--contract`, `--dataset`, `--aliases`, `--judge-all`, — всё
остальное совпадает. С 2026-08-11 клиент и промпты — вендорные
байт-копии оригинала (`src/vendored_vllm_client.py`, `src/vendored_prompts.py`),
поэтому форк работает без соседнего `Q-RAG-feedback`; их синхронность с
оригиналом проверяет `tests/test_vendored.py` там, где оригинал есть.

`--contract v1` — прежнее поведение бит в бит: `EM`/`F1` против единственного
`answer`, судья на каждом примере. Им сняты все 65 ранов до 2026-08-08, и
пересуживать их не нужно.

`--contract v2` (умолчание с 2026-08-08) — награда по вариантам ответа:
`em_alias` и `f1_alias` максимумом по списку допустимых, судья только там, где
`em_alias == 0`, с эталоном `" | ".join(варианты)`, и
`reward = 1.0 если em_alias иначе вердикт`. Варианты берутся из таблицы
датасета (`--dataset <ключ>` через `runlib.DATASETS` либо `--aliases <путь>`),
**не из `retrieval.jsonl`**: стадия ретривала алиасы роняет. Там, где судью не
звали, `LLM_Judge_Score` вменена единицей и помечена `judge_imputed: true`.

`--judge-all` зовёт судью на каждом примере независимо от `em_alias` — нужен
для диагностики расхождения EM и вердикта.

Зафиксированные параметры оценки:

| Параметр | Значение |
|---|---|
| Reader model | Qwen3-4B |
| Judge model | Qwen3-4B |
| System prompts | `sys_qa` и `sys_judge` (вендорная копия `src/vendored_prompts.py`) |
| Context | все `pred_texts`, разделённые двумя переводами строки |
| Thinking | false для reader и judge |
| Temperature | 0.0 |
| Reader max tokens | 1 000 |
| Judge max tokens | 100 |
| Samples | 7 405 |
| Parallel clients | 32 |
| API | OpenAI-compatible `/v1/chat/completions` |

Один процесс `answer_judge_llms.py` делает запросы последовательно. Для полного
прогона retrieval JSONL делился на 32 последовательных shard-а, каждый shard
обрабатывался штатным скриптом, после чего JSON-массивы объединялись в исходном
порядке. Параллелизм влияет только на скорость, но не на prompts или decoding.

Команда для steps 2/4/6:

```bash
cd /home/a.anokhin/Judge/Q-RAG-feedback
set -euo pipefail

for qrag_steps in 2 4 6; do
  case "$qrag_steps" in
    2)
      qrag_input=/home/a.anokhin/Judge/full-wiki/runs/fullwiki_qrag_fixed.jsonl
      qrag_output=/home/a.anokhin/Judge/full-wiki/runs/fullwiki_qrag_fixed_answer_judge_mt1000.json
      ;;
    4)
      qrag_input=/home/a.anokhin/Judge/full-wiki/runs/fullwiki_qrag_fixed_steps4.jsonl
      qrag_output=/home/a.anokhin/Judge/full-wiki/runs/fullwiki_qrag_fixed_steps4_answer_judge_mt1000.json
      ;;
    6)
      qrag_input=/home/a.anokhin/Judge/full-wiki/runs/fullwiki_qrag_fixed_steps6.jsonl
      qrag_output=/home/a.anokhin/Judge/full-wiki/runs/fullwiki_qrag_fixed_steps6_answer_judge_mt1000.json
      ;;
  esac

  qrag_parts=$(mktemp -d "/tmp/qrag-steps${qrag_steps}-judge.XXXXXX")

  split \
    -n l/32 \
    -d \
    -a 2 \
    --additional-suffix=.jsonl \
    "$qrag_input" \
    "$qrag_parts/input-"

  find "$qrag_parts" -name 'input-*.jsonl' -print0 \
    | sort -z \
    | xargs -0 -P 32 -I '{}' bash -c '
        qrag_part_input=$1
        qrag_part_output=${qrag_part_input%.jsonl}.json

        env -u VLLM_API_KEY \
          /home/a.anokhin/venvs/gpu/bin/python \
          /home/a.anokhin/Judge/full-wiki/src/answer_judge.py \
          --contract v1 \
          --retriever-logfile "$qrag_part_input" \
          --output-file "$qrag_part_output" \
          --base-url http://127.0.0.1:8010/v1 \
          --answer-model Qwen3-4B \
          --max-samples 100000 \
          --max-tokens 1000 \
          --judge-max-tokens 100
      ' _ '{}'

  jq -s 'add' "$qrag_parts"/input-*.json \
    | tee "$qrag_output" >/dev/null

  jq '{
    samples: length,
    mean_em: (map(.EM) | add / length),
    mean_f1: (map(.F1) | add / length),
    mean_judge: (map(.LLM_Judge_Score) | add / length),
    empty_predictions: (map(select((.prediction // "") == "")) | length)
  }' "$qrag_output"
done
```

Ручная команда выше оставлена на контракте v1 намеренно: она воспроизводит
старые раны, а `exp.py` для новых берёт `v2` умолчанием.

Перед повторным запуском нужно выбрать новые output names или явно удалить/
архивировать старые файлы: `answer_judge.py` и `tee` перезаписывают output.

### Определение answer-метрик

Перед EM/F1 prediction и ground truth приводятся к нижнему регистру, из них
удаляются английские артикли `a/an/the`, пунктуация и лишние пробелы.

- **EM** равен 1 при полном совпадении нормализованных строк.
- **F1** — гармоническое среднее precision и recall по multiset-пересечению
  нормализованных токенов.
- **LLM Judge** получает question, prediction и ground truth. Score равен 1,
  только если текст ответа judge после `final_answer` и `.upper()` **строго
  посимвольно** равен `CORRECT`; `normalize_answer` к судье не применяется.

Reader output после `</think>` и последнего маркера `Final Answer:` отделяется
функцией `final_answer`. Thinking при этом явно отключён через
`chat_template_kwargs={"enable_thinking": false}`.

Строгость судейского сравнения проверена: ответ вида `CORRECT.` с точкой был
бы засчитан как неверный. `judge_diagnostics.py` повторяет судейские запросы
на уже сохранённых предсказаниях с теми же промптом и параметрами и
сохраняет сырые ответы:

```bash
cd /home/a.anokhin/Judge/full-wiki

env -u VLLM_API_KEY /home/a.anokhin/venvs/gpu/bin/python src/judge_diagnostics.py \
  --input runs/fullwiki_qrag_fixed_steps6_answer_judge_mt1000.json \
  --output runs/judge_diagnostics_steps6.json \
  --samples 500
```

На выборке из 500 записей: 0 неканоничных ответов, дельта скора при
нормализации ровно 0.0. Метрику менять не нужно, и сравнимость с
опубликованными числами сохраняется.

## 7. Проверка готовых результатов

Пересчёт итоговой таблицы:

```bash
for qrag_result in \
  /home/a.anokhin/Judge/full-wiki/runs/fullwiki_qrag_fixed_answer_judge_mt1000.json \
  /home/a.anokhin/Judge/full-wiki/runs/fullwiki_qrag_fixed_steps4_answer_judge_mt1000.json \
  /home/a.anokhin/Judge/full-wiki/runs/fullwiki_qrag_fixed_steps6_answer_judge_mt1000.json
do
  echo "$qrag_result"
  jq '{
    samples: length,
    mean_em: (map(.EM) | add / length),
    mean_f1: (map(.F1) | add / length),
    mean_judge: (map(.LLM_Judge_Score) | add / length),
    empty_predictions: (map(select((.prediction // "") == "")) | length)
  }' "$qrag_result"
done
```

Во всех восьми канонических файлах:

```text
samples = 7405
empty_predictions = 0
missing metrics = 0
```

Проверка, что порядок ID после 32-way evaluation совпадает с retrieval JSONL:

```bash
diff \
  <(jq -r '.id' /path/to/retrieval.jsonl) \
  <(jq -r '.[].id' /path/to/answer_judge.json)
```

Пустой output и exit code 0 означают полное совпадение.

### Сборка итоговой таблицы

`report_phase0.py` пересчитывает все строки из файлов, включая title-метрики
(из сохранённых `gold_titles` и `retrieved_titles`), поэтому раны, сделанные
до введения нормализации, оцениваются тем же определением, что и новые.

```bash
cd /home/a.anokhin/Judge/full-wiki

/home/a.anokhin/venvs/gpu/bin/python src/report_phase0.py \
  --coverage runs/hotpotqa_dev_title_coverage.json \
  --output runs/phase0_report.json \
  --run "no retrieval=runs/fullwiki_no_retrieval_answer_judge_mt1000.json" \
  --run "GTE top-2=runs/fullwiki_gte_only_steps2_answer_judge_mt1000.json" \
  --run "GTE top-4=runs/fullwiki_gte_only_steps4_answer_judge_mt1000.json" \
  --run "GTE top-6=runs/fullwiki_gte_only_steps6_answer_judge_mt1000.json" \
  --run "Q-RAG 2 шага=runs/fullwiki_qrag_fixed_answer_judge_mt1000.json" \
  --run "Q-RAG 4 шага=runs/fullwiki_qrag_fixed_steps4_answer_judge_mt1000.json" \
  --run "Q-RAG 6 шагов=runs/fullwiki_qrag_fixed_steps6_answer_judge_mt1000.json" \
  --run "oracle (gold sentences)=runs/fullwiki_oracle_sf_answer_judge_mt1000.json" \
  --pair "GTE top-2" "Q-RAG 2 шага" \
  --pair "GTE top-4" "Q-RAG 4 шага" \
  --pair "GTE top-6" "Q-RAG 6 шагов" \
  --pair "no retrieval" "GTE top-6"
```

`--pair` принимает две метки отдельными аргументами, а не через `=`: метка —
человеческий текст и сама может содержать знак равенства («…, N=2»), и тогда
разрез по нему молча даёт несуществующую метку.

`--pair` даёт парный тест МакНемара по EM на одних и тех же вопросах. EM
бинарен, обе конфигурации отвечают на идентичный набор, поэтому всю
информацию несут расхождения; при нуле разница распределена как
Binomial(расхождения, 0.5).

Руками эту команду набирать не нужно: `make report` собирает её сам из
манифестов, по одному вызову на датасет со своим `--coverage`, и вклеивает
результат между маркерами в RESULTS.md.

### Срез рана для чтения

`answer_judge.json` содержит всё, но неудобен: титул статьи спрятан первой
строкой внутри текста чанка, рядом лежат покандидатные дампы каждого хопа, а
алиасов ответа нет вовсе.

```bash
python src/export_eval_json.py --run 2026-08-05-lineA-best-2wiki
```

Кладёт `runs/<run_id>/export.json`: вопрос, голд-ответ с алиасами, ответ
модели, извлечённые чанки по полям `title`/`text`/`score`/`row` и метрики
примера, включая `em_alias` и `f1_alias`. Алиасы берутся сами: у MuSiQue из
поля `answer_aliases` примера, у 2Wiki — через `answer_id` из
`id_aliases.json` по соседству с датасетом, у HotpotQA их нет.

Основные `EM` и `F1` скрипт не пересчитывает, а **сверяет** с
`answer_judge.json` на каждой записи и падает при расхождении: если
нормализация ответа разъедется с судейской, alias-версии станут несравнимы с
колонкой EM в таблице, и лучше узнать об этом до публикации.

### Регрессия

Прогон с дефолтами должен воспроизводить канонический файл:

```bash
cd /home/a.anokhin/Judge/full-wiki

# ... retrieve с --steps 2 и без --reranker/--log-candidates,
#     вывод в runs/regression_qrag_fixed_steps2.jsonl

diff <(jq -c '.pred_idx'       runs/fullwiki_qrag_fixed.jsonl) \
     <(jq -c '.pred_idx'       runs/regression_qrag_fixed_steps2.jsonl)
diff <(jq -c '.title_em'       runs/fullwiki_qrag_fixed.jsonl) \
     <(jq -c '.title_em_exact' runs/regression_qrag_fixed_steps2.jsonl)
diff <(jq -c '.retrieval_hops' runs/fullwiki_qrag_fixed.jsonl) \
     <(jq -c '.retrieval_hops' runs/regression_qrag_fixed_steps2.jsonl)
```

Проверено 28 июля 2026: все три `diff` пустые. Единственное отличие схемы —
добавленные поля `reranker`, `title_recall_exact` и `title_em_exact`.

