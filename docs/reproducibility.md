# Воспроизводимость

Что именно зафиксировано на входе эксперимента: корпус, датасет, чекпоинт,
encoder, окружение и хеши артефактов.

## Зафиксированные входные данные и модели

### Корпус Wiki-18

```text
path:   /home/a.anokhin/Judge/datasets/data_sources/full-wiki/data00/
        jiajie_jin/flashrag_indexes/wiki_dpr_100w/wiki_dump.jsonl
rows:   21 015 324
size:   14 393 573 105 bytes
sha256: 43d7d3f58d01d711d95b00b70584211eea639fa46802905a4b7e11cf0617752d
```

Каждая JSONL-строка должна содержать поля `id` и `contents`. В данном корпусе
`int(item["id"])` совпадает с zero-based номером строки. Это соответствие
критично: FAISS ID, GTE shard ID, Q-RAG action shard ID и row ID корпуса должны
обозначать один и тот же чанк.

### Evaluation dataset

```text
path:   /home/a.anokhin/Judge/datasets/data_sources/hotpotqa/
        hotpot_dev_fullwiki_v1.json
items:  7 405
size:   47 454 698 bytes
sha256: 2f1f3e594a3066a3084cc57950ca2713c24712adaad03af6ccce18d1846d5618
```

### Остальные датасеты эвала

Раны на них появились после первой версии этого документа. Хеши берутся из
`inputs.dataset.sha256` манифестов и совпадают у всех ранов соответствующего
датасета. Пути — `runs/shared/` для производных наборов, остальное во внешнем
каталоге датасетов.

| Датасет | Файл | items | SHA-256 |
|---|---|---:|---|
| 2WikiMultiHopQA dev | `dev.json` | 12 576 | `79f77ae104088ea8e25b1a65dbece768d45771194663bc5660ec9a98070dadf5` |
| MuSiQue-Ans dev | `runs/shared/musique_ans_dev_hotpot_schema.jsonl` | 2 417 | `a2996d330e2a82274282ad39d3664977d80b75f2d1d9db6b9d0584495ba7d5d0` |
| NQ (Search-R1 test) | `runs/shared/searchr1/nq.jsonl` | 3 610 | `655f62f00b64f0ec49d734a5f60c146aaf1968f420e9f2c7c6701e5cca4e903a` |
| TriviaQA (Search-R1 test) | `runs/shared/searchr1/triviaqa.jsonl` | 11 313 | `7dc518b458bbcc798bbcc35895080cd2cb8e9d176bfe0120e3169a0e4fe5a41e` |
| PopQA (Search-R1 test) | `runs/shared/searchr1/popqa.jsonl` | 14 267 | `4eda18303a72c77a5b2f69d4f2f792713eb355969f14a8a58c37695b808ca36f` |
| HotpotQA (Search-R1 test) | `runs/shared/searchr1/hotpotqa.jsonl` | 7 405 | `ea8c6f33107557e49b0f15fe88bf9efff0e12ae116ffb704f444317c2281ccd8` |
| 2Wiki (Search-R1 test) | `runs/shared/searchr1/2wikimultihopqa.jsonl` | 12 576 | `728afd8ef846b2c39211bea4d29bfb0a5c00dad8b8a33d88fb597a5a4e44d2cc` |
| MuSiQue (Search-R1 test) | `runs/shared/searchr1/musique.jsonl` | 2 417 | `c3565d54d3c86adb5f54aec420db14998fb65297266793f1edb11954faaa851b` |
| Bamboogle (Search-R1 test) | `runs/shared/searchr1/bamboogle.jsonl` | 125 | `a5542f8aa57a77d567105a6a7ec92a226dfa097c1ed91cb19457bfdef64661fa` |

Семь нижних нарезаны из одного `test.parquet` Search-R1
(`src/build_searchr1_eval.py`), его собственный sha256 проверяется на месте —
см. `SEARCHR1_TEST_SHA256` в `src/build_train_mix.py`. Вопросы там
нормализованы, голд взят из `golden_answers`, gold-титулов нет, поэтому
`coverage: None` и главная метрика — `em_alias`.

### Q-RAG checkpoints

**Реранкинг над top-100 (июльская линия).** Из read-only оригинала:

```text
run:        Jul21_19-44-36_QRAG_HotPotQA+2WikiMultihopQA
checkpoint: model_best.pt
section:    critic
state:      state_embed.*
action:     action_embed.*
sha256:     865fc68fb9aa023224aac122257cc0671dcf0cedd343683cdcf40f8c00b55fa3
path:       /home/a.anokhin/Judge/Q-RAG-feedback/runs/
            Jul21_19-44-36_QRAG_HotPotQA+2WikiMultihopQA/model_best.pt
```

**Прямой поиск state-башней (линия A и её ветки).** Именно на них получены
итоговые числа `RESULTS.md`; каждый файл — 7 333 936 023 байта,
`state_source: critic`. В ветке от каждого рана лежат конфиг-снимок и
кривые обучения (`Q-RAG_for_full-wiki/runs/<ран>/`), сами чекпоинты — только
на машине лаборатории, в каталоге прежней копии кода
(`/home/a.anokhin/Judge/full-wiki/Q-RAG_for_full-wiki/runs/<ран>/`):

| Обучающий ран | Чекпоинт | SHA-256 |
|---|---|---|
| `Aug03_12-44-38_lineA_main` | `model_best.pt` | `c8b2212d4b1e04449623a35acabe2ea8abc06d09acc22b6a87e29a2ea5c9db36` |
| `Aug03_12-44-38_lineA_main` | `model_last.pt` | `a7f424ea65b63ee13bca0e8a1348a6197ef9c00e0a8ee0ceb7435978f87826a8` |
| `Aug08_20-39-33_rawmix4b` | `model_best.pt` | `29c9097608e7721b8f3b12ec7ad7b7975ec537decefbff3055cc0a3102e84c43` |
| `Aug08_20-39-33_rawmix4b` | `model_last.pt` | `8c78b181dc5b9899c0026264e0e2c57c500f9a7d1afdd152057c984c1bb7dbb7` |
| `Aug08_20-39-40_rawmix8b` | `model_best.pt` | `fd461e266898b76ba88686f160baa0b5abbb6205f431c7d3a7e8a21c9b3c4639` |
| `Aug08_20-39-40_rawmix8b` | `model_last.pt` | `760f556a1c4f4e3d5175def7a8683e9dd32cb8d07f45190ec021372dbb9cd18e` |

Опубликованные 37.70 / 30.31 / 10.14 EM получены первой строкой —
`Aug03_12-44-38_lineA_main/model_best.pt`. `lineA_main` обучался на
HotpotQA + 2Wiki, ветка `rawmix` — на сырой смеси NQ + HotpotQA двумя руками
(4B и 8B считали награду своей моделью, поэтому их числа читаются только
внутри своей руки). Какой чекпоинт стоит за конкретной строкой таблицы, всегда
видно в `config.retrieve.checkpoint` манифеста этого рана. Обучающая смесь и holdout — `train_data/`,
собирается `src/build_train_mix.py`; решения по обучению —
[training.md](training.md).

### Базовый encoder

```text
model:              Alibaba-NLP/gte-multilingual-base
requested revision: main
resolved revision:  9bbca17d9273fd0d03d5725c7a4b0f6b45142062
dimension:          768
max sequence length: 256
```

Для новых сборок следует передавать resolved commit SHA, а не изменяемый
`main`. `trust_remote_code=True` допустим только для этого проверенного и
зафиксированного snapshot; retrieval загружает его с `local_files_only=True`.

## Требования к ресурсам

Фактическая среда сборки:

```text
Python:                3.11.14
PyTorch:               2.10.0
NumPy:                 2.2.6
pandas:                3.0.1
FAISS:                 1.14.3
sentence-transformers: 5.2.3
transformers:          4.57.6
GPU:                   NVIDIA H200, 143 771 MiB
Python environment:    /home/a.anokhin/venvs/gpu
```

Каждый полный float32 embedding cache и каждый FlatIP-индекс занимают примерно
60.1 GiB. Для двух embedding cache и двух индексов требуется около 241 GiB,
не считая корпуса, action encoder и offsets. Backend `torch-cuda` держит полную
GTE-матрицу `21 015 324 × 768 × float32` на GPU, то есть примерно 60.1 GiB.

CPU-вариант `--candidate-backend faiss-cpu` поддерживается, но exact search на
установленном generic `faiss-cpu` существенно медленнее. Итоговые эксперименты
получены с `torch-cuda`.

## Хеши артефактов

| Объект | SHA-256 / revision |
|---|---|
| Wiki-18 corpus | `43d7d3f58d01d711d95b00b70584211eea639fa46802905a4b7e11cf0617752d` |
| HotpotQA FullWiki dev | `2f1f3e594a3066a3084cc57950ca2713c24712adaad03af6ccce18d1846d5618` |
| HotpotQA distractor dev (oracle-контекст) | `4e9ecb5c8d3b719f624d66b60f8d56bf227f03914f5f0753d6fa1b359d7104ea` |
| GTE revision | `9bbca17d9273fd0d03d5725c7a4b0f6b45142062` |
| Q-RAG checkpoint (Jul21, реранкинг) | `865fc68fb9aa023224aac122257cc0671dcf0cedd343683cdcf40f8c00b55fa3` |
| Extracted action encoder | `459169e68999ed97a46d901b49c621d23d942c671ac09a1f54780b867010d286` |
| Qwen3-4B snapshot | `1cfa9a7208912126459214e8b04321603b3df60c` |
| Трейн линии A `hotpotqa_2wiki_candidate_train_q0_s1.jsonl` | `831b9007161394fdafd11e3ffdff50dc662f528d47c581f3583d24908c5e7311` |

### Скрипты пайплайна

Хеши **на 2026-08-12**. Список — ровно `runlib.PIPELINE_SCRIPTS`: те файлы,
чьи хеши `exp.py` пишет в `code.scripts` каждого манифеста. Правка любого из
них, даже в докстринге, меняет хеш; расхождение этой таблицы с манифестом рана
означает, что ран снят другой версией, и это ожидаемое поведение, а не сбой.

| Скрипт | SHA-256 |
|---|---|
| `exp.py` | `50e8d506fc13c2a1761bc4025b0c79411717024b7e46c7dc51b32126d289a427` |
| `src/fullwiki_qrag.py` | `6aa59f0e1face211fc9a65e55511dae8818253cf985fab33f6c0c22c34c44b0e` |
| `src/answer_judge.py` | `88aa0cdda9d6aec6cfdc4493aa03c21e2e1906a90605fa576b98bb47a880cb36` |
| `src/build_eval_variants.py` | `1109f9479ca25cfd0fc1aab39dc951890963ef3f8988df9982914e624267bb26` |
| `src/build_index_wiki_gte.py` | `7d5df70435ecc7cbca923501f3727e43b66e62e1248203120a9d21138d5188c6` |
| `src/build_index_wiki_qrag.py` | `d26d29176aa9a582c6ed071aaac4289da433a2bdc804e9fc9b2717552663ea0d` |
| `src/candidate_pool.py` | `bcb1f24c8b00326865a9e424aa330a26de9c795a6bfb95c8b62fda0e87f5563f` |
| `src/report_phase0.py` | `0ac77189416aca8c842a349e7e576378c1bad0721873869dc9bffacfe3da68b4` |
| `answer_judge_llms.py` (read-only оригинал) | `cb1d6ee785b370a85818a03d160864606f49953c83a76b7a03bc48bb94b03ffd` |

`corpus_title_coverage.py` и `judge_diagnostics.py` из этой таблицы убраны
намеренно: они считают диагностику рядом с ранами, но в `code.scripts` не
попадают, и держать их хеши здесь значило обещать сверку, которой нет.

Три канонических Q-RAG-рана получены версией `fullwiki_qrag.py` с хешем
`c9c5db13ce9b70b463daf5913ba3aa0ab2ee78996acc5a3b2d1851e71f2b5418`. Текущая
версия добавляет `--reranker`, `--log-candidates` и нормализацию титулов;
регрессия из [pipeline.md](pipeline.md) §7 подтверждает, что с дефолтными флагами она
воспроизводит те раны по `pred_idx`, `retrieval_hops` и `title_*_exact`.
`answer_judge_llms.py` не менялся: с 2026-08-08 `exp.py` ходит через форк
`answer_judge.py`, а оригинал остаётся read-only эталоном регрессии. Форк на
`--contract v1` воспроизводит его посимвольно — сверка запросов ридера и
судьи на 50 записях реального ретривала в `test_answer_judge.py`.

Machine-readable manifests и build-state являются главным источником истины
для encoder/index параметров. Документация намеренно дублирует ключевые значения,
чтобы эксперимент можно было понять и запустить без ручного разбора JSON.
