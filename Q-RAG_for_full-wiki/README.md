# Q-RAG для full-wiki

Форк Q-RAG-feedback, в котором state-башня Q-RAG обучена как итеративный
плотный ретривер по Wiki-18 (линия A лаборатории full-wiki). Каталог живёт
внутри ветки `full-wiki_v2`; его собственная история влита subtree-merge.

Этот README описывает форк. Что здесь унаследовано от оригинала — смотрите
ветку `Q-RAG-feedback` этого же репозитория; её старые документы в
[docs/](docs/) оставлены как справка по унаследованному ядру и текущее
состояние форка не описывают.

## Точка форка

Копия сделана 2026-08-01 с клона ветки `Q-RAG-feedback` на коммите
`a41ec9f9c325478da570b3fba301ce5e9a444d7c` (2026-07-29, «Merge branch
'base_qrag_fixes' into Q-RAG-feedback» — состояние `origin/Q-RAG-feedback`
на 30.07.2026). До базового коммита форка (2026-08-02) в копии успели
измениться шесть файлов — `envs/candidate_dataset_adapter.py`,
`envs/qa_dataset_adapter.py`, `envs/utils.py`, `rl/bert_predictor.py`
(CLS-пулинг и L2-нормировка вместо masked mean), `rl/feedback/llm_answer.py`,
`.gitignore` — и добавиться пять: `envs/dataloaders/nq_hotpotqa.py`,
`rl/feedback/occ_status.py`, `vLLM_clients/vllm_occ.py`, `q-rag.def`,
`run_rsync.sh`. Всё дальнейшее читается дифами обычной истории git.

## Как читать историю

`git log -- Q-RAG_for_full-wiki/` покажет один merge-коммит — это свойство
subtree-merge, а не отсутствие истории. Сама история форка (14 коммитов):

```bash
git log dfb959a^2            # вся история форка
git show 3311ec0             # правки rl-ядра: четыре дефекта PQN,
                             # действия без перекодирования текста,
                             # SearchBoltzmannPolicy
```

Мотивировка каждой правки —
в сообщении соответствующего коммита и в [../docs/training.md](../docs/training.md).

## Чем обучены опубликованные башни

Точка входа — `train_q_rag_search.py` (не `train_q_rag.py`: действия даёт
поиск `s @ M.T` по 21 015 324 строкам, а не кандидаты из датасета):

```bash
cd Q-RAG_for_full-wiki
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
python train_q_rag_search.py                              # линия A: configs/training_fullwiki_search.yaml
python train_q_rag_search.py training_fullwiki_rawmix.yaml  # rawmix; руки 4B/8B разводятся env-переменными
```

Записи трёх опубликованных ранов — конфиг-снимок и кривые обучения — лежат
в [runs/](runs/): `Aug03_12-44-38_lineA_main` (его `model_best` дал
37.70 / 30.31 / 10.14 EM), `Aug08_20-39-33_rawmix4b`,
`Aug08_20-39-40_rawmix8b`. Чекпоинты (7.3 ГБ каждый) не публикуются; их
sha256 — в [../docs/reproducibility.md](../docs/reproducibility.md).
Полный контекст обучения — данные, награда, ресурсы, решения —
[../docs/training.md](../docs/training.md).

Ридер и судья награды ходят в vLLM; контракт целиком через переменные
окружения `VLLM_MODEL`, `VLLM_BASE_URL`, `VLLM_API_KEY`. Реальные ключи в
конфиги не записываются.

## Окружение

Фактическое окружение обучения — то же, что у эвала лаборатории
(версии: [../docs/reproducibility.md](../docs/reproducibility.md)). Поверх
семи пакетов лабораторного `pyproject.toml` обучению дополнительно нужны:
`hydra-core` 1.3.2, `omegaconf` 2.3.0, `einops` 0.8.2,
`rotary-embedding-torch` 0.8.9, `tensorboard` 2.20.0, `rouge` 1.0.1,
`openai` 2.21.0 + `httpx` (клиент vLLM), `pyarrow` (rawmix-parquet), `tqdm`.
Файлы `requirements.txt` и `requirements-cu118.txt` в этом каталоге —
дофорковые записи оригинала, текущее окружение они не описывают.

## Тесты

```bash
cd Q-RAG_for_full-wiki
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python -m pytest tests/ -q
```

Ожидается `75 passed, 2 skipped` без GPU и сети: два пропуска — офлайн-регрессии
награды, которым нужны сохранённые раны обучения целиком. В свежем клоне без
rawmix-смеси (`train_data/raw` в git не хранится) пропусков семь — пять
rawmix-тестов подсказывают команду пересборки `src/build_train_mix.py`.
Из сбора тестов лаборатории каталог исключён (`norecursedirs` в корневом
`pyproject.toml`), тесты гоняются отсюда отдельно.
