# Q-RAG Documentation

Эта папка — сжатая карта проекта, чтобы в следующей сессии не перечитывать весь код.

## Что это за проект

Q-RAG: retrieval-agent для long-context / multi-hop QA. Основная идея: не дообучать большой reader LLM, а обучать легкий value-based retrieval agent в пространстве текстовых embedding-ов. Агент пошагово выбирает chunks из контекста, получает reward за найденные supporting facts или за улучшение ответа reader LLM, после чего выбранные chunks можно передать отдельному LLM для ответа.

Кодовая база — `Q-RAG-feedback/` (эта документация лежит внутри неё, в `docs/`).

С 2026-07-10 `Q-RAG-feedback/` — git-репозиторий (история ветки `Judged`, текущая ветка `feedback-integration`):

- база — upstream-коммит `e41c29a` (2026-03-20), поверх него снапшот локальной работы;
- смержен `upstream/Judged` (фикс TD-λ таргета в PQN, фикс потери последних переходов rollout, удаление legacy-файлов);
- remote `upstream` = `https://github.com/griver/Q-RAG-feedback-project.git` (приватный, для fetch нужен classic PAT со scope `repo`);
- GIG-пайплайн перенесён из корня `~/Judge` в `gig_pipeline/`.

Соседняя копия `Q-RAG/`, упоминавшаяся в старых версиях этих доков, удалена — рабочее дерево одно.

## Документы

Все документы этой папки — дофорковые: они описывают унаследованное ядро
Q-RAG-feedback (Group IG, реранкер над кандидатами из датасета) и не знают
про линию A. Форк и обучение опубликованных башен описаны уровнем выше —
[../README.md](../README.md) и `docs/training.md` лаборатории. Устаревшие
целиком `NEXT_SESSION_BRIEF.md`, `RUN_MODES.md` и `DATA_AND_CONFIGS.md`
удалены; их можно прочитать в ветке `Q-RAG-feedback` или в истории git.

- `CODE_MAP.md` - карта директорий и назначение ключевых файлов.
- `ARCHITECTURE.md` - как связаны agent, env, datasets, feedback и evaluation.
- `IG_GROUP_IG_METHOD.md` - отдельное описание методики IG / Group IG и связанного кода.
- `KNOWN_ISSUES.md` - замеченные шероховатости, stale-файлы и места, где легко ошибиться.
- `FEEDBACK_STABLE_MERGE.md` - инвентарь перенесённой и сопоставленной функциональности.
- `SMOKE_TESTS.md` - короткие ручные training/eval/vLLM smoke-тесты.
- `../gig_pipeline/README.md` - краткая шпаргалка по порядку запуска GIG-скриптов.

## Рекомендуемый порядок чтения

1. `IG_GROUP_IG_METHOD.md`
2. `ARCHITECTURE.md`
3. `KNOWN_ISSUES.md`
