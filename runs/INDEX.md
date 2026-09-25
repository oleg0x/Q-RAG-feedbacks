# Реестр ранов

Генерируется `python exp.py index`. Руками не править.

| Ран | Датасет | Метка | Реранкер | Шагов | Статус | EM | F1 | Judge | title EM |
|---|---|---|---|---:|---|---:|---:|---:|---:|
| [`2026-07-28-no-retrieval`](2026-07-28-no-retrieval/manifest.json) | hotpotqa_dev_fullwiki | no retrieval | none | — | ok (backfill) | 15.53 | 23.46 | 23.35 | 0.00 |
| [`2026-08-07-sr1-noctx-2wiki`](runs_history/2026-08-07-sr1-noctx-2wiki/manifest.json) | sr1_2wiki | no retrieval | none | — | ok | 21.16 | 25.93 | 33.26 | — |
| [`2026-08-07-sr1-noctx-bamboogle`](runs_history/2026-08-07-sr1-noctx-bamboogle/manifest.json) | sr1_bamboogle | no retrieval | none | — | ok | 8.80 | 15.00 | 11.20 | — |
| [`2026-08-07-sr1-noctx-hotpotqa`](runs_history/2026-08-07-sr1-noctx-hotpotqa/manifest.json) | sr1_hotpotqa | no retrieval | none | — | ok | 15.52 | 23.48 | 23.28 | — |
| [`2026-08-07-sr1-noctx-musique`](runs_history/2026-08-07-sr1-noctx-musique/manifest.json) | sr1_musique | no retrieval | none | — | ok | 1.74 | 9.70 | 5.05 | — |
| [`2026-08-07-sr1-noctx-nq`](runs_history/2026-08-07-sr1-noctx-nq/manifest.json) | sr1_nq | no retrieval | none | — | ok | 9.25 | 16.45 | 18.61 | — |
| [`2026-08-07-sr1-noctx-popqa`](runs_history/2026-08-07-sr1-noctx-popqa/manifest.json) | sr1_popqa | no retrieval | none | — | ok | 9.27 | 13.91 | 14.19 | — |
| [`2026-08-07-sr1-noctx-triviaqa`](runs_history/2026-08-07-sr1-noctx-triviaqa/manifest.json) | sr1_triviaqa | no retrieval | none | — | ok | 6.75 | 18.50 | 25.02 | — |
| [`2026-07-28-gte-only-steps2`](runs_history/2026-07-28-gte-only-steps2/manifest.json) | hotpotqa_dev_fullwiki | GTE top-2 | truncate | 2 | ok (backfill) | 25.86 | 35.09 | 36.12 | 10.10 |
| [`2026-07-28-gte-only-steps4`](runs_history/2026-07-28-gte-only-steps4/manifest.json) | hotpotqa_dev_fullwiki | GTE top-4 | truncate | 4 | ok (backfill) | 27.89 | 37.04 | 38.06 | 19.05 |
| [`2026-07-28-gte-only-steps6`](runs_history/2026-07-28-gte-only-steps6/manifest.json) | hotpotqa_dev_fullwiki | GTE top-6 | none | 6 | ok (backfill) | 28.74 | 38.05 | 39.20 | 23.57 |
| [`2026-08-02-gte-refresh-q1024-steps2`](runs_history/2026-08-02-gte-refresh-q1024-steps2/manifest.json) | hotpotqa_dev_fullwiki | GTE refresh 1024, 2 шага | none | 2 | ok | 25.12 | 33.84 | 35.22 | 8.87 |
| [`2026-08-02-gte-refresh-q1024-steps4`](runs_history/2026-08-02-gte-refresh-q1024-steps4/manifest.json) | hotpotqa_dev_fullwiki | GTE refresh 1024, 4 шага | none | 4 | ok | 25.10 | 33.80 | 35.27 | 15.92 |
| [`2026-08-02-gte-refresh-q1024-steps6`](runs_history/2026-08-02-gte-refresh-q1024-steps6/manifest.json) | hotpotqa_dev_fullwiki | GTE refresh 1024, 6 шагов | none | 6 | ok | 26.21 | 35.19 | 36.77 | 19.54 |
| [`2026-08-02-gte-refresh-q256-steps6`](runs_history/2026-08-02-gte-refresh-q256-steps6/manifest.json) | hotpotqa_dev_fullwiki | GTE refresh 256, 6 шагов | none | 6 | ok | 26.43 | 35.26 | 36.84 | 20.26 |
| [`2026-08-02-gte-only-n2-steps4`](runs_history/2026-08-02-gte-only-n2-steps4/manifest.json) | hotpotqa_dev_fullwiki | GTE top-4, 2 чанка на титул | truncate | 4 | ok | 27.71 | 37.04 | 38.37 | 14.18 |
| [`2026-08-02-gte-only-n2-steps6`](2026-08-02-gte-only-n2-steps6/manifest.json) | hotpotqa_dev_fullwiki | GTE top-6, 2 чанка на титул | none | 6 | ok | 29.29 | 39.03 | 40.09 | 19.42 |
| [`2026-08-02-search-zeroshot-steps6`](2026-08-02-search-zeroshot-steps6/manifest.json) | hotpotqa_dev_fullwiki | Прямой поиск, zero-shot GTE, 6 шагов, N=2 | — | 6 | ok | 27.16 | 36.15 | 37.45 | 15.22 |
| [`2026-08-05-search-zeroshot-2wiki`](2026-08-05-search-zeroshot-2wiki/manifest.json) | 2wiki_dev | Прямой поиск, zero-shot GTE, 6 шагов, N=2 | — | 6 | ok | 15.83 | 20.53 | 25.93 | 13.61 |
| [`2026-08-05-search-zeroshot-musique`](2026-08-05-search-zeroshot-musique/manifest.json) | musique_ans_dev | Прямой поиск, zero-shot GTE, 6 шагов, N=2 | — | 6 | ok | 4.96 | 11.27 | 9.10 | 3.85 |
| [`2026-08-07-sr1-zeroshot-2wiki`](runs_history/2026-08-07-sr1-zeroshot-2wiki/manifest.json) | sr1_2wiki | Прямой поиск, zero-shot GTE, 6 шагов, N=2 | — | 6 | ok | 13.84 | 18.64 | 25.40 | — |
| [`2026-08-07-sr1-zeroshot-bamboogle`](runs_history/2026-08-07-sr1-zeroshot-bamboogle/manifest.json) | sr1_bamboogle | Прямой поиск, zero-shot GTE, 6 шагов, N=2 | — | 6 | ok | 19.20 | 29.08 | 26.40 | — |
| [`2026-08-07-sr1-zeroshot-hotpotqa`](runs_history/2026-08-07-sr1-zeroshot-hotpotqa/manifest.json) | sr1_hotpotqa | Прямой поиск, zero-shot GTE, 6 шагов, N=2 | — | 6 | ok | 27.05 | 36.08 | 37.69 | — |
| [`2026-08-07-sr1-zeroshot-musique`](runs_history/2026-08-07-sr1-zeroshot-musique/manifest.json) | sr1_musique | Прямой поиск, zero-shot GTE, 6 шагов, N=2 | — | 6 | ok | 4.96 | 11.20 | 9.02 | — |
| [`2026-08-07-sr1-zeroshot-nq`](runs_history/2026-08-07-sr1-zeroshot-nq/manifest.json) | sr1_nq | Прямой поиск, zero-shot GTE, 6 шагов, N=2 | — | 6 | ok | 27.37 | 37.91 | 38.39 | — |
| [`2026-08-07-sr1-zeroshot-popqa`](runs_history/2026-08-07-sr1-zeroshot-popqa/manifest.json) | sr1_popqa | Прямой поиск, zero-shot GTE, 6 шагов, N=2 | — | 6 | ok | 32.05 | 39.58 | 38.31 | — |
| [`2026-08-07-sr1-zeroshot-triviaqa`](runs_history/2026-08-07-sr1-zeroshot-triviaqa/manifest.json) | sr1_triviaqa | Прямой поиск, zero-shot GTE, 6 шагов, N=2 | — | 6 | ok | 14.38 | 32.14 | 40.95 | — |
| [`2026-08-08-search-zeroshot-2wiki-n1`](runs_history/2026-08-08-search-zeroshot-2wiki-n1/manifest.json) | 2wiki_dev | Прямой поиск, zero-shot GTE, 6 шагов, N=1 | — | 6 | ok | 16.13 | 20.60 | 26.16 | 16.21 |
| [`2026-08-08-search-zeroshot-hotpotqa-n1`](runs_history/2026-08-08-search-zeroshot-hotpotqa-n1/manifest.json) | hotpotqa_dev_fullwiki | Прямой поиск, zero-shot GTE, 6 шагов, N=1 | — | 6 | ok | 25.78 | 34.49 | 36.21 | 19.00 |
| [`2026-08-08-search-zeroshot-musique-n1`](runs_history/2026-08-08-search-zeroshot-musique-n1/manifest.json) | musique_ans_dev | Прямой поиск, zero-shot GTE, 6 шагов, N=1 | — | 6 | ok | 4.96 | 11.66 | 9.23 | 4.55 |
| [`2026-08-08-search-zeroshot-2wiki-noquota`](runs_history/2026-08-08-search-zeroshot-2wiki-noquota/manifest.json) | 2wiki_dev | Прямой поиск, zero-shot GTE, 6 шагов, без квоты | — | 6 | ok | 15.36 | 20.02 | 25.17 | 10.72 |
| [`2026-08-08-search-zeroshot-hotpotqa-noquota`](runs_history/2026-08-08-search-zeroshot-hotpotqa-noquota/manifest.json) | hotpotqa_dev_fullwiki | Прямой поиск, zero-shot GTE, 6 шагов, без квоты | — | 6 | ok | 26.67 | 35.62 | 37.12 | 10.84 |
| [`2026-08-08-search-zeroshot-musique-noquota`](runs_history/2026-08-08-search-zeroshot-musique-noquota/manifest.json) | musique_ans_dev | Прямой поиск, zero-shot GTE, 6 шагов, без квоты | — | 6 | ok | 4.55 | 10.66 | 8.52 | 2.73 |
| [`2026-08-05-lineA-best-2wiki`](2026-08-05-lineA-best-2wiki/manifest.json) | 2wiki_dev | Прямой поиск, обученная башня, 6 шагов, N=2 | — | 6 | ok | 30.31 | 37.24 | 41.38 | 30.51 |
| [`2026-08-05-lineA-best-hotpotqa`](2026-08-05-lineA-best-hotpotqa/manifest.json) | hotpotqa_dev_fullwiki | Прямой поиск, обученная башня, 6 шагов, N=2 | — | 6 | ok | 37.70 | 48.87 | 50.22 | 45.48 |
| [`2026-08-05-lineA-best-musique`](2026-08-05-lineA-best-musique/manifest.json) | musique_ans_dev | Прямой поиск, обученная башня, 6 шагов, N=2 | — | 6 | ok | 10.14 | 18.00 | 15.06 | 11.58 |
| [`2026-08-07-sr1-best-2wiki`](runs_history/2026-08-07-sr1-best-2wiki/manifest.json) | sr1_2wiki | Прямой поиск, обученная башня, 6 шагов, N=2 | — | 6 | ok | 25.95 | 33.31 | 38.71 | — |
| [`2026-08-07-sr1-best-bamboogle`](runs_history/2026-08-07-sr1-best-bamboogle/manifest.json) | sr1_bamboogle | Прямой поиск, обученная башня, 6 шагов, N=2 | — | 6 | ok | 28.80 | 38.69 | 36.80 | — |
| [`2026-08-07-sr1-best-hotpotqa`](runs_history/2026-08-07-sr1-best-hotpotqa/manifest.json) | sr1_hotpotqa | Прямой поиск, обученная башня, 6 шагов, N=2 | — | 6 | ok | 37.74 | 48.87 | 50.28 | — |
| [`2026-08-07-sr1-best-musique`](runs_history/2026-08-07-sr1-best-musique/manifest.json) | sr1_musique | Прямой поиск, обученная башня, 6 шагов, N=2 | — | 6 | ok | 10.05 | 17.89 | 15.06 | — |
| [`2026-08-07-sr1-best-nq`](runs_history/2026-08-07-sr1-best-nq/manifest.json) | sr1_nq | Прямой поиск, обученная башня, 6 шагов, N=2 | — | 6 | ok | 26.23 | 37.19 | 38.84 | — |
| [`2026-08-07-sr1-best-popqa`](runs_history/2026-08-07-sr1-best-popqa/manifest.json) | sr1_popqa | Прямой поиск, обученная башня, 6 шагов, N=2 | — | 6 | ok | 33.62 | 40.96 | 40.04 | — |
| [`2026-08-07-sr1-best-triviaqa`](runs_history/2026-08-07-sr1-best-triviaqa/manifest.json) | sr1_triviaqa | Прямой поиск, обученная башня, 6 шагов, N=2 | — | 6 | ok | 14.58 | 32.36 | 41.18 | — |
| [`2026-08-08-lineA-best-2wiki-n1`](runs_history/2026-08-08-lineA-best-2wiki-n1/manifest.json) | 2wiki_dev | Прямой поиск, обученная башня, 6 шагов, N=1 | — | 6 | ok | 29.34 | 35.92 | 39.96 | 29.92 |
| [`2026-08-08-lineA-best-hotpotqa-n1`](runs_history/2026-08-08-lineA-best-hotpotqa-n1/manifest.json) | hotpotqa_dev_fullwiki | Прямой поиск, обученная башня, 6 шагов, N=1 | — | 6 | ok | 36.19 | 47.10 | 48.41 | 45.94 |
| [`2026-08-08-lineA-best-musique-n1`](runs_history/2026-08-08-lineA-best-musique-n1/manifest.json) | musique_ans_dev | Прямой поиск, обученная башня, 6 шагов, N=1 | — | 6 | ok | 9.81 | 17.47 | 14.48 | 12.78 |
| [`2026-08-08-lineA-best-2wiki-noquota`](runs_history/2026-08-08-lineA-best-2wiki-noquota/manifest.json) | 2wiki_dev | Прямой поиск, обученная башня, 6 шагов, без квоты | — | 6 | ok | 30.57 | 37.54 | 41.58 | 30.40 |
| [`2026-08-08-lineA-best-hotpotqa-noquota`](runs_history/2026-08-08-lineA-best-hotpotqa-noquota/manifest.json) | hotpotqa_dev_fullwiki | Прямой поиск, обученная башня, 6 шагов, без квоты | — | 6 | ok | 37.91 | 49.21 | 50.41 | 45.10 |
| [`2026-08-08-lineA-best-musique-noquota`](runs_history/2026-08-08-lineA-best-musique-noquota/manifest.json) | musique_ans_dev | Прямой поиск, обученная башня, 6 шагов, без квоты | — | 6 | ok | 10.30 | 18.29 | 15.64 | 11.46 |
| [`2026-08-06-lineA-last-2wiki`](runs_history/2026-08-06-lineA-last-2wiki/manifest.json) | 2wiki_dev | Прямой поиск, последний чекпоинт, 6 шагов, N=2 | — | 6 | ok | 31.45 | 38.72 | 42.84 | 33.17 |
| [`2026-08-06-lineA-last-hotpotqa`](runs_history/2026-08-06-lineA-last-hotpotqa/manifest.json) | hotpotqa_dev_fullwiki | Прямой поиск, последний чекпоинт, 6 шагов, N=2 | — | 6 | ok | 37.58 | 49.06 | 50.44 | 44.21 |
| [`2026-08-06-lineA-last-musique`](runs_history/2026-08-06-lineA-last-musique/manifest.json) | musique_ans_dev | Прямой поиск, последний чекпоинт, 6 шагов, N=2 | — | 6 | ok | 10.59 | 17.93 | 15.35 | 11.92 |
| [`2026-07-28-qrag-fixed-steps2`](runs_history/2026-07-28-qrag-fixed-steps2/manifest.json) | hotpotqa_dev_fullwiki | Q-RAG, 2 шага | qrag | 2 | ok (backfill) | 22.89 | 30.87 | 31.92 | 14.98 |
| [`2026-07-28-qrag-fixed-steps4`](runs_history/2026-07-28-qrag-fixed-steps4/manifest.json) | hotpotqa_dev_fullwiki | Q-RAG, 4 шага | qrag | 4 | ok (backfill) | 24.94 | 33.25 | 33.98 | 24.15 |
| [`2026-07-28-qrag-fixed-steps6`](runs_history/2026-07-28-qrag-fixed-steps6/manifest.json) | hotpotqa_dev_fullwiki | Q-RAG, 6 шагов | qrag | 6 | ok (backfill) | 26.36 | 34.96 | 35.99 | 26.70 |
| [`2026-08-02-pool-oracle-titles-k2`](runs_history/2026-08-02-pool-oracle-titles-k2/manifest.json) | hotpotqa_dev_fullwiki | oracle по титулам, 2 чанка | — | 2 | ok | 34.02 | 44.56 | 46.24 | 43.79 |
| [`2026-08-02-pool-oracle-chunks-k2`](runs_history/2026-08-02-pool-oracle-chunks-k2/manifest.json) | hotpotqa_dev_fullwiki | oracle по чанкам, 2 чанка | — | 2 | ok | 35.80 | 46.74 | 48.39 | 43.79 |
| [`2026-08-02-pool-oracle-titles-k4`](runs_history/2026-08-02-pool-oracle-titles-k4/manifest.json) | hotpotqa_dev_fullwiki | oracle по титулам, 4 чанка | — | 4 | ok | 33.29 | 43.61 | 45.16 | 43.79 |
| [`2026-08-02-pool-oracle-chunks-k4`](runs_history/2026-08-02-pool-oracle-chunks-k4/manifest.json) | hotpotqa_dev_fullwiki | oracle по чанкам, 4 чанка | — | 4 | ok | 35.00 | 45.73 | 47.43 | 43.79 |
| [`2026-08-02-pool-oracle-titles-k6`](runs_history/2026-08-02-pool-oracle-titles-k6/manifest.json) | hotpotqa_dev_fullwiki | oracle по титулам, 6 чанков | — | 6 | ok | 33.21 | 43.55 | 44.83 | 43.79 |
| [`2026-08-02-pool-oracle-chunks-k6`](runs_history/2026-08-02-pool-oracle-chunks-k6/manifest.json) | hotpotqa_dev_fullwiki | oracle по чанкам, 6 чанков | — | 6 | ok | 34.71 | 45.46 | 47.06 | 43.79 |
| [`2026-08-02-pool-oracle-chunks-n2-k2`](runs_history/2026-08-02-pool-oracle-chunks-n2-k2/manifest.json) | hotpotqa_dev_fullwiki | oracle по чанкам, 2 чанка, 2 на титул | — | 2 | ok | 36.15 | 47.07 | 48.83 | 43.79 |
| [`2026-08-02-pool-oracle-chunks-n2-k4`](runs_history/2026-08-02-pool-oracle-chunks-n2-k4/manifest.json) | hotpotqa_dev_fullwiki | oracle по чанкам, 4 чанка, 2 на титул | — | 4 | ok | 35.56 | 46.39 | 48.01 | 43.79 |
| [`2026-08-02-pool-oracle-chunks-n2-k6`](runs_history/2026-08-02-pool-oracle-chunks-n2-k6/manifest.json) | hotpotqa_dev_fullwiki | oracle по чанкам, 6 чанков, 2 на титул | — | 6 | ok | 35.65 | 46.48 | 48.01 | 43.79 |
| [`2026-07-28-oracle-sf`](runs_history/2026-07-28-oracle-sf/manifest.json) | hotpotqa_dev_fullwiki | oracle (gold sentences) | oracle | — | ok (backfill) | 63.70 | 78.86 | 81.72 | 100.00 |
| [`2026-08-10-sr1-rawmix4b-best-nq`](runs_history/2026-08-10-sr1-rawmix4b-best-nq/manifest.json) | sr1_nq | Сырая смесь NQ+HotpotQA, Qwen3-4B, пик (шаг ~19 000) | — | 6 | ok | 25.73 | 36.24 | 51.58 | — |
| [`2026-08-10-sr1-rawmix4b-best-triviaqa`](runs_history/2026-08-10-sr1-rawmix4b-best-triviaqa/manifest.json) | sr1_triviaqa | Сырая смесь NQ+HotpotQA, Qwen3-4B, пик (шаг ~19 000) | — | 6 | ok | 14.41 | 32.01 | 71.23 | — |
| [`2026-08-10-sr1-rawmix4b-best-popqa`](runs_history/2026-08-10-sr1-rawmix4b-best-popqa/manifest.json) | sr1_popqa | Сырая смесь NQ+HotpotQA, Qwen3-4B, пик (шаг ~19 000) | — | 6 | ok | 33.05 | 40.01 | 47.52 | — |
| [`2026-08-10-sr1-rawmix4b-best-hotpotqa`](runs_history/2026-08-10-sr1-rawmix4b-best-hotpotqa/manifest.json) | sr1_hotpotqa | Сырая смесь NQ+HotpotQA, Qwen3-4B, пик (шаг ~19 000) | — | 6 | ok | 36.41 | 47.35 | 49.22 | — |
| [`2026-08-10-sr1-rawmix4b-best-2wiki`](runs_history/2026-08-10-sr1-rawmix4b-best-2wiki/manifest.json) | sr1_2wiki | Сырая смесь NQ+HotpotQA, Qwen3-4B, пик (шаг ~19 000) | — | 6 | ok | 25.49 | 32.38 | 42.62 | — |
| [`2026-08-10-sr1-rawmix4b-best-musique`](runs_history/2026-08-10-sr1-rawmix4b-best-musique/manifest.json) | sr1_musique | Сырая смесь NQ+HotpotQA, Qwen3-4B, пик (шаг ~19 000) | — | 6 | ok | 9.97 | 17.78 | 18.33 | — |
| [`2026-08-10-sr1-rawmix4b-best-bamboogle`](runs_history/2026-08-10-sr1-rawmix4b-best-bamboogle/manifest.json) | sr1_bamboogle | Сырая смесь NQ+HotpotQA, Qwen3-4B, пик (шаг ~19 000) | — | 6 | ok | 24.80 | 36.37 | 33.60 | — |
| [`2026-08-10-sr1-rawmix4b-last-nq`](runs_history/2026-08-10-sr1-rawmix4b-last-nq/manifest.json) | sr1_nq | Сырая смесь NQ+HotpotQA, Qwen3-4B, последний (шаг 30 000) | — | 6 | ok | 24.71 | 34.46 | 47.98 | — |
| [`2026-08-10-sr1-rawmix4b-last-triviaqa`](runs_history/2026-08-10-sr1-rawmix4b-last-triviaqa/manifest.json) | sr1_triviaqa | Сырая смесь NQ+HotpotQA, Qwen3-4B, последний (шаг 30 000) | — | 6 | ok | 14.18 | 31.49 | 70.52 | — |
| [`2026-08-10-sr1-rawmix4b-last-popqa`](runs_history/2026-08-10-sr1-rawmix4b-last-popqa/manifest.json) | sr1_popqa | Сырая смесь NQ+HotpotQA, Qwen3-4B, последний (шаг 30 000) | — | 6 | ok | 33.04 | 39.89 | 47.16 | — |
| [`2026-08-10-sr1-rawmix4b-last-hotpotqa`](runs_history/2026-08-10-sr1-rawmix4b-last-hotpotqa/manifest.json) | sr1_hotpotqa | Сырая смесь NQ+HotpotQA, Qwen3-4B, последний (шаг 30 000) | — | 6 | ok | 36.12 | 47.23 | 48.59 | — |
| [`2026-08-10-sr1-rawmix4b-last-2wiki`](runs_history/2026-08-10-sr1-rawmix4b-last-2wiki/manifest.json) | sr1_2wiki | Сырая смесь NQ+HotpotQA, Qwen3-4B, последний (шаг 30 000) | — | 6 | ok | 23.89 | 30.88 | 41.14 | — |
| [`2026-08-10-sr1-rawmix4b-last-musique`](runs_history/2026-08-10-sr1-rawmix4b-last-musique/manifest.json) | sr1_musique | Сырая смесь NQ+HotpotQA, Qwen3-4B, последний (шаг 30 000) | — | 6 | ok | 9.23 | 17.11 | 17.71 | — |
| [`2026-08-10-sr1-rawmix4b-last-bamboogle`](runs_history/2026-08-10-sr1-rawmix4b-last-bamboogle/manifest.json) | sr1_bamboogle | Сырая смесь NQ+HotpotQA, Qwen3-4B, последний (шаг 30 000) | — | 6 | ok | 20.80 | 29.82 | 25.60 | — |
| [`2026-08-10-sr1-rawmix8b-best-nq`](runs_history/2026-08-10-sr1-rawmix8b-best-nq/manifest.json) | sr1_nq | Сырая смесь NQ+HotpotQA, Qwen3-8B, пик (шаг ~14 000) | — | 6 | ok | 25.04 | 36.88 | 50.44 | — |
| [`2026-08-10-sr1-rawmix8b-best-triviaqa`](runs_history/2026-08-10-sr1-rawmix8b-best-triviaqa/manifest.json) | sr1_triviaqa | Сырая смесь NQ+HotpotQA, Qwen3-8B, пик (шаг ~14 000) | — | 6 | ok | 14.21 | 32.94 | 73.14 | — |
| [`2026-08-10-sr1-rawmix8b-best-popqa`](runs_history/2026-08-10-sr1-rawmix8b-best-popqa/manifest.json) | sr1_popqa | Сырая смесь NQ+HotpotQA, Qwen3-8B, пик (шаг ~14 000) | — | 6 | ok | 33.03 | 41.04 | 48.63 | — |
| [`2026-08-10-sr1-rawmix8b-best-hotpotqa`](runs_history/2026-08-10-sr1-rawmix8b-best-hotpotqa/manifest.json) | sr1_hotpotqa | Сырая смесь NQ+HotpotQA, Qwen3-8B, пик (шаг ~14 000) | — | 6 | ok | 37.68 | 49.50 | 51.51 | — |
| [`2026-08-10-sr1-rawmix8b-best-2wiki`](runs_history/2026-08-10-sr1-rawmix8b-best-2wiki/manifest.json) | sr1_2wiki | Сырая смесь NQ+HotpotQA, Qwen3-8B, пик (шаг ~14 000) | — | 6 | ok | 23.21 | 30.27 | 38.53 | — |
| [`2026-08-10-sr1-rawmix8b-best-musique`](runs_history/2026-08-10-sr1-rawmix8b-best-musique/manifest.json) | sr1_musique | Сырая смесь NQ+HotpotQA, Qwen3-8B, пик (шаг ~14 000) | — | 6 | ok | 10.26 | 18.86 | 18.74 | — |
| [`2026-08-10-sr1-rawmix8b-best-bamboogle`](runs_history/2026-08-10-sr1-rawmix8b-best-bamboogle/manifest.json) | sr1_bamboogle | Сырая смесь NQ+HotpotQA, Qwen3-8B, пик (шаг ~14 000) | — | 6 | ok | 27.20 | 33.65 | 32.00 | — |
| [`2026-08-10-sr1-rawmix8b-last-nq`](runs_history/2026-08-10-sr1-rawmix8b-last-nq/manifest.json) | sr1_nq | Сырая смесь NQ+HotpotQA, Qwen3-8B, последний (шаг 30 000) | — | 6 | ok | 25.04 | 36.18 | 49.78 | — |
| [`2026-08-10-sr1-rawmix8b-last-triviaqa`](runs_history/2026-08-10-sr1-rawmix8b-last-triviaqa/manifest.json) | sr1_triviaqa | Сырая смесь NQ+HotpotQA, Qwen3-8B, последний (шаг 30 000) | — | 6 | ok | 14.22 | 32.70 | 72.60 | — |
| [`2026-08-10-sr1-rawmix8b-last-popqa`](runs_history/2026-08-10-sr1-rawmix8b-last-popqa/manifest.json) | sr1_popqa | Сырая смесь NQ+HotpotQA, Qwen3-8B, последний (шаг 30 000) | — | 6 | ok | 33.08 | 40.90 | 48.55 | — |
| [`2026-08-10-sr1-rawmix8b-last-hotpotqa`](runs_history/2026-08-10-sr1-rawmix8b-last-hotpotqa/manifest.json) | sr1_hotpotqa | Сырая смесь NQ+HotpotQA, Qwen3-8B, последний (шаг 30 000) | — | 6 | ok | 38.69 | 50.17 | 52.37 | — |
| [`2026-08-10-sr1-rawmix8b-last-2wiki`](runs_history/2026-08-10-sr1-rawmix8b-last-2wiki/manifest.json) | sr1_2wiki | Сырая смесь NQ+HotpotQA, Qwen3-8B, последний (шаг 30 000) | — | 6 | ok | 23.73 | 30.98 | 38.95 | — |
| [`2026-08-10-sr1-rawmix8b-last-musique`](runs_history/2026-08-10-sr1-rawmix8b-last-musique/manifest.json) | sr1_musique | Сырая смесь NQ+HotpotQA, Qwen3-8B, последний (шаг 30 000) | — | 6 | ok | 10.14 | 19.80 | 18.99 | — |
| [`2026-08-10-sr1-rawmix8b-last-bamboogle`](runs_history/2026-08-10-sr1-rawmix8b-last-bamboogle/manifest.json) | sr1_bamboogle | Сырая смесь NQ+HotpotQA, Qwen3-8B, последний (шаг 30 000) | — | 6 | ok | 26.40 | 32.93 | 30.40 | — |
| [`2026-08-02-gte-pool-diag`](runs_history/2026-08-02-gte-pool-diag/manifest.json) | hotpotqa_dev_fullwiki | GTE top-6 (диагностика пула) | none | 6 | ok | — | — | — | — |
| [`2026-08-02-gte-pool-diag-k500`](runs_history/2026-08-02-gte-pool-diag-k500/manifest.json) | hotpotqa_dev_fullwiki | GTE пул top-500 (диагностика) | none | 1 | ok | — | — | — | — |
| [`2026-08-02-gte-pool-diag-k1000`](runs_history/2026-08-02-gte-pool-diag-k1000/manifest.json) | hotpotqa_dev_fullwiki | GTE пул top-1000 (диагностика) | none | 1 | ok | — | — | — | — |

## Прежние имена файлов

До 2026-07-28 раны лежали в плоском `runs/` с параметрами в именах.
Миграция (`src/migrate_runs.py`) перенесла файлы в каталоги ранов;
симлинки со старыми именами, какое-то время сохранявшиеся ради команд
из `docs/pipeline.md`, давно удалены. Точные команды каждого рана — в
его `cmd.sh`.

## Архив

Раны, исключённые из сравнения, `exp.py archive` уносит в
`runs/_archive/` и записывает причину в его README; каталог
появляется вместе с первым архивированным раном.

