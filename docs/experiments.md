# Эксперименты

Общие правила — раздел 10 плана проекта: фиксировать версию корпуса, модели,
chunking, коллекцию, параметры retrieval, reranker, LLM, prompt, датасет и
experiment ID; не выдумывать метрики; менять один параметр за раз.

## Developer B: chunking и embedding-модели

План: `configs/experiments/b_chunking_embeddings.yaml`.
Базовая конфигурация: BGE-small, `paragraph`, 512 символов, без overlap.

| Группа | Что меняется | Значения |
| --- | --- | --- |
| chunking_strategy | стратегия | fixed / overlap (10 %) / paragraph / section |
| chunk_size | размер, символы | 256 / 512 / 1024 |
| overlap_percent | overlap для стратегии overlap | 0 / 10 / 20 % |
| embedding_model | модель | BGE-small vs multilingual-E5-small |

Запуск (после `grab` и `prepare`, с установленным `.[embeddings]`):

```powershell
.\.venv\Scripts\python.exe experiments\runners\b_indexing_experiments.py --dry-run
.\.venv\Scripts\python.exe experiments\runners\b_indexing_experiments.py --group chunk_size
.\.venv\Scripts\python.exe experiments\runners\b_indexing_experiments.py --store qdrant
```

По умолчанию используется точный in-memory поиск (cosine, как в Qdrant), чтобы
эксперименты не зависели от Docker; `--store qdrant` пишет каждую конфигурацию в
отдельную коллекцию `exp_b__…`. `--limit-docs N` — быстрый прогон на части корпуса
(указывается в результате через `documents`).

Каждый прогон сохраняет `experiments/results/<experiment_id>.json` (условия, метрики
индексации: число chunks, длины, время chunking/эмбеддинга/записи, chunks/s) и строку
в `experiments/results/b_indexing_summary.csv`.

Retrieval-метрики (Recall@K, MRR, Precision@K, Hit Rate) вычисляются функциями
Developer A из `java_docs_rag.evaluation` на `data/evaluation/questions.json`. Пока они
не реализованы, результат содержит `"retrieval": null` и причину — значения не
подставляются вручную. Релевантность размечается на уровне `document_id`, поэтому одна
разметка подходит ко всем стратегиям chunking.

## Developer A: retrieval, reranker, prompts, LLM

Заполняет Developer A.
