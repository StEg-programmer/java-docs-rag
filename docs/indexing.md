# Индексация (Developer B): запуск, Qdrant, Langfuse

Поток: официальные Java docs → `data/raw` → parse/clean/normalize →
`data/processed/*.jsonl` → chunks → embeddings → Qdrant.
Решения по ID, хешам и схеме — в `docs/decisions/0001-indexing-identity-and-chunking.md`.

## Установка

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"            # httpx, bs4, pyyaml, pytest, ruff
.\.venv\Scripts\python.exe -m pip install -e ".[embeddings]"     # sentence-transformers (+torch), ~1-2 ГБ
.\.venv\Scripts\python.exe -m pip install -e ".[observability]"  # langfuse, по необходимости
```

Unit-тесты не требуют ни моделей, ни сети, ни Docker.

## 1. Сбор документации (grab)

```powershell
.\.venv\Scripts\python.exe -m java_docs_rag.pipeline.indexing_pipeline -v grab --max-pages 50
```

Границы корпуса — `configs/sources.yaml` (домен, префиксы путей, стартовые страницы,
исключения, `max_pages`). Краулер соблюдает `robots.txt`, делает паузу между запросами
(`--delay`, по умолчанию 0.5 с) и повторяет запрос при 429/5xx. Результат:
`data/raw/javase21/...html` и `data/manifests/java-se-21-api.raw.json`.
Для первой проверки начните с `--max-pages 50`; полный корпус (до 1500 страниц) с паузой 0.5 с —
примерно 15–30 минут.

## 2. Подготовка (prepare)

```powershell
.\.venv\Scripts\python.exe -m java_docs_rag.pipeline.indexing_pipeline prepare
```

Пишет `data/processed/java-se-21-api.jsonl` и печатает `corpus_version` —
его фиксируйте в экспериментах.

## 3. Qdrant

```powershell
docker compose up -d qdrant          # или команда docker run из плана проекта
curl http://localhost:6333/readyz
```

Dashboard: http://localhost:6333/dashboard. Данные — в `qdrant_storage/` (в .gitignore).

## 4. Индексация (index)

```powershell
.\.venv\Scripts\python.exe -m java_docs_rag.pipeline.indexing_pipeline index `
    --model BAAI/bge-small-en-v1.5 --strategy paragraph --chunk-size 1000
```

- Модель скачивается с Hugging Face при первом запуске (кэш вне репозитория, `HF_HOME`),
  далее работает локально на CPU. Второй вариант: `--model intfloat/multilingual-e5-small`.
- Коллекция по умолчанию: `java_docs__bge-small-en-v1-5__paragraph-s1000-o0`
  (своя для каждой модели и конфигурации chunking).
- Повторный запуск переиндексирует только изменённые документы; `--full` — всё заново.
- Переменные: `QDRANT_URL`, `QDRANT_COLLECTION`, `QDRANT_API_KEY`, `EMBEDDING_MODEL`,
  `EMBEDDING_DEVICE` (`.env` не загружается автоматически — задайте их в окружении или
  в Run Configuration PyCharm).

Developer A подключается к той же коллекции через `QdrantVectorStore` и
`create_embedder` с той же моделью:

```python
from java_docs_rag.common.config import EmbeddingConfig, VectorStoreConfig
from java_docs_rag.embeddings.factory import create_embedder
from java_docs_rag.vectorstore.qdrant_store import QdrantVectorStore

embedder = create_embedder(EmbeddingConfig("BAAI/bge-small-en-v1.5"))
store = QdrantVectorStore(
    VectorStoreConfig(collection="java_docs__bge-small-en-v1-5__paragraph-s1000-o0")
)
results = store.search(embedder.embed_query("How do I get the size of a List?"), top_k=5)
```

## Langfuse (наблюдаемость)

Трассировка выключена по умолчанию (`ObservabilityConfig(enabled=False)` →
`NoOpTracer`). Включение:

1. Получите ключи проекта: Langfuse Cloud (бесплатный тариф, https://cloud.langfuse.com)
   или self-hosted по официальной инструкции Langfuse (docker compose из их репозитория:
   v3 требует Postgres, ClickHouse, Redis и S3/MinIO, поэтому в наш `docker-compose.yml`
   не добавлен).
2. Установите `.[observability]` и задайте переменные окружения
   `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, `LANGFUSE_HOST`.
3. Запускайте этапы с флагом `--trace`:
   `... indexing_pipeline --trace index ...`

Спаны индексации: `indexing.grab`, `indexing.prepare`, `indexing.run` →
`indexing.embed_and_upsert` (число chunks, батчей, время эмбеддинга и записи).

Для Developer A (трассировка retrieval и generation) — тот же адаптер:

```python
from java_docs_rag.observability.langfuse_client import create_tracer

tracer = create_tracer(config.observability)
with tracer.span("rag.retrieve", metadata={"top_k": 10}, input=question) as span:
    results = retriever.retrieve(question, 10)
    span.update(output={"chunk_ids": [r.chunk.chunk_id for r in results]})
tracer.flush()
```

Ошибки Langfuse только логируются и не прерывают pipeline; адаптер поддерживает SDK v2
и v3.
