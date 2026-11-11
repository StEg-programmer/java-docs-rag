# ADR 0001 — Идентификаторы, хеши, chunking и схема Qdrant

Статус: предложено Developer B, требует согласования Developer A.

## Контекст

`docs/architecture.md` оставил открытыми: стабильные document/chunk ID, алгоритм
`content_hash`, формат `updated_at`, ключи metadata/filters, меру сходства, единицы
`chunk_size`, правила хранения векторов разных моделей. Эти решения нужны индексации
(B) и потребляются retrieval/evaluation (A).

## Решения

**Document ID** — выводится только из URL, не зависит от содержимого и времени
запуска: `javase{версия}:{путь внутри docs/api без .html}`, например
`javase21:java.base/java/util/List`. Фрагмент `#...` и query отбрасываются. Для прочих
сайтов — `{host}:{путь}`. Реализация: `grabber/identity.py::document_id_from_url`.
Эти ID используются в `EvaluationQuestion.relevant_document_ids`.

**content_hash** — SHA-256 (hex) от UTF-8 текста `Document.content`. Каждый этап,
меняющий текст (cleaner, normalizer), пересчитывает хеш, поэтому хеш всегда
соответствует содержимому. В raw-manifest хранится SHA-256 исходных HTML-байтов.

**updated_at** — значение `<meta name="dc.created">` страницы javadoc
(`YYYY-MM-DD`), иначе `None`. Время скачивания хранится в raw-manifest.

**Metadata документа** (все значения — строки): `source`, `java_version`, `module`,
`package`, `kind` (class/interface/enum/record/…), `type_name`, `page_type`.
Chunk добавляет `chunking_strategy`, `chunk_size`, `chunk_overlap`, `char_start`,
`char_end`, `content_hash`.

**Формат текста** — парсер сохраняет структуру в облегчённом Markdown: заголовки
`#`…`######`, блоки кода в ```` ``` ````, строки таблиц `a | b | c`. Section chunking и
поле `section` опираются на заголовки вне блоков кода.

**chunk_size / overlap — в символах.** 256/512/1024 символа ≈ 60–250 токенов и не
превышают окно BGE/E5 (512 токенов), поэтому текст не обрезается моделью молча.
`overlap` учитывается только стратегией `overlap`; остальные стратегии работают с 0.

**Chunk ID** — `{document_id}#{strategy}-s{size}-o{overlap}#{position}`. Разные
конфигурации дают разные ID, повторный запуск — те же.

**section** — путь заголовков в начале chunk: `Interface List<E> > Method Details > size`;
`None` до первого заголовка.

**Текст для эмбеддинга** — по умолчанию `"{title} — {section}\n{text}"`, чтобы фрагмент
метода `size()` содержал имя класса. Сохраняемый `chunk.text` не меняется.
Отключается флагом `--no-context`. Префиксы моделей (E5: `query: `/`passage: `,
BGE: инструкция для запросов) добавляет Embedder, а не вызывающий код.

**Qdrant**
- Одна коллекция = одна embedding-модель + одна конфигурация chunking:
  `{QDRANT_COLLECTION}__{модель}__{strategy-sN-oM}`. Смешивание моделей в одной
  коллекции исключено; при несовпадении размерности — `VectorStoreError`.
- ID точки — UUIDv5 от `chunk_id` (повторный upsert перезаписывает).
- Payload — все поля `DocumentChunk`, `metadata` вложенным объектом, плюс
  `embedding_model` и `chunking`. Keyword-индексы: `document_id`,
  `metadata.java_version`, `metadata.package`.
- Distance = Cosine, векторы нормализованы; `similarity_score` = score Qdrant
  (больше — ближе), порядок — по убыванию. Пустая выдача — пустой список.
- `filters`: ключи полей chunk (`document_id`, `title`, `section`, …) — точное
  совпадение по полю; любые другие — по `metadata.<ключ>`.

**Инкрементальная индексация** — `data/manifests/index__<collection>.json` хранит
`document_id → content_hash`. Изменённый документ: сначала `delete_document`,
затем новые chunks (не остаётся «хвостов» от более длинной версии). Пропавшие из
корпуса документы удаляются из коллекции.

**Версия корпуса** — `corpus_version` = первые 16 hex SHA-256 от отсортированных пар
`(document_id, content_hash)`. Пишется в результаты каждого эксперимента.

## Последствия

- A может использовать `document_id` как метку релевантности для всех стратегий
  chunking (document-level метрики).
- Смена `configs/sources.yaml` меняет `corpus_version`; сравнивать эксперименты можно
  только при одинаковой версии.
- Изменения в общей зоне (требуют одобрения A): зависимости `httpx`, `beautifulsoup4`,
  `pyyaml` в `pyproject.toml`, extras `embeddings` и `observability`; расширенный
  `configs/sources.yaml`; новые `configs/models.yaml`, `configs/experiments/…`;
  переменные в `.env.example`.

## Предложения к общим контрактам (не внесены, обсудить отдельным PR)

1. `common/config.py::load_config` не реализован — CLI индексации пока принимает
   параметры аргументами/переменными окружения.
2. `EmbeddingConfig` стоит дополнить `batch_size`, `device`; `VectorStoreConfig` —
   `api_key`. Сейчас это параметры конструкторов.
3. Подключить подкоманды `grab/prepare/index` в общий `cli.py` (сейчас точка входа —
   `python -m java_docs_rag.pipeline.indexing_pipeline`).
4. `IndexingPipeline.run` теперь возвращает `IndexingReport` вместо `None`
   (совместимо: вызывающему коду результат можно игнорировать).
