# Архитектура и контракты

## Границы компонентов

Поток индексации (B): crawler обнаруживает разрешённые URL, downloader получает
raw HTML, repository хранит его локально, manifest отслеживает изменения.
Parser создаёт Document; cleaner и normalizer подготавливают его к chunking.
Chunker создаёт DocumentChunk, Embedder вычисляет векторы, VectorStore хранит их.
Indexer будет отвечать за batch-запись, indexing_pipeline — за связывание этапов.
Сейчас `IndexingPipeline.run` принимает уже подготовленные Document; полный
маршрут от URL потребует согласованного расширения точки входа.

Поток ответа (A): query_processor подготавливает вопрос, Retriever возвращает
SearchResult, Reranker переоценивает их, filtering выбирает доказательства.
ContextBuilder и PromptBuilder формируют запрос к LLMProvider; AnswerGenerator
собирает RAGAnswer с источниками и insufficient_context. RAGPipeline будет
управлять этим маршрутом. Все эти операции сейчас являются заглушками.

Evaluation использует EvaluationQuestion и отдельные метрики поиска и ответа.
Observability будет опциональной; конфигурация по умолчанию её отключает.
Сетевая обработка и экспорт трассировок в каркасе отсутствуют.

## Общие модели

| Модель | Назначение |
| --- | --- |
| Document | Документ, URL, хеш, текст, время обновления, строковые метаданные |
| DocumentChunk | Фрагмент, связь с документом, раздел и позиция |
| SearchResult | Фрагмент, similarity_score, опциональный rerank_score |
| RAGAnswer | Вопрос, ответ, использованные источники, результаты поиска, модель, отказ |
| EvaluationQuestion | Эталон, релевантные document ID, категория, answerable |

Модели — frozen dataclass с type hints, без runtime-валидации типов.
Неизменяемость поверхностная: вложенные list/dict сохраняют типы из задания.
Пустые metadata создаются через default_factory; общих mutable defaults нет.
`updated_at` — nullable строка; формат времени ещё предстоит согласовать.

## Межмодульные интерфейсы

Все Protocol синхронные, структурные, без зависимости от SDK поставщиков.
`Embedder.model_id`, `dimension` и `LLMProvider.model_id` доступны для чтения.
Batch embedding должен сохранять порядок входных текстов; dimension должна
совпадать с длиной векторов и размерностью коллекции.
`VectorStore.upsert` получает параллельные chunks/vectors; `search` принимает
строковые фильтры или None, возвращает SearchResult; `delete_document` удаляет
все chunks одного документа. Retriever и Reranker возвращают списки результатов.
LLMProvider принимает system_prompt/user_prompt и возвращает строку.

До реализации нужно согласовать в ADR: стабильные document/chunk ID, алгоритм
content_hash, формат updated_at, допустимые ключи metadata/filters, меру сходства
и порядок выдачи, семантику rerank_score и поведение пустой выдачи.
Далее — правила abstention, бюджет контекста, единицы chunk_size (пока не
определены) и document-level агрегацию метрик при нескольких chunks документа.

## Конфигурация и ограничения

`common/config.py` описывает значения `configs/default.yaml`; `sources.yaml`
содержит отдельный список источников. `load_config` зарезервирован и не реализован.
`.env.example` — справочник параметров будущих интеграций. Секреты нельзя хранить
в YAML/Git. Выбор embedding и generation моделей отложен (model_id = None).
Пока не определена модель и размерность, индексирование не является рабочим.

Документы, векторы, результаты, модели и состояние Qdrant не коммитятся.
Небольшие синтетические fixtures и версионируемые эталоны можно хранить в
`tests/fixtures`; тяжёлые наборы — в игнорируемом `data/evaluation`.
Архитектурные решения размещаются в `docs/decisions/NNNN-topic.md`:
контекст, решение, последствия, согласование A/B.
