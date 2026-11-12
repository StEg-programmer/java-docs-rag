# Java Documentation RAG System

Университетская лабораторная работа: собственная Retrieval-Augmented Generation
система для официальной документации Java. Цель — исследовать влияние chunking,
retrieval, reranking и filtering на ответы с проверяемыми источниками и отказом
при недостаточном контексте. Начальный источник — Java SE 21 API, заданный в
`configs/sources.yaml`.

## Архитектура

```text
Java docs → grabber → preprocessing/chunking → embeddings → Qdrant
                                                             ↓
question → query_processor → retrieval → reranking → filtering
                                                    ↓
                                       context/prompt → Ollama → RAGAnswer

pipeline: orchestration     common: models, Protocol, config, exceptions
evaluation: quality        observability: optional tracing
```

Исходники находятся в `src/java_docs_rag`. Полная структура каталогов и созданных
файлов приведена в [docs/project-tree.md](docs/project-tree.md).
Детали границ и контрактов: [architecture.md](docs/architecture.md).

## Распределение задач

| Разработчик | Модули |
| --- | --- |
| A — Codex | retrieval, reranking, filtering, generation, evaluation, rag_pipeline |
| B — Claude Code | grabber, preprocessing, embeddings, vectorstore, observability, indexing_pipeline |
| Вместе | common, CLI, конфигурации, зависимости, CI и документация |

Правила совместной работы: [AGENTS.md](AGENTS.md) и [CLAUDE.md](CLAUDE.md).
Изменения чужих модулей и общих контрактов требуют согласования.

## Установка

Требуется Python **3.11**. Используйте существующее проектное `.venv`.
Команды для PowerShell из корня репозитория:

```powershell
.\.venv\Scripts\python.exe --version
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m java_docs_rag.cli --help
```

На новой машине сначала создайте окружение по инструкции [setup.md](docs/setup.md).
При необходимости скопируйте `.env.example` в `.env`, сохранив существующий `.env`.
В PyCharm выберите `.venv/Scripts/python.exe`. Активация окружения необязательна.

Runtime-зависимости: httpx, beautifulsoup4, pyyaml; extra `dev` — pytest и Ruff.
Unit-тесты используют fake models и HTTP, сервисы им не нужны. Для реальных
embeddings/reranking нужен существующий optional extra `embeddings` и заранее
подготовленные модели. Их установка/скачивание и запуск сервисов — отдельный шаг.
`.env` и `configs/*.yaml` не загружаются автоматически: используйте аргументы CLI
и переменные окружения. `configs/models.yaml` документирует модели.

## Проверки

Все локальные проверки выполняйте через Python из `.venv`:

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
```

GitHub Actions запускает эти проверки на Python 3.11. Unit-тесты проверяют
общие модели без Docker, сети, моделей или API-ключей. Каталоги integration и
fixtures зарезервированы для дальнейших тестов.

## Текущий статус

Реализованы индексация B и ответный pipeline A, объединённые через CLI.
`java-docs-rag` (после editable install) и `python -m java_docs_rag.cli` эквивалентны.
Общий API моделей/Protocol сохранён. Подробности запуска и результаты проверки —
в [integration handoff](docs/reports/integration-handoff.md), индексация —
в [docs/indexing.md](docs/indexing.md).

```powershell
# Команды требуют готовых локальных моделей и запущенных Qdrant/Ollama.
# Эти переменные запрещают Hugging Face downloads в данном сеансе.
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
.\.venv\Scripts\python.exe -m java_docs_rag.cli index `
  --html tests/fixtures/javadoc/java.base/java/util/List.html `
  --source-url https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/util/List.html
.\.venv\Scripts\python.exe -m java_docs_rag.cli ask "What does List.size() return?" `
  --ollama-model YOUR_INSTALLED_OLLAMA_MODEL
# Старый B CLI доступен без изменения порядка его аргументов:
.\.venv\Scripts\python.exe -m java_docs_rag.cli indexing --data-dir data prepare
```

`index` и `ask` используют одни defaults: BGE-small, 384 dimensions, paragraph
1000 chars, коллекцию `java_docs__bge-small-en-v1-5__paragraph-s1000-o0`.
При изменении модели/chunking повторите одинаковые параметры у обеих команд.
`--collection` задаёт точное имя (для smoke выберите отдельную test collection).
Новый `index` проверяет совместимость и строит изолированную коллекцию, сохраняя
остальные документы. После проверки записанных точек атомарно переключает alias;
старые и неудачные builds не удаляются. `collection` в JSON — стабильный alias,
`build_collection` — опубликованная physical collection. При ошибке загрузки
модели/размерности/записи прежняя searchable collection сохраняется.
Для существующей physical collection используйте новый alias:
`index ... --collection NEW_ALIAS --from-collection OLD_PHYSICAL`, затем
`ask ... --collection NEW_ALIAS`. Одно имя не может быть physical collection и alias.
Подробности, single-writer ограничения и неоднозначность timeout при публикации:
[ADR безопасной публикации](docs/decisions/0002-safe-collection-publication.md).
B `indexing ... index` сохраняет прежний incremental manifest и delete-first
поведение; он не даёт гарантию безопасной замены нового маршрута.
`ask` выдаёт JSON с вопросом, ответом, флагом отказа, raw retrieval и sources.
Источники — переданные модели доказательства, а не автоматическая проверка её цитат.

У `index` и `ask` есть `--trace` для count-only Langfuse telemetry. По умолчанию
tracing выключен; настройте endpoint/project keys локально через `LANGFUSE_HOST`,
`LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY` (не помещайте ключи в репозиторий).
Передаются durations, counts и outcomes без текста вопросов, документов, prompts
и ответов. Подробные команды и фактическая проверка получения — в integration handoff.
