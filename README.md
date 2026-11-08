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

Runtime-зависимостей пока нет. Extra `dev` содержит только pytest и Ruff.
ML-библиотеки, модели и сервисы не требуются. `.env` и YAML пока являются
образцами конфигурации: автоматическая загрузка ещё не реализована.
`docker-compose.yml` описывает только Qdrant; контейнер не требуется для каркаса.

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

Созданы структура модулей, frozen dataclass-модели, Protocol-интерфейсы,
конфигурационные dataclass, YAML-образцы, исключения, CLI справки, тесты и CI.
Методы будущих компонентов поднимают `NotImplementedError`. Реальные RAG и
indexing pipelines, парсинг, загрузка YAML/.env, метрики и интеграции отсутствуют.
Контейнеры не запускались, модели не загружались.

Следующий шаг A: согласовать семантику scores/filters, реализовать retrieval
с поддельными зависимостями, затем reranking, grounding и оценку.
Следующий шаг B: согласовать ID/hash и метаданные, реализовать локальное
хранение, parser и chunking на небольших HTML fixtures, затем адаптеры индексации.
