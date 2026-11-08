# Локальная настройка

## Python и PyCharm

Нужен установленный Python 3.11; 3.10 и 3.12+ не входят в поддерживаемый диапазон.
В текущем проекте уже есть `.venv`: проверьте
`.\.venv\Scripts\python.exe --version`. Используйте этот путь для всех локальных
проверок и установки зависимостей. Сообщение launcher `No Python` вместе с
`Access is denied` может означать ограничение sandbox, а не отсутствие Python.

Если существующая `.venv` ссылается на отсутствующий Python, закройте процессы,
которые используют окружение. Сохраните его при необходимости под другим именем
внутри проекта, затем создайте новое `.venv` с исправным Python 3.11.
Не переносите пакеты из старого окружения вручную. Исходный код и `.idea`
при пересоздании окружения менять не требуется.

Только на новой машине без окружения создайте его: `py -3.11 -m venv .venv`.
Существующее исправное окружение пересоздавать не нужно. Далее из корня проекта:

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev]"
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check .
.\.venv\Scripts\python.exe -m ruff format --check .
.\.venv\Scripts\python.exe -m java_docs_rag.cli --help
```

При необходимости выполните `Copy-Item .env.example .env` только при отсутствии
собственного `.env`.
В PyCharm: Settings → Project → Python Interpreter → Existing environment →
`.venv/Scripts/python.exe`. Working directory — корень репозитория, test runner —
pytest. Editable install подключает src-layout; pytest дополнительно использует
`pythonpath = ["src"]` из pyproject.toml.

## Конфигурация и данные

YAML и `.env` в текущем каркасе не загружаются. Они задают будущий формат.
Все Python-зависимости разработки лёгкие: pytest и Ruff. PyTorch,
sentence-transformers, Qwen, Ollama models и платные API не используются.
Конфигурации экспериментов помещайте в `configs/experiments`, локальные
результаты — в `experiments/results`, отчёты — в `experiments/reports`.

## Сервисы

Compose описывает только Qdrant: HTTP 6333, gRPC 6334 на loopback и локальное
`qdrant_storage/`. Образ закреплён на версии; обновление версии — отдельная задача.
Для тестов каркаса Docker не нужен. Конфигурация создана без запуска контейнеров.
Ollama и Langfuse не входят в Compose и не устанавливаются этой инструкцией.
Не запускайте сервисы и не загружайте модели до отдельной задачи интеграции.

## Проверки CI

Workflow `tests.yml` использует Python 3.11, устанавливает `.[dev]`, затем
выполняет Ruff lint/format и pytest. Integration-тестов пока нет. При их добавлении
сервисы должны включаться явно, а отсутствие сервиса — приводить к skip.
