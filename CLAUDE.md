# Инструкции Claude Code / Developer B

Прочитайте [AGENTS.md](AGENTS.md): это общий обязательный контракт работы.
Проект — учебный Java Documentation RAG, Python 3.11, src-layout.
Текущий статус — каркас, реальная индексация и генерация не реализованы.

Ваша зона: `grabber`, `preprocessing` (включая chunking), `embeddings`,
`vectorstore`, `observability`, `pipeline/indexing_pipeline.py` и их тесты.
Поток: официальные Java docs → raw → parse/clean/normalize → chunks →
embeddings → Qdrant. Для межмодульной связи используйте `common.models` и
`common.interfaces`; подключайте зависимости явно и без I/O при импорте.

Developer A (Codex) владеет retrieval, reranking, filtering, generation,
evaluation и `pipeline/rag_pipeline.py`. Не изменяйте его модули без согласования.
Изменения `common`, CLI, конфигураций, зависимостей и CI согласуйте с A.
Архитектурные решения фиксируйте в `docs/decisions`.

До изменений изучите рабочее дерево и сохраняйте чужие правки.
Для локальных проверок и установки используйте исключительно
`.\.venv\Scripts\python.exe` (Python 3.11), не системный Python 3.10.
Запускайте его с `-m pytest`, `-m ruff check .` и `-m ruff format --check .`;
тесты не должны требовать сеть или модели.
Git: ветки `feature/b-...`, небольшие PR; без commit/push/merge и переписывания
истории без поручения. Не устанавливайте тяжёлые библиотеки, не скачивайте
модели, не запускайте Docker и не используйте платные API без отдельной задачи.
Не изменяйте файлы вне проекта. Отчитывайтесь о проверках и обнаруженных проблемах.
