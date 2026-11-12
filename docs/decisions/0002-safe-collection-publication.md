# ADR 0002 — Безопасная публикация index через Qdrant alias

Статус: реализовано для нового интеграционного CLI `index`.
Общие Protocol, `IndexingPipeline.run` и storage adapters B не изменены.

## Причина

Передавать фиктивные previous_hashes ради замены опасно: B pipeline удаляет
старые чанки до embedding/upsert. Ошибка модели, несовместимая collection или
частичный отказ записи могут убрать ранее searchable документ.

## Решение

Новый административный слой `pipeline/collection_rebuild.py` использует прежний
B pipeline только с новым изолированным store, без previous_hashes/delete_missing.
Порядок:

0. Проверка всего replacement batch: каждый document_id должен дать хотя бы один
   непустой после strip чанк. Ошибка называет ID и происходит до Qdrant-запросов.
   Cached chunks передаются pipeline через локальный Chunker adapter без повторного split.
1. Read-only проверка существующего target/source: unnamed Cosine vectors,
   размерность, payload embedding_model, chunking tag, chunk/point IDs, vectors.
2. Read-only snapshot исходных points с vectors/payload, включая pagination.
3. Создание уникальной physical collection `<alias>__build_<uuid>`.
4. Копирование не заменяемых документов и индексация заменяемых документов B pipeline.
5. Scroll/read-back: полный набор IDs/payload, vector dimensions/finite values,
   совпадение vectors сохранённых документов с учётом float tolerance.
6. Атомарный набор alias actions delete_alias/create_alias публикует новую версию.

Решение использует [атомарное переключение aliases Qdrant](https://qdrant.tech/documentation/manage-data/collections/#collection-aliases).
Перед публикацией проверяется, что alias всё ещё указывает на исходную версию.
Ни старые, ни failed physical collections автоматически не удаляются.
Читатели могут продолжать использовать alias без смены параметров запроса.

Ошибки модели/размерности/записи/верификации до публикации оставляют прежний alias
и physical collection неизменными. При timeout/разрыве соединения **во время**
alias request outcome может быть неоднозначным: сервер мог уже переключить alias
на полностью проверенную версию. Проверить `/aliases` перед повтором; старый
physical build всё равно сохранён и доступен по имени из previous_collection/error.

## Миграция и согласование с B

Нельзя безопасно сделать alias с именем уже существующей physical collection.
Такая операция отвергается до mutation. Миграция без удаления:

```powershell
.\.venv\Scripts\python.exe -m java_docs_rag.cli index --input data/processed/java-se-21-api.jsonl `
  --collection java_docs_safe --from-collection java_docs__bge-small-en-v1-5__paragraph-s1000-o0
.\.venv\Scripts\python.exe -m java_docs_rag.cli ask "What does List.size() return?" `
  --collection java_docs_safe --ollama-model YOUR_INSTALLED_OLLAMA_MODEL
```

Модель/chunking должны совпадать; migration source не меняется. Это новый
локальный composition API, а не изменение shared indexing contract B.
`IndexingPipeline.run` сохраняет смысл previous_hashes/delete_missing; источник
передаётся через административный слой, pipeline получает только stage store.
Старый CLI B не переписан. Распространение такой гарантии на его incremental
route, общий transactional replace Protocol или cleanup policy требует отдельного
совместного решения A/B. Новый безопасный CLI не рекламирует старый route как atomic.

## Ограничения и последствия

- Один writer на alias; исходная physical collection должна быть quiescent.
  Проверка alias перед switch не является distributed lock/CAS. Прямые записи
  старым CLI во время snapshot и параллельные publishers не поддерживаются.
- Требуются права на создание collection, scroll с vectors и управление aliases.
- Snapshot/read-back хранятся в памяти; subset rebuild копирует весь retained
  корпус. Увеличиваются время работы и расход диска. Cleanup только вручную
  после отдельного согласования; output/error сообщает build names.
- Верифицируются сохранность данных и публикация, а не качество embeddings/ответов.
- Offline тесты используют реальный B Qdrant adapter и fake REST transport:
  model-load failure, несовпадение dimension, partial-write failure, ошибочный
  successful response без записи и failed alias publication сохраняют старый поиск.
  Success удаляет устаревшие tail chunks только в новой версии и сохраняет другие документы.
