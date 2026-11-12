# Интеграция Developer A и Developer B

Дата: 2026-11-11. Ветка `feature/a-01-semantic-retriever`, HEAD `3991da1`.
Исходные изменения A были восстановлены пользователем из stash и сохранены.
Локальная резервная копия изменений сохранилась. Commit/push/merge и смена
ветки в этой задаче не выполняются.

## Изменения интеграции

- `src/java_docs_rag/cli.py`: поддерживаемые команды `index`, `ask`, forwarding
  `indexing` к существующему CLI B. JSON-вывод включает collection, embedding
  model и dimension. У index `collection` — stable alias, `build_collection` —
  новая проверенная physical collection, `previous_collection` — старая версия
  alias (если была). Общая функция разрешения ресурсов использует factories B.
- `src/java_docs_rag/pipeline/collection_rebuild.py`: read-only preflight,
  изолированный rebuild, проверка записанных points и атомарная alias publication.
  Обоснование и миграция: [ADR 0002](../decisions/0002-safe-collection-publication.md).
- `src/java_docs_rag/reranking/sentence_transformer.py`: настоящий lazy callable
  для `CrossEncoderReranker`, использующий `CrossEncoder.predict`; в unit-тестах
  подменяется только модель. Импорт/конструктор не загружают ML и не выполняют сеть.
  Настоящий loader требует локальных весов (`local_files_only=True`).
- `configs/models.yaml`: задокументированы reranker и настройки Ollama.
- `README.md`, `docs/indexing.md`: актуальные поддерживаемые точки входа.
- `tests/unit/test_offline_integration.py`: fixture → DocumentPreparer → factories
  B → IndexingPipeline → InMemoryVectorStore → SemanticRetriever →
  CrossEncoderReranker → RAGPipeline → grounded AnswerGenerator.
- `tests/unit/test_cli.py`: реальные CLI index/ask, общие settings, ошибки
  размерности/отсутствующей коллекции, передача аргументов CLI B и сохранность
  прежнего searchable документа при ошибках замены.
- `tests/unit/test_collection_rebuild.py`: fake REST boundary с aliases,
  pagination, partial writes и fault injection для тестов настоящего B adapter.
- `tests/integration/test_live_smoke.py`: opt-in live smoke через настоящий CLI;
  normal context settings, fresh alias, cited List.size() evidence и модельный
  отказ на unsupported вопрос. По умолчанию skipped, JSON сохраняется в ignored
  `experiments/reports/<alias>.json`, включая частичный результат при ошибке.
- `tests/unit/test_sentence_transformer_scorer.py`: lazy loading, scalar scores,
  пакетный порядок и ошибки загрузки/невалидных scores.
- `experiments/runners/b_indexing_experiments.py` и
  `tests/unit/pipeline/test_b_experiments.py`: точечная интеграционная правка
  потребителя A-метрик в коде B. Убран предварительный срез по чанкам перед
  document-level k; вопросы без relevant IDs исключаются из среднего, как в A.
  Устаревшее ожидание «A ещё не реализован» заменено причиной пустого dataset;
  добавлен regression test с двумя чанками первого документа перед релевантным
  вторым документом. Сам indexing pipeline, factories и storage adapters B не менялись.

Ранее восстановленные A-файлы остаются частью будущего общего PR. Их полное
описание: [developer-a-handoff.md](developer-a-handoff.md). Этот старый отчёт
описывает состояние **до** fast-forward B; его утверждения об отсутствии
реализации B и 179 тестах исторические. Для интеграции актуален настоящий отчёт.

## Точки входа и конфигурация

Из корня проекта через проектную Python 3.11 `.venv`:

```powershell
.\.venv\Scripts\python.exe -m java_docs_rag.cli --help
.\.venv\Scripts\python.exe -m java_docs_rag.cli index --help
.\.venv\Scripts\python.exe -m java_docs_rag.cli ask --help
```

После editable install эквивалент: `.\.venv\Scripts\java-docs-rag.exe ...`.
Прежний `python -m java_docs_rag.pipeline.indexing_pipeline` сохранён.
Его grab/prepare/index доступны также через prefix `java_docs_rag.cli indexing`:

```powershell
# grab обращается к сети; приведён для будущего разрешённого запуска.
.\.venv\Scripts\python.exe -m java_docs_rag.cli indexing --data-dir data grab --max-pages 5
.\.venv\Scripts\python.exe -m java_docs_rag.cli indexing --data-dir data prepare
.\.venv\Scripts\python.exe -m java_docs_rag.cli index --input data/processed/java-se-21-api.jsonl
.\.venv\Scripts\python.exe -m java_docs_rag.cli ask "What does List.size() return?" `
  --ollama-model YOUR_INSTALLED_OLLAMA_MODEL
```

### Безопасная замена: исправление перед финальным review

Первоначальный delete-first вариант нового index **удалён**. Теперь до mutation
проверяются unnamed Cosine config, dimension, canonical embedding_model,
chunking tag и исходные point/chunk IDs/vectors. Existing collection с чужой
моделью того же размера также отвергается, а не считается совместимой по size.

Дополнительно весь input batch проверяется **по каждому document_id** до любых
Qdrant-запросов: каждый документ обязан дать хотя бы один чанк с непустым после
strip текстом. Иначе ValueError называет проблемный document_id; наличие других
валидных документов не разрешает пустую замену. Проверенные chunks кешируются
и через локальный Chunker adapter передаются B pipeline, без повторного split.
Контракт B не меняется. Regression через `index --input` начинает с searchable
документа и подаёт его пустую/whitespace замену вместе с другим валидным документом
в обоих порядках: нет новых REST requests, staging collections и alias switch,
оригинальный search result остаётся прежним (4 cases).

Затем из read-only snapshot создаётся `<alias>__build_<uuid>`. В неё копируются
не заменяемые документы с payload/vectors и через настоящий B IndexingPipeline
индексируются новые версии. Устаревшие tail chunks заменяемых документов в новую
collection не копируются. Scroll/read-back проверяет полный набор IDs/payload,
размерности/конечность vectors и сохранность retained vectors. Только после этого
одним atomic alias request публикуется новая версия. Старый physical build
сохраняется и не мутируется ни при успехе, ни при ошибке. Failed builds тоже
не удаляются; их имена сообщаются в ошибке для отдельного cleanup review.

При timeout самого alias request исход публикации может быть неоднозначным:
сервер мог уже переключить alias на полностью проверенный новый build.
Перед повтором проверить `/aliases`; старый physical build остаётся доступен.
Требуется один writer на alias и отсутствие прямых записей в source collection
во время snapshot; alias recheck не является distributed lock/CAS.

Нельзя атомарно заменить **physical** collection alias-ом с тем же именем без
destructive migration. Такой target отвергается до mutation; безопасная миграция:

```powershell
.\.venv\Scripts\python.exe -m java_docs_rag.cli index --input data/processed/java-se-21-api.jsonl `
  --collection java_docs_safe --from-collection java_docs__bge-small-en-v1-5__paragraph-s1000-o0
.\.venv\Scripts\python.exe -m java_docs_rag.cli ask "What does List.size() return?" `
  --collection java_docs_safe --ollama-model YOUR_INSTALLED_OLLAMA_MODEL
```

Старый источник при migration сохраняется. `--from-collection` допустим только
для нового target alias. У нового index нет incremental manifest: rebuilding
копирует весь retained корпус. Нужны дополнительные память/диск и права на aliases.
`IndexingPipeline.run` B получает **только isolated stage store**, без фиктивных
previous_hashes и без delete_missing. Контракт B/VectorStore не менялся, поэтому
миграция его потребителей/согласование новой shared сигнатуры не нужны.

Legacy B `indexing ... index` оставлен с прежним manifest/skip unchanged и
delete-first поведением; безопасность замены этой команды не заявляется.
Распространение alias publication на legacy route или изменение transactional
indexing contract требует отдельного совместного решения A/B. Используйте новый
`index` для описанной здесь безопасной замены.

| Настройка | Default / источник |
| --- | --- |
| Embedding model | --model, затем EMBEDDING_MODEL, затем BAAI/bge-small-en-v1.5 |
| Embedding dimension | из registry B: BGE/E5 384; неизвестной модели нужен --dimension |
| Device | --device, затем EMBEDDING_DEVICE, затем cpu |
| Chunking | --strategy paragraph, --chunk-size 1000, --overlap 150; у paragraph effective overlap=0 |
| Qdrant URL | --qdrant-url, затем QDRANT_URL, затем http://localhost:6333 |
| Collection | --collection — точное логическое имя (alias для нового index); иначе collection_name_for(QDRANT_COLLECTION или java_docs, canonical model_id, chunker.config_tag) |
| Migration source | index --from-collection EXISTING_PHYSICAL при новом --collection NEW_ALIAS |
| Qdrant API key | QDRANT_API_KEY; в вывод не попадает |
| Reranker | --reranker-model, затем RERANKER_MODEL, затем cross-encoder/ms-marco-MiniLM-L-6-v2 |
| Ollama | --ollama-url / OLLAMA_BASE_URL / http://localhost:11434 |
| Ollama model | --ollama-model или OLLAMA_MODEL; обязателен, автоматического pull нет |
| Retrieval | --top-k 10, --top-n 5, --min-similarity None |
| Context / HTTP timeout | --max-context-chars 8000, --timeout 60 seconds |

YAML и `.env` не загружаются автоматически; `load_config` не менялся.
`configs/models.yaml` — reference registry, не ещё один источник overrides.
При смене `--model`, strategy, size, overlap повторите одинаковые параметры у
index и ask; aliases B разрешаются до построения имени collection.
Явный `--collection` обходит name isolation. Новый index проверяет source config
и model/chunking payload. При ask из произвольной legacy collection пользователю
всё ещё нужно подтвердить модель: одна размерность не доказывает совместимость.

Реальный scorer использует существующий extra `embeddings`, новых зависимостей
не добавлено. API `CrossEncoder.predict` сверен с
[официальной документацией Sentence Transformers](https://www.sbert.net/docs/package_reference/cross_encoder/model.html).
Single-score model возвращает один конечный float на пару в исходном порядке.
Scores используются только для ранга, а не как similarity cutoff/вероятности.
ML loading/inference failure — явная ошибка, не успешный отказ.

## Проверенные контракты и границы

По коду ADR 0001 и B adapters подтверждены:

- `create_embedder` для BGE-small и multilingual E5-small возвращает dimension=384,
  canonical model_id и соответствующие query/document prefixes. Индексация
  использует normalized embeddings. `DocumentIndexer` проверяет длину batch и vectors.
- Общий CLI resolver index/ask создаёт один и тот же model/dimension/chunking/target
  по одинаковым параметрам. У ask до embedding-запроса проверяется существование
  collection и её размерность; отсутствующая/несовпадающая collection даёт ошибку,
  не создаётся пустая collection и не возвращается ложный отказ.
- Staged Qdrant создаётся B adapter с unnamed vector и distance=Cosine; upsert проверяет
  размерность существующей collection. `similarity_score = float(point['score'])`,
  без инверсии; выше означает ближе. Memory store также считает cosine.
- `document_id`: `javase21:java.base/java/util/List` из URL;
  `chunk_id`: `{document_id}#paragraph-sN-o0#{position}` (или другая strategy).
  UUIDv5 point ID не заменяет payload chunk ID. Все поля DocumentChunk/metadata
  сохраняются payload и восстанавливаются search, включая source_url.
- `SemanticRetriever` передаёт SearchResult дальше без изменения. Pipeline хранит
  raw retrieval в retrieved_chunks, затем cutoff → rerank → dedup → budget.
  `sources` — выбранные чанки, переданные модели, citation [S1] соответствует
  sources[0]. Это traceability, не автоматическое доказательство достоверности ответа.
- Пустое evidence вызывает отказ без provider; точный marker INSUFFICIENT_CONTEXT
  после strip даёт model-declared отказ с сохранёнными переданными sources.

Ограничения: сам B adapter проверяет size, но не удостоверяет embedding_model
и distance. У нового index это проверяет administrative preflight до mutation;
для ask по произвольной legacy physical collection нужна ручная проверка
Qdrant config/payload. Нельзя утверждать, что чужой same-dimension индекс совместим.
Named vectors этим маршрутом не поддержаны. ADR ошибочно обещает, что характерный
char chunk-size гарантирует token window: char budget этого не гарантирует;
нужно отдельно измерить truncation на выбранной модели.

Общие models/Protocol/config dataclass, B indexing pipeline/factories/storage
не менялись (совместимая правка experiment consumer перечислена выше). Новых shared
контрактов, требующих миграции A/B, нет. ADR 0001 всё ещё имеет статус предложенного;
по ID/Cosine/collection mapping реализация A совместима с ним, формальное совместное
утверждение и будущие изменения token-budget/config-loader — пункты PR review.

## Offline проверка

Первый baseline после fast-forward: pytest собрал 239 items и остановился на
4 collection errors: отсутствуют `bs4` и `yaml` в проектной `.venv`.
Это зависимости уже существующего pyproject B, не новые ML зависимости.
До установки подготовлены реализация, fake-model тесты и документация, выполнена
статическая проверка. После предоставленного разрешения выполнено:
`.\.venv\Scripts\python.exe -m pip install -e ".[dev]"`.
Установлены существующие runtime dependencies и их транзитивные зависимости;
torch, sentence-transformers, модели и сервисы не устанавливались.

Первый полный запуск после установки: 272 passed, 1 failed — устаревшее
ожидание B про неготовую evaluation A. После описанной совместимой правки:
промежуточный результат был 274 passed. После исправления безопасности замены
результат был 286 passed. После per-document preflight (до Langfuse дополнений) результат:
**290 passed in 2.95s** до добавления opt-in smoke. После его добавления и
установки approved ML extra тогдашний offline запуск: **290 passed, 1 skipped in 2.31s**;
skip — только не включённый live smoke. Python 3.11.9, pytest 8.4.2.
Новые integration-focused offline тесты содержат 27 passed cases
(CLI 21, fixture round trip 1, scorer 5); ещё 1 новый regression case — в тестах
B experiment runner. Прежние тесты retrieval/query/reranker A сохранены.

Ruff lint: **All checks passed!**; Ruff format: **98 files already formatted**.
`git diff --check` — exit 0. Для каждого нового файла выполнен
`git diff --no-index --check`: whitespace diagnostics = 0; exit 1 из-за различий
с `/dev/null` не считается whitespace ошибкой. Module CLI `--help` успешно
показывает `indexing`, `index`, `ask`.
Общий command batch также запускал `pip show sentence-transformers torch`,
который отдельно дал exit 1 / `Package(s) not found`; это результат проверки
готовности live окружения, не провал pytest/Ruff.

Повторяемый offline round trip:

```powershell
.\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider tests/unit/test_offline_integration.py tests/unit/test_cli.py tests/unit/test_sentence_transformer_scorer.py
.\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider
.\.venv\Scripts\python.exe -m ruff check --no-cache .
.\.venv\Scripts\python.exe -m ruff format --check --no-cache .
git diff --check
$integrationNewFiles = git ls-files --others --exclude-standard
foreach ($integrationNewFile in $integrationNewFiles) {
  git diff --no-index --check -- /dev/null $integrationNewFile
}
```

Fixture — существующий синтетический `tests/fixtures/javadoc/java.base/java/util/List.html`.
Работают настоящий parser/cleaner/normalizer, chunker, indexer/indexing pipeline,
SentenceTransformerEmbedder (с fake encode), memory cosine search, retriever,
CrossEncoderReranker с настоящим scorer adapter (fake predict), все фильтры,
generator/prompts. Проверяются dimension=384, prefix flow, ID/metadata/URL,
score preservation, источник ответа List.size(), два пути отказа и отсутствие
лишних scoring/LLM вызовов. После изменения документа проверено удаление старых
tail chunks. Это offline unit/component test, поэтому integration marker не нужен.

Ключевые новые regression cases в CLI работают с настоящим B QdrantVectorStore
и stateful fake REST boundary. Сначала индексируется document и сохраняется его
search result. Затем принудительно вызываются model-load failure, collection
dimension mismatch, неверный dimension модели, partial batch write error,
ложный successful response без записи и alias publication error. После каждого
возвращается ошибка CLI, alias и содержимое старой physical collection не меняются,
а original document по-прежнему находится search. Отдельные тесты проверяют
read-only rejection чужой distance/model/chunking, migration legacy collection
и success со сменой alias, удалением stale tail только в новой версии и сохранением
не заменяемых документов. Это проверяет fault behavior, а не наличие live Qdrant.

## Live smoke — отдельный сценарий, не offline результат

Начальные read-only probes (историческое состояние до разрешённого setup):

| Проверка | Фактический результат |
| --- | --- |
| `.venv python -m pip show sentence-transformers torch` | оба package не найдены |
| GET http://localhost:6333/readyz, timeout 2 s | «Время ожидания завершения операции истекло.» |
| GET http://localhost:11434/api/tags, timeout 2 s | «Время ожидания завершения операции истекло.» |
| `Get-Command ollama` | исполняемый файл не найден в PATH |

Повторный read-only inventory после per-document исправления подтвердил:
torch/sentence-transformers отсутствуют; у Docker Desktop Linux engine отсутствует
named pipe `dockerDesktopLinuxEngine` (`The system cannot find the file specified`).
Docker Desktop executable присутствует. Ollama executable не найден также в
`%LOCALAPPDATA%\Programs\Ollama` и `C:\Program Files\Ollama`.
В стандартном `%USERPROFILE%\.cache\huggingface\hub` snapshots BGE-small и
ms-marco-MiniLM-L-6-v2 не найдены. Оба localhost endpoint снова дали timeout.
Для продолжения запрашиваются отдельные разрешения на конкретные setup actions;
последующие фактические результаты записываются ниже отдельно от offline counts.

На этом начальном этапе live smoke не был выполнен. Timeout подтверждал недоступность в момент probe,
а не доказывает отсутствие установленных сервисов. Кэш моделей и payload/config
живой collection не проверены: ML packages отсутствуют, endpoints недоступны.
Изолированная collection не создавалась; исходные Qdrant данные не менялись.
Никакие сервисы или модели автоматически не устанавливались/запускались.

Исторический список требовавшихся setup-разрешений (продолжение ниже):

1. Установить существующий тяжёлый extra:
   `.\.venv\Scripts\python.exe -m pip install -e ".[embeddings]"`.
2. Проверить локальный кэш BGE и reranker; если весов нет, отдельно разрешить
   загрузку. Пример команды загрузки после разрешения (без offline env flags):

   ```powershell
   .\.venv\Scripts\python.exe -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-small-en-v1.5'); snapshot_download('cross-encoder/ms-marco-MiniLM-L-6-v2')"
   ```

3. Разрешить восстановление/запуск Qdrant: `docker compose up -d qdrant`, затем
   успешный GET `/readyz`. Docker CLI присутствует; Docker engine сейчас не проверялся.
4. Уточнить путь/установку Ollama, выбрать модель и отдельно разрешить startup
   `ollama serve`. После успешного `/api/tags` использовать уже имеющуюся модель;
   если нужной нет, отдельно разрешить `ollama pull <СОГЛАСОВАННОЕ_ИМЯ>`.
5. Выполнить команды ниже на новой collection, записать оба JSON и проверить
   supporting text/citations. Не считать недоступность сервиса успешным отказом.

После отдельно разрешённой подготовки зависимостей, кэша embedding/reranker и
локальной Ollama-модели выполнить **в изолированной новой collection**:

```powershell
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$smokeCollection = "java_docs_ab_smoke_" + (Get-Date -Format "yyyyMMddHHmmss")
# Имя модели должно уже присутствовать в Ollama /api/tags, никаких auto pull.
$smokeModel = "YOUR_INSTALLED_OLLAMA_MODEL"
.\.venv\Scripts\python.exe -m java_docs_rag.cli index `
  --html tests/fixtures/javadoc/java.base/java/util/List.html `
  --source-url https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/util/List.html `
  --model bge-small --collection $smokeCollection
.\.venv\Scripts\python.exe -m java_docs_rag.cli ask "What does List.size() return?" `
  --model bge-small --collection $smokeCollection --ollama-model $smokeModel
.\.venv\Scripts\python.exe -m java_docs_rag.cli ask "What is tomorrow's weather?" `
  --model bge-small --collection $smokeCollection --ollama-model $smokeModel
```

`$smokeCollection` теперь stable alias. Проверить `/aliases` и physical
`build_collection` из index JSON: size=384, distance=Cosine, canonical embedding_model
в payload, только fixture document IDs. Первый ответ должен иметь отказ=false,
содержать supported утверждение о количестве элементов и citation [Sx],
совпадающий source должен иметь ожидаемые document_id, chunk_id, URL и supporting text.
Для второго ожидать недостаточный контекст=true; при cutoff=None evidence может
быть непустым и от модели требуется точный marker. Иная формулировка или ответ из
внешних знаний означает провал smoke, его нельзя превращать в «passed» вручную.
Проверку forced empty context (`--max-context-chars 0`) можно выполнить отдельно,
но она не заменяет unsupported-question smoke с обычными настройками.
Collection не удалять автоматически; cleanup только после отдельного согласования.

### Продолжение после предоставленных разрешений

Выполнены approved actions:

- `python -m pip install -e ".[embeddings]"`: успешно. В `.venv` установлены
  sentence-transformers 5.7.0, torch 2.15.0, transformers 5.22.2 и зависимости.
- Docker Desktop запущен в фоне; engine 28.3.3 доступен.
- `docker compose up -d qdrant`: image v1.12.5 загружен, контейнер
  `java-docs-rag-qdrant-1` запущен. Прежние коллекции не удалялись.
- Одобрены и выполнены загрузка official `ollama/ollama:latest` и CPU-only запуск
  `java-docs-rag-smoke-ollama` на `127.0.0.1:11434`. Bind cache — ignored
  `models/ollama`. Image digest:
  `sha256:910d0d8107b538fb0894e63163eb195de0cc7aefcbc666dc62d3b18f71124592`.
- Одобрены и успешно скачаны конфиги/tokenizers/safetensors моделей
  BAAI/bge-small-en-v1.5 и cross-encoder/ms-marco-MiniLM-L-6-v2 в ignored
  `models/huggingface` (`HF_HOME`). ONNX/OpenVINO и дублирующие pickle weights
  не скачивались. Windows cache работает без symlinks; это warning, не ошибка.
- Одобрена загрузка `qwen2.5:1.5b` в контейнер через
  `docker exec java-docs-rag-smoke-ollama ollama pull qwen2.5:1.5b`.
  Загрузка успешно завершена; digest модели:
  `65ec06548149b04c096a120e4a6da9d4017ea809c91734ea5631e89f96ddc57b`.

Фактические версии сервисов: Qdrant 1.12.5, Ollama 0.40.16.
Кэшированные HF revisions:

- BGE-small: `5c38ec7c405ec4b44b94cc5a9bb96e735b38267a`.
- Reranker: `233902d25c440f23af6f7d6e94d2946bac0bee0a`.

Повторяемый live тест после готовности Ollama и модели:

```powershell
$env:HF_HOME = Join-Path (Get-Location) "models/huggingface"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:JAVA_DOCS_RAG_LIVE = "1"
$env:OLLAMA_MODEL = "qwen2.5:1.5b"
.\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider -s tests/integration/test_live_smoke.py
```

Тест использует default context=8000 chars, top_k=10/top_n=5, min_similarity=None;
изменён только transport timeout до 120 секунд. Unsupported проверка не подменяется
forced empty context или результатом fake модели.

### Фактические live результаты

Первая попытка: **1 failed in 92.01s**. Индексация успешно записала два чанка,
а модель ответила `List.size() returns the number of elements in this list.`
Ответ подтверждался возвращённым JavaDoc, но модель не добавила требуемую `[Sx]`
цитату: строгая citation assertion провалилась. В этой версии harness остановился
до второго вопроса. Первоначальный результат сохранён локально в каталоге experiments/reports.

Harness исправлен так, чтобы записывать **оба** ответа до quality assertions.
Citation assertion, prompts, model и context settings не ослаблялись/не менялись.
Повторный live запуск на **новом** alias: **1 passed in 89.51s**.
Результаты повторного прогона:

- Один fixture document, два чанка. GET реальной collection подтвердил
  `size=384`, `distance=Cosine`; GET `/aliases` подтвердил указанную связь.
- Вопрос: `What does List.size() return?`
  Ответ: `List.size() returns the number of elements in the list. [S1]`.
  `insufficient_context=false`.
- `[S1]` соответствует chunk
  `javase21:java.base/java/util/List#paragraph-s1000-o0#0`, document
  `javase21:java.base/java/util/List`, URL
  `https://docs.oracle.com/en/java/javase/21/docs/api/java.base/java/util/List.html`.
  В сохранённом тексте действительно есть описание `size()`:
  `Returns the number of elements in this list.` Ответ просмотрен и подтверждается
  этим evidence; это локальный синтетический fixture, не результат сетевого crawling.
- Вопрос: `What is tomorrow's weather?`
  Ответ: `Insufficient context to answer the question.`,
  **`insufficient_context=true`**. В `sources` осталось **2** JavaDoc chunks:
  отказ получен от реальной модели при непустом evidence и нормальном бюджете.

Строгий smoke прошёл в повторной попытке, но начальный citation failure остаётся
известным ограничением выбранной небольшой модели/default sampling. Один pass
не гарантирует стабильное соблюдение citation format. Автоматический citation
repair/entailment validator не добавлялся; первоначальный failure не скрыт.

Оба JSON — локальные ignored artifacts. Модели, caches, Qdrant state и secret
файлы Ollama не добавлялись в Git. Test aliases/builds и оба контейнера оставлены
для review/повторного запуска; Docker Desktop также работает. Cleanup/остановка
не выполнялись автоматически. Текущего setup blocker для этого smoke больше нет.

## Langfuse: integrated index/ask и подтверждение получения

Дополнение текущей задачи: SDK `langfuse 3.15.0` установлен с разрешения через
существующий extra `python -m pip install -e ".[observability]"`; constraint
`langfuse>=2.50,<4` и общие контракты не менялись. Qdrant/Ollama и кэш моделей
повторно использованы без перезапуска или новых загрузок моделей.

Изменённые/новые файлы этого дополнения:

- `src/java_docs_rag/cli.py`: opt-in `index --trace` / `ask --trace`, общий root span,
  проверка наличия трёх environment variables, correlation IDs в JSON.
- `src/java_docs_rag/pipeline/rag_pipeline.py`: optional tracer в конце конструктора,
  timing/count/outcome для retrieval, reranking, фильтров/evidence и generation.
- `src/java_docs_rag/pipeline/collection_rebuild.py`: передача tracer в существующий
  B IndexingPipeline; безопасная публикация и per-document preflight сохранены.
- `src/java_docs_rag/observability/safe_tracing.py`: privacy allowlist и изоляция
  ошибок telemetry. Реализует существующий Tracer API поверх прежнего adapter.
- `src/java_docs_rag/observability/langfuse_client.py`: согласованное задачей
  усиление существующего B adapter — warning содержит тип ошибки, а не её текст,
  который мог бы содержать credentials. Публичные сигнатуры/поведение SDK не менялись.
- `src/java_docs_rag/observability/receipt.py`: отдельная explicit read-back проверка
  уже известных IDs, не вызывается обычным CLI и не создаёт новые traces.
- `tests/unit/test_safe_tracing.py`, `tests/unit/test_receipt.py`, расширенный
  `tests/unit/test_cli.py`; opt-in `tests/integration/test_langfuse_receipt.py`.
- README и `docs/indexing.md`: флаги, privacy defaults, безопасный пример без input text.

### Схема и privacy defaults

Флаг отсутствует → NoOpTracer, Langfuse SDK не импортируется/не вызывается,
environment `LANGFUSE_ENABLED=true` само по себе не включает integrated tracing.
Флаг присутствует → нужны локальные `LANGFUSE_HOST`, `LANGFUSE_PUBLIC_KEY`,
`LANGFUSE_SECRET_KEY`. Пользователь настроил их в Windows user environment;
в существующий agent process они были прочитаны локально без вывода значений.
Ключи не записывались в файлы проекта/отчёт/Git.

| Span | Передаваемые counts/outcomes |
| --- | --- |
| index.command | documents_total/indexed/skipped/deleted, chunks, published, outcome |
| indexing.run | documents, documents_total/indexed/skipped/deleted, chunks |
| indexing.embed_and_upsert | chunks, batches, embed_seconds, upsert_seconds |
| ask.command / rag.run | raw_count, source_count, insufficient_context, outcome |
| rag.retrieve | top_k, raw_count |
| rag.similarity_filter | eligible_count |
| rag.rerank | top_n, ranked_count |
| rag.evidence_selection | unique_count, source_count, selected_chars, max_context_chars |
| rag.generate | provider_called, source_count, insufficient_context, outcome |

Каждый span получает `duration_ms` и фиксированный outcome (`ok`, `error`,
`abstained`; `skipped` зарезервирован). Не передаются question/document text,
prompts, answers, IDs/URL документов, произвольные metadata или credentials.
Inputs всегда None, outputs и metadata фильтруются по allowlist числовых/boolean
значений и фиксированных outcomes. Не применяется скрытое хеширование контента.

Business exceptions распространяются исходными объектами, но SDK context закрывается
без exception/traceback: в telemetry остаётся только `outcome=error`. Failures
telemetry enter/update/close/flush не меняют business result. Сообщения SDK adapter
не содержат текст исключения. Это privacy слой integrated routes; прямой доступ
к сырому SDK с произвольными аргументами остаётся ответственностью вызывающего кода.

CLI `telemetry.trace_ids` — correlation IDs, `receipt_verified=false` даже после
flush: flush/ID не являются доказательством видимости на сервере. `errors` считает
исключения, пойманные privacy wrapper; внутренние best-effort ошибки exporter
могут логироваться SDK отдельно. Доказательство получения — только read-back.

### Фактические offline checks

После первоначальной реализации: 305 passed, 1 skipped. После receipt verifier:
**315 passed, 2 skipped in 3.48s**, 317 collected; оба skips — opt-in live тесты.
Ruff lint: **All checks passed!**; формат: **103 files already formatted**.
Это финальный полный offline запуск после исправления диагностики receipt;
`git diff --check` прошёл, whitespace diagnostics новых файлов = 0.
Unit-тесты не использовали keys, сеть, Qdrant, Ollama или Langfuse.

Проверены все trace stages, durations, counts, abstention/error outcomes,
исключение sentinel text/credentials из metadata/input/output/logs, сохранение
исходного business exception и fail-open поведение при start/update/close/flush.
Receipt tests покрывают delayed v2 data без лишнего legacy fallback, v1 для
несовместимого сервера, Retry-After, HTTP 401/403/429/500, transport errors,
неожиданные I/O fields и запрет unfiltered reads при некорректных IDs.

### Live команды и receipt

Для прогона создана отдельная тестовая коллекция.
Выполнены настоящие integrated `index --trace` (локальный JavaDoc fixture) и
`ask --trace` (`What does List.size() return?`, qwen2.5:1.5b). Обе команды дали
exit 0; ответ имел непустые sources и `insufficient_context=false`.
Начальный opt-in pytest закончился **1 failed in 78.76s** на receipt API HTTP 429,
а не на индексации/ответе. Тест и результат не выданы за pass.

Verifier исправлен: v1 fallback используется только при 404/405/410 от v2,
а не при ещё пустой v2 выдаче; Retry-After соблюдается в пределах deadline.
Проверены **те же IDs**, без повторного создания traces. Первая bounded проверка
подтвердила index, но QA stages не были видны за 180 секунд. После ingestion delay
следующая read-only проверка получила HTTP 200 для обоих trace IDs и полный набор
stage names с корректными durations/outcomes и пустыми inputs.

| Команда | Проверка сервером |
| --- | --- |
| index --trace | Доставка подтверждена, видны все 3 ожидаемых этапа |
| ask --trace | Доставка подтверждена, видны все 7 ожидаемых этапов |

Endpoint подтверждения: `GET /api/public/v2/observations?traceId=<id>`
с Basic Auth из локальных environment keys. Старый SDK/server compatibility
может давать задержку видимости; новые IDs/flush сами по себе не подтверждают receipt.
См. [официальные сведения о read APIs](https://langfuse.com/docs/api-and-data-platform/features/public-api).
Default timeout нового opt-in теста — 600 s (override `LANGFUSE_RECEIPT_TIMEOUT`);
каждая операция ожидания короткая, API polling ограничен deadline.

Проверенные реальные значения: index.command ~27011.62 ms, 1 document/2 chunks,
published=true; retrieval ~688.51 ms, raw_count=2; rerank ~337.50 ms, ranked_count=2;
evidence selection ~0.27 ms, source_count=2, selected_chars=1206; generation
~36416.90 ms, provider_called=true, insufficient_context=false. Root outcomes — ok.
API вернул также совместимое дублирующее представление root; verifier проверяет
уникальные expected names и не суммирует дубли как дополнительные business calls.

Локальный отчёт в каталоге experiments/reports содержит результат проверки
receipt_verified=true и статусы HTTP для обеих трасс.

Повторяемая команда с уже настроенным process environment и кэшем:

```powershell
$env:HF_HOME = Join-Path (Get-Location) "models/huggingface"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:OLLAMA_MODEL = "qwen2.5:1.5b"
$env:JAVA_DOCS_RAG_LANGFUSE_LIVE = "1"
.\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider -s tests/integration/test_langfuse_receipt.py
```

Флаги обычных команд: `... cli index --html <fixture> --source-url <URL> --trace`
и `... cli ask "What does List.size() return?" --collection <alias> --ollama-model
qwen2.5:1.5b --trace`. Test aliases/builds оставлены для review, прежние коллекции
не удалялись. Branch/HEAD/stash сохранены; commit/push не выполнялись.

## Checklist финального PR

- [ ] Проверить общий diff A + integration, включая untracked тесты/fixture/reports.
- [ ] Подтвердить неизменность HEAD/ветки/stash и отсутствие случайного staging/секретов.
- [ ] Запустить full pytest, оба Ruff, whitespace check для tracked и новых файлов.
- [ ] Сверить CLI defaults и configs/models.yaml; повторить model/chunking на index/ask.
- [ ] Проверить non-destructive subset indexing, explicit collection overrides и ID round trip.
- [ ] Повторить failed replacement regression; проверить ADR 0002, legacy migration,
  single-writer policy, retained build names и неопределённость alias timeout.
- [ ] Подтвердить статус ADR A/B и согласовать обновления shared docs/config отдельно от API.
- [ ] Приложить фактический live JSON или оставить явные blockers; не смешивать с offline counts.
- [ ] Только после единого review пользователь выполняет один финальный commit; stash пока сохранять.
