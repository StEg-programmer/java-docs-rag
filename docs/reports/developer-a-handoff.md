# Передача реализации Developer A

Дата: 2026-11-11. Ветка: `feature/a-01-semantic-retriever`.
Изменения оставлены в рабочем дереве; commit, push и merge не выполнялись.

## Состояние и границы

Перед работой прочитаны `AGENTS.md`, README, `docs/architecture.md`, контракты,
конфигурация, исходники и имеющиеся unit-тесты; проверены ветка, status и diff.
Сохранены ранее реализованные `SemanticRetriever`/`VectorRetriever`, `QueryProcessor`,
`CrossEncoderReranker` и их тесты. В рамках этой задачи они не редактировались.
Новые изменения относятся к filtering, generation, evaluation, `RAGPipeline`,
их тестам, синтетическому fixture и этому отчёту.

Расхождения с исходным планом и документацией:

- README и architecture описывают исходный каркас и уже не отражают реализацию A.
  Общие документы, CLI, config и контракты не менялись: их обновление требует
  согласования A/B. Этот явно запрошенный отчёт фиксирует текущее поведение A.
- В `docs/decisions` есть только `.gitkeep`; утверждённых ADR и существующего
  формата evaluation dataset не найдено. Поэтому использован JSON ниже.
  Описанные здесь правила бюджета, отказа и метрик реализованы по текущей задаче;
  перенос в совместные ADR остаётся интеграционным пунктом, не утверждением
  о согласии Developer B.
- В доступной ветке embedder, Qdrant store, indexer, parser, chunker и indexing
  pipeline Developer B остаются заглушками. Рабочую индексацию и живые ответы
  с ними подтвердить сейчас нельзя. Эти файлы не изменялись.

## 1. Фильтры

Файлы реализации:

- `src/java_docs_rag/filtering/similarity.py`
- `src/java_docs_rag/filtering/deduplication.py`
- `src/java_docs_rag/filtering/context_filter.py`

`SimilarityFilter.filter(results, threshold)` оставляет результаты с
`similarity_score >= threshold`, включая точное равенство. `None` отключает cutoff.
Порядок сохраняется; `rerank_score` на cutoff не влияет. Некорректные типы порога,
bool, NaN и infinity отвергаются. Порог не предполагает диапазон [0, 1].

`DeduplicationFilter` оставляет первое вхождение каждого `chunk_id` после reranking.
Два разных чанка одного документа допустимы. Никакой повторной сортировки нет.

`ContextFilter.filter(results, max_chars)` выбирает целые чанки в порядке ранга.
Бюджет — сумма `len(chunk.text)` выбранных чанков: Unicode code points Python,
включая все пробелы, переводы строк и табуляцию. Бюджет не считает JSON-экранирование,
заголовки, citation labels, URL, вопрос, system prompt и разделители. Это бюджет
текста доказательств, а не размера сериализованного prompt, байтов или токенов.
Не помещающийся чанк пропускается, следующие рассматриваются. Ровно помещающийся
принимается. Допустим целый бюджет >= 0; bool/другие типы дают `TypeError`,
отрицательное число — `ValueError`. Нулевой бюджет обычно даёт пустой контекст;
чанки с пустым текстом имеют нулевую стоимость.

Входные последовательности и объекты не изменяются.
Тесты: `tests/unit/test_filtering.py`, 20 passed. Проверены границы порога,
дубликаты, сохранение порядка/идентичности, точное попадание в бюджет,
пропуск большого чанка с выбором последующего, Unicode/whitespace, пустые входы
и ошибки параметров. Общие Ruff-проверки прошли, см. раздел 5.

## 2. Grounded generation

Файлы реализации:

- `src/java_docs_rag/generation/context_builder.py`
- `src/java_docs_rag/generation/prompt_builder.py`
- `src/java_docs_rag/generation/generator.py`
- `src/java_docs_rag/generation/ollama_provider.py`

`ContextBuilder.build(chunks)` формирует JSON-массив в порядке выбранных чанков.
Каждая запись содержит `citation` вида `[S1]`, `chunk_id`, `document_id`,
`source_url`, `title`, `section`, `text`. Labels локальны для конкретного ответа;
они соответствуют индексам `sources`, начиная с 1. JSON сохраняет текст после
десериализации и отделяет его от служебных полей.

`PromptBuilder.build(question, context)` возвращает отдельные system/user строки.
System требует отвечать только по доказательствам, цитировать labels, трактовать
документы/metadata как данные, игнорировать инструкции внутри документов,
не дополнять пробелы внешними знаниями. User содержит вопрос как JSON-строку
и evidence JSON. Это инструкция модели, не доказанная защита от prompt injection.

`AnswerGenerator(provider, context_builder=..., prompt_builder=...)` сохраняет
совместимость с прежним `AnswerGenerator(provider)`. Без результатов возвращает
`insufficient_context=True`, пустые sources и сообщение
`Insufficient context to answer the question.` без вызова `provider.generate()`.
Если провайдер после `strip()` возвращает **ровно** `INSUFFICIENT_CONTEXT`, это
модельный отказ: тот же явный текст и `insufficient_context=True`, но `sources`
содержит переданные модели чанки. Маркер внутри обычной фразы отказом не считается;
свободные формулировки отказа автоматически не распознаются. Blank/non-string
ответ провайдера — `GenerationError`, а не успешный отказ.

Гарантия `sources`: это все доказательства, переданные модели, в порядке labels.
Поле не означает, что модель использовала каждое доказательство, правильно
сослалась на него или что каждый факт ответа проверен. Проверка entailment и
валидности цитат не реализована. При отдельном вызове генератора `question`
равен переданному вопросу, `retrieved_chunks` содержит выбранные результаты;
именно pipeline располагает исходной выдачей и заменяет это поле ею.

`OllamaProvider(model_id, base_url='http://localhost:11434', timeout=60.0,
transport=urlopen)` сохраняет первые два параметра конструктора.
Используется стандартная библиотека `urllib`, без новых зависимостей.
Значения берутся из существующего `GenerationConfig` при сборке приложения:

```python
# generation_config: GenerationConfig; model_id нужно явно выбрать заранее.
provider = OllamaProvider(
    model_id=generation_config.model_id,
    base_url=generation_config.base_url,
    timeout=60.0,
)
```

`model_id=None` или blank отклоняется; это выявляет незаполненную конфигурацию,
а не выбирает/скачивает модель. URL должен быть HTTP(S), timeout — положительным
конечным числом. Импорты и конструктор не выполняют сеть. Только `generate()`
делает POST `/api/chat` с `model`, двумя `messages` (system/user) и `stream=false`.
Формат сверен с [официальным Ollama Chat API](https://docs.ollama.com/api/chat).
Timeout явно передаётся HTTP transport и ограничивает ожидание операций сокета;
это не строгий общий deadline всего выполнения.

Transport/HTTP failures, disconnect, timeout, некорректный UTF-8/JSON,
неполный ответ (`done` не true), error response, отсутствие assistant message,
missing/non-string/blank content дают `GenerationError`. Transport/parse errors
сохраняют причину через exception chaining. Повторов запросов нет.
`transport` можно явно подменить callable с интерфейсом `urlopen(request,
timeout=...)`, возвращающим context manager с `status` и `read()`.

Тесты: `tests/unit/test_generation.py`, 33 passed. Проверены traceability,
сериализация потенциальных инструкций как текста, разделение prompts, источники,
оба пути отказа, маркер в обычной фразе, bad provider content, request JSON,
timeout и ошибки transport/HTTP/response. Все HTTP-вызовы подменены; Ollama
и модели не запускались. Ruff прошёл, см. раздел 5.

## 3. Pipeline ответа

Файл: `src/java_docs_rag/pipeline/rag_pipeline.py`.

```text
question → QueryProcessor.process → prepared query
         → Retriever.retrieve(top_k) → raw retrieval snapshot
         → SimilarityFilter(min_similarity)
         → Reranker.rerank(rerank_top_n)
         → DeduplicationFilter(chunk_id, first)
         → ContextFilter(max_context_chars)
         → AnswerGenerator → RAGAnswer с raw retrieved_chunks
```

Cutoff выполняется до reranking: неподходящие similarity не занимают top_n.
Reranker оценивает все оставшиеся кандидаты и возвращает top_n; дедупликация
выполняется после этого, поэтому число итоговых источников может быть меньше
top_n. Дозаполнение из отброшенных reranker кандидатов не выполняется.

Конструктор `RAGPipeline(retriever, reranker, provider)` сохранён. Дополнительно
можно передать `config: RetrievalConfig`, `max_context_chars`, query processor,
каждый фильтр и `answer_generator`. При отсутствии последнего создаётся
`AnswerGenerator(self.provider)`; явно переданный генератор управляет своим
provider и builders. Внешние адаптеры передаются явно, автоматически не создаются.

Поля pipeline-ответа:

- `question`: подготовленный вопрос после `strip()`; регистр и внутренний текст
  не изменяются. Исходные boundary whitespace не сохраняются.
- `retrieved_chunks`: новая list с исходными SearchResult retrieval до cutoff,
  reranking, дедупликации и бюджета. Порядок, similarity и исходные объекты
  сохранены; rerank scores из дальнейшего этапа сюда не подмешиваются.
- `sources`: выбранные и переданные модели чанки; при отсутствии evidence пусто,
  при модельном отказе evidence сохраняется. Гарантии описаны выше.
- `model`: model_id провайдера генератора, в том числе при отказе без его вызова.

Nested metadata остаётся поверхностно изменяемой по общим dataclass-контрактам;
этапы A её не мутируют. Ошибки адаптеров распространяются, не маскируются отказом.

Действующие defaults:

| Параметр | Значение | Где задан |
| --- | --- | --- |
| top_k | 10 | существующий RetrievalConfig |
| rerank_top_n | 5 | существующий RetrievalConfig |
| min_similarity | None, cutoff выключен | существующий RetrievalConfig |
| max_context_chars | 8000 | новый необязательный аргумент RAGPipeline |
| Ollama base_url | http://localhost:11434 | GenerationConfig / OllamaProvider |
| Ollama model_id | None в config; явная непустая строка в provider | выбор при интеграции |
| Ollama timeout | 60 секунд | новый необязательный аргумент provider |
| EvaluationRunner.k | 10 | новый необязательный аргумент runner |

`max_context_chars` и timeout не добавлялись в shared config/YAML. Их включение
в конфигурационный loader и CLI требует совместного изменения. Loader и CLI
этой ветки всё ещё каркас; использовать Python API с явной сборкой.

Тесты: `tests/unit/test_rag_pipeline.py`, 7 passed. Реальные QueryProcessor,
SemanticRetriever, CrossEncoderReranker, фильтры и генератор работают вместе;
fake — только embedder, store, scorer и LLM provider. Проверены normalized query,
cutoff до rerank slots, дедупликация/бюджет после rerank, raw snapshot, sources,
пустая retrieval, отбрасывание cutoff/бюджетом без вызова LLM, модельный отказ
и распространение EmbeddingError. Ruff прошёл, см. раздел 5.

## 4. Evaluation

Файлы реализации: `src/java_docs_rag/evaluation/dataset.py`,
`retrieval_metrics.py`, `answer_metrics.py`, `runner.py`.
Fixture: `tests/fixtures/evaluation/synthetic.json` (три синтетических вопроса,
включая Unicode, отсутствующий reference и out-of-scope; это не benchmark).

Формат dataset — UTF-8 JSON array, каждая запись имеет ровно поля
`EvaluationQuestion`:

```json
[
  {
    "question_id": "list",
    "question": "What is a List?",
    "expected_answer": "An ordered collection.",
    "relevant_document_ids": ["java-list"],
    "category": "api",
    "answerable": true
  }
]
```

`question_id`, `question`, `category` — непустые строки; `expected_answer` —
непустая строка либо null; relevant IDs — массив непустых строк;
`answerable` — bool. ID вопросов должны быть уникальны. Отсутствующие/лишние
поля, ошибочные типы и malformed JSON/UTF-8 дают ValueError; ошибки файловой
системы распространяются. Пустой массив допустим. Значения не нормализуются
loader-ом. Повторяющиеся relevant IDs допустимы и сворачиваются метриками.

`evaluate_retrieval(results, relevant_document_ids, k)` сначала убирает повторы
**document_id** из всей доступной выдачи, сохраняя первое место, затем берёт k
уникальных документов. Это отличается от взятия k чанков с последующим unique.
Пусть L — полученный список, R — множество relevant IDs, h = |set(L) ∩ R|:

- Precision@k = h / k. Недостающие ранги считаются промахами: при одном найденном
  релевантном документе и k=10 precision равен 0.1, а не 1.
- Recall@k = h / |R|; при пустом R равен 0.
- MRR@k = 1 / rank первого релевантного документа в L; если такого нет, 0.

k — положительный int, bool/нецелые типы запрещены. Пустая выдача или пустое R
дают все нули. Runner передаёт именно raw `answer.retrieved_chunks`, не sources
и не reranked evidence. Если raw top_k чанков содержит мало уникальных документов,
evaluation не делает дополнительные запросы, чтобы восполнить k.

`evaluate_answer` возвращает:

- `abstention_accuracy` для любого вопроса: 1, когда insufficient_context
  соответствует `not answerable`, иначе 0. Проверяется явный флаг, не свободный
  текст ответа; правильность ответа при answerable этим не доказывается.
- `exact_match` только для answerable с expected_answer != null: сравнение после
  casefold и схлопывания Unicode whitespace. Пунктуация и citation labels не
  удаляются. Отказ на answerable reference всегда получает 0, даже если строки
  случайно совпали. Для missing reference/unanswerable ключ отсутствует.

Exact match не измеряет семантическую эквивалентность, полноту, groundedness
или корректность цитат. Парафраз и правильный ответ с добавленным `[S1]` могут
не совпасть с кратким эталоном. Casefold пригоден как простой baseline, но
может скрыть значимые различия регистра Java identifiers.

`EvaluationRunner(pipeline, k=10).run(questions)` вызывает pipeline один раз
для каждого вопроса и вычисляет macro average:

| Группа метрик | Какие вопросы входят |
| --- | --- |
| precision_at_k, recall_at_k, mrr_at_k | answerable и непустые relevant IDs |
| abstention_accuracy | все вопросы |
| exact_match | answerable с ненулевым expected_answer |

Возвращаются средние, `<metric>_count` и `question_count`; counts представлены
float ради сохранения `dict[str, float]`. При нулевом размере группы среднее
равно 0 и count равен 0. У пустого dataset все значения нулевые. Исключения
pipeline не перехватываются: выполнение прерывается, успешный отчёт вместо
сервисной ошибки не возвращается. Category-stratified metrics пока не вычисляются.

Тесты: `tests/unit/test_evaluation.py`, 32 passed. Проверены JSON/Unicode,
schema и duplicate question IDs, пустой dataset, отсутствующий файл, повторные
document IDs, cutoff k, знаменатели, missing reference, unanswerable и неверный
отказ, punctuation exact match, применимость групп, raw retrieval и ошибки runner.
Ruff прошёл, см. раздел 5.

## 5. Фактическая проверка

Проверки выполнены из корня проекта через Python 3.11.9 проектной `.venv`.
Первый запуск завершился ошибкой доступа к базовому интерпретатору Python 3.11.
Это ограничение доступа, не отсутствие интерпретатора: после предоставленного
разрешения проверки успешно выполнены вне sandbox. Окружение не пересоздавалось.

```powershell
.\.venv\Scripts\python.exe -B -m pytest -p no:cacheprovider
.\.venv\Scripts\python.exe -m ruff check --no-cache .
.\.venv\Scripts\python.exe -m ruff format --check --no-cache .
git diff --check
```

Результаты: **179 passed**; Ruff lint — **All checks passed!**;
Ruff format — **68 files already formatted**; `git diff --check` — exit 0.
Ошибок вне scope не обнаружено. Ни одного integration-теста или живого сервиса
эти результаты не подтверждают. Unit-тесты используют fake adapters и mock HTTP.

| Тестовый файл | Passed |
| --- | ---: |
| test_models.py | 16 |
| test_retriever.py | 25 |
| test_query_processor.py | 19 |
| test_cross_encoder.py | 27 |
| test_filtering.py | 20 |
| test_generation.py | 33 |
| test_rag_pipeline.py | 7 |
| test_evaluation.py | 32 |
| Всего | 179 |

Новый `tests/unit/conftest.py` содержит синтетические candidates для новых тестов;
прежние тесты не изменены. Новые JSON/test/report файлы просмотрены отдельно от
tracked diff. Перед передачей проверены итоговый status, diff и whitespace.

## 6. Контракты с Developer B и pending integration

| Пункт | Подтверждено доступным кодом | Осталось проверить/согласовать |
| --- | --- | --- |
| Embedder | `embed_query(str) -> list[float]`, `embed_documents(Sequence[str]) -> list[list[float]]`, read-only model_id/dimension | реальная модель и размерность; одинаковая модель/нормализация для индекса и запроса; порядок batch |
| VectorStore | upsert(chunks, vectors), search(vector, top_k, filters=None) -> list[SearchResult], delete_document(document_id) | рабочая ветка B; payload mapping; round-trip всех chunk fields; delete/upsert |
| Collection | config URL `http://localhost:6333`, collection `java_docs`; Docker image `qdrant/qdrant:v1.12.5`, localhost ports 6333/6334 | реально созданная collection, размерность vectors, distance metric и named vector settings |
| Embedding dimension | defaults `model_id=None`, `dimension=None`; stub принимает явные значения | числовое значение установить по модели B и сверить с длиной каждого вектора и collection; сейчас оно неизвестно |
| Similarity | A сохраняет score store и использует inclusive >=; reranker score — отдельное поле, выше лучше | B должен подтвердить «выше = релевантнее», шкалу/метрику и порядок поиска; distance при необходимости преобразовать на границе store; до калибровки cutoff=None |
| IDs | строковые document_id и chunk_id в frozen dataclass; A не генерирует и не преобразует их | стабильность и уникальность ID, правила версий/URL/content_hash; fixtures ID условные, не production-алгоритм |
| Metadata | dict[str, str], A не мутирует | допустимые ключи фильтров и сохранение metadata в Qdrant |
| Reranker | model_id и injected scorer; один score на пару в исходном порядке, стабильная descending сортировка | локальная модель/адаптер scoring, калибровка; ничего не скачано |
| Ollama | HTTP boundary проверен fake transport | заранее установленная выбранная модель, живой endpoint, реальные ответы/отказы и citations |
| Shared config/CLI/docs | существующие dataclass использованы без изменения | совместная сборка CLI, loader YAML/.env, новые timeout/budget settings, обновление README/architecture и ADR |

Конструкторы не должны делать network/model loading. Для A передаются готовые
Embedder/VectorStore; `SemanticRetriever(embedder, vectorstore)` вызывает только
embed_query и search. Векторные типы синхронные, float lists; batch сохраняет
порядок входных текстов. Индекс и retrieval обязаны использовать согласованные
dimension, model_id и collection. Сейчас реальное соблюдение этого в ветке B
не проверено. Слияние чужой ветки или изменение shared контрактов не выполнялось.

## 7. План практического integration smoke test — НЕ ВЫПОЛНЕН

Предусловия: коллега предоставляет рабочие indexing/chunking/embedder/store;
совместно фиксируются ID, модель/dimension и score semantics. Пользователь
отдельно разрешает запуск Docker/Ollama и, если нужны, загрузки моделей.
Выбрать тестовую collection отдельно от пользовательских данных; provisioning
и cleanup согласовать. Никакого удаления существующей collection по умолчанию.

1. Подготовить маленький локальный Java sample с URL документации List и текстом
   о порядке элементов и доступе по индексу. Можно сохранить минимальный HTML
   и пропустить его через HTMLDocumentParser B; проверить title, URL и content.
   Для первого smoke допустим заранее подготовленный Document с явно тестовыми ID.
2. Передать Document через рабочий `IndexingPipeline(chunker, embedder, store)`.
   Проверить непустые чанки, уникальные chunk_id, общий document_id, URL и текст.
   Проверить `len(vector) == embedder.dimension == collection vector size` и
   корректное соответствие batch chunks/vectors. Затем проверить retrieval
   вопросом `What does a Java List preserve?`: найден именно тестовый документ.
3. Подключить готовый локальный scorer и выбранный OllamaProvider. Собрать A
   pipeline. Следующий код — функция интеграционного сценария, **не команда,
   уже успешно выполненная с текущими заглушками**:

```python
from java_docs_rag.common.config import RetrievalConfig
from java_docs_rag.pipeline.indexing_pipeline import IndexingPipeline
from java_docs_rag.pipeline.rag_pipeline import RAGPipeline
from java_docs_rag.reranking.cross_encoder import CrossEncoderReranker
from java_docs_rag.retrieval.retriever import SemanticRetriever


def smoke(document, chunker, embedder, store, scorer, provider):
    # Все аргументы подготовлены коллегами; модели и сервисы уже разрешены/готовы.
    # store указывает на отдельную согласованную тестовую collection.
    IndexingPipeline(chunker, embedder, store).run([document])
    pipeline = RAGPipeline(
        SemanticRetriever(embedder, store),
        CrossEncoderReranker("agreed-local-reranker", scorer),
        provider,
        config=RetrievalConfig(top_k=10, rerank_top_n=5, min_similarity=None),
        max_context_chars=8000,
    )
    grounded = pipeline.run("  What does a Java List preserve?  ")
    assert not grounded.insufficient_context
    assert grounded.question == "What does a Java List preserve?"
    assert grounded.sources
    assert any(s.document_id == document.document_id for s in grounded.sources)
    assert "[S" in grounded.answer
    assert grounded.retrieved_chunks

    outside = pipeline.run("What will the weather be here tomorrow?")
    assert outside.insufficient_context
    return grounded, outside
```

4. Вручную сверить утверждение о порядке элементов с текстом evidence и убедиться,
   что каждый использованный `[Sx]` существует в `sources`. URL должен вести к
   нужному Java sample. Проверить реально использованные collection/model/dimension.
5. Для out-of-scope вопроса ожидать отказ. При cutoff=None нерелевантные чанки
   могут попасть в prompt; тогда LLM должна вернуть точный marker. Если она
   отвечает из внешних знаний или пишет отказ иными словами, smoke считается
   неуспешным: зафиксировать ответ и согласовать модель/prompt/калибровку cutoff.
   Источники при модельном отказе могут быть непустыми по контракту.
6. Отдельно проверить бездоказательный путь: пустая тестовая выдача или полностью
   отфильтрованная выдача даёт отказ без обращения к LLM. Уже подтверждено unit,
   но при живой сборке проверить счётчиком wrapper-провайдера.
7. Составить небольшой dataset с **реальными** document IDs этой collection,
   answerable и unanswerable вопросами. Запустить
   `EvaluationRunner(pipeline, k=10).run(load_dataset(path))`; сохранить средние
   вместе с counts и параметрами эксперимента. Не использовать условные ID
   синтетического fixture как ID реального индекса.

План smoke не выдаётся за результат проверки. Сейчас подтверждена реализация A
и её unit-поведение; подключение ветки B и live services остаётся pending.
