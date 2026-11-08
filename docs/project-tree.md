# Дерево проекта

Полный перечень 77 файлов исходного каркаса. Все перечисленные файлы созданы
в рамках Bootstrap. Существующие `.idea/` и `.venv/` сохранены; их содержимое,
служебный `.git/`, caches и генерируемые editable-install metadata не показаны.
`.gitkeep` сохраняют пустые каталоги в Git.

```text
java-docs-rag/
├── .env.example
├── .github/
│   └── workflows/
│       └── tests.yml
├── .gitignore
├── AGENTS.md
├── CLAUDE.md
├── configs/
│   ├── default.yaml
│   ├── experiments/
│   │   └── .gitkeep
│   └── sources.yaml
├── data/
│   ├── evaluation/
│   │   └── .gitkeep
│   ├── manifests/
│   │   └── .gitkeep
│   ├── processed/
│   │   └── .gitkeep
│   └── raw/
│       └── .gitkeep
├── docker-compose.yml
├── docs/
│   ├── architecture.md
│   ├── decisions/
│   │   └── .gitkeep
│   ├── project-tree.md
│   └── setup.md
├── experiments/
│   ├── reports/
│   │   └── .gitkeep
│   ├── results/
│   │   └── .gitkeep
│   └── runners/
│       └── .gitkeep
├── pyproject.toml
├── README.md
├── src/
│   └── java_docs_rag/
│       ├── __init__.py
│       ├── cli.py
│       ├── common/
│       │   ├── __init__.py
│       │   ├── config.py
│       │   ├── exceptions.py
│       │   ├── interfaces.py
│       │   └── models.py
│       ├── embeddings/
│       │   ├── __init__.py
│       │   ├── factory.py
│       │   └── sentence_transformer.py
│       ├── evaluation/
│       │   ├── __init__.py
│       │   ├── answer_metrics.py
│       │   ├── dataset.py
│       │   ├── retrieval_metrics.py
│       │   └── runner.py
│       ├── filtering/
│       │   ├── __init__.py
│       │   ├── context_filter.py
│       │   ├── deduplication.py
│       │   └── similarity.py
│       ├── generation/
│       │   ├── __init__.py
│       │   ├── context_builder.py
│       │   ├── generator.py
│       │   ├── ollama_provider.py
│       │   └── prompt_builder.py
│       ├── grabber/
│       │   ├── __init__.py
│       │   ├── crawler.py
│       │   ├── downloader.py
│       │   ├── manifest.py
│       │   └── repository.py
│       ├── observability/
│       │   ├── __init__.py
│       │   └── langfuse_client.py
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── indexing_pipeline.py
│       │   └── rag_pipeline.py
│       ├── preprocessing/
│       │   ├── __init__.py
│       │   ├── chunking/
│       │   │   ├── __init__.py
│       │   │   ├── base.py
│       │   │   ├── fixed.py
│       │   │   ├── overlap.py
│       │   │   ├── paragraph.py
│       │   │   └── section.py
│       │   ├── cleaner.py
│       │   ├── normalizer.py
│       │   └── parser.py
│       ├── reranking/
│       │   ├── __init__.py
│       │   └── cross_encoder.py
│       ├── retrieval/
│       │   ├── __init__.py
│       │   ├── query_processor.py
│       │   └── retriever.py
│       └── vectorstore/
│           ├── __init__.py
│           ├── indexer.py
│           └── qdrant_store.py
└── tests/
    ├── fixtures/
    │   └── .gitkeep
    ├── integration/
    │   └── .gitkeep
    └── unit/
        └── test_models.py
```
