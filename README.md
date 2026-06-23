# Learn Retrieval-Augmented Generation by building one in plain Go

## Introduction

### Introduction

- RAG: retrieval augmented generation
  - embedding model
  - vector store
  - retriever
  - storer
- RAG extends LLNs

## Getting started

### Go app setup

[OpenAI for Go](https://github.com/openai/openai-go)

```sh
go mod init github.com/mariolazzari/go-rag
go mod tidy
```

### Cloud models

[Ollama](https://ollama.com/)
[Pricing](https://ollama.com/pricing)

### Setting up model

```sh
curl -fsSL https://ollama.com/install.sh | sh
ollama list
ollama pull gemma3
```

## Vector store

### System prompt

```sh
./prompts/system-custom.md
```

### Vector store in Postgres

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE documents (
    id text PRIMARY KEY,
    content text NOT NULL,
    metadata jsonb NOT NULL DEFAULT '{}'::jsonb,
    embedding vector(768) NOT NULL,
    created_at timestamp with time zone NOT NULL DEFAULT now()
);

CREATE INDEX documents_embedding_idx ON documents USING hnsw (embedding vector_cosine_ops);
```

```yaml
services:
  postgres:
    # Official image: Postgres 18 with the pgvector extension preinstalled.
    image: pgvector/pgvector:pg18
    container_name: rag-course-postgres
    restart: unless-stopped
    environment:
      POSTGRES_USER: rag
      POSTGRES_PASSWORD: rag
      POSTGRES_DB: rag
    ports:
      - "5432:5432"
    volumes:
      # Postgres 18+ stores data in a major-version subdirectory and
      # expects the mount at /var/lib/postgresql (the parent), not at
      # /var/lib/postgresql/data as in PG 17 and earlier.
      - pgdata:/var/lib/postgresql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U rag -d rag"]
      interval: 5s
      timeout: 3s
      retries: 10

volumes:
  pgdata:
```
