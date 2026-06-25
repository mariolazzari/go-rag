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

### Vector store interface

```go
// This file defines the storage abstraction we'll use to talk to the database.
//
// Concrete backends (pgvector for now, possibly weaviate/qdrant/etc. later) live
// in subpackages so callers depend only on the interface.
//
// What is a vector store, conceptually?
//
// A vector store is a database that indexes high-dimensional float32
// vectors (the embeddings produced by an LLM embedder) and answers
// "give me the K rows whose vector is closest to this one" in
// sub-linear time. That nearest-neighbor search is the lookup half of
// RAG.
//
// You could implement Store with a flat in-memory slice for a few
// thousand chunks — just compute cosine distance against every row.
// That works and is sometimes the right answer for a course demo.
// The pgvector backend in this project uses an HNSW index instead, so
// the same code scales to millions of chunks without changing.
//
// Three things every backend has to handle:
//
//	dimension match  —  the query vector and stored vectors must be the
//	                    same length, set at ingest time.
//	distance metric  —  cosine, dot product, or Euclidean. We use cosine,
//	                    which is the right default for embedding models.
//	top-K ranking    —  return the closest K rows, plus a similarity
//	                    score the caller can filter or display.
package vector

import "context"

// Document is a single ingestible unit — typically a chunk of a larger
// source file. Embedding is populated by an llm.Embedder before the
// document reaches the store.
type Document struct {
	// ID is a stable identifier the store uses for upsert/delete. A
	// good default is "<source-path>#<chunk-index>".
	ID string

	// Content is the text that was embedded. Stored verbatim so it can
	// be returned to a RAG prompt assembler at query time.
	Content string

	// Metadata is arbitrary structured data associated with the chunk
	// (source filename, page number, ingest timestamp, ...). Backends
	// are expected to round-trip it without inspection.
	Metadata map[string]string

	// Embedding is the vector representation of Content. All documents
	// in a single store must share the same dimension.
	Embedding []float32
}

// Result is one hit from a similarity query.
type Result struct {
	Document

	// Score is the similarity between the query vector and the stored
	// vector. Higher is more similar; the exact metric (cosine,
	// inner-product, ...) depends on the backend's index configuration.
	//
	// Cosine similarity (what pgvector returns here) is in [-1, 1] for
	// arbitrary vectors but in [0, 1] for the normalized vectors that
	// almost every modern embedding model produces. A useful rule of
	// thumb for OpenAI/Nomic-style embeddings:
	//
	//	> 0.80   strongly related
	//	  0.60-0.80  related
	//	  0.40-0.60  weakly related
	//	  < 0.40   probably noise
	//
	// These thresholds are not universal; they shift with the
	// embedding model and the corpus.
	Score float32
}

// Store is the contract every vector backend implements. Methods take a
// context so callers can enforce timeouts and cancellation, which is
// especially important for ingest pipelines.
type Store interface {
	// Upsert inserts new documents or replaces existing ones by ID.
	// Implementations should perform this in a single transaction where
	// the backend supports it.
	Upsert(ctx context.Context, docs []Document) error

	// Query returns the topK documents most similar to the supplied
	// embedding. The embedding's dimension must match the store's
	// configured dimension; mismatches must surface as an error rather
	// than silent truncation.
	Query(ctx context.Context, embedding []float32, topK int) ([]Result, error)

	// Delete removes documents by ID. Missing IDs are not an error.
	Delete(ctx context.Context, ids []string) error

	// DeleteBySource removes every document whose "source" metadata
	// equals source. Used by the ingest pipeline to clear stale
	// chunks before re-upserting an edited file — without it, a file
	// re-ingested with fewer chunks than before would leave the
	// trailing chunks orphaned in the store.
	//
	// A source with no matching rows is not an error.
	DeleteBySource(ctx context.Context, source string) error

	// Close releases any underlying resources (DB pools, network
	// connections). Calling Close on an already-closed Store is a
	// no-op.
	Close() error
}
```

## Adding documents 

### Embedder

```go
// This file adds embedding support to
// the llm package — a prerequisite for the ingest pipeline, which
// must turn chunk text into vectors before upserting them into the
// store.
package llm

import (
	"context"
	"fmt"

	"github.com/openai/openai-go/v3"
)

// Embedder turns text into dense vector representations
// suitable for similarity search. Implementations must be safe to call
// concurrently.
//
// What's an embedding, in 30 seconds:
//
//   - You hand the model a string.
//   - It hands you back an array of floats — typically 384, 768, 1536,
//     or 3072 numbers. That array is a point in a high-dimensional
//     vector space.
//   - Strings with similar MEANING land near each other in that space;
//     unrelated strings land far apart. "vampires hate garlic" and
//     "vampire weaknesses" will have a small cosine distance; "vampires
//     hate garlic" and "the price of tea in China" will have a large one.
//   - That spatial proximity is what makes retrieval work: embed the
//     user's question, find the chunks whose embeddings are closest to
//     it, hand those chunks to the chat model.
//
// Two practical constraints:
//
//  1. The dimension is fixed per model. text-embedding-3-small produces
//     1536 floats; nomic-embed-text produces 768. The vector store's
//     column type is locked to one number — switching models means
//     re-ingesting everything.
//  2. You must use the SAME embedding model for queries that you used
//     for ingest. Different models live in different vector spaces and
//     their coordinates are not interchangeable.
type Embedder interface {
	// Embed returns one vector per input string, in the same order. All
	// returned vectors share the same dimension. Implementations should
	// batch requests internally if the backend supports it.
	Embed(ctx context.Context, texts []string) ([][]float32, error)
}

// Embed implements Embedder against an OpenAI-compatible
// /embeddings endpoint. The model is taken from Config.EmbeddingModel;
// the URL and credentials come from whichever constructor built this
// client (see New vs NewEmbedder in client.go).
//
// We always send the array form (OfArrayOfStrings) even when there is
// only one input; the server returns one embedding per input. Batching
// matters for ingest performance — one round trip embeds a whole
// document's worth of chunks instead of N.
//
// Two SDK quirks worth knowing:
//
//   - The API spec allows the server to return embeddings in arbitrary
//     order, so we index into the result by d.Index rather than trust
//     positional matching.
//   - The SDK decodes embeddings as []float64 (matching the JSON wire
//     format), but pgvector and the rest of this codebase work in
//     float32 to halve memory and bandwidth. We narrow at this boundary;
//     embedding values are well within float32 range so the cast is
//     lossless in practice.
func (c *Client) Embed(ctx context.Context, texts []string) ([][]float32, error) {
	if len(texts) == 0 {
		return nil, nil
	}

	resp, err := c.sdk.Embeddings.New(ctx, openai.EmbeddingNewParams{
		Model: c.cfg.Model,
		Input: openai.EmbeddingNewParamsInputUnion{OfArrayOfStrings: texts},
	})
	if err != nil {
		return nil, err
	}

	if len(resp.Data) != len(texts) {
		return nil, fmt.Errorf("embeddings: expected %d vectors; got %d", len(texts), len(resp.Data))
	}

	vecs := make([][]float32, len(texts))
	for _, d := range resp.Data {
		idx := int(d.Index)
		if idx < 0 || idx >= len(vecs) {
			return nil, fmt.Errorf("embeddings: index %d out of range", idx)
		}
		vec := make([]float32, len(d.Embedding))
		for i, f := range d.Embedding {
			vec[i] = float32(f)
		}
		vecs[idx] = vec
	}

	return vecs, nil
}
```
