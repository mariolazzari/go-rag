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
```
