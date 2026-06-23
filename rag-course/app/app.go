package app

// Package app wires the pieces of lesson 1 together: the LLM client
// and the chat REPL. Keeping orchestration here means cmd/rag/main.go
// stays a thin shim that's easy to read.
//
// The shape of the program in lesson 1, viewed from this file:
//
//	llm.Client      talks HTTP to the chat-completions endpoint
//	chat.RunREPL    foreground: terminal chat loop
//
// Later lessons will grow this file: a vector store will be opened
// here, a background ingest watcher will be spawned, an optional web
// server will start alongside the REPL. The wiring point is always
// the same — Run takes a Config, builds the components, and starts
// the foreground loop.
import (
	"context"
	"rag-course/chat"
	"rag-course/config"
	"rag-course/llm"
)

// Run is the program's main loop. In lesson 1 there is only the
// foreground REPL, so Run constructs the LLM client and hands it
// straight to chat.RunREPL.
func Run(ctx context.Context, cfg config.Config) error {
	client := llm.New(cfg)
	return chat.RunREPL(ctx, client, chat.Options{
		SystemPromptFile: cfg.SystemPromptFile,
	})
}