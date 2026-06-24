// Package config loads runtime configuration from environment
// variables. In lesson 1 there is exactly one thing to configure: how
// to reach an OpenAI-compatible chat endpoint (base URL, API key, and
// model name).
//
// Why env-driven and not flags or a YAML file? Three reasons:
//
//   - LLM apps usually run in containers or PaaS where env vars are
//     the universal config knob.
//   - Secrets (API keys) belong in env, not on the CLI.
//   - It's the simplest thing that works, and "the simplest thing
//     that works" is the rule for lesson 1.
//
// Later lessons will add embedding settings, a database URL, and a
// system-prompt path. Each one slots in as another field and another
// os.Getenv call — no architectural change required.
package config

import (
	"os"

	"github.com/joho/godotenv"
)

// Config holds settings for the LLM client. SystemPromptFile is here
// from the start because writing one good system prompt is a real
// part of the lesson; the rest of the project's settings will arrive
// in later lessons.
type Config struct {
	// BaseURL points at any OpenAI-compatible chat-completions
	// server: api.openai.com, a local Ollama at :11434/v1, LM
	// Studio, Groq, Together, vLLM, and so on. The wire protocol is
	// the same; only the URL and model name change.
	BaseURL          string

	// APIKey is sent as `Authorization: Bearer <key>` when non-empty.
	// Local servers usually accept any value (or none); hosted
	// providers require their own key.
	APIKey           string

	// Model is the chat-completions model identifier. Defaults to
	// gpt-4o-mini so a fresh OpenAI key works with no further setup.
	Model            string

	// SystemPromptFile is the path to a text/markdown file whose
	// contents become the conversation's system message. A missing
	// file is silently treated as "no system prompt".
	SystemPromptFile string
}

// Load reads configuration from the environment, applying defaults
// for any value left unset. A .env file in the working directory is
// loaded first if present; values already exported in the process
// environment take precedence ("real env wins over .env" — the
// twelve-factor convention).
//
// Supported variables:
//
//	OPENAI_BASE_URL     defaults to https://api.openai.com/v1
//	OPENAI_API_KEY      no default; sent as a bearer token when set
//	OPENAI_MODEL        defaults to gpt-4o-mini
//	SYSTEM_PROMPT_FILE	no default
func Load() Config {
	_ = godotenv.Load()

	cfg := Config {
		BaseURL: os.Getenv("OPENAI_BASE_URL"),
		APIKey: os.Getenv("OPENAI_API_KEY"),
		Model: os.Getenv("OPENAI_MODEL"),
		SystemPromptFile: os.Getenv("SYSTEM_PROMPT_FILE"),
	}

	if cfg.BaseURL == "" {
		cfg.BaseURL = "https://api.openai.com/v1"
	}

	if cfg.Model == "" {
		cfg.Model = "gpt-4o-mini"
	}

	return cfg
}