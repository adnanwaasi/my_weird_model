#!/usr/bin/env bash
# Register a fine-tuned Qwen2.5 GGUF with Ollama, reusing the official qwen2.5 chat/tool template
# so Ollama can parse <tool_call> output into structured tool calls.
set -euo pipefail

GGUF=${1:?usage: ./make_ollama_model.sh path/to/model.gguf [model-name]}
NAME=${2:-market-researcher}
BASE=qwen2.5:3b

ollama show "$BASE" --template >/dev/null 2>&1 || ollama pull "$BASE"

MODELFILE=$(mktemp)
trap 'rm -f "$MODELFILE"' EXIT
{
  echo "FROM $(realpath "$GGUF")"
  echo "TEMPLATE \"\"\"$(ollama show "$BASE" --template)\"\"\""
  ollama show "$BASE" --parameters | sed '/^\s*$/d; s/^/PARAMETER /'
  echo "PARAMETER num_ctx 16384"
  echo "PARAMETER temperature 0.3"
} > "$MODELFILE"

ollama create "$NAME" -f "$MODELFILE"
echo "Created Ollama model '$NAME'. Try: ollama run $NAME"
