# Unsloth fine-tune: market-researcher

`unsloth_market_researcher.ipynb` fine-tunes **Qwen2.5-3B-Instruct** with LoRA. The training data is
Hermes function-calling (for tool use) plus finance-alpaca (for domain knowledge). It then exports a Q4_K_M GGUF.

1. Open the notebook in Colab with a **T4 GPU** and run all cells (~45-60 min).
2. Download the `.gguf` file it produces.
3. Register it with Ollama:
   ```bash
   ./make_ollama_model.sh ~/Downloads/<file>.gguf market-researcher
   ```
4. The research agent uses `--model market-researcher` by default.

To try more or fewer steps, or to change the data mix, edit `max_steps`, `N_TOOL` and `N_FINANCE` in the notebook.
