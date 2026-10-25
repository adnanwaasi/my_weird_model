"""One place to create every chat model the project uses.

Models are named as "provider:model" specs, e.g.
    ollama:market-researcher        (your Unsloth fine-tune)
    ollama:qwen2.5:3b
    anthropic:claude-opus-5-5

Each *role* in the agent (researcher, writer, ...) has a default spec, which you can
override with an environment variable (RESEARCHER_LLM=ollama:qwen2.5:3b) or a CLI flag.

Adding a provider later (OpenAI, Groq, Gemini, vLLM, ...) = write one builder function
and add it to PROVIDERS. Nothing else in the codebase changes.
"""

import os

from langchain_core.language_models import BaseChatModel

# ---------------------------------------------------------------- roles

ROLE_DEFAULTS = {
    # cheap, local, high-volume: planning queries and choosing tool calls
    "researcher": "ollama:market-researcher",
    # strongest model: turns evidence into the final cited report
    "writer": "anthropic:claude-opus-5-5",
}


def spec_for(role: str) -> str:
    """Spec for a role: env var <ROLE>_LLM wins, else the default above."""
    return os.environ.get(f"{role.upper()}_LLM", ROLE_DEFAULTS[role])


# ---------------------------------------------------------------- providers


def _ollama(model: str, **kw) -> BaseChatModel:
    from langchain_ollama import ChatOllama

    return ChatOllama(
        model=model,
        base_url=kw.pop("base_url", os.environ.get("OLLAMA_HOST", "http://localhost:11434")),
        num_ctx=kw.pop("num_ctx", 16384),
        temperature=kw.pop("temperature", 0.3),
        **kw,
    )


def _anthropic(model: str, **kw) -> BaseChatModel:
    from langchain_anthropic import ChatAnthropic

    # Credentials: ANTHROPIC_API_KEY env var.
    return ChatAnthropic(
        model=model,
        max_tokens=kw.pop("max_tokens", 64000),
        # Claude Opus 5.5 always thinks; effort controls how hard. Its default is "medium".
        # "fallbacks": if a safety classifier declines, the API retries on a recommended model.
        betas=kw.pop("betas", ["server-side-fallback-2026-07-01"]),
        model_kwargs={
            "output_config": {"effort": kw.pop("effort", "high")},
            "fallbacks": "default",
            **kw.pop("model_kwargs", {}),
        },
        **kw,
    )


PROVIDERS = {
    "ollama": _ollama,
    "anthropic": _anthropic,
}


# ---------------------------------------------------------------- entry point


def get_llm(spec_or_role: str, **kwargs) -> BaseChatModel:
    """Build a chat model from a role name ("writer") or a spec ("ollama:qwen2.5:3b").

    Extra kwargs are passed to the provider's LangChain class (temperature, num_ctx, ...).
    """
    spec = spec_for(spec_or_role) if spec_or_role in ROLE_DEFAULTS else spec_or_role
    provider, sep, model = spec.partition(":")  # split on the first ":" only; ollama tags keep theirs
    if not sep or not model:
        raise ValueError(f"LLM spec must look like 'provider:model', got {spec!r}")
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider {provider!r}; available: {sorted(PROVIDERS)}")
    return PROVIDERS[provider](model, **kwargs)


if __name__ == "__main__":
    # Quick smoke test: python get_llm.py ollama:qwen2.5:3b
    import sys

    llm = get_llm(sys.argv[1] if len(sys.argv) > 1 else "researcher")
    print(llm.invoke("Say hi in five words.").content)
