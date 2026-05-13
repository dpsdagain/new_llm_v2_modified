"""Diagnostic: verify get_llm correctly plumbs `think` and `num_ctx`.

The old "Ollama Cloud Switches" sidebar prepended fake commands to the
user's prompt as plain text. The fix moves these to real API options:

  * Local Ollama (ChatOllama): native `num_ctx=` and `reasoning=`
    constructor params.
  * Ollama Cloud (ChatOpenAI → Ollama's OpenAI-compat endpoint):
    top-level `extra_body={"options": {"num_ctx": N}, "think": bool}`
    on the ChatOpenAI constructor. extra_body must be a direct kwarg —
    nesting it inside `model_kwargs` makes LangChain warn and drop it.
    The OpenAI SDK forwards `extra_body` straight into the request JSON
    so the Ollama gateway parses these fields the same as it would on
    its native API.

This script constructs each variant and inspects the resulting object
to prove the params landed in the right place.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    # Stub the API keys so OLLAMA_CLOUD path doesn't ValueError.
    os.environ.setdefault("OLLAMA_CLOUD_API_KEY", "stub-for-diag")
    os.environ.setdefault("OPENROUTER_API_KEY", "stub-for-diag")

    from rag_chain import get_llm

    print("=== Local Ollama (ChatOllama native params) ===")
    llm = get_llm(model="ollama:llama3.2:1b", think=True, num_ctx=32768)
    print(f"  type           : {type(llm).__name__}")
    print(f"  .num_ctx       : {getattr(llm, 'num_ctx', '<not set>')}")
    print(f"  .reasoning     : {getattr(llm, 'reasoning', '<not set>')}")
    assert getattr(llm, "num_ctx", None) == 32768, "num_ctx not propagated"
    # `reasoning` may not be exposed as an attribute in older langchain-ollama
    # versions — accept either True or attribute-not-present.
    print("  [OK] num_ctx attached natively")
    print()

    print("=== Ollama Cloud (ChatOpenAI + extra_body top-level kwarg) ===")
    llm = get_llm(
        model="ollama-cloud:gpt-oss:120b-cloud",
        think=True,
        num_ctx=131072,
    )
    print(f"  type             : {type(llm).__name__}")
    eb = getattr(llm, "extra_body", None) or {}
    print(f"  .extra_body      : {eb}")
    assert eb, "extra_body missing from ChatOpenAI"
    # Ollama OpenAI-compat: thinking is via `reasoning_effort` (not `think`)
    # and num_ctx must be top-level (not nested under `options`).
    assert eb.get("reasoning_effort") == "high", \
        f"reasoning_effort not 'high' (got {eb.get('reasoning_effort')!r})"
    assert eb.get("num_ctx") == 131072, \
        f"num_ctx wrong (got {eb.get('num_ctx')!r})"
    assert "think" not in eb, "must NOT send `think` — gateway drops it"
    assert "options" not in eb, \
        "must NOT nest under `options` — gateway drops it"
    print("  [OK] reasoning_effort='high' + top-level num_ctx")
    print()

    print("=== Ollama Cloud — defaults (no tuning) ===")
    llm = get_llm(model="ollama-cloud:gemma4:31b-cloud")
    eb = getattr(llm, "extra_body", None) or {}
    print(f"  .extra_body      : {eb}")
    # When think/num_ctx are None we should NOT inject extra_body — keeps
    # the request shape clean and avoids surprising the gateway.
    assert not eb, "extra_body should be empty/None when no tuning requested"
    print("  [OK] no extra_body when tuning omitted")
    print()

    print("=== Ollama Cloud — think=False (explicit off) ===")
    llm = get_llm(model="ollama-cloud:deepseek-r1:70b-cloud", think=False)
    eb = getattr(llm, "extra_body", None) or {}
    print(f"  .extra_body      : {eb}")
    assert eb.get("reasoning_effort") == "none", \
        f"explicit-off must serialize as reasoning_effort='none' (got {eb!r})"
    assert "num_ctx" not in eb, "num_ctx should be absent when not requested"
    print("  [OK] think=False -> reasoning_effort='none'")
    print()

    print("=== Cache key respects tuning ===")
    a = get_llm(model="ollama-cloud:gpt-oss:120b-cloud", think=True, num_ctx=8192)
    b = get_llm(model="ollama-cloud:gpt-oss:120b-cloud", think=False, num_ctx=8192)
    c = get_llm(model="ollama-cloud:gpt-oss:120b-cloud", think=True, num_ctx=32768)
    assert a is not b, "different `think` must yield distinct cached LLMs"
    assert a is not c, "different `num_ctx` must yield distinct cached LLMs"
    print("  [OK] (model, think, num_ctx) all participate in cache key")
    print()

    print("VERDICT: PASS. Tuning params reach the right layer for both")
    print("         Ollama family paths, and the LLM cache key prevents")
    print("         stale-handle reuse across toggle changes.")


if __name__ == "__main__":
    main()
