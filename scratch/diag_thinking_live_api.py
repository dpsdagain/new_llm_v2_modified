"""Live-API smoke test for the Thinking toggle.

This is the ONLY way to prove `reasoning_effort: high` actually
produces a thinking trace from Ollama Cloud. The earlier plumbing
diagnostic only showed the request shape was correct — it can't tell
you whether the gateway consumed the field.

Requires: OLLAMA_CLOUD_API_KEY in your .env (the same key the app uses).

Picks a known reasoning model (gpt-oss:120b) and asks one question
twice — once with thinking off, once with thinking on. Then prints
both responses + their `additional_kwargs` and `response_metadata`.

What "working" looks like:
  - With thinking ON, the response either contains a <|channel>thought
    block in .content OR populates .additional_kwargs["reasoning_content"]
    / .response_metadata["reasoning"].
  - With thinking OFF, neither artifact is present.
  - If both responses look identical, the gateway is dropping the
    reasoning_effort field — either an Ollama Cloud version bug or
    your account tier doesn't enable thinking for this model.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load .env so the API key is available.
from dotenv import load_dotenv
load_dotenv(override=True)

if not os.getenv("OLLAMA_CLOUD_API_KEY"):
    print("ERROR: OLLAMA_CLOUD_API_KEY not in environment. Aborting.")
    sys.exit(1)

from rag_chain import get_llm
from langchain_core.messages import HumanMessage

# A reasoning-class model. Swap if your account doesn't have access.
MODEL = "ollama-cloud:gpt-oss:120b-cloud"
QUESTION = (
    "I have 3 apples. I give half to a friend, then eat 1. Then I buy "
    "twice as many as I have left. How many apples do I have? Show your "
    "work step by step."
)


def call(label: str, *, think):
    print(f"\n=== {label} (think={think}) ===")
    llm = get_llm(model=MODEL, think=think, streaming=False)
    print(f"  request extra_body: {getattr(llm, 'extra_body', None)}")
    try:
        resp = llm.invoke([HumanMessage(content=QUESTION)])
    except Exception as e:
        print(f"  ERROR: {type(e).__name__}: {e}")
        return None

    content = resp.content if isinstance(resp.content, str) else str(resp.content)
    addl = getattr(resp, "additional_kwargs", {}) or {}
    meta = getattr(resp, "response_metadata", {}) or {}

    # Heuristics for "thinking output is present".
    has_channel_marker = "<|channel" in content or "<think" in content.lower()
    has_reasoning_kw = any(
        k in addl for k in ("reasoning_content", "reasoning", "thought")
    ) or any(k in meta for k in ("reasoning", "reasoning_content"))

    print(f"  content length     : {len(content)} chars")
    print(f"  has channel marker : {has_channel_marker}")
    print(f"  has reasoning kwarg: {has_reasoning_kw}")
    if addl:
        print(f"  additional_kwargs  : {json.dumps({k: str(v)[:120] for k, v in addl.items()}, indent=2)}")
    if meta:
        meta_preview = {k: (str(v)[:120] if not isinstance(v, dict) else list(v.keys())) for k, v in meta.items()}
        print(f"  response_metadata  : {json.dumps(meta_preview, indent=2)}")
    print(f"  content preview    : {content[:300]!r}")
    return {"content": content, "addl": addl, "meta": meta,
            "has_thinking": has_channel_marker or has_reasoning_kw}


def main():
    off = call("THINKING OFF", think=False)
    on  = call("THINKING ON",  think=True)
    if off is None or on is None:
        print("\nVERDICT: live API call failed — check API key / model access.")
        return

    print("\n=== VERDICT ===")
    if on["has_thinking"] and not off["has_thinking"]:
        print("PASS. reasoning_effort=high produced a thinking trace,")
        print("      reasoning_effort=none suppressed it.")
        print("      Toggle works end-to-end on this model.")
    elif on["has_thinking"] and off["has_thinking"]:
        print("PARTIAL. Both responses contain reasoning artifacts —")
        print("      the model emits thinking by default and the toggle")
        print("      may not be needed (or 'none' isn't suppressing it).")
    elif not on["has_thinking"] and not off["has_thinking"]:
        print("FAIL. Neither response had a thinking trace. Either:")
        print("      (a) the gateway is dropping reasoning_effort,")
        print("      (b) this model doesn't support thinking,")
        print("      (c) your account tier doesn't enable it.")
    else:
        print("WEIRD. OFF has thinking, ON does not — investigate.")


if __name__ == "__main__":
    main()
