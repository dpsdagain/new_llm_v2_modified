"""Diagnostic: Clear Chat must reset the sentinel cooldown.

Pre-fix: _sentinel_cooldown lived in build_rag_chain's closure. After
Clear Chat the turn counter restarted at 0 but last_turn was stuck at
N (where the sentinel last fired). The trigger condition
(turn_count - last_turn) >= SENTINEL_INTERVAL stayed False until the
new conversation overshot N, making the summarizer go silent for many
turns after every Clear Chat.

Post-fix: app.py Clear Chat sets st.session_state.rag_chain = None,
which forces build_rag_chain() on the next turn — the new closure has
a fresh _sentinel_cooldown = {"last_turn": 0}.

This script proves the closure-isolation behavior end-to-end without
needing Streamlit: build the chain twice, mutate the cooldown the
first time, build a second chain, verify they don't share state.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# We want to introspect the closure cells of _full_context_cache_chain.
# Easiest: build_rag_chain returns a RunnableLambda; the closure is on
# the wrapped function, accessible via .__closure__ and .func.__closure__.

from langchain_core.runnables import RunnableLambda


def get_cooldown_cell(rag_chain):
    """Walk the RunnableLambda → wrapped fn → closure to find _sentinel_cooldown."""
    fn = rag_chain.func if hasattr(rag_chain, "func") else rag_chain
    if fn.__closure__ is None:
        return None
    names = fn.__code__.co_freevars
    for name, cell in zip(names, fn.__closure__):
        if name == "_sentinel_cooldown":
            return cell.cell_contents
    return None


def main():
    # Stub: avoid loading real Chroma by passing None and trapping early.
    # build_rag_chain dereferences `db` lazily so we can pass a sentinel
    # and bail out if it tries to use it.
    from rag_chain import build_rag_chain

    print("=== Sentinel cooldown closure isolation ===")
    chain1 = build_rag_chain(db=None, model="claude-sonnet-4-6")
    cd1 = get_cooldown_cell(chain1)
    print(f"chain1 cooldown initial: {cd1}")
    assert cd1 == {"last_turn": 0}

    # Simulate the sentinel firing on turn 7 of the first conversation.
    cd1["last_turn"] = 7
    print(f"chain1 cooldown after firing: {cd1}")

    # Simulate Clear Chat: app.py sets st.session_state.rag_chain = None
    # The next chat-input call will rebuild via build_rag_chain.
    chain2 = build_rag_chain(db=None, model="claude-sonnet-4-6")
    cd2 = get_cooldown_cell(chain2)
    print(f"chain2 cooldown initial: {cd2}")

    if cd2 == {"last_turn": 0} and cd1 is not cd2:
        print()
        print("VERDICT: PASS. New chain has a fresh cooldown closure.")
        print("  After Clear Chat clears st.session_state.rag_chain,")
        print("  the rebuilt chain starts at last_turn=0 — sentinel can")
        print("  fire on the very first qualifying turn of the next chat.")
    else:
        print()
        print(f"VERDICT: FAIL. cd2={cd2}, same-cell={cd1 is cd2}")


if __name__ == "__main__":
    main()
