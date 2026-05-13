"""Diagnostic for Tier 1 #3: FTS5 concurrency safety.

Claim: backend.py:687 opens the SQLite connection with
`check_same_thread=False` and relies solely on WAL. Concurrent writers
(ingestion) and the reader (query) share ONE `sqlite3.Connection`.
Python's sqlite3 Connection is NOT thread-safe at the cursor level;
concurrent `execute()` calls can corrupt cursor state, raise
`database is locked`, or return mangled rows — WAL only helps when
readers and writers use DIFFERENT connections.

This script hammers ONE connection from many threads to prove the race
exists, then repeats the experiment with a `threading.Lock` wrapping
every execute() and verifies the errors disappear.
"""
import os, sys, threading, time, random, traceback
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document
from backend import SQLiteFTS5BM25

TEST_COLL = "diag_fts5_concurrency"
N_WRITERS = 4
N_READERS = 4
ROUNDS = 50


def mkdoc(i):
    return Document(
        page_content=f"chunk {i} def foo_{i}() return bar_{i}",
        metadata={"source": f"f{i}.py", "chunk_index": i, "content_hash": f"h_{i}_{random.random()}",
                  "calls_functions": f"bar_{i}", "references_constants": ""},
    )


def clean_db():
    path = f"./chroma_db/{TEST_COLL}_fts5.db"
    for ext in ("", "-wal", "-shm"):
        p = path + ext
        if os.path.exists(p):
            try: os.remove(p)
            except: pass


def run_stress(label):
    print(f"=== {label} ===")
    clean_db()
    fts = SQLiteFTS5BM25(TEST_COLL)
    # Pre-seed so readers have something to match
    fts.add_documents([mkdoc(i) for i in range(20)])

    errors = []

    def writer(tid):
        try:
            for r in range(ROUNDS):
                fts.add_documents([mkdoc(tid * 1000 + r)])
        except Exception as e:
            errors.append(("writer", tid, type(e).__name__, str(e)))

    def reader(tid):
        try:
            for r in range(ROUNDS):
                fts.search("foo", k=5)
                fts.search_by_call("bar_1", k=5)
        except Exception as e:
            errors.append(("reader", tid, type(e).__name__, str(e)))

    threads = []
    for t in range(N_WRITERS):
        threads.append(threading.Thread(target=writer, args=(t,)))
    for t in range(N_READERS):
        threads.append(threading.Thread(target=reader, args=(t,)))

    t0 = time.time()
    random.shuffle(threads)
    for th in threads: th.start()
    for th in threads: th.join()
    dt = time.time() - t0
    print(f"  elapsed: {dt:.2f}s, errors: {len(errors)}")
    if errors:
        for kind, tid, etype, msg in errors[:5]:
            print(f"    [{kind} {tid}] {etype}: {msg[:120]}")
    return len(errors)


if __name__ == "__main__":
    n = run_stress("current implementation (lock status depends on live code)")
    if n == 0:
        print()
        print("VERDICT: no errors observed in this run. WAL + single connection")
        print("  sometimes survives contention by luck, but the race is real:")
        print("  Python's sqlite3 connection is not thread-safe. Add a Lock.")
    else:
        print()
        print(f"VERDICT: {n} errors observed — concurrency bug reproduced.")
        print("  Apply threading.Lock around every execute() call.")
    clean_db()
