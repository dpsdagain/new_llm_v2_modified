# Project Instructions

## Debugging Protocol: Prove Before You Fix

When diagnosing a bug or test failure in a multi-stage pipeline:

1. **NEVER propose a fix based only on code reading.**
   Write a diagnostic script first that traces the ACTUAL data through
   each pipeline stage. Print what goes in and what comes out at every
   step. The script is the diagnosis — the code is just a theory.

2. **Verify every assumption with a concrete test.**
   Before writing any fix, state your assumptions as testable claims:
     - "BM25 will return config.py for this query" → run the query, print results
     - "ChromaDB filter will match this metadata" → run the filter, print results
     - "This function runs during ingestion" → add a print, ingest, check output
   If you can't prove the assumption in <10 lines of Python, you don't
   understand the system well enough to fix it.

3. **Trace bottom-up, not top-down.**
   Start from the failing output and walk BACKWARD through the pipeline:
     - What was in the final context? (print it)
     - What survived reranking? (print it)
     - What entered the candidate pool? (print it)
     - What did BM25 return? (print it)
     - What was ingested? (print it)
   The first stage where the expected data disappears is the bug.
   Everything downstream is a symptom.

4. **One diagnostic script before any code change.**
   For every failed test, write a standalone script that reproduces
   the failure by querying the actual pipeline with the actual test
   input. This script becomes the regression test for the fix.
