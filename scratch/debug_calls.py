import ast as _ast
import re

def _extract_called_functions(code: str) -> list[str]:
    print(f"DEBUG: Parsing code snippet (len {len(code)})")
    try:
        tree = _ast.parse(code)
        calls = []
        for node in _ast.walk(tree):
            if isinstance(node, _ast.Call):
                if isinstance(node.func, _ast.Name):
                    calls.append(node.func.id)
                elif isinstance(node.func, _ast.Attribute):
                    calls.append(node.func.attr)
        print(f"DEBUG: AST found {len(calls)} calls")
        return list(set(calls))
    except Exception as e:
        print(f"DEBUG: AST failed: {e}. Falling back to regex.")
        res = list(set(re.findall(r"\b([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", code)))
        print(f"DEBUG: Regex found {len(res)} calls")
        return res

test_code = """
// File: test.py
def my_func():
    result = _content_hash(doc)
    return result
"""

print(f"RESULT: {_extract_called_functions(test_code)}")
