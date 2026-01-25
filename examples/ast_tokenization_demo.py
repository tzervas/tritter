#!/usr/bin/env python3
"""
Demonstration of AST-aware code tokenization for Tritter.

This script shows how AST tokenization preserves code structure and semantics
compared to plain text tokenization.
"""

from tritter.tokenization import (
    ASTTokenizer,
    CodeLanguage,
    ModalityType,
    MultiModalTokenizer,
)


def main() -> None:
    """Run AST tokenization demonstration."""
    print("=" * 80)
    print("AST-Aware Code Tokenization Demo")
    print("=" * 80)

    # Example Python code
    python_code = """
def factorial(n: int) -> int:
    '''Calculate factorial recursively.'''
    if n <= 1:
        return 1
    return n * factorial(n - 1)

class Calculator:
    def add(self, x: int, y: int) -> int:
        return x + y
"""

    # Example Rust code
    rust_code = """
fn fibonacci(n: u32) -> u32 {
    if n <= 1 {
        n
    } else {
        fibonacci(n - 1) + fibonacci(n - 2)
    }
}

struct Point {
    x: i32,
    y: i32,
}
"""

    # 1. Direct AST tokenization
    print("\n1. AST Tokenizer (Python)")
    print("-" * 80)
    ast_tokenizer = ASTTokenizer()
    python_tokens = ast_tokenizer.tokenize(python_code, CodeLanguage.PYTHON)

    print(f"Total tokens: {len(python_tokens)}")
    print("\nToken breakdown:")
    token_type_counts = {}
    for token in python_tokens:
        token_type_counts[token.type.value] = token_type_counts.get(token.type.value, 0) + 1

    for token_type, count in sorted(token_type_counts.items(), key=lambda x: -x[1]):
        print(f"  {token_type:15s}: {count:3d}")

    print("\nFirst 15 tokens:")
    for i, token in enumerate(python_tokens[:15]):
        print(f"  {i+1:2d}. {token.type.value:12s} | {token.text!r:20s}")

    # 2. AST tokenization for Rust
    print("\n2. AST Tokenizer (Rust)")
    print("-" * 80)
    rust_tokens = ast_tokenizer.tokenize(rust_code, CodeLanguage.RUST)

    print(f"Total tokens: {len(rust_tokens)}")
    print("\nFirst 15 tokens:")
    for i, token in enumerate(rust_tokens[:15]):
        print(f"  {i+1:2d}. {token.type.value:12s} | {token.text!r:20s}")

    # 3. MultiModalTokenizer integration
    print("\n3. MultiModalTokenizer Integration")
    print("-" * 80)
    tokenizer = MultiModalTokenizer()

    simple_code = "def hello(name): return f'Hello, {name}!'"

    # Encode as CODE (uses AST)
    code_tokens = tokenizer.encode(simple_code, ModalityType.CODE, language="python")
    print(f"\nCode: {simple_code}")
    print(f"CODE modality tokens ({len(code_tokens)}): {code_tokens}")

    # Encode as TEXT (uses BPE)
    text_tokens = tokenizer.encode(simple_code, ModalityType.TEXT)
    print(f"TEXT modality tokens ({len(text_tokens)}): {text_tokens}")

    print(f"\nToken count: CODE={len(code_tokens)}, TEXT={len(text_tokens)}")

    # 4. Language detection
    print("\n4. Language Detection")
    print("-" * 80)

    test_snippets = [
        ("def main(): pass", None, "Python (heuristic)"),
        ("fn main() {}", None, "Rust (heuristic)"),
        ("x = 42", ".py", "Python (extension)"),
        ("let x = 42;", ".rs", "Rust (extension)"),
    ]

    for code, ext, description in test_snippets:
        detected = ast_tokenizer.detect_language(code, ext)
        print(f"  {description:25s}: {code:20s} -> {detected.value}")

    # 5. Error handling
    print("\n5. Error Handling (Malformed Code)")
    print("-" * 80)

    malformed_codes = [
        "def incomplete(",
        "fn missing_brace {",
        "// JavaScript (unsupported)\nfunction test() {}",
    ]

    for malformed in malformed_codes:
        try:
            tokens = tokenizer.encode(malformed, ModalityType.CODE)
            print(f"  Handled: {malformed[:30]:30s} -> {len(tokens)} tokens (fallback)")
        except Exception as e:
            print(f"  Error: {malformed[:30]:30s} -> {e}")

    print("\n" + "=" * 80)
    print("Demo complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
