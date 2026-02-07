"""
AST-aware code tokenization using tree-sitter.

Provides semantic tokenization that respects code structure rather than treating
code as plain text. Extracts meaningful units like function names, keywords,
operators, and preserves syntactic boundaries.

Why: Traditional BPE tokenization splits code at arbitrary byte boundaries, breaking
identifiers mid-name ("my_function" → ["my_", "func", "tion"]) and losing structural
information like function scopes and control flow. AST-aware tokenization respects
semantic boundaries, enabling better code understanding for tasks like code completion,
refactoring, and documentation generation. Tree-sitter provides fast, incremental,
error-tolerant parsing for multiple languages with a unified API.

This is critical for the embedding-prediction paradigm where the model operates in
continuous embedding space - we want embeddings that correspond to semantic units
(functions, classes, statements) not arbitrary substrings.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

try:
    from tree_sitter import Language, Node, Parser
    from tree_sitter_python import language as python_language
    from tree_sitter_rust import language as rust_language

    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    Language = Any
    Parser = Any
    Node = Any


class CodeLanguage(Enum):
    """Supported programming languages for AST tokenization.

    Why: Explicit enumeration ensures type safety and provides clear interface
    for language detection. Each language requires its own tree-sitter parser
    but shares the same tokenization strategy (keywords, identifiers, operators).
    """

    PYTHON = "python"
    RUST = "rust"
    UNKNOWN = "unknown"


class TokenType(Enum):
    """Token types for structured code representation.

    Why: Categorizing tokens by syntactic role (keyword vs identifier vs operator)
    enables the model to learn different embedding patterns for different token types.
    For example, keywords like 'def' and 'class' have fixed semantics, while identifiers
    are user-defined and need context-dependent embeddings. This structured representation
    is essential for code understanding tasks.
    """

    KEYWORD = "keyword"  # def, class, fn, struct, if, while, etc.
    IDENTIFIER = "identifier"  # Variable/function/class names
    OPERATOR = "operator"  # +, -, *, /, =, ==, etc.
    LITERAL = "literal"  # String, number, boolean literals
    COMMENT = "comment"  # Single-line and multi-line comments
    PUNCTUATION = "punctuation"  # ( ) { } [ ] , ; :
    INDENT = "indent"  # Indentation (Python-specific)
    DEDENT = "dedent"  # De-indentation (Python-specific)
    NEWLINE = "newline"  # Line breaks
    WHITESPACE = "whitespace"  # Other whitespace


@dataclass
class CodeToken:
    """A single token from AST-aware tokenization.

    Why: Wrapping tokens in a dataclass provides type safety and makes the code
    more maintainable. Storing both the token type and text enables different
    encoding strategies - keywords map to dedicated vocab IDs, identifiers use
    BPE subtokens, etc. Position tracking (optional) enables span-based operations
    like code navigation and refactoring.

    Attributes:
        type: The syntactic category of this token
        text: The actual source text for this token
        start_byte: Start position in source (optional, for debugging)
        end_byte: End position in source (optional, for debugging)
    """

    type: TokenType
    text: str
    start_byte: int | None = None
    end_byte: int | None = None


class ASTTokenizer:
    """AST-aware tokenizer using tree-sitter for semantic code tokenization.

    Parses source code into an abstract syntax tree and extracts semantic tokens
    that respect language structure. Supports multiple languages through tree-sitter's
    unified interface.

    Why: AST tokenization provides semantic boundaries that align with how developers
    think about code (functions, classes, statements) rather than arbitrary byte chunks.
    This improves model understanding for code-related tasks. Tree-sitter's error
    recovery handles incomplete/malformed code gracefully, essential for real-world
    code completion where partial code is common. Fast incremental parsing (<1ms for
    typical edits) enables interactive applications.

    Example:
        ```python
        tokenizer = ASTTokenizer()
        tokens = tokenizer.tokenize("def foo(): return 42", CodeLanguage.PYTHON)
        # Returns: [
        #     CodeToken(type=KEYWORD, text="def"),
        #     CodeToken(type=IDENTIFIER, text="foo"),
        #     CodeToken(type=PUNCTUATION, text="("),
        #     ...
        # ]
        ```
    """

    # Python keywords (from Python 3.12)
    PYTHON_KEYWORDS = {
        "False",
        "None",
        "True",
        "and",
        "as",
        "assert",
        "async",
        "await",
        "break",
        "class",
        "continue",
        "def",
        "del",
        "elif",
        "else",
        "except",
        "finally",
        "for",
        "from",
        "global",
        "if",
        "import",
        "in",
        "is",
        "lambda",
        "nonlocal",
        "not",
        "or",
        "pass",
        "raise",
        "return",
        "try",
        "while",
        "with",
        "yield",
    }

    # Rust keywords (from Rust 1.75)
    RUST_KEYWORDS = {
        "as",
        "async",
        "await",
        "break",
        "const",
        "continue",
        "crate",
        "dyn",
        "else",
        "enum",
        "extern",
        "false",
        "fn",
        "for",
        "if",
        "impl",
        "in",
        "let",
        "loop",
        "match",
        "mod",
        "move",
        "mut",
        "pub",
        "ref",
        "return",
        "self",
        "Self",
        "static",
        "struct",
        "super",
        "trait",
        "true",
        "type",
        "unsafe",
        "use",
        "where",
        "while",
    }

    # Tree-sitter node types for operators
    OPERATOR_TYPES = {
        "binary_operator",
        "unary_operator",
        "comparison_operator",
        "boolean_operator",
        "assignment_operator",
        "+",
        "-",
        "*",
        "/",
        "%",
        "=",
        "==",
        "!=",
        "<",
        ">",
        "<=",
        ">=",
        "and",
        "or",
        "not",
        "&",
        "|",
        "^",
        "~",
        "<<",
        ">>",
    }

    # Tree-sitter node types for punctuation
    PUNCTUATION_TYPES = {"(", ")", "{", "}", "[", "]", ",", ";", ":", ".", "->", "=>"}

    def __init__(self) -> None:
        """Initialize AST tokenizer with tree-sitter parsers.

        Why: We initialize parsers for all supported languages at construction time
        rather than lazily to fail fast if tree-sitter bindings are missing. This
        makes debugging easier - ImportError at construction is clearer than random
        failures during tokenization. The parsers are lightweight (~1KB each) so
        pre-initialization doesn't impact memory.

        Raises:
            ImportError: If tree-sitter or language bindings are not installed
        """
        if not TREE_SITTER_AVAILABLE:
            raise ImportError(
                "tree-sitter is not available. Install with: "
                "uv pip install tree-sitter tree-sitter-python tree-sitter-rust"
            )

        # Initialize parsers for supported languages
        self.parsers: dict[CodeLanguage, Parser] = {}

        # Python parser (tree-sitter 0.20+ uses Language in Parser constructor)
        self.parsers[CodeLanguage.PYTHON] = Parser(Language(python_language()))

        # Rust parser
        self.parsers[CodeLanguage.RUST] = Parser(Language(rust_language()))

    def detect_language(self, code: str, file_extension: str | None = None) -> CodeLanguage:
        """Detect programming language from code or file extension.

        Args:
            code: Source code string
            file_extension: Optional file extension (e.g., ".py", ".rs")

        Returns:
            Detected language or UNKNOWN if unsupported

        Why: Automatic language detection enables seamless handling of code from
        mixed repositories. Extension-based detection is fast and reliable for most
        cases. We could add heuristic-based detection (looking for language-specific
        patterns) but that adds complexity with marginal benefit - in practice, code
        datasets include file paths with extensions.
        """
        if file_extension:
            ext = file_extension.lower()
            if ext in (".py", ".pyi", ".pyw"):
                return CodeLanguage.PYTHON
            elif ext in (".rs",):
                return CodeLanguage.RUST

        # Heuristic-based detection (simple patterns)
        # This is a fallback when no extension is provided
        if "def " in code or "import " in code or "class " in code:
            return CodeLanguage.PYTHON
        elif "fn " in code or "struct " in code or "impl " in code:
            return CodeLanguage.RUST

        return CodeLanguage.UNKNOWN

    def tokenize(
        self, code: str, language: CodeLanguage | None = None, file_extension: str | None = None
    ) -> list[CodeToken]:
        """Tokenize source code using AST-aware parsing.

        Args:
            code: Source code string to tokenize
            language: Programming language (auto-detected if None)
            file_extension: Optional file extension for language detection

        Returns:
            List of semantic code tokens

        Why: Returns CodeToken objects rather than raw strings to preserve semantic
        information (token type, position). This enables different encoding strategies
        downstream - keywords get dedicated vocab IDs, identifiers use BPE, etc.
        Auto-detection makes the API convenient for most use cases while explicit
        language parameter allows override when detection is ambiguous.

        Note: Falls back to simple whitespace tokenization if language is unsupported
        or parsing fails. This ensures robustness for malformed code or unsupported
        languages rather than failing completely.
        """
        # Detect language if not provided
        if language is None:
            language = self.detect_language(code, file_extension)

        # Fallback to simple tokenization for unsupported languages
        if language == CodeLanguage.UNKNOWN or language not in self.parsers:
            return self._fallback_tokenize(code)

        try:
            # Parse code with tree-sitter
            parser = self.parsers[language]
            tree = parser.parse(bytes(code, "utf8"))
            root_node = tree.root_node

            # Extract tokens from AST
            tokens = self._extract_tokens(root_node, code, language)
            return tokens

        except Exception:
            # Fallback on any parsing error (malformed code, tree-sitter issues)
            return self._fallback_tokenize(code)

    def _extract_tokens(
        self, node: "Node", source_code: str, language: CodeLanguage
    ) -> list[CodeToken]:
        """Extract semantic tokens from tree-sitter AST.

        Args:
            node: Root node of the AST
            source_code: Original source code
            language: Programming language being parsed

        Returns:
            List of semantic tokens in source order

        Why: Recursive tree traversal extracts tokens in source order while preserving
        semantic boundaries. We visit leaf nodes (terminals) to extract actual text
        while using parent node types to determine token categories (keyword vs identifier).
        This approach handles nested structures correctly (e.g., function calls within
        function calls) and preserves all source text including comments and whitespace.
        """
        tokens: list[CodeToken] = []
        keywords = self._get_keywords(language)

        def visit(n: "Node") -> None:
            """Recursively visit AST nodes to extract tokens.

            Why: Inner function avoids passing language/keywords on every recursive call.
            Tree-sitter nodes have type (e.g., 'function_definition') and text (source span).
            Leaf nodes represent actual tokens, internal nodes represent syntactic structure.
            """
            # Skip nodes with no source text
            if n.start_byte >= n.end_byte:
                return

            # Get node text
            node_text = source_code[n.start_byte : n.end_byte]

            # Leaf nodes are actual tokens
            if n.child_count == 0:
                token_type = self._classify_token(n, node_text, keywords)
                tokens.append(
                    CodeToken(
                        type=token_type,
                        text=node_text,
                        start_byte=n.start_byte,
                        end_byte=n.end_byte,
                    )
                )
            else:
                # Internal nodes: recursively process children
                for child in n.children:
                    visit(child)

        visit(node)
        return tokens

    def _classify_token(self, node: "Node", text: str, keywords: set[str]) -> TokenType:
        """Classify a token based on its tree-sitter node type and text.

        Args:
            node: Tree-sitter AST node
            text: Token text from source
            keywords: Set of language keywords

        Returns:
            Token type classification

        Why: Tree-sitter node types (e.g., 'identifier', 'string') provide semantic
        information but need refinement - an identifier might actually be a keyword
        depending on the language. We cross-reference with keyword sets to correctly
        classify tokens. This ensures 'def' is tagged as KEYWORD not IDENTIFIER.
        """
        node_type = node.type

        # Check if it's a keyword (language-specific)
        if text in keywords:
            return TokenType.KEYWORD

        # Comment
        if "comment" in node_type:
            return TokenType.COMMENT

        # String or number literal
        if node_type in ("string", "number", "integer", "float", "boolean"):
            return TokenType.LITERAL
        if node_type.endswith("_literal"):
            return TokenType.LITERAL

        # Operators
        if text in self.OPERATOR_TYPES or node_type in self.OPERATOR_TYPES:
            return TokenType.OPERATOR

        # Punctuation
        if text in self.PUNCTUATION_TYPES or node_type in self.PUNCTUATION_TYPES:
            return TokenType.PUNCTUATION

        # Whitespace
        if text.isspace():
            if text == "\n":
                return TokenType.NEWLINE
            return TokenType.WHITESPACE

        # Identifier (default for named entities)
        if node_type in ("identifier", "type_identifier", "field_identifier"):
            return TokenType.IDENTIFIER

        # Default: treat as identifier
        return TokenType.IDENTIFIER

    def _get_keywords(self, language: CodeLanguage) -> set[str]:
        """Get keyword set for a specific language.

        Args:
            language: Programming language

        Returns:
            Set of language keywords

        Why: Centralizes keyword sets to avoid scattering them across methods.
        Makes it easy to add new languages - just add a new keyword set here.
        """
        if language == CodeLanguage.PYTHON:
            return self.PYTHON_KEYWORDS
        elif language == CodeLanguage.RUST:
            return self.RUST_KEYWORDS
        return set()

    def _fallback_tokenize(self, code: str) -> list[CodeToken]:
        """Fallback tokenization using simple whitespace splitting.

        Args:
            code: Source code string

        Returns:
            List of tokens from simple whitespace tokenization

        Why: When tree-sitter is unavailable or parsing fails (malformed code,
        unsupported language), we fall back to basic whitespace tokenization.
        This ensures robustness - the system degrades gracefully rather than
        failing completely. For unsupported languages (JavaScript, Go, C++),
        this provides basic functionality until we add more tree-sitter parsers.

        Note: This loses semantic structure but preserves token boundaries better
        than treating code as plain text would. Better than nothing for error recovery.
        """
        tokens: list[CodeToken] = []
        current_pos = 0

        for line in code.split("\n"):
            # Tokenize by whitespace
            for word in line.split():
                tokens.append(
                    CodeToken(
                        type=TokenType.IDENTIFIER,  # Default type
                        text=word,
                        start_byte=current_pos,
                        end_byte=current_pos + len(word),
                    )
                )
                current_pos += len(word) + 1  # +1 for space

            # Add newline token
            if current_pos < len(code):
                tokens.append(
                    CodeToken(
                        type=TokenType.NEWLINE,
                        text="\n",
                        start_byte=current_pos,
                        end_byte=current_pos + 1,
                    )
                )
                current_pos += 1

        return tokens
