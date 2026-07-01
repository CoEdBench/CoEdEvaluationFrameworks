"""
Minimal tree-sitter based parsers for languages without custom .scm queries.

These parsers create a basic tree-sitter AST.  All nodes are typed as ``CODE``
since no language-specific query files are loaded, but the resulting module
still provides line-number-based block lookups — sufficient to eliminate
"Could not find module" warnings in ReadFile/ViewCode for non-Python/Java
files used in the FIM pipeline.
"""

import tree_sitter_go as tsgo
import tree_sitter_javascript as tsjs
import tree_sitter_typescript as tsts
from tree_sitter import Language

from moatless.codeblocks.parser.parser import CodeParser


class GoParser(CodeParser):
    def __init__(self, **kwargs):
        language = Language(tsgo.language())
        super().__init__(language, **kwargs)
        self.queries = []

    @property
    def language(self):
        return "go"


class TypeScriptParser(CodeParser):
    def __init__(self, **kwargs):
        language = Language(tsts.language_typescript())
        super().__init__(language, **kwargs)
        self.queries = []

    @property
    def language(self):
        return "typescript"


class TsxParser(CodeParser):
    def __init__(self, **kwargs):
        language = Language(tsts.language_tsx())
        super().__init__(language, **kwargs)
        self.queries = []

    @property
    def language(self):
        return "tsx"


class JavaScriptParser(CodeParser):
    def __init__(self, **kwargs):
        language = Language(tsjs.language())
        super().__init__(language, **kwargs)
        self.queries = []

    @property
    def language(self):
        return "javascript"
