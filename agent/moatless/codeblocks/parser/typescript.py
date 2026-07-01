from tree_sitter import Language

from moatless.codeblocks.parser.parser import CodeParser


class TypeScriptParser(CodeParser):
    def __init__(self, **kwargs):
        try:
            import tree_sitter_typescript as tstypescript
        except ImportError as e:
            raise ImportError(
                "TypeScript parser requires `tree-sitter-typescript`. Install it via `pip install tree-sitter-typescript`."
            ) from e

        super().__init__(Language(tstypescript.language_typescript()), **kwargs)
        self.queries = []
        self.queries.extend(self._build_queries("typescript.scm"))
        self.gpt_queries = []

    @property
    def language(self):
        return "typescript"


class TSXParser(CodeParser):
    def __init__(self, **kwargs):
        try:
            import tree_sitter_typescript as tstypescript
        except ImportError as e:
            raise ImportError(
                "TSX parser requires `tree-sitter-typescript`. Install it via `pip install tree-sitter-typescript`."
            ) from e

        super().__init__(Language(tstypescript.language_tsx()), **kwargs)
        self.queries = []
        self.queries.extend(self._build_queries("typescript.scm"))
        self.gpt_queries = []

    @property
    def language(self):
        return "tsx"
