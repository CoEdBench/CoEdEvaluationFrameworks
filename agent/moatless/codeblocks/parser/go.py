from tree_sitter import Language

from moatless.codeblocks.parser.parser import CodeParser


class GoParser(CodeParser):
    def __init__(self, **kwargs):
        try:
            import tree_sitter_go as tsgo
        except ImportError as e:
            raise ImportError(
                "Go parser requires `tree-sitter-go`. Install it via `pip install tree-sitter-go`."
            ) from e

        super().__init__(Language(tsgo.language()), **kwargs)
        self.queries = []
        self.queries.extend(self._build_queries("go.scm"))
        self.gpt_queries = []

    @property
    def language(self):
        return "go"
