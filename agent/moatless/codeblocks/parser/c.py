from tree_sitter import Language

from moatless.codeblocks.parser.parser import CodeParser


class CParser(CodeParser):
    def __init__(self, **kwargs):
        try:
            import tree_sitter_c as tsc
        except ImportError as e:
            raise ImportError("C parser requires `tree-sitter-c`. Install it via `pip install tree-sitter-c`.") from e

        super().__init__(Language(tsc.language()), **kwargs)
        self.queries = []
        self.queries.extend(self._build_queries("c.scm"))
        self.gpt_queries = []

    @property
    def language(self):
        return "c"
