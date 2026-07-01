from tree_sitter import Language

from moatless.codeblocks.parser.parser import CodeParser


class RustParser(CodeParser):
    def __init__(self, **kwargs):
        try:
            import tree_sitter_rust as tsrust
        except ImportError as e:
            raise ImportError(
                "Rust parser requires `tree-sitter-rust`. Install it via `pip install tree-sitter-rust`."
            ) from e

        super().__init__(Language(tsrust.language()), **kwargs)
        self.queries = []
        self.queries.extend(self._build_queries("rust.scm"))
        self.gpt_queries = []

    @property
    def language(self):
        return "rust"
