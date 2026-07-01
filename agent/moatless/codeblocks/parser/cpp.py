from tree_sitter import Language

from moatless.codeblocks.parser.parser import CodeParser


class CppParser(CodeParser):
    def __init__(self, **kwargs):
        try:
            import tree_sitter_cpp as tscpp
        except ImportError as e:
            raise ImportError(
                "C++ parser requires `tree-sitter-cpp`. Install it via `pip install tree-sitter-cpp`."
            ) from e

        super().__init__(Language(tscpp.language()), **kwargs)
        self.queries = []
        self.queries.extend(self._build_queries("cpp.scm"))
        self.gpt_queries = []

    @property
    def language(self):
        return "cpp"
