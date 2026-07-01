from tree_sitter import Language

from moatless.codeblocks.parser.parser import CodeParser


class JavaScriptParser(CodeParser):
    def __init__(self, **kwargs):
        try:
            import tree_sitter_javascript as tsjavascript
        except ImportError as e:
            raise ImportError(
                "JavaScript parser requires `tree-sitter-javascript`. Install it via `pip install tree-sitter-javascript`."
            ) from e

        super().__init__(Language(tsjavascript.language()), **kwargs)
        self.queries = []
        self.queries.extend(self._build_queries("javascript.scm"))
        self.gpt_queries = []

    @property
    def language(self):
        return "javascript"
