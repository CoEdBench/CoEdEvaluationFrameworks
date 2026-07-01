from moatless.codeblocks.parser.java import JavaParser
from moatless.codeblocks.parser.parser import CodeParser
from moatless.codeblocks.parser.python import PythonParser


TS_LIKE_EXTS = {".ts", ".tsx", ".mts", ".cts"}
JS_LIKE_EXTS = {".js", ".jsx", ".mjs", ".cjs"}
GO_LIKE_EXTS = {".go"}
C_LIKE_EXTS = {".c", ".h"}
CPP_LIKE_EXTS = {".cc", ".cpp", ".cxx", ".c++", ".hh", ".hpp", ".hxx", ".h++"}
RUST_LIKE_EXTS = {".rs"}


def is_supported(language: str) -> bool:
    return language and language in [
        "python",
        "java",
        "go",
        "c",
        "cpp",
        "c++",
        "rust",
        "typescript",
        "tsx",
        "javascript",
        "jsx",
        "js",
    ]


def create_parser_by_ext(ext: str, **kwargs) -> CodeParser | None:
    ext = (ext or "").lower()

    if ext == ".py":
        return PythonParser(**kwargs)
    elif ext == ".java":
        return JavaParser(**kwargs)
    elif ext in GO_LIKE_EXTS:
        from moatless.codeblocks.parser.go import GoParser

        return GoParser(**kwargs)
    elif ext in C_LIKE_EXTS:
        from moatless.codeblocks.parser.c import CParser

        return CParser(**kwargs)
    elif ext in CPP_LIKE_EXTS:
        from moatless.codeblocks.parser.cpp import CppParser

        return CppParser(**kwargs)
    elif ext in RUST_LIKE_EXTS:
        from moatless.codeblocks.parser.rust import RustParser

        return RustParser(**kwargs)
    elif ext == ".tsx":
        from moatless.codeblocks.parser.typescript import TSXParser

        return TSXParser(**kwargs)
    elif ext in TS_LIKE_EXTS:
        from moatless.codeblocks.parser.typescript import TypeScriptParser

        return TypeScriptParser(**kwargs)
    elif ext in JS_LIKE_EXTS:
        from moatless.codeblocks.parser.javascript import JavaScriptParser

        return JavaScriptParser(**kwargs)

    raise NotImplementedError(f"Extension {ext} is not supported.")


def create_parser(language: str, **kwargs) -> CodeParser | None:
    if language == "python":
        return PythonParser(**kwargs)
    elif language == "java":
        return JavaParser(**kwargs)
    elif language == "go":
        from moatless.codeblocks.parser.go import GoParser

        return GoParser(**kwargs)
    elif language == "c":
        from moatless.codeblocks.parser.c import CParser

        return CParser(**kwargs)
    elif language in ["cpp", "c++"]:
        from moatless.codeblocks.parser.cpp import CppParser

        return CppParser(**kwargs)
    elif language == "rust":
        from moatless.codeblocks.parser.rust import RustParser

        return RustParser(**kwargs)
    elif language == "typescript":
        from moatless.codeblocks.parser.typescript import TypeScriptParser

        return TypeScriptParser(**kwargs)
    elif language == "tsx":
        from moatless.codeblocks.parser.typescript import TSXParser

        return TSXParser(**kwargs)
    elif language in ["javascript", "jsx", "js"]:
        from moatless.codeblocks.parser.javascript import JavaScriptParser

        return JavaScriptParser(**kwargs)

    raise NotImplementedError(f"Language {language} is not supported.")
