import logging
import os
import re

from moatless.codeblocks.codeblocks import CodeBlock, CodeBlockType
from moatless.codeblocks.parser.create import create_parser
from moatless.codeblocks.parser.java import JavaParser
from moatless.codeblocks.parser.parser import CodeParser
from moatless.codeblocks.parser.python import PythonParser

logger = logging.getLogger(__name__)


_JS_LIKE_EXTS = {".js", ".jsx", ".mjs", ".cjs"}
_TS_LIKE_EXTS = {".ts", ".tsx", ".mts", ".cts"}
_GO_LIKE_EXTS = {".go"}
_C_LIKE_EXTS = {".c", ".h"}
_CPP_LIKE_EXTS = {".cc", ".cpp", ".cxx", ".c++", ".hh", ".hpp", ".hxx", ".h++"}
_RUST_LIKE_EXTS = {".rs"}
_SUPPORTED_EXTS = (
    {".py", ".java"}
    | _GO_LIKE_EXTS
    | _TS_LIKE_EXTS
    | _JS_LIKE_EXTS
    | _C_LIKE_EXTS
    | _CPP_LIKE_EXTS
    | _RUST_LIKE_EXTS
)


def _normalize_parser_path(file_path: str) -> str:
    """
    Normalize LLM-produced file paths before parser dispatch.
    Handles whitespace, quotes, and common line suffixes like ':123' or '#L123'.
    """
    if not file_path:
        return ""

    path = file_path.strip().strip("'\"")
    path = path.split("#", 1)[0]

    # Strip trailing :line[:column] suffixes, e.g. foo.go:12 or foo.ts:10:5
    match = re.match(r"^(.*\.[A-Za-z0-9]+):\d+(?::\d+)?$", path)
    if match:
        path = match.group(1)

    return path


def _get_ext(file_path: str) -> str:
    normalized = _normalize_parser_path(file_path).lower()
    return os.path.splitext(normalized)[1]


def supports_codeblocks(path: str):
    return _get_ext(path) in _SUPPORTED_EXTS


def get_parser_by_path(file_path: str) -> CodeParser | None:
    try:
        ext = _get_ext(file_path)

        if ext == ".py":
            return PythonParser()
        elif ext == ".java":
            return JavaParser()
        elif ext in _GO_LIKE_EXTS:
            from moatless.codeblocks.parser.go import GoParser

            return GoParser()
        elif ext in _C_LIKE_EXTS:
            from moatless.codeblocks.parser.c import CParser

            return CParser()
        elif ext in _CPP_LIKE_EXTS:
            from moatless.codeblocks.parser.cpp import CppParser

            return CppParser()
        elif ext in _RUST_LIKE_EXTS:
            from moatless.codeblocks.parser.rust import RustParser

            return RustParser()
        elif ext == ".tsx":
            from moatless.codeblocks.parser.typescript import TSXParser

            return TSXParser()
        elif ext in _TS_LIKE_EXTS:
            from moatless.codeblocks.parser.typescript import TypeScriptParser

            return TypeScriptParser()
        elif ext in _JS_LIKE_EXTS:
            from moatless.codeblocks.parser.javascript import JavaScriptParser

            return JavaScriptParser()
        else:
            return None
    except Exception as e:
        logger.warning("Parser init failed for %s: %s", file_path, e)
        return None
