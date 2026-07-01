import importlib.util

import pytest

from moatless.codeblocks import get_parser_by_path, supports_codeblocks


def test_supports_codeblocks_includes_c_cpp_javascript_rust():
    assert supports_codeblocks("a.c")
    assert supports_codeblocks("a.h")
    assert supports_codeblocks("a.cpp")
    assert supports_codeblocks("a.hpp")
    assert supports_codeblocks("a.js")
    assert supports_codeblocks("a.jsx")
    assert supports_codeblocks("a.rs")
    assert supports_codeblocks("A.CPP:45")
    assert supports_codeblocks('"src/main.rs#L9"')


@pytest.mark.skipif(importlib.util.find_spec("tree_sitter_c") is None, reason="tree_sitter_c is not installed")
def test_c_parser_can_parse_simple_file():
    parser = get_parser_by_path("main.c")
    assert parser is not None

    module = parser.parse(
        "int add(int a, int b) {\n    return a + b;\n}\n",
        file_path="main.c",
    )

    assert module is not None
    assert module.language == "c"


@pytest.mark.skipif(importlib.util.find_spec("tree_sitter_cpp") is None, reason="tree_sitter_cpp is not installed")
def test_cpp_parser_can_parse_simple_file():
    parser = get_parser_by_path("main.cpp")
    assert parser is not None

    module = parser.parse(
        "int add(int a, int b) {\n    return a + b;\n}\n",
        file_path="main.cpp",
    )

    assert module is not None
    assert module.language == "cpp"


@pytest.mark.skipif(
    importlib.util.find_spec("tree_sitter_javascript") is None,
    reason="tree_sitter_javascript is not installed",
)
def test_javascript_parser_can_parse_simple_file():
    parser = get_parser_by_path("index.js")
    assert parser is not None

    module = parser.parse(
        "export function sum(a, b) {\n    return a + b;\n}\n",
        file_path="index.js",
    )

    assert module is not None
    assert module.language == "javascript"


@pytest.mark.skipif(importlib.util.find_spec("tree_sitter_rust") is None, reason="tree_sitter_rust is not installed")
def test_rust_parser_can_parse_simple_file():
    parser = get_parser_by_path("lib.rs")
    assert parser is not None

    module = parser.parse(
        "fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n",
        file_path="lib.rs",
    )

    assert module is not None
    assert module.language == "rust"
