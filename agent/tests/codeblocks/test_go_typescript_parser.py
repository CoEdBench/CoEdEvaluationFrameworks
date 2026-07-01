import importlib.util

import pytest

from moatless.codeblocks import get_parser_by_path, supports_codeblocks


def test_supports_codeblocks_includes_go_and_typescript():
    assert supports_codeblocks("a.go")
    assert supports_codeblocks("a.ts")
    assert supports_codeblocks("a.tsx")
    assert supports_codeblocks("a.mts")
    assert supports_codeblocks("a.js")
    assert supports_codeblocks("A.GO:123")
    assert supports_codeblocks('"path/to/file.ts#L12"')


@pytest.mark.skipif(importlib.util.find_spec("tree_sitter_go") is None, reason="tree_sitter_go is not installed")
def test_go_parser_can_parse_simple_file():
    parser = get_parser_by_path("main.go")
    assert parser is not None

    module = parser.parse(
        "package main\n\nfunc main() {\n    println(\"hi\")\n}\n",
        file_path="main.go",
    )

    assert module is not None
    assert module.language == "go"


@pytest.mark.skipif(
    importlib.util.find_spec("tree_sitter_typescript") is None,
    reason="tree_sitter_typescript is not installed",
)
def test_typescript_parser_can_parse_simple_file():
    parser = get_parser_by_path("index.ts")
    assert parser is not None

    module = parser.parse(
        "export function sum(a: number, b: number) { return a + b; }\n",
        file_path="index.ts",
    )

    assert module is not None
    assert module.language == "typescript"


@pytest.mark.skipif(
    importlib.util.find_spec("tree_sitter_typescript") is None,
    reason="tree_sitter_typescript is not installed",
)
def test_typescript_parser_handles_mts_and_js():
    mts_parser = get_parser_by_path("vitest.config.mts")
    assert mts_parser is not None
    mts_module = mts_parser.parse("export default { test: {} }\n", file_path="vitest.config.mts")
    assert mts_module is not None

    js_parser = get_parser_by_path("foo.js")
    assert js_parser is not None
    js_module = js_parser.parse("function x(){ return 1 }\n", file_path="foo.js")
    assert js_module is not None
