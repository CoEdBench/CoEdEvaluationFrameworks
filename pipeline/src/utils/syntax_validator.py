"""
Multilingual syntax validation module (tree-sitter >= 0.23 new API)
"""
import ast
import logging
from typing import Literal

logger = logging.getLogger(__name__)

LangType = Literal["python", "typescript", "tsx", "go", "java"]

# ── Global parser cache ───────────────────────────────────────────────────────
_PARSER_CACHE: dict | None = None
_PARSER_INIT_FAILED: bool = False   # ← New: track whether init failed, avoid repeated retries


def _build_parser_map() -> dict:
    """
    Build {lang_name: Parser} mapping.
    tree-sitter >= 0.23 correct usage:
      language = Language(tree_sitter_xxx.language())
      parser   = Parser(language)
    """
    from tree_sitter import Language, Parser
    import tree_sitter_python     as tsp
    import tree_sitter_typescript as tsts
    import tree_sitter_go         as tsg
    import tree_sitter_java       as tsj

    lang_objects = {
        "python":     Language(tsp.language()),
        "typescript": Language(tsts.language_typescript()),
        "tsx":        Language(tsts.language_tsx()),
        "go":         Language(tsg.language()),
        "java":       Language(tsj.language()),
    }
    return {name: Parser(lang) for name, lang in lang_objects.items()}


def _get_parser_map() -> dict | None:
    """
    Get the global parser cache.
    If initialization ever failed, returns None immediately (no retries), avoiding repeated exceptions.
    """
    global _PARSER_CACHE, _PARSER_INIT_FAILED

    if _PARSER_INIT_FAILED:
        return None                        # ← known failure, fast return

    if _PARSER_CACHE is None:
        try:
            _PARSER_CACHE = _build_parser_map()
            logger.info("✅ tree-sitter parser map initialized successfully")
        except Exception as e:
            _PARSER_INIT_FAILED = True     # ← mark failure, no more retries
            logger.warning(
                f"tree-sitter init failed ({e}), will use fallback validators"
            )
            return None

    return _PARSER_CACHE


# ── tree-sitter core validation ───────────────────────────────────────────────────

def _has_error(node) -> bool:
    """
    Recursively check if there is an ERROR node in the syntax tree.
    Note: only check ERROR, not MISSING.
    MISSING nodes appear frequently in code snippets (not complete files) and would cause false positives.
    """
    if node.type == "ERROR":               # ← only check ERROR, skip MISSING
        return True
    return any(_has_error(child) for child in node.children)


def _ts_validate(source: str, lang_name: str) -> bool:
    """Parse source with tree-sitter, return whether there are no syntax errors."""
    parser_map = _get_parser_map()

    if parser_map is None:
        # tree-sitter unavailable, fall back to fallback
        return _fallback_validate(source, lang_name)

    parser = parser_map.get(lang_name)
    if parser is None:
        logger.warning(f"tree-sitter: unsupported lang '{lang_name}', skip syntax check")
        return True

    try:
        tree = parser.parse(source.encode("utf-8"))
        valid = not _has_error(tree.root_node)
        if not valid:
            logger.debug(f"Syntax ERROR node detected in {lang_name} source")
        return valid
    except Exception as e:
        logger.warning(f"tree-sitter parse error ({e}), falling back")
        return _fallback_validate(source, lang_name)


# ── Fallback: language-specific validation logic ────────────────────────────────

def _fallback_validate(source: str, lang_name: str) -> bool:
    validators = {
        "python":     _validate_python,
        "typescript": _validate_typescript,
        "tsx":        _validate_typescript,
        "go":         _validate_go,
        "java":       _validate_java,
    }
    validator = validators.get(lang_name)
    if validator is None:
        logger.warning(f"No fallback validator for lang='{lang_name}', skipping")
        return True
    return validator(source)


def _validate_python(source: str) -> bool:
    try:
        ast.parse(source)
        return True
    except Exception as e:
        logger.warning(f"python syntax check failed: {e}")
        return False


def _validate_typescript(source: str) -> bool:
    import subprocess
    try:
        result = subprocess.run(
            ["node", "--input-type=module", "--check"],
            input=source, capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except FileNotFoundError:
        logger.warning("node not found, skipping TypeScript syntax check")
        return True
    except Exception as e:
        logger.warning(f"TypeScript syntax check failed: {e}")
        return True


def _validate_go(source: str) -> bool:
    import subprocess
    try:
        result = subprocess.run(
            ["gofmt"], input=source, capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except FileNotFoundError:
        logger.warning("gofmt not found, skipping Go syntax check")
        return True
    except Exception as e:
        logger.warning(f"Go syntax check failed: {e}")
        return True


def _validate_java(source: str) -> bool:
    try:
        import javalang
        javalang.parse.parse(source)
        return True
    except ImportError:
        logger.warning("javalang not installed, skipping Java syntax check")
        return True
    except javalang.parser.JavaSyntaxError:
        return False
    except Exception as e:
        logger.warning(f"Java syntax check failed: {e}")
        return True


# ── Unified external interface ─────────────────────────────────────────────────

_LANG_ALIAS: dict[str, str] = {
    "py":     "python",
    "ts":     "typescript",
    "golang": "go",
}


def check_syntax_valid(source: str, lang: str) -> bool:
    """
    Unified external interface: validate whether source is syntactically valid in the given language.
    Returns True  -> syntax is valid (or passes through if validation is unavailable)
    Returns False -> syntax error detected
    """
    lang = lang.lower().strip()
    lang = _LANG_ALIAS.get(lang, lang)
    return _ts_validate(source, lang)