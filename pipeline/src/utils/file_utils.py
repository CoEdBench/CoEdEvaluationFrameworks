def _norm_path(p: str) -> str:
    """Normalize path separator to /, strip leading/trailing whitespace"""
    return p.strip().replace("\\", "/")


def _get_field(obj, key: str) -> str:
    """Compatible with dict and dataclass, get field and normalize path"""
    raw = obj.get(key) if isinstance(obj, dict) else getattr(obj, key)
    return _norm_path(raw) if isinstance(raw, str) else raw
