"""Shared JSON object storage; commit writes only after serialization succeeds."""
import json
from pathlib import Path


def read_object(path, error_message):
    path = Path(path)
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError(error_message)
    return data


def write_object(path, data):
    path = Path(path)
    payload = json.dumps(data, ensure_ascii=False, indent=2) + '\n'
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    try:
        temporary.write_text(payload, encoding='utf-8')
        temporary.replace(path)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            # A cleanup failure must not replace the original write error.
            pass
