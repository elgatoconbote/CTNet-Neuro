from __future__ import annotations

import json
from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None


def _scalarize(value: Any) -> Any:
    if np is not None and isinstance(value, np.generic):
        return value.item()
    return value


def _summarize_material_ref(material_ref: Any) -> dict[str, Any] | None:
    if material_ref is None:
        return None

    out: dict[str, Any] = {
        "kind": type(material_ref).__name__,
    }

    for key in ("snap_path", "source_path", "readout_A", "readout_B", "runtime_step"):
        if hasattr(material_ref, key):
            out[key] = _scalarize(getattr(material_ref, key))

    for key in ("neurons", "glia", "is_inh", "pacer", "mask_a", "mask_b", "outside_mask"):
        if hasattr(material_ref, key):
            value = getattr(material_ref, key)
            try:
                out[f"{key}_len"] = int(len(value))
            except Exception:
                pass
            if np is not None:
                try:
                    out[f"{key}_sum"] = float(np.sum(value))
                except Exception:
                    pass

    if hasattr(material_ref, "runtime_summary") and material_ref.runtime_summary:
        out["runtime_summary"] = {
            str(k): _scalarize(v) for k, v in material_ref.runtime_summary.items()
        }

    return out


def _json_safe(value: Any) -> Any:
    value = _scalarize(value)

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if np is not None and isinstance(value, np.ndarray):
        return {
            "kind": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "mean": float(value.mean()) if value.size else 0.0,
            "std": float(value.std()) if value.size else 0.0,
        }

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_json_safe(v) for v in value]

    if is_dataclass(value):
        out = {}
        for f in fields(value):
            current = getattr(value, f.name)
            if f.name == "material_ref":
                out[f.name] = _summarize_material_ref(current)
            else:
                out[f.name] = _json_safe(current)
        return out

    if hasattr(value, "__dict__"):
        return {str(k): _json_safe(v) for k, v in value.__dict__.items()}

    return repr(value)


def save_state_json(state: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_safe(state)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
