from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path


def save_atlas_json(orchestrator, path: str) -> None:
    payload = {
        "step": orchestrator.step_idx,
        "nodes": [asdict(n.state) for n in orchestrator.atlas.nodes],
    }
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")
