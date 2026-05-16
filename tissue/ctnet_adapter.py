from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from contextlib import contextmanager
import importlib.util
import sys
import os
import math

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

NEEDLES = [
    "load_latest_snapshot",
    "build_terminal_tissue",
    "assign_inhibitory_types",
    "assign_pacemakers",
    "build_sparse_graph",
    "build_ng_neighbors",
    "build_patch_mask",
]


@dataclass
class CTNetMaterial:
    snap_path: str
    source_path: str
    neurons: Any
    glia: Any
    is_inh: Any
    pacer: Any
    edges: Any
    ng_neigh: Any
    gn_neigh: Any
    mask_a: Any
    mask_b: Any
    outside_mask: Any
    a: Any | None = None
    h: Any | None = None
    tag: Any | None = None
    g: Any | None = None
    gmem: Any | None = None
    readout_A: float = 0.0
    readout_B: float = 0.0
    runtime_step: int = 0
    runtime_summary: dict[str, float] | None = None


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _score_file(py: Path) -> int:
    try:
        text = py.read_text(encoding="utf-8")
    except Exception:
        return -1
    return sum(1 for n in NEEDLES if n in text)


def _iter_candidate_files():
    for py in REPO_ROOT.rglob("*.py"):
        if "cerebro_virtual_v0" in py.parts:
            continue
        score = _score_file(py)
        if score >= 4:
            yield score, py


def _load_module_from_path(py: Path):
    mod_name = f"_cv0_ctnet_{py.stem}"
    spec = importlib.util.spec_from_file_location(mod_name, py)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No pude crear spec para {py}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _has_required_api(obj: Any) -> bool:
    return all(hasattr(obj, name) for name in NEEDLES)


def _resolve_base_namespace():
    errors = []
    candidates = sorted(_iter_candidate_files(), key=lambda t: (t[0], str(t[1])), reverse=True)

    for score, py in candidates:
        try:
            mod = _load_module_from_path(py)

            if _has_required_api(mod):
                return mod, py

            if hasattr(mod, "base") and _has_required_api(mod.base):
                return mod.base, py

            for value in mod.__dict__.values():
                if _has_required_api(value):
                    return value, py

        except Exception as e:
            rel = py.relative_to(REPO_ROOT)
            errors.append(f"{rel} -> {type(e).__name__}: {e}")

    preview = "\n".join(errors[:12]) if errors else "sin errores capturados"
    raise RuntimeError(
        "No pude resolver un namespace CTNet base utilizable.\n"
        f"Intentos fallidos:\n{preview}"
    )


@contextmanager
def _pushd(path: Path):
    old = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def _find_snapshot_candidates():
    patterns = [
        "outputs/**/snapshot_*.json",
        "outputs/**/*.json",
    ]
    found = []
    for pattern in patterns:
        found.extend(REPO_ROOT.glob(pattern))
    found = [x for x in found if x.is_file() and "snapshot" in x.name]
    found.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return found


def _topk_mean(values: np.ndarray, frac: float = 0.35) -> float:
    if values.size == 0:
        return 0.0
    k = max(1, int(math.ceil(values.size * frac)))
    idx = np.argpartition(values, -k)[-k:]
    return float(np.mean(values[idx]))


def _bootstrap_runtime(material: CTNetMaterial) -> None:
    if material.a is not None:
        return

    nN = len(material.neurons)
    nG = len(material.glia)

    a = np.zeros(nN, dtype=np.float32)
    h = np.zeros(nN, dtype=np.float32)
    tag = np.zeros(nN, dtype=np.float32)
    g = np.zeros(nG, dtype=np.float32)
    gmem = np.zeros(nG, dtype=np.float32)

    for i, x in enumerate(material.neurons):
        drive = (
            0.08 * x["v"]
            + 0.06 * x["ca"]
            + 0.02 * (x["na"] - x["k"])
            + 0.01 * x["m_na"]
            - 0.005 * x["n_k"]
        )
        a[i] = 0.16 * np.tanh(drive)
        h[i] = np.clip(0.02 * abs(a[i]) + 0.01 * x["ca"], 0.0, 1.0)

    for j, x in enumerate(material.glia):
        g[j] = np.clip(0.24 * x["e"] + 0.03 * abs(x["xi"]), 0.0, 1.0)

    material.a = a
    material.h = h
    material.tag = tag
    material.g = g
    material.gmem = gmem
    material.readout_A = 0.0
    material.readout_B = 0.0
    material.runtime_step = 0
    material.runtime_summary = None


def build_ctnet_material(seed: int = 42) -> CTNetMaterial:
    base, source_py = _resolve_base_namespace()

    try:
        with _pushd(REPO_ROOT):
            snap_path, data = base.load_latest_snapshot()
    except Exception as e:
        candidates = _find_snapshot_candidates()
        preview = "\n".join(str(x.relative_to(REPO_ROOT)) for x in candidates[:10])
        raise RuntimeError(
            "CTNet encontró el módulo base, pero no pudo resolver snapshots desde la raíz del repo.\n"
            f"Repo root: {REPO_ROOT}\n"
            f"Primeros candidatos detectados:\n{preview if preview else 'ninguno'}\n"
            f"Error original: {type(e).__name__}: {e}"
        ) from e

    neurons, glia = base.build_terminal_tissue(data)
    is_inh = base.assign_inhibitory_types(neurons, frac_inh=0.18, seed=seed)
    pacer = base.assign_pacemakers(neurons, frac=0.01)

    try:
        pacer[is_inh == 1] = 0
    except Exception:
        pass

    edges = base.build_sparse_graph(neurons, is_inh, radius=4.2, sigma=2.0)
    ng_neigh, gn_neigh = base.build_ng_neighbors(neurons, glia, radius=4.5)
    mask_a = base.build_patch_mask(neurons, center=(50, 18), radius=8.0)
    mask_b = base.build_patch_mask(neurons, center=(14, 46), radius=8.0)
    outside_mask = ((1 - mask_a) * (1 - mask_b)).astype(np.int32)

    material = CTNetMaterial(
        snap_path=str(snap_path),
        source_path=str(source_py.relative_to(REPO_ROOT)),
        neurons=neurons,
        glia=glia,
        is_inh=is_inh,
        pacer=pacer,
        edges=edges,
        ng_neigh=ng_neigh,
        gn_neigh=gn_neigh,
        mask_a=mask_a,
        mask_b=mask_b,
        outside_mask=outside_mask,
    )
    _bootstrap_runtime(material)
    return material


def step_material(
    material: CTNetMaterial,
    pulse_a: float,
    pulse_b: float,
    stress: float = 0.0,
    substeps: int = 6,
) -> dict[str, float]:
    _bootstrap_runtime(material)

    a = material.a
    h = material.h
    tag = material.tag
    g = material.g
    gmem = material.gmem

    for _ in range(substeps):
        a_new = np.zeros_like(a)
        h_new = np.zeros_like(h)
        tag_new = np.zeros_like(tag)
        g_new = np.zeros_like(g)
        gmem_new = np.zeros_like(gmem)

        for j, x in enumerate(material.glia):
            near_n = material.gn_neigh[j]
            local_neur = float(np.mean(np.abs(a[near_n]))) if near_n else 0.0
            local_tag = float(np.mean(tag[near_n])) if near_n else 0.0

            g_new[j] = float(np.clip(
                0.92 * g[j]
                + 0.030 * local_neur
                + 0.018 * x["e"] * (1.0 - 0.25 * stress),
                0.0,
                1.0,
            ))

            gmem_boost = 0.0
            if near_n:
                frac_a = float(np.mean([material.mask_a[i] for i in near_n]))
                frac_b = float(np.mean([material.mask_b[i] for i in near_n]))
                if pulse_a > 0.0:
                    gmem_boost += 0.040 * pulse_a * frac_a * local_neur + 0.020 * local_tag
                if pulse_b > 0.0:
                    gmem_boost += 0.040 * pulse_b * frac_b * local_neur + 0.020 * local_tag

            gmem_new[j] = float(np.clip(
                0.985 * gmem[j] + (1.0 - gmem[j]) * gmem_boost,
                0.0,
                1.0,
            ))

        phase = material.runtime_step / 12.0

        for i, x in enumerate(material.neurons):
            recur = 0.0
            mem_recur = 0.0
            for j, w in material.edges[i]:
                recur += w * float(a[j])
                if w > 0.0:
                    mem_recur += w * float(tag[j])

            near_g = material.ng_neigh[i]
            glia_support = float(np.mean(g_new[near_g])) if near_g else 0.0
            glia_memory = float(np.mean(gmem_new[near_g])) if near_g else 0.0

            pulse = 0.0
            if material.pacer[i] == 1:
                pulse += 0.016 + 0.012 * math.sin(2.0 * math.pi * phase + 0.17 * i)
            if material.mask_a[i] == 1:
                pulse += pulse_a
            if material.mask_b[i] == 1:
                pulse += pulse_b

            ion_drive = (
                0.05 * x["v"]
                + 0.03 * x["ca"]
                + 0.015 * (x["na"] - x["k"])
                + 0.015 * x["m_na"]
                - 0.008 * x["n_k"]
            )
            inh_self = 0.07 if material.is_inh[i] == 1 else 0.0

            a_new[i] = (
                0.74 * float(a[i])
                + 0.30 * recur
                + 0.10 * mem_recur
                + 0.12 * glia_support
                + 0.08 * glia_memory
                + 0.42 * pulse
                + ion_drive
                - inh_self
                - 0.08 * stress
            )

            h_new[i] = np.clip(
                0.92 * float(h[i])
                + 0.05 * abs(a_new[i])
                + 0.03 * glia_support,
                0.0,
                1.0,
            )

            tag_in = (
                0.020 * h_new[i]
                + 0.018 * glia_memory
                + 0.010 * max(0.0, a_new[i])
            )
            tag_new[i] = np.clip(
                0.970 * float(tag[i]) + (1.0 - float(tag[i])) * tag_in,
                0.0,
                1.0,
            )

        mean_drive = float(np.mean(a_new))
        a_new = np.tanh(1.15 * (a_new - 0.55 * mean_drive))

        a = a_new
        h = h_new
        tag = tag_new
        g = g_new
        gmem = gmem_new
        material.runtime_step += 1

    material.a = a
    material.h = h
    material.tag = tag
    material.g = g
    material.gmem = gmem

    idx_a = np.where(material.mask_a == 1)[0]
    idx_b = np.where(material.mask_b == 1)[0]
    idx_o = np.where(material.outside_mask == 1)[0]
    idx_inh = np.where(material.is_inh == 1)[0]

    a_ctr = a - float(np.mean(a))
    patch_a_vals = a_ctr[idx_a] if len(idx_a) else np.array([], dtype=np.float32)
    patch_b_vals = a_ctr[idx_b] if len(idx_b) else np.array([], dtype=np.float32)
    outside_vals = a_ctr[idx_o] if len(idx_o) else np.array([], dtype=np.float32)

    patch_a_mean = _topk_mean(patch_a_vals, frac=0.40)
    patch_b_mean = _topk_mean(patch_b_vals, frac=0.40)
    outside_mean = float(np.mean(outside_vals)) if outside_vals.size else 0.0
    tag_a_mean = float(np.mean(tag[idx_a])) if len(idx_a) else 0.0
    tag_b_mean = float(np.mean(tag[idx_b])) if len(idx_b) else 0.0

    material.readout_A = float(np.tanh(
        0.78 * material.readout_A
        + 0.72 * patch_a_mean
        - 0.20 * patch_b_mean
        - 0.08 * outside_mean
        + 0.08 * tag_a_mean
    ))
    material.readout_B = float(np.tanh(
        0.78 * material.readout_B
        + 0.72 * patch_b_mean
        - 0.20 * patch_a_mean
        - 0.08 * outside_mean
        + 0.08 * tag_b_mean
    ))

    out = {
        "mean_a": float(np.mean(a_ctr)),
        "mean_abs_a": float(np.mean(np.abs(a_ctr))),
        "inh_activity": float(np.mean(np.abs(a_ctr[idx_inh]))) if len(idx_inh) else 0.0,
        "glia_mean": float(np.mean(g)),
        "tag_mean": float(np.mean(tag)),
        "patch_a_mean": patch_a_mean,
        "patch_b_mean": patch_b_mean,
        "outside_mean": outside_mean,
        "patch_delta": float(patch_b_mean - patch_a_mean),
        "tag_delta": float(tag_b_mean - tag_a_mean),
        "readout_a": float(material.readout_A),
        "readout_b": float(material.readout_B),
        "readout_delta": float(material.readout_B - material.readout_A),
        "active_frac": float(np.mean(np.abs(a_ctr) > 0.12)),
        "mean_raw_a": float(np.mean(a)),
    }
    material.runtime_summary = out
    return out


def summarize_material(material: CTNetMaterial) -> dict[str, float | int | str]:
    out = {
        "snap_path": material.snap_path,
        "source_path": material.source_path,
        "num_neurons": int(len(material.neurons)) if hasattr(material.neurons, "__len__") else -1,
        "num_glia": int(len(material.glia)) if hasattr(material.glia, "__len__") else -1,
        "num_inh": int(np.sum(material.is_inh)) if material.is_inh is not None else -1,
        "num_pacers": int(np.sum(material.pacer)) if material.pacer is not None else -1,
        "patch_a_size": int(np.sum(material.mask_a)) if material.mask_a is not None else -1,
        "patch_b_size": int(np.sum(material.mask_b)) if material.mask_b is not None else -1,
    }
    if material.runtime_summary:
        for k, v in material.runtime_summary.items():
            out[f"runtime_{k}"] = v
    return out
