#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTNet V6 + Perfect v11 Fusion
=============================

Fusion operativa entre:
- CTNet_V6.py: CTNet-Babel fused system, con Babel reflexivo, memoria topológica,
  coherencia dinámica, masa semántica, verificadores y lector global.
- ctnet_perfect_v11.py.bak_fix_unlock_logic: capa perfect/executive bridge,
  con PerspectiveAtlas, MacroFieldHead, ExecutiveCompetition, HomeostasisCore,
  StructuralMemoryUpdater, memoria episódica y lógica de unlock anti-lock.

No usa checkpoint externo. Usa el PerfectBridge como metacontrol ejecutivo/homeostático
encima de un DummyBase identitario para regular CTNet V6.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import io
import json
import math
import os
import re
import sys
import unicodedata
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
V6_PATH = os.path.join(HERE, "CTNet_V6.py")
PERFECT_PATH = os.path.join(HERE, "ctnet_perfect_v11.py.bak_fix_unlock_logic")


def _load_module(name: str, path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    # spec_from_file_location can return None for extensionless backup files.
    # SourceFileLoader forces Python-source loading for .bak files.
    from importlib.machinery import SourceFileLoader
    loader = SourceFileLoader(name, path)
    spec = importlib.util.spec_from_loader(name, loader)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"No puedo importar {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    return mod


V6 = _load_module("CTNet_V6_module", V6_PATH)
PERFECT = _load_module("ctnet_perfect_v11_module", PERFECT_PATH)


def strip_accents(s: str) -> str:
    return ''.join(c for c in unicodedata.normalize('NFD', s or '') if unicodedata.category(c) != 'Mn')


def norm(s: str) -> str:
    s = strip_accents((s or '').lower())
    s = re.sub(r"[^a-z0-9_+/*^().,=\-\s?¿¡!:/]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def stable_seed(key: str) -> int:
    raw = hashlib.blake2b(key.encode('utf-8'), digest_size=8).digest()
    return int.from_bytes(raw, 'big') & 0x7FFFFFFF


def stable_vec(key: str, dim: int) -> np.ndarray:
    rng = np.random.default_rng(stable_seed(key))
    v = rng.normal(0.0, 1.0, size=dim).astype(np.float32)
    n = np.linalg.norm(v)
    if n > 1e-12:
        v = v / n
    return v


def text_tensor(text: str, n_tokens: int = 128, d_model: int = 32, device: str = "cpu") -> torch.Tensor:
    """Deterministic [1,N,D] tensor from text, not learned."""
    t = norm(text)
    toks = [x for x in re.split(r"\s+", t) if x]
    if not toks:
        toks = ["<basal>"]
    arr = np.zeros((n_tokens, d_model), dtype=np.float32)
    whole = stable_vec("whole:" + t[:512], d_model)
    for i in range(n_tokens):
        tok = toks[i % len(toks)]
        phase = stable_vec(f"tok:{tok}:pos:{i}", d_model)
        drift = stable_vec(f"pos:{i}", d_model) * (0.15 + 0.15 * math.sin(i / 7.0))
        arr[i] = 0.72 * phase + 0.20 * whole + 0.08 * drift
    arr = np.tanh(arr)
    return torch.tensor(arr, dtype=torch.float32, device=device).unsqueeze(0)


def scalar(x: Any) -> float:
    if torch.is_tensor(x):
        return float(x.detach().mean().cpu().item())
    try:
        return float(x)
    except Exception:
        return 0.0


class DummyBase(nn.Module):
    """Identity-like base model used so PerfectBridge can operate without checkpoint."""
    def __init__(self, d_model: int = 32):
        super().__init__()
        self.mix = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        # Deterministic safe init: close to identity but not dead.
        torch.manual_seed(1337)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, state: Any = None):
        z = torch.tanh(x + 0.08 * self.mix(x))
        return {"z": z, "out": z}, state


@dataclass
class PerfectSignals:
    intent: float
    explore: float
    decide: float
    reflect: float
    stabilize: float
    energy: float
    saturation: float
    fatigue: float
    fragility: float
    consolidation_need: float
    incoherence_debt: float
    atlas: List[float]
    macro: Dict[str, float]
    unlock_log: str
    tick: int

    @property
    def dominant_exec(self) -> str:
        pairs = {
            "intent": self.intent,
            "explore": self.explore,
            "decide": self.decide,
            "reflect": self.reflect,
            "stabilize": self.stabilize,
        }
        return max(pairs.items(), key=lambda kv: kv[1])[0]

    def compact(self) -> Dict[str, Any]:
        return {
            "dominant_exec": self.dominant_exec,
            "intent": round(self.intent, 6),
            "explore": round(self.explore, 6),
            "decide": round(self.decide, 6),
            "reflect": round(self.reflect, 6),
            "stabilize": round(self.stabilize, 6),
            "energy": round(self.energy, 6),
            "saturation": round(self.saturation, 6),
            "fatigue": round(self.fatigue, 6),
            "fragility": round(self.fragility, 6),
            "consolidation_need": round(self.consolidation_need, 6),
            "incoherence_debt": round(self.incoherence_debt, 6),
            "atlas": [round(x, 6) for x in self.atlas],
            "macro": {k: round(v, 6) for k, v in self.macro.items()},
            "tick": self.tick,
        }


class PerfectController:
    def __init__(self, d_model: int = 32, n_tokens: int = 128, device: str = "cpu") -> None:
        self.d_model = d_model
        self.n_tokens = n_tokens
        self.device = device
        self.bridge = PERFECT.CTNetPerfectBridge(
            base_model=DummyBase(d_model=d_model),
            d_model=d_model,
            n_tokens=n_tokens,
            episodic_slots=8,
            n_charts=4,
            device=device,
        ).to(device)
        self.state = self.bridge.init_state(batch_size=1)
        self.bridge.eval()

    def step(self, text: str) -> PerfectSignals:
        x = text_tensor(text, self.n_tokens, self.d_model, self.device)
        buf = io.StringIO()
        with torch.no_grad(), contextlib.redirect_stdout(buf):
            out, self.state = self.bridge(x, self.state)
        unlock_log = buf.getvalue().strip()
        atlas = out.get("atlas_gates", torch.zeros(1, 4, device=self.device))
        macro = {
            "S": scalar(out.get("macro_S", 0.0)),
            "Es": scalar(out.get("macro_Es", 0.0)),
            "U": scalar(out.get("macro_U", 0.0)),
            "A": scalar(out.get("macro_A", 0.0)),
            "D": scalar(out.get("macro_D", 0.0)),
            "R": scalar(out.get("macro_R", 0.0)),
            "I": scalar(out.get("macro_I", 0.0)),
        }
        return PerfectSignals(
            intent=scalar(out.get("exec_intent", 0.0)),
            explore=scalar(out.get("exec_explore", 0.0)),
            decide=scalar(out.get("exec_decide", 0.0)),
            reflect=scalar(out.get("exec_reflect", 0.0)),
            stabilize=scalar(out.get("exec_stabilize", 0.0)),
            energy=scalar(out.get("homeo_energy", 0.0)),
            saturation=scalar(out.get("homeo_saturation", 0.0)),
            fatigue=scalar(out.get("homeo_fatigue", 0.0)),
            fragility=scalar(out.get("homeo_fragility", 0.0)),
            consolidation_need=scalar(out.get("homeo_consolidation_need", 0.0)),
            incoherence_debt=scalar(out.get("homeo_incoherence_debt", 0.0)),
            atlas=[float(v) for v in atlas.detach().reshape(-1).cpu().tolist()[:4]],
            macro=macro,
            unlock_log=unlock_log,
            tick=int(getattr(self.state, "tick", 0)),
        )


class CTNetV6PerfectFusion:
    def __init__(
        self,
        dim: int = 128,
        fractions: int = 18,
        cycles: int = 2,
        capacity: int = 160,
        device: str = "cpu",
    ) -> None:
        self.v6 = V6.CTNetBabelFusedSystem(dim=dim, fractions=fractions, cycles=cycles, capacity=capacity)
        self.perfect = PerfectController(d_model=32, n_tokens=128, device=device)
        self.turn = 0
        self.history: List[Dict[str, Any]] = []

    def _control_prompt(self, text: str, sig: PerfectSignals) -> str:
        """Inject executive/homeostatic state as control context, not as factual claim."""
        mode = sig.dominant_exec
        # Keep it compact to avoid drowning the user request.
        return (
            f"[control ejecutivo CTNetPerfect: modo={mode}; "
            f"explore={sig.explore:.3f}; decide={sig.decide:.3f}; reflect={sig.reflect:.3f}; "
            f"stabilize={sig.stabilize:.3f}; fragilidad={sig.fragility:.3f}; "
            f"deuda_incoherencia={sig.incoherence_debt:.3f}; consolidacion={sig.consolidation_need:.3f}]\n"
            f"{text}"
        )

    def _decide_policy(self, sig: PerfectSignals) -> Dict[str, Any]:
        mode = sig.dominant_exec
        hard_boundary = sig.incoherence_debt > 0.68 or sig.fragility > 0.72
        stabilize = mode == "stabilize" or sig.consolidation_need > 0.67
        reflect = mode == "reflect" and not hard_boundary
        explore = mode == "explore" and not hard_boundary
        decide = mode == "decide" or (sig.decide > 0.24 and not reflect and not explore)
        return {
            "mode": mode,
            "hard_boundary": hard_boundary,
            "stabilize": stabilize,
            "reflect": reflect,
            "explore": explore,
            "decide": decide,
        }

    def _postprocess_answer(self, answer: str, pre: PerfectSignals, post: PerfectSignals, policy: Dict[str, Any]) -> str:
        prefix = ""
        if policy["hard_boundary"]:
            prefix = (
                "[Control Perfect] Señal alta de fragilidad/deuda de incoherencia: "
                "la respuesta queda tratada como cierre provisional o frontera hasta verificación.\n\n"
            )
        elif policy["stabilize"]:
            prefix = "[Control Perfect] Modo estabilización/consolidación: priorizo invariantes y reduzco expansión.\n\n"
        elif policy["reflect"]:
            prefix = "[Control Perfect] Modo reflexión: priorizo metaestructura, plantillas e invariantes.\n\n"
        elif policy["explore"]:
            prefix = "[Control Perfect] Modo exploración: permito más apertura de cartas, manteniendo verificación.\n\n"
        elif policy["decide"]:
            prefix = "[Control Perfect] Modo decisión: cierro salida concreta y reduzco deriva reflexiva.\n\n"

        trace = (
            "\n\n[Fusión V6+Perfect] "
            f"exec_pre={pre.dominant_exec}, exec_post={post.dominant_exec}, "
            f"homeo_debt={post.incoherence_debt:.4f}, frag={post.fragility:.4f}, "
            f"cons={post.consolidation_need:.4f}, atlas="
            f"[{', '.join(f'{x:.3f}' for x in post.atlas)}]."
        )
        return prefix + answer.rstrip() + trace

    def respond(self, text: str = "") -> Dict[str, Any]:
        self.turn += 1
        # Perfect v11 first: interpret state of the incoming u/p.
        pre = self.perfect.step(text or "<sin entrada externa>")
        policy = self._decide_policy(pre)

        # V6 receives the user request plus executive control context.
        controlled = self._control_prompt(text or "", pre)
        v6_result = self.v6.respond(controlled)

        # Perfect v11 sees the emitted response, updates episodic/homeostatic memory.
        post = self.perfect.step((text or "<sin entrada externa>") + "\n" + v6_result["answer"][:4000])
        policy_post = self._decide_policy(post)
        policy = {**policy, **{f"post_{k}": v for k, v in policy_post.items()}}

        answer = self._postprocess_answer(v6_result["answer"], pre, post, policy_post)
        out = {
            "turn": self.turn,
            "input": text,
            "answer": answer,
            "policy_pre": self._decide_policy(pre),
            "policy_post": policy_post,
            "perfect_pre": pre.compact(),
            "perfect_post": post.compact(),
            "v6_dynamic": v6_result.get("dynamic_status", {}),
            "v6_structure": v6_result.get("reflective_structure", {}),
            "v6_raw": v6_result,
        }
        self.history.append({k: out[k] for k in ("turn", "input", "policy_pre", "policy_post", "perfect_post", "v6_dynamic", "v6_structure")})
        self.history = self.history[-32:]
        return out

    def status(self) -> Dict[str, Any]:
        last = self.history[-1] if self.history else {}
        return {
            "turns": self.turn,
            "last": last,
            "perfect_tick": int(getattr(self.perfect.state, "tick", 0)),
        }


def run_demo() -> List[Dict[str, Any]]:
    eng = CTNetV6PerfectFusion(device="cpu", cycles=2, capacity=128)
    qs = [
        "",
        "Que eres tras fusionar CTNet V6 y ctnet_perfect_v11?",
        "Cual es la capital de Francia?",
        "Crea codigo Python para factorial con tests",
        "Haz un programa que siempre termine y tenga un bucle infinito obligatorio",
        "Reflexiona sobre la estructura que ya tienes y dime que ha crecido",
        "Dame auditoria del metabolismo ejecutivo: atlas, homeostasis, modo y memoria",
        "Cual es la capital de Japon?",
        "Si entras en bucle reflexivo, como te desbloqueas?",
    ]
    return [eng.respond(q) for q in qs]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("message", nargs="?", default=None)
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--cycles", type=int, default=2)
    ap.add_argument("--capacity", type=int, default=160)
    args = ap.parse_args()

    if args.demo:
        rows = run_demo()
        if args.json:
            print(json.dumps(rows, ensure_ascii=False, indent=2, default=str))
        else:
            for i, r in enumerate(rows, 1):
                inp = r["input"] if r["input"] else "<sin entrada externa>"
                print(f"TURNO {i}")
                print(f"ENTRADA: {inp}")
                print("CTNET V6 PERFECT FUSION:")
                print(r["answer"])
                print("PERFECT_POST:", json.dumps(r["perfect_post"], ensure_ascii=False))
                print("V6_DYNAMIC:", json.dumps(r["v6_dynamic"], ensure_ascii=False))
                print("V6_STRUCTURE:", json.dumps(r["v6_structure"], ensure_ascii=False))
                print()
        return

    eng = CTNetV6PerfectFusion(device=args.device, cycles=args.cycles, capacity=args.capacity)
    if args.interactive:
        print("CTNet V6 Perfect Fusion. Ctrl-D para salir.")
        while True:
            try:
                msg = input("u/p> ")
            except EOFError:
                break
            r = eng.respond(msg)
            print(r["answer"])
        return

    msg = args.message if args.message is not None else ""
    r = eng.respond(msg)
    if args.json:
        print(json.dumps(r, ensure_ascii=False, indent=2, default=str))
    else:
        print(r["answer"])


if __name__ == "__main__":
    main()
