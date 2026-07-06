#!/usr/bin/env python3
"""
DYCONET Flask wrapper for the Dayplanner app.

Endpoint:
    POST /api/dyconet

Supported request formats:
    1. Compact app/API payload:
       {"needs": {...}, "valences": {...}, "stressors": {...}, "tolerances": {...}}

    2. LimeSurvey-style single response:
       {"id": 1, "PROFILE": "...", "MOBIL": "...", "APP": "...", "emoval": "..."}

    3. LimeSurvey-style export:
       {"responses": [{...}, {...}]}

The current baseline model intentionally does not use stressors inside the ODE.
Tolerances are still used in the Cognitive Passport profile and contextual routing flags.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from flask import Flask, jsonify, request

import torch

from DynamicTopology_Symmetric_HOTCO_Grossberg_baseline import (
    DeepHOTCO_v4,
    NEED_NAMES,
    STRESSOR_NAMES,
    VALUE_NAMES,
    load_data,
)

try:
    from parsers.parser_support import EMPIRICAL_POP_MEANS
except Exception:  # pragma: no cover - only used if parser_support is missing
    EMPIRICAL_POP_MEANS = {
        "car": {"pro_env": -0.419, "physical": -0.786, "privacy": 0.746, "autonomy": 0.559, "cost": 0.102, "speed": 0.674, "safety_accident": 0.511, "safety_crime": 0.766, "comfort": 0.728, "reliable": 0.648, "health_infection": 0.424},
        "bike": {"pro_env": 0.910, "physical": 0.874, "privacy": 0.694, "autonomy": 0.761, "cost": 0.719, "speed": 0.517, "safety_accident": 0.225, "safety_crime": 0.604, "comfort": 0.095, "reliable": 0.771, "health_infection": 0.544},
        "pt": {"pro_env": 0.542, "physical": -0.360, "privacy": 0.028, "autonomy": -0.070, "cost": 0.436, "speed": 0.205, "safety_accident": 0.696, "safety_crime": 0.333, "comfort": 0.198, "reliable": 0.139, "health_infection": 0.333},
        "walk": {"pro_env": 0.977, "physical": 0.871, "privacy": 0.714, "autonomy": 0.705, "cost": 0.792, "speed": -0.254, "safety_accident": 0.489, "safety_crime": 0.421, "comfort": 0.125, "reliable": 0.820, "health_infection": 0.819},
    }

# Keep CPU inference predictable on small server containers.
torch.set_num_threads(int(os.environ.get("HOTCO_TORCH_NUM_THREADS", "1")))

APP_DIR = Path(__file__).resolve().parent
DEVICE = os.environ.get("DYCONET_DEVICE", "cpu")
T_MAX = float(os.environ.get("DYCONET_T_MAX", "10.0"))
DT_EVAL = float(os.environ.get("DYCONET_DT_EVAL", "0.02"))
CONFIG_PATH = os.environ.get("DYCONET_CONFIG", str(APP_DIR / "config.yaml"))

LEGACY_MODES = ["car", "bike", "pt", "walk"]
_MODEL_CACHE: Dict[Tuple[int, Tuple[str, ...]], DeepHOTCO_v4] = {}
_MODEL_LOCK = threading.Lock()

logging.basicConfig(level=os.environ.get("DYCONET_LOG_LEVEL", "INFO"))
logger = logging.getLogger("dyconet_api")

app = Flask(__name__)


def _as_float(value: Any, default: float = 0.5) -> float:
    """Convert API values to finite floats and fall back safely."""
    if value is None or value == "NA" or value == "":
        return default
    try:
        x = float(value)
    except (TypeError, ValueError):
        return default
    if x != x or x in (float("inf"), float("-inf")):
        return default
    return x


def _clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _first_number(source: Mapping[str, Any], keys: Iterable[str], default: float = 0.5) -> float:
    for key in keys:
        if key in source:
            return _as_float(source.get(key), default)
    return default


def _get_model(n_modes: int, mode_names: List[str]) -> DeepHOTCO_v4:
    """Create/reuse a model instance for this topology."""
    cache_key = (int(n_modes), tuple(mode_names))
    model = _MODEL_CACHE.get(cache_key)
    if model is None:
        model = DeepHOTCO_v4(
            n_modes=int(n_modes),
            n_needs=len(NEED_NAMES),
            mode_names=mode_names,
            t_max=T_MAX,
            dt_eval=DT_EVAL,
        ).to(DEVICE)
        model.eval()
        _MODEL_CACHE[cache_key] = model
    return model


def _is_limesurvey_payload(payload: Any) -> bool:
    if isinstance(payload, list):
        return True
    if not isinstance(payload, dict):
        return False
    return "responses" in payload or any(k in payload for k in ("MOBIL", "APP", "emoval", "PROFILE", "POI"))


def _compact_needs_to_vector(needs_in: Mapping[str, Any]) -> List[float]:
    aliases = {
        "pro_env": ["pro_env", "env", "environment", "environmental"],
        "physical": ["physical", "health_activity", "activity"],
        "privacy": ["privacy"],
        "autonomy": ["autonomy", "flex", "flexibility"],
        "cost": ["cost"],
        "speed": ["speed", "time"],
        "safety_accident": ["safety_accident", "traffic_safety", "safety"],
        "safety_crime": ["safety_crime", "personal_security", "safety"],
        "comfort": ["comfort", "comfort_physical"],
        "reliable": ["reliable", "reliability"],
        "health_infection": ["health_infection", "infection"],
    }
    return [_clamp(_first_number(needs_in, aliases.get(name, [name]), 0.5), 0.0, 1.0) for name in NEED_NAMES]


def _compact_valences_to_vector(valences_in: Mapping[str, Any], mode_names: List[str]) -> List[float]:
    aliases = {
        "car": ["car", "car_driver"],
        "bike": ["bike"],
        "pt": ["pt", "public_transport", "bus", "tram"],
        "walk": ["walk", "walking"],
    }
    return [_clamp(_first_number(valences_in, aliases.get(mode, [mode]), 0.0), -1.0, 1.0) for mode in mode_names]


def _compact_availability_to_vector(payload: Mapping[str, Any], mode_names: List[str]) -> List[float]:
    source = payload.get("availability") or payload.get("feasibility") or {}
    if not isinstance(source, Mapping):
        return [1.0] * len(mode_names)
    return [_clamp(_as_float(source.get(mode), 1.0), 0.0, 1.0) for mode in mode_names]


def _compact_tolerances_to_vector(tolerances_in: Mapping[str, Any]) -> List[float]:
    return [_clamp(_as_float(tolerances_in.get(name), 0.5), 0.0, 1.0) for name in STRESSOR_NAMES]


def _compact_values_to_vector(values_in: Mapping[str, Any]) -> List[float]:
    return [_clamp(_as_float(values_in.get(name), 0.5), 0.0, 1.0) for name in VALUE_NAMES]


def _compact_beliefs_to_tensor(payload: Mapping[str, Any], mode_names: List[str]) -> torch.Tensor:
    """
    Build [1, n_modes, n_needs] beliefs.

    If the payload includes beliefs as {mode: {need: value}}, use them.
    Missing cells fall back to the empirical population means used by the parser.
    """
    beliefs_in = payload.get("beliefs") if isinstance(payload.get("beliefs"), Mapping) else {}
    rows: List[List[float]] = []
    for mode in mode_names:
        mode_source = beliefs_in.get(mode, {}) if isinstance(beliefs_in, Mapping) else {}
        row: List[float] = []
        for need in NEED_NAMES:
            default = float(EMPIRICAL_POP_MEANS.get(mode, {}).get(need, 0.0))
            if isinstance(mode_source, Mapping) and need in mode_source:
                val = _as_float(mode_source[need], default)
            else:
                val = default
            row.append(_clamp(val, -1.5, 1.5))
        rows.append(row)
    return torch.tensor([rows], dtype=torch.float32, device=DEVICE)


def _run_compact_payload(payload: Mapping[str, Any]) -> Dict[str, Any]:
    needs_in = payload.get("needs")
    valences_in = payload.get("valences")
    if not isinstance(needs_in, Mapping) or not isinstance(valences_in, Mapping):
        raise ValueError("Compact DYCONET payload must include object fields 'needs' and 'valences'.")

    mode_names = LEGACY_MODES
    n_modes = len(mode_names)
    n_needs = len(NEED_NAMES)
    n_nodes = n_needs + 2 * n_modes

    needs = torch.tensor([_compact_needs_to_vector(needs_in)], dtype=torch.float32, device=DEVICE)
    valences = torch.tensor([_compact_valences_to_vector(valences_in, mode_names)], dtype=torch.float32, device=DEVICE)
    availability = torch.tensor([_compact_availability_to_vector(payload, mode_names)], dtype=torch.float32, device=DEVICE)
    tolerances = torch.tensor([_compact_tolerances_to_vector(payload.get("tolerances", {}) if isinstance(payload.get("tolerances"), Mapping) else {})], dtype=torch.float32, device=DEVICE)
    values = torch.tensor([_compact_values_to_vector(payload.get("values", {}) if isinstance(payload.get("values"), Mapping) else {})], dtype=torch.float32, device=DEVICE)
    beliefs = _compact_beliefs_to_tensor(payload, mode_names)

    initial_state = torch.zeros(1, n_nodes, dtype=torch.float32, device=DEVICE)
    initial_state[:, :n_needs] = needs
    initial_state[:, n_needs:n_needs + n_modes] = torch.sigmoid(valences) * 0.2 * availability
    initial_state[:, n_needs + n_modes:] = valences

    model = _get_model(n_modes=n_modes, mode_names=mode_names)
    with _MODEL_LOCK, torch.no_grad():
        final_state, traces = model(initial_state, beliefs, availability, return_trace=True)
        metadata = {
            "profile": {},
            "pois": {"n_pois": 0, "mode_accessibility": {}},
            "ranking": sorted(
                zip(NEED_NAMES, needs[0].detach().cpu().tolist()),
                key=lambda x: x[1],
                reverse=True,
            )[:3],
        }
        metadata["ranking"] = [name for name, _score in metadata["ranking"]]
        passport_json = model.generate_cognitive_passport(
            agent_id=payload.get("id", "api_request"),
            initial_state=initial_state[0],
            beliefs=beliefs[0],
            tolerances=tolerances[0],
            availability=availability[0],
            values=values[0],
            metadata=metadata,
            trace=traces[0],
            final_state=final_state[0],
        )
    return json.loads(passport_json)


def _normalise_limesurvey_payload(payload: Any) -> Dict[str, Any]:
    """Convert single-response or list payloads into {'responses': [...]} for the parser."""
    if isinstance(payload, list):
        return {"responses": payload}
    if isinstance(payload, dict) and "responses" in payload:
        if not isinstance(payload["responses"], list):
            raise ValueError("Field 'responses' must be a list.")
        return payload
    if isinstance(payload, dict):
        return {"responses": [payload]}
    raise ValueError("Unsupported JSON payload. Expected an object or a list of responses.")


def _select_availability(data: Mapping[str, Any]) -> torch.Tensor:
    for key in ("effective_availability", "availability", "feasibility", "mode_active_mask"):
        value = data.get(key)
        if torch.is_tensor(value):
            return value.to(DEVICE).float()
    n_agents = int(data["needs"].shape[0])
    n_modes = int(data.get("n_modes", data["beliefs"].shape[1]))
    return torch.ones(n_agents, n_modes, dtype=torch.float32, device=DEVICE)


def _run_limesurvey_payload(payload: Any) -> Dict[str, Any]:
    wrapped = _normalise_limesurvey_payload(payload)
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".json", encoding="utf-8", delete=False) as tmp:
            json.dump(wrapped, tmp, ensure_ascii=False)
            tmp_path = tmp.name

        data = load_data(tmp_path, device=DEVICE, config_path=CONFIG_PATH)
        n_agents = int(data["needs"].shape[0])
        n_modes = int(data.get("n_modes", data["beliefs"].shape[1]))
        mode_names = list(data.get("mode_names", data.get("modes", LEGACY_MODES[:n_modes])))
        availability = _select_availability(data)

        model = _get_model(n_modes=n_modes, mode_names=mode_names)
        with _MODEL_LOCK, torch.no_grad():
            final_state, traces = model(
                data["initial_state"].to(DEVICE),
                data["beliefs"].to(DEVICE),
                availability,
                return_trace=True,
            )
            passports: List[Dict[str, Any]] = []
            metadata_list = data.get("metadata", [{} for _ in range(n_agents)])
            tolerances = data.get("tolerances")
            values = data.get("values")
            for i in range(n_agents):
                agent_meta = metadata_list[i] if i < len(metadata_list) and isinstance(metadata_list[i], dict) else {}
                agent_id = agent_meta.get("response_id", i)
                passport_json = model.generate_cognitive_passport(
                    agent_id=agent_id,
                    initial_state=data["initial_state"][i].to(DEVICE),
                    beliefs=data["beliefs"][i].to(DEVICE),
                    tolerances=tolerances[i].to(DEVICE) if torch.is_tensor(tolerances) else None,
                    availability=availability[i].to(DEVICE),
                    values=values[i].to(DEVICE) if torch.is_tensor(values) else None,
                    metadata=agent_meta,
                    trace=traces[i],
                    final_state=final_state[i],
                )
                passports.append(json.loads(passport_json))

        if len(passports) == 1:
            return passports[0]
        return {"passports": passports}
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass


def run_dyconet(payload: Any) -> Dict[str, Any]:
    if _is_limesurvey_payload(payload):
        return _run_limesurvey_payload(payload)
    if isinstance(payload, Mapping):
        return _run_compact_payload(payload)
    raise ValueError("Request body must be a JSON object.")


@app.get("/health")
def health() -> Any:
    return jsonify({"status": "ok", "service": "dyconet", "device": DEVICE})


@app.post("/api/dyconet")
def dyconet_route() -> Any:
    try:
        payload = request.get_json(force=True)
        result = run_dyconet(payload)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - operational safety net
        logger.exception("DYCONET inference failed")
        return jsonify({"error": "DYCONET inference failed", "detail": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=os.environ.get("FLASK_DEBUG", "0") == "1", host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))
