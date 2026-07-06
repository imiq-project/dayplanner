"""
=============================================================================
PARSER SUPPORT — HotCo-C / DeepHOTCO v4.2
=============================================================================
Reusable constants, data, and utility functions for the LimeSurvey parser.
Has NO dependency on survey-specific parsing logic — safe to import from
any context.

INDEX
─────
  §1  Topology constants          MODES_4, MODES_5, NEED_NAMES, STRESSOR_NAMES, VALUE_NAMES
  §2  JSON ↔ Model mappings       NEED_MAPPING, JSON_MODE_MAPPING
  §3  Stressor perturbation       STRESSOR_NODE_TARGETS
  §4  Belief imputation system    EMPIRICAL_POP_MEANS, IMPUTATION_COEFFICIENTS,
                                  MEAN_FREQ, BELIEF_CONSTRAINTS, LITERATURE_BELIEFS
  §5  Imputation functions        impute_full_beliefs(), estimate_valence_from_beliefs()
  §6  Config loading              load_config()
  §7  Validation helpers          validate_response()
  §8  JSON utilities              safe_json_load(), extract_nested_value()
  §9  Scale transformations       normalize_range(), center_scale(), inverse_scale()
  §10 Missing-data handling       handle_missing()
  §11 Aggregation & statistics    aggregate_frequencies(), compute_icc(),
                                  cronbachs_alpha(), detect_outliers_iqr/zscore()
  §12 Belief consistency          check_belief_consistency()
  §13 Mode prevalence             check_mode_prevalence()
  §14 Demographics encoding       encode_demographic()
  §15 POI parsing                 CATEGORY_TO_TYPE_MAP, parse_poi_list(),
                                  _map_category_to_type(), compute_poi_accessibility(),
                                  infer_home_poi(), generate_synthetic_poi(),
                                  validate_poi_coordinates(), poi_summary()
  §16 Reporting                   NumpyEncoder, generate_parsing_report(), save_report()

Consolidated from: constants.py · utils.py · poi_utils.py
Duplicate functions resolved (normalize_range, aggregate_frequencies,
parse_poi_list, compute_poi_accessibility: single canonical version kept here).

Version: 4.2.0
=============================================================================
"""

import json
import math
import logging

import numpy as np
import yaml
from scipy.stats import pearsonr
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("parser_support")

# =============================================================================
# §1  TOPOLOGY CONSTANTS
# =============================================================================

MODES_4 = ['car', 'bike', 'pt', 'walk']
MODES_5 = ['car', 'bike', 'pt', 'walk', 'car_green']

NEED_NAMES = [
    'pro_env',          # 0: Environmental protection
    'physical',         # 1: Physical activity
    'privacy',          # 2: Privacy / lack of crowding
    'autonomy',         # 3: Flexibility / control
    'cost',             # 4: Monetary cost
    'speed',            # 5: Time / Speed
    'safety_accident',  # 6: Traffic safety
    'safety_crime',     # 7: Personal security
    'comfort',          # 8: Physical comfort
    'reliable',         # 9: Reliability
    'health_infection'  # 10: Health (Infection risk)
]

STRESSOR_NAMES = ['rain', 'crowding', 'darkness', 'traffic', 'temperature']
VALUE_NAMES    = ['biospheric', 'altruistic', 'egoistic', 'hedonic']

# =============================================================================
# §2  JSON ↔ MODEL MAPPINGS
# =============================================================================

# Model need name → LimeSurvey JSON key
NEED_MAPPING = {
    'pro_env':          'env',
    'physical':         'health_activity',
    'privacy':          'crowding',     # Survey key is 'crowding'; NO inversion needed
    'autonomy':         'flex',
    'cost':             'cost',
    'speed':            'time',
    'safety_accident':  'safety_accident',
    'safety_crime':     'safety_crime',
    'comfort':          'comfort_physical',
    'reliable':         'reliable',
    'health_infection': 'health_infection',
}

# Internal mode name → candidate JSON keys (first match wins)
JSON_MODE_MAPPING = {
    'car':          ['car_driver', 'car'],
    'car_passenger':['car_pass', 'car_passenger'],
    'bike':         ['bike'],
    'pt':           ['pt', 'public_transport', 'bus', 'tram'],
    'walk':         ['walk', 'walking'],
    'train':        ['train'],
    'car_green':    ['ev', 'electric_car', 'hybrid'],
}

# =============================================================================
# §3  STRESSOR PERTURBATION TARGETS (VBN Theory)
# =============================================================================

STRESSOR_NODE_TARGETS = {
    'rain': {
        'need_comfort':  -0.5,
        'need_physical': -0.3,
        'valence_bike':  -0.8,
        'valence_walk':  -0.6,
        'valence_car':   +0.4,
        'valence_pt':    +0.2,
    },
    'crowding': {
        'need_privacy':  -0.6,
        'need_comfort':  -0.5,
        'valence_pt':    -0.7,
        'valence_walk':  -0.3,
    },
    'darkness': {
        'need_safety_crime': +0.5,
        'valence_walk':      -0.6,
        'valence_bike':      -0.5,
        'valence_car':       +0.2,
    },
    'traffic': {
        'need_speed':    +0.4,
        'need_autonomy': -0.3,
        'valence_car':   -0.6,
        'valence_bike':  +0.4,
        'valence_pt':    +0.2,
    },
    'temperature': {
        'need_comfort':  -0.6,
        'need_physical': -0.4,
        'valence_bike':  -0.5,
        'valence_walk':  -0.5,
        'valence_car':   +0.3,
    },
}

# =============================================================================
# §4  BELIEF IMPUTATION SYSTEM  (Empirical, v4.2)
# =============================================================================
#
# Three-layer strategy:
#   Layer 1: EMPIRICAL_POP_MEANS — observed means (N=232, Magdeburg survey)
#   Layer 2: Per-agent linear adjustment (need_importance × freq_dev + valence)
#   Layer 3: WARN-only constraint audit (no value modification)
#
# RMSE improvement vs literature heuristic: 0.453 vs 0.624 (−27.4%)
# Formula:
#   belief = pop_mean + α_eff × need_importance × freq_dev + β × valence
# =============================================================================

EMPIRICAL_POP_MEANS = {
    'car': {
        'pro_env':         -0.419,  # n=66
        'physical':        -0.786,  # n=70
        'privacy':         +0.746,  # n=21 (small n)
        'autonomy':        +0.559,  # n=99
        'cost':            +0.102,  # n=164
        'speed':           +0.674,  # n=139
        'safety_accident': +0.511,  # n=62
        'safety_crime':    +0.766,  # n=77
        'comfort':         +0.728,  # n=49
        'reliable':        +0.648,  # n=162
        'health_infection':+0.424,  # n=33
    },
    'bike': {
        'pro_env':         +0.910,  # n=37
        'physical':        +0.874,  # n=45
        'privacy':         +0.694,  # n=12 (small n)
        'autonomy':        +0.761,  # n=60
        'cost':            +0.719,  # n=102
        'speed':           +0.517,  # n=80
        'safety_accident': +0.225,  # n=37
        'safety_crime':    +0.604,  # n=48
        'comfort':         +0.095,  # n=28
        'reliable':        +0.771,  # n=96
        'health_infection':+0.544,  # n=19
    },
    'pt': {
        'pro_env':         +0.542,  # n=80
        'physical':        -0.360,  # n=76
        'privacy':         +0.028,  # n=24 (small n)
        'autonomy':        -0.070,  # n=119
        'cost':            +0.436,  # n=237
        'speed':           +0.205,  # n=185
        'safety_accident': +0.696,  # n=90
        'safety_crime':    +0.333,  # n=114
        'comfort':         +0.198,  # n=54
        'reliable':        +0.139,  # n=202
        'health_infection':+0.333,  # n=48
    },
    'walk': {
        'pro_env':         +0.977,  # n=43
        'physical':        +0.871,  # n=44
        'privacy':         +0.714,  # n=14 (small n)
        'autonomy':        +0.705,  # n=70
        'cost':            +0.792,  # n=125
        'speed':           -0.254,  # n=101
        'safety_accident': +0.489,  # n=47
        'safety_crime':    +0.421,  # n=61
        'comfort':         +0.125,  # n=32
        'reliable':        +0.820,  # n=109
        'health_infection':+0.819,  # n=24
    },
    'car_green': {
        'pro_env':         +0.171,  # n=37
        'physical':        -0.875,  # n=32
        'privacy':         +0.750,  # n=12 (small n)
        'autonomy':        +0.590,  # n=48
        'cost':            -0.346,  # n=78
        'speed':           +0.613,  # n=62
        'safety_accident': +0.493,  # n=25
        'safety_crime':    +0.481,  # n=27
        'comfort':         +0.710,  # n=23
        'reliable':        +0.572,  # n=60
        'health_infection':+0.410,  # n=26
    },
}

IMPUTATION_COEFFICIENTS = {
    'alpha_need_freq': 0.40,   # need_importance × frequency deviation
    'beta_valence':    0.10,   # emotional valence
}

MEAN_FREQ = {
    'car':       0.1511,
    'bike':      0.3260,
    'pt':        0.3149,
    'walk':      0.5835,
    'car_green': 0.0094,
}

BELIEF_CONSTRAINTS = {
    # (mode, need): (min_allowed, max_allowed)
    ('bike',      'physical'):       (+0.0, +1.0),
    ('bike',      'pro_env'):        (+0.0, +1.0),
    ('car',       'comfort'):        (+0.0, +1.0),
    ('car',       'physical'):       (-1.0, +0.0),
    ('car',       'privacy'):        (+0.0, +1.0),
    ('car',       'pro_env'):        (-1.0, +0.0),
    ('car_green', 'comfort'):        (+0.0, +1.0),
    ('car_green', 'physical'):       (-1.0, +0.0),
    ('car_green', 'privacy'):        (+0.0, +1.0),
    ('pt',        'physical'):       (-1.0, +0.0),
    ('pt',        'pro_env'):        (+0.0, +1.0),
    ('walk',      'cost'):           (+0.0, +1.0),
    ('walk',      'physical'):       (+0.0, +1.0),
    ('walk',      'pro_env'):        (+0.0, +1.0),
    ('walk',      'speed'):          (-1.0, +0.0),
}

# Legacy literature heuristics (v4.1) — preserved for paper comparison
LITERATURE_BELIEFS = {
    'walk':      {'pro_env':+1.0,'physical':+1.0,'privacy':+0.6,'autonomy':+0.9,
                  'cost':+1.0,'speed':-0.8,'safety_accident':+0.3,'safety_crime':-0.6,
                  'comfort':-0.4,'reliable':+0.5,'health_infection':+0.7},
    'bike':      {'pro_env':+0.9,'physical':+1.0,'privacy':+0.8,'autonomy':+1.0,
                  'cost':+0.8,'speed':+0.3,'safety_accident':-0.4,'safety_crime':-0.5,
                  'comfort':-0.3,'reliable':+0.6,'health_infection':+0.8},
    'pt':        {'pro_env':+0.7,'physical':-0.2,'privacy':-0.7,'autonomy':-0.4,
                  'cost':+0.4,'speed':+0.1,'safety_accident':+0.5,'safety_crime':-0.3,
                  'comfort':-0.1,'reliable':-0.2,'health_infection':-0.6},
    'car':       {'pro_env':-0.9,'physical':-0.8,'privacy':+1.0,'autonomy':+0.7,
                  'cost':-0.7,'speed':+0.6,'safety_accident':-0.2,'safety_crime':+0.4,
                  'comfort':+0.8,'reliable':+0.3,'health_infection':+0.9},
    'car_green': {'pro_env':+0.3,'physical':-0.8,'privacy':+1.0,'autonomy':+0.6,
                  'cost':-0.4,'speed':+0.6,'safety_accident':-0.2,'safety_crime':+0.4,
                  'comfort':+0.8,'reliable':+0.2,'health_infection':+0.9},
}

# Backward-compatible alias
DEFAULT_BELIEFS = EMPIRICAL_POP_MEANS

# =============================================================================
# §5  IMPUTATION FUNCTIONS
# =============================================================================

def impute_full_beliefs(
    mode: str,
    partial_beliefs: Dict,
    need_ranking: Optional[List[str]] = None,
    need_importance: Optional[Dict] = None,
    frequency: Optional[float] = None,
    valence: Optional[float] = None,
) -> Dict:
    """
    Impute full belief vector (11 needs) using 3-layer empirical strategy.

    Survey design constraint: only 3 beliefs are observed per mode (top-ranked
    needs). The remaining 8 are imputed. 73% of the belief matrix is imputed.

    Layer 1 — Population baseline:
        Start from EMPIRICAL_POP_MEANS(mode, need) across N=612 agents.

    Layer 2 — Ranking-weighted per-agent adjustment:
        For each imputed need:
            adjustment = alpha_eff(need) × need_importance(need) × freq_dev
                       + beta × valence

        alpha_eff is modulated by the need's position in the agent's ranking:
            Rank 1-3   → observed (override below); alpha_eff = 1.0 (unused)
            Rank 4-6   → alpha × 1.5  (agent signalled relevance explicitly)
            Rank 7-11  → alpha × 0.7  (agent de-prioritised this need)
            Unranked   → alpha × 1.0  (no positional information)

    Override — Observed survey data (absolute precedence):
        The 3 directly elicited beliefs replace imputed values unconditionally.
        Scale: 1-7 → (-1, +1) via (x-4)/3.

    Layer 3 — WARN-only constraint audit:
        Sign constraints are checked but NOT enforced. Violations are returned
        as metadata in _constraint_warnings for the audit script only.

    Args:
        mode            : internal mode name ('car', 'bike', 'pt', 'walk', ...)
        partial_beliefs : {json_key: raw_1_to_7} — the 3 observed beliefs
        need_ranking    : list of model need names in agent's ranked order
        need_importance : {need: float [0,1]} — all 11 need importance ratings
        frequency       : float [0,1] — normalized usage frequency for this mode
        valence         : float [-1,+1] — emotional valence for this mode

    Returns:
        dict {model_need: float [-1,+1]}  — all 11 needs filled.
        Contains _constraint_warnings key if violations detected (audit only).
    """
    # ── Layer 1: Empirical population means ───────────────────────────────────
    final_beliefs = EMPIRICAL_POP_MEANS.get(mode, {}).copy()
    for need in NEED_NAMES:
        if need not in final_beliefs:
            final_beliefs[need] = 0.0

    # ── Layer 2: Ranking-weighted per-agent linear adjustment ─────────────────
    rank_pos: Dict[str, int] = {}
    if need_ranking:
        for pos, need in enumerate(need_ranking, start=1):
            rank_pos[need] = pos

    def _alpha_multiplier(need: str) -> float:
        pos = rank_pos.get(need)
        if pos is None:
            return 1.0
        if pos <= 3:
            return 1.0    # will be overridden by observed data anyway
        if pos <= 6:
            return 1.5    # upper-mid rank — more adjustment weight
        return 0.7         # lower rank — less adjustment weight

    observed_needs: set = set()

    if need_importance is not None and (frequency is not None or valence is not None):
        alpha_base = IMPUTATION_COEFFICIENTS['alpha_need_freq']
        beta       = IMPUTATION_COEFFICIENTS['beta_valence']
        mean_f     = MEAN_FREQ.get(mode, 0.3)
        freq_dev   = (frequency - mean_f) if frequency is not None else 0.0
        val        = valence if valence is not None else 0.0

        for need in NEED_NAMES:
            ni         = need_importance.get(need, 0.5)
            alpha_eff  = alpha_base * _alpha_multiplier(need)
            adjustment = alpha_eff * ni * freq_dev + beta * val
            final_beliefs[need] = final_beliefs[need] + adjustment

    # ── Override: observed survey data (absolute precedence) ──────────────────
    for model_need, json_key in NEED_MAPPING.items():
        if json_key in partial_beliefs:
            try:
                raw_val = float(partial_beliefs[json_key])
                final_beliefs[model_need] = (raw_val - 4.0) / 3.0
                observed_needs.add(model_need)
            except (ValueError, TypeError):
                pass

    # ── Layer 3: WARN-only constraint audit ───────────────────────────────────
    constraint_warnings: list = []
    for need in NEED_NAMES:
        constraint = BELIEF_CONSTRAINTS.get((mode, need))
        if constraint is None:
            final_beliefs[need] = np.clip(final_beliefs[need], -1.0, 1.0)
            continue
        lo, hi = constraint
        v = final_beliefs[need]
        if v < lo or v > hi:
            src = 'observed' if need in observed_needs else 'imputed'
            constraint_warnings.append({
                'mode': mode, 'need': need,
                'value': round(float(v), 4),
                'expected': [lo, hi], 'source': src,
            })
        final_beliefs[need] = np.clip(final_beliefs[need], -1.0, 1.0)

    if constraint_warnings:
        final_beliefs['_constraint_warnings'] = constraint_warnings

    return final_beliefs


def estimate_valence_from_beliefs(beliefs: Dict, values: Dict) -> float:
    """
    Estimate emotional valence [0, 1] from value-belief alignment.
    Used as fallback when EMOVAL data is missing.
    Returns 0.5 (neutral) when beliefs or values are empty.
    """
    score = 0.0
    if 'pro_env' in beliefs:
        score += beliefs['pro_env'] * values.get('biospheric', 0.5)
    if 'cost' in beliefs:
        score += beliefs['cost'] * values.get('egoistic', 0.5)
    if 'comfort' in beliefs:
        score += beliefs['comfort'] * values.get('hedonic', 0.5)
    return float(np.clip(0.5 + score * 0.3, 0.0, 1.0))

# =============================================================================
# §6  CONFIG LOADING
# =============================================================================

def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

# =============================================================================
# §7  VALIDATION HELPERS
# =============================================================================

def validate_response(response: Dict, config: Dict) -> Tuple[bool, Optional[str]]:
    """
    Validate a single survey response.

    Returns:
        (is_valid, exclusion_reason)  — exclusion_reason is None if valid.
    """
    required = config['validation']['required_sections']
    for section in required:
        if not response.get(section):
            return False, f"missing_{section}"

    # Attention check in APP section
    if response.get('APP'):
        try:
            app_raw = response['APP']
            app_data = app_raw if isinstance(app_raw, dict) else None
            if app_data is None and isinstance(app_raw, str):
                try:
                    app_data = json.loads(app_raw)
                    if isinstance(app_data, str):
                        app_data = json.loads(app_data)
                except (json.JSONDecodeError, TypeError):
                    return False, "app_parse_error"
            if not isinstance(app_data, dict):
                return False, "app_structure_invalid"
            attn = app_data.get('answers', {}).get('ratings', {}).get('attn_check')
            if attn is not None and attn != 7:
                return False, "attention_check_failed"
        except Exception:
            return False, "app_parse_error"

    # Completion rate
    sections = ['PROFILE', 'MOBIL', 'APP', 'emoval', 'values']
    filled = sum(1 for s in sections if response.get(s))
    if filled / len(sections) < config['validation']['min_completion']:
        return False, f"low_completion_{filled/len(sections):.2f}"

    return True, None

# =============================================================================
# §8  JSON UTILITIES
# =============================================================================

def safe_json_load(json_input: Any) -> Optional[Dict]:
    """
    Safely parse JSON string or dict. Handles double-encoded strings.
    Returns None if input is falsy or unparseable.
    """
    if not json_input:
        return None
    if isinstance(json_input, dict):
        return json_input
    if isinstance(json_input, str):
        try:
            data = json.loads(json_input)
            if isinstance(data, str):
                data = json.loads(data)
            return data if isinstance(data, dict) else None
        except (json.JSONDecodeError, TypeError):
            return None
    return None


def extract_nested_value(data: Dict, path: str, default=None):
    """
    Extract nested value using dot-notation path.

    Example:
        extract_nested_value(data, 'answers.ratings.cost', default=4.0)
    """
    current = data
    for key in path.split('.'):
        if isinstance(current, dict) and key in current:
            current = current[key]
        else:
            return default
    return current

# =============================================================================
# §9  SCALE TRANSFORMATIONS
# =============================================================================

def normalize_range(
    value: float,
    old_min: float,
    old_max: float,
    new_min: float = 0.0,
    new_max: float = 1.0,
) -> float:
    """
    Normalize value from [old_min, old_max] to [new_min, new_max].

    Example:
        normalize_range(4, 1, 7, 0, 1) → 0.5
    """
    if old_max == old_min:
        return new_min
    return ((value - old_min) / (old_max - old_min)) * (new_max - new_min) + new_min


def center_scale(value: float, center: float, scale_range: float) -> float:
    """Center and scale value. Example: center_scale(5, 4, 3) → 0.333"""
    return (value - center) / scale_range


def inverse_scale(value: float, scale_min: float, scale_max: float) -> float:
    """Invert scale for reverse-coded items. Example: inverse_scale(2, 1, 7) → 6"""
    return scale_max - value + scale_min

# =============================================================================
# §10  MISSING-DATA HANDLING
# =============================================================================

def handle_missing(
    value: Any,
    strategy: str = 'neutral',
    context: Optional[Dict] = None,
) -> float:
    """
    Handle missing / NA values.

    Strategies:
        'neutral'        → 0.5 (midpoint of [0,1])
        'mean'           → context['population_mean']
        'belief_derived' → estimate_valence_from_beliefs(context)
    """
    if value is None or value == 'NA' or value == '':
        if strategy == 'neutral':
            return 0.5
        if strategy == 'mean' and context and 'population_mean' in context:
            return context['population_mean']
        if strategy == 'belief_derived' and context:
            return estimate_valence_from_beliefs(
                context.get('beliefs', {}), context.get('values', {})
            )
        return 0.5
    return value

# =============================================================================
# §11  AGGREGATION & STATISTICS
# =============================================================================

def aggregate_frequencies(
    s1_freq: float,
    s2_freq: float,
    method: str = 'mean',
) -> float:
    """
    Aggregate S1 and S2 frequencies.

    Methods: 'mean' | 'max' | 'weighted' (0.6*S1 + 0.4*S2)
    """
    if method == 'mean':
        return (s1_freq + s2_freq) / 2.0
    if method == 'max':
        return max(s1_freq, s2_freq)
    if method == 'weighted':
        return 0.6 * s1_freq + 0.4 * s2_freq
    raise ValueError(f"Unknown aggregation method: {method}")


def compute_icc(data1: np.ndarray, data2: np.ndarray) -> Dict:
    """
    Compute Intraclass Correlation Coefficient (test-retest reliability).

    Args:
        data1: [N, n_modes] from S1
        data2: [N, n_modes] from S2

    Returns:
        {mode: {'icc', 'p_value', 'n'}}
    """
    results = {}
    for i, mode in enumerate(MODES_4):
        mask = ~(np.isnan(data1[:, i]) | np.isnan(data2[:, i]))
        if mask.sum() > 10:
            corr, p = pearsonr(data1[mask, i], data2[mask, i])
            results[mode] = {'icc': corr, 'p_value': p, 'n': int(mask.sum())}
        else:
            results[mode] = {'icc': np.nan, 'p_value': np.nan, 'n': int(mask.sum())}
    return results


def cronbachs_alpha(items: np.ndarray) -> float:
    """
    Compute Cronbach's alpha for scale reliability.

    Args:
        items: [n_respondents, n_items]
    """
    n = items.shape[1]
    if n < 2:
        return 0.0
    item_var   = np.var(items, axis=0, ddof=1)
    total_var  = np.var(items.sum(axis=1), ddof=1)
    if total_var == 0:
        return 0.0
    return (n / (n - 1)) * (1 - item_var.sum() / total_var)


def detect_outliers_iqr(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """Return boolean mask where True = IQR outlier."""
    q25, q75 = np.percentile(data, 25), np.percentile(data, 75)
    iqr = q75 - q25
    return (data < q25 - threshold * iqr) | (data > q75 + threshold * iqr)


def detect_outliers_zscore(data: np.ndarray, threshold: float = 3.0) -> np.ndarray:
    """Return boolean mask where True = z-score outlier."""
    z = np.abs((data - data.mean()) / (data.std() + 1e-10))
    return z > threshold

# =============================================================================
# §12  BELIEF CONSISTENCY
# =============================================================================

def check_belief_consistency(
    beliefs: Dict[str, Dict[str, float]],
    needs: Dict[str, float],
    threshold: float = 0.5,
) -> bool:
    """
    Check if beliefs are consistent with declared needs.

    Theory: high cost sensitivity → at least one mode should have strong
    positive belief_cost.
    """
    cost_need = needs.get('cost', 0.5)
    if cost_need > 0.7:
        mode_costs = {m: b.get('cost', 0.0) for m, b in beliefs.items()}
        if mode_costs:
            best_cost = max(mode_costs.values())
            if best_cost < threshold:
                return False
    return True

# =============================================================================
# §13  MODE PREVALENCE CHECK
# =============================================================================

def check_mode_prevalence(
    frequencies: Dict[str, np.ndarray],
    mode: str,
    min_users: int = 30,
    min_percentage: float = 0.10,
) -> bool:
    """
    Check if a mode has sufficient prevalence to model separately.

    Returns True if mode meets both min_users and min_percentage criteria.
    """
    if mode not in frequencies:
        return False
    freq_arr   = frequencies[mode]
    n_users    = int(np.sum(freq_arr >= 2))
    n_total    = len(freq_arr)
    percentage = n_users / n_total if n_total > 0 else 0.0
    logger.info(f"Mode '{mode}': {n_users}/{n_total} users ({percentage*100:.1f}%)")
    return n_users >= min_users and percentage >= min_percentage

# =============================================================================
# §14  DEMOGRAPHICS ENCODING
# =============================================================================

def encode_demographic(value: Any, variable: str, config: Dict) -> int:
    """Encode demographic variable to integer using config encoding rules."""
    encoding = config['metadata']['demographics']['encoding'].get(variable, {})
    return encoding.get(value, 0)

# =============================================================================
# §15  POI PARSING
# =============================================================================

# Category → Game Tab type mapping (Magdeburg context)
CATEGORY_TO_TYPE_MAP: Dict[str, str] = {
    # Home / residential
    'residential': 'home', 'house': 'home', 'suburb': 'home', 'home': 'home',
    # Work / professional
    'university': 'work', 'Mensa': 'work', 'Cafeteria Uni Bibliothek': 'work',
    'Campus OvGU': 'work', 'Sportzentrum der OVGU': 'work',
    'research_institute': 'work',
    'Max-Planck-Institut für Dynamik komplexer technischer Systeme': 'work',
    'Neoscan Solutions': 'work',
    'hospital': 'work', 'doctors': 'work',
    'Prof. Vorwerk – Zentrum für Augenheilkunde': 'work',
    'Universitätsklinikum Magdeburg': 'work',
    'government': 'work', 'Jugendamt': 'work', 'Zentraler Kriminaldienst': 'work',
    'Verwaltungsbibliothek der Landeshauptstadt Magdeburg': 'work',
    'bank': 'work', 'Deutsche Bank': 'work',
    'Sparkasse MagdeBurg - Geschäftsstelle Alter Markt': 'work',
    'industrial': 'work', 'commercial': 'work', 'office': 'work',
    'station': 'work', 'Neustädter Bahnhof': 'work',
    # School / childcare
    'school': 'school', 'kindergarten': 'school', 'childcare': 'school',
    'Grundschule': 'school', 'Gymnasium': 'school', 'Hegel-Gymnasium': 'school',
    'Domgrundschule': 'school', 'Grundschule am Westring': 'school',
    'Neue Schule Magdeburg': 'school',
    'Kindertagesstätte \"Am Storchennest\"': 'school',
    'Kita \"GETEC\"': 'school', 'Kinder-Eltern-Zentrum Nordwest': 'school',
    # Leisure / recreation (food, retail, sports, culture, transport stops)
    'restaurant': 'leisure', 'cafe': 'leisure', 'bar': 'leisure',
    'pub': 'leisure', 'fast_food': 'leisure',
    'Starbucks': 'leisure', 'Subway': 'leisure', 'Nordsee': 'leisure',
    'Ristorante Da Nino Dolce Vita': 'leisure', 'Dolce & Caffè Bar': 'leisure',
    'Cafe Del Sol': 'leisure', 'Pipper Kiezcafé': 'leisure', 'Cô Ba': 'leisure',
    'supermarket': 'leisure', 'EDEKA': 'leisure', 'Lidl': 'leisure',
    'REWE': 'leisure', 'Kaufland': 'leisure', 'Aldi': 'leisure',
    'PENNY': 'leisure', 'Netto Marken-Discount': 'leisure', 'NP': 'leisure',
    'convenience': 'leisure', 'mall': 'leisure', 'POLO': 'leisure',
    'Eastside': 'leisure', 'attraction': 'leisure', 'park': 'leisure',
    'lake': 'leisure', 'theatre': 'leisure', 'cinema': 'leisure',
    'Wallonerkirche': 'leisure', 'Gruson-Gewächshäuser': 'leisure',
    'Kavalier I \"Scharnhorst\"': 'leisure',
    'sports_centre': 'leisure', 'fitness_centre': 'leisure',
    'FitX': 'leisure', 'McFit': 'leisure', 'EasyFitness': 'leisure',
    'LuckyFitness': 'leisure', 'Vitopia': 'leisure',
    'Steps-dance Center': 'leisure', 'GETEC-Arena': 'leisure',
    'Stadion Schöppensteg': 'leisure', 'Guts-Muths-Stadion': 'leisure',
    'SC Magdeburg': 'leisure', 'events_venue': 'leisure',
    'Studentenclub Kiste': 'leisure', 'The Fan - Sportsbar': 'leisure',
    'Blocschmiede': 'leisure',
    'tram_stop': 'leisure', 'bus_stop': 'leisure',
    'halt': 'leisure', 'stop': 'leisure', 'platform': 'leisure',
    'POI': 'leisure', 'yes': 'leisure',
}


def _map_category_to_type(category: str) -> str:
    """
    Map LimeSurvey POI category to Game Tab type.

    Priority:
        1. Exact match in CATEGORY_TO_TYPE_MAP
        2. Case-insensitive partial match
        3. Keyword heuristics (home/school/work)
        4. Default: 'leisure'
    """
    if category in CATEGORY_TO_TYPE_MAP:
        return CATEGORY_TO_TYPE_MAP[category]
    cat_lower = category.lower()
    for key, val in CATEGORY_TO_TYPE_MAP.items():
        if key.lower() in cat_lower or cat_lower in key.lower():
            return val
    if any(w in cat_lower for w in ['home', 'house', 'wohnung', 'residenz', 'apartment']):
        return 'home'
    if any(w in cat_lower for w in ['school', 'schule', 'kita', 'kindergarten', 'daycare']):
        return 'school'
    if any(w in cat_lower for w in ['work', 'office', 'büro', 'firma', 'company']):
        return 'work'
    logger.debug(f"Category '{category}' → 'leisure' (default)")
    return 'leisure'


def parse_poi_list(poi_json: Optional[str]) -> List[Dict]:
    """
    Parse POI JSON string and convert to Game Tab format.

    Input (LimeSurvey):
        {"id":"poi-1","name":"EDEKA","category":"supermarket",
         "lat":52.14,"lng":11.64,"transportMode":"WALKING","frequencyIndex":2}

    Output (Game Tab):
        {"type":"leisure","x":11.64,"y":52.14,"_meta":{...}}

    Ordering:
        1. POIs type 'home' first (Game Tab spawn point)
        2. Then by frequencyIndex descending
    """
    if not poi_json or str(poi_json).strip() in ('', 'null'):
        return []
    try:
        raw = poi_json if isinstance(poi_json, list) else json.loads(str(poi_json))
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"POI JSON parse error: {e}")
        return []
    if not isinstance(raw, list):
        return []

    parsed = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            continue
        if 'lat' not in item or 'lng' not in item:
            continue
        try:
            lat = float(item['lat'])
            lng = float(item['lng'])
        except (ValueError, TypeError):
            continue
        if not (51.8 <= lat <= 52.5 and 11.4 <= lng <= 12.0):
            logger.warning(f"POI {i} outside Magdeburg range: lat={lat:.4f} lng={lng:.4f}")
        category = item.get('category', 'POI')
        parsed.append({
            'type': _map_category_to_type(category),
            'x': lng,
            'y': lat,
            '_meta': {
                'name':      item.get('name', 'Unknown'),
                'category':  category,
                'frequency': item.get('frequencyIndex', 0),
                'transport': item.get('transportMode', 'UNKNOWN'),
                'poi_id':    item.get('id', f'poi-{i}'),
            },
        })

    if not parsed:
        return []
    parsed.sort(key=lambda p: (
        p['type'] != 'home',
        -p.get('_meta', {}).get('frequency', 0),
    ))
    logger.info(f"Parsed {len(parsed)} POIs. Types: {_count_poi_types(parsed)}")
    return parsed


def _count_poi_types(pois: List[Dict]) -> Dict[str, int]:
    """Count POIs by type."""
    counts: Dict[str, int] = {'home': 0, 'work': 0, 'leisure': 0, 'school': 0}
    for p in pois:
        t = p.get('type', 'leisure')
        counts[t] = counts.get(t, 0) + 1
    return counts


def compute_poi_accessibility(pois: List[Dict]) -> Dict[str, float]:
    """
    Calculate modal accessibility scores from POI transport-mode metadata.

    Maps survey transport codes (WALKING, CYCLING, BUS, …) to DeepHOTCO
    modes and normalises to [0.2, 1.0]. Returns 0.5 for all modes when
    no transport metadata is available.
    """
    default = {m: 0.5 for m in MODES_5}
    if not pois:
        return default

    counts = {k: 0 for k in (
        'WALKING','CYCLING','E_BIKE','BUS','TRAM','TRAIN',
        'CAR_DRIVER','CAR_PASSENGER','E_SCOOTER','MOTORBIKE','TAXI',
    )}
    for p in pois:
        t = p.get('_meta', {}).get('transport')
        if t and t in counts:
            counts[t] += 1

    total = sum(counts.values())
    if total == 0:
        return default

    acc = {
        'car':       (counts['CAR_DRIVER'] + counts['CAR_PASSENGER'] + counts['TAXI'])  / total,
        'bike':      (counts['CYCLING']    + counts['E_BIKE']        + counts['E_SCOOTER']) / total,
        'pt':        (counts['BUS']        + counts['TRAM']          + counts['TRAIN'])  / total,
        'walk':       counts['WALKING'] / total,
        'car_green':  counts['CAR_DRIVER'] / total,
    }
    return {m: max(0.2, min(1.0, v)) for m, v in acc.items()}


def infer_home_poi(pois: List[Dict]) -> Optional[Dict]:
    """
    Infer home POI from type and frequency.

    Returns the highest-frequency 'home' POI, or the highest-frequency POI
    overall as a fallback. Returns None for empty lists.
    """
    if not pois:
        return None
    homes = [p for p in pois if p['type'] == 'home']
    pool  = homes if homes else pois
    return max(pool, key=lambda p: p.get('_meta', {}).get('frequency', 0))


def generate_synthetic_poi(
    center_lat: float = 52.120278,
    center_lng: float = 11.627778,
) -> List[Dict]:
    """
    Generate a synthetic home POI at Magdeburg city centre.
    Used as fallback when agent has no real POIs.
    """
    logger.info("Generating synthetic POI (agent has no real POIs)")
    return [{
        'type': 'home',
        'x': center_lng,
        'y': center_lat,
        '_meta': {
            'name':      'Synthetic Home (Magdeburg Center)',
            'category':  'synthetic',
            'frequency': 3,
            'transport': 'SYNTHETIC',
            'poi_id':    'poi-synthetic-home',
        },
    }]


def validate_poi_coordinates(
    pois: List[Dict],
    min_x: float = 11.5, max_x: float = 11.8,
    min_y: float = 52.0, max_y: float = 52.3,
) -> bool:
    """
    Validate that all POIs fall within the specified bounding box.
    Returns True for empty lists (technically valid).
    """
    if not pois:
        return True
    valid = True
    for i, p in enumerate(pois):
        x, y = p.get('x'), p.get('y')
        if x is None or y is None:
            logger.error(f"POI {i} missing x/y coordinates")
            valid = False
        elif not (min_x <= x <= max_x and min_y <= y <= max_y):
            logger.warning(
                f"POI {i} outside bounding box: x={x:.4f} y={y:.4f}. "
                f"Meta: {p.get('_meta', {})}"
            )
            valid = False
    return valid


def poi_summary(pois: List[Dict]) -> str:
    """Return a one-line human-readable summary string for logging."""
    if not pois:
        return "0 POIs"
    counts  = _count_poi_types(pois)
    home    = infer_home_poi(pois)
    summary = f"{len(pois)} POIs: "
    summary += ", ".join(f"{n} {t}" for t, n in counts.items() if n > 0)
    if home:
        name = home.get('_meta', {}).get('name', 'Unknown')
        freq = home.get('_meta', {}).get('frequency', 0)
        summary += f". Home: {name} (freq={freq})"
    return summary

# =============================================================================
# §16  REPORTING
# =============================================================================

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles NumPy int64, float64, ndarray, and bool_."""
    def default(self, obj):
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
        if isinstance(obj, (np.bool_, bool)): return bool(obj)
        return super().default(obj)


def generate_parsing_report(stats: Dict) -> Dict:
    """Generate a structured parsing report from accumulated stats."""
    return {
        'timestamp':          str(np.datetime64('now')),
        'parser_version':     '4.2.0',
        'summary': {
            'n_total_responses': stats.get('n_total', 0),
            'n_valid':           stats.get('n_valid', 0),
            'n_excluded':        stats.get('n_excluded', 0),
            'exclusion_rate':    stats.get('n_excluded', 0) / max(stats.get('n_total', 1), 1),
        },
        'exclusion_reasons': stats.get('exclusion_reasons', {}),
        'missing_data':      stats.get('missing_data', {}),
        'mode_prevalence':   stats.get('mode_prevalence', {}),
        'icc_results':       stats.get('icc_results', {}),
        'cronbach_alpha':    stats.get('cronbach_alpha', {}),
        'outliers':          stats.get('outliers', {}),
    }


def save_report(report: Dict, output_path: str) -> None:
    """Save report dict to JSON file using NumpyEncoder."""
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder)
    logger.info(f"Report saved to {output_path}")
