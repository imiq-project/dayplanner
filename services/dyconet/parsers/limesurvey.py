"""
=============================================================================
LIMESURVEY PARSER — HotCo-C / DeepHOTCO v4.2
=============================================================================
Survey-specific parsing logic: frequencies, needs, beliefs, valences,
tolerances, demographics, and POIs.

Consolidated from: main_parser.py · mobil_parser.py · app_parser.py
                   section_parsers.py

Public API (unchanged from original package):
    load_data_from_limesurvey(json_path, config_path=None, device='cpu')
    LimeSurveyParser(config_path=None)

All support utilities (constants, imputation, normalization, POI helpers)
are imported from parser_support.py which lives alongside this file.

INDEX
─────
  §A  Imports and logging setup
  §B  MobilParser          — MOBIL section: frequencies, tolerances, stressors, feasibility
  §C  AppParser            — APP section: needs, beliefs, ranking, imputation wrapper
  §D  EmovalParser         — Emotional valences for modes
  §E  ValuesParser         — Schwartz value dimensions
  §F  ProfileParser        — Demographic profile
  §G  POIParser            — Points of Interest (delegates to parser_support)
  §H  LimeSurveyParser     — Orchestrator: all sections → PyTorch tensors
  §I  Public convenience   — load_data_from_limesurvey()

Version: 4.2.0
=============================================================================
"""

# =============================================================================
# §A  IMPORTS AND LOGGING
# =============================================================================

import json
import math
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from scipy.optimize import minimize

# All reusable helpers live in parser_support — no inline duplication
from .parser_support import (
    # Constants
    MODES_4, MODES_5, NEED_NAMES, STRESSOR_NAMES, VALUE_NAMES,
    NEED_MAPPING, JSON_MODE_MAPPING,
    EMPIRICAL_POP_MEANS, IMPUTATION_COEFFICIENTS, MEAN_FREQ, BELIEF_CONSTRAINTS,
    # Imputation
    impute_full_beliefs, estimate_valence_from_beliefs,
    # Config and validation
    load_config, validate_response,
    # JSON helpers
    safe_json_load, extract_nested_value,
    # Scale transforms
    normalize_range,
    # Statistics
    compute_icc, cronbachs_alpha, check_mode_prevalence,
    # Demographics
    encode_demographic,
    # POI
    parse_poi_list, compute_poi_accessibility, generate_synthetic_poi,
    # Reporting
    generate_parsing_report, save_report,
    # Missing data
    handle_missing,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
)
logging.getLogger("parser_support").setLevel(logging.ERROR)
logger = logging.getLogger("limesurvey_parser")

# Default fallback config when none is provided
_DEFAULT_CONFIG = {
    'frequencies': {'scale_min': 1, 'scale_max': 5, 'aggregation': 'mean'},
    'stressors': {
        'tolerances': {
            'rain':        {'variables': ['RA1', 'RA2', 'RA3'], 'scale': [1, 7]},
            'crowding':    {'variables': ['CA1', 'CA2', 'CA3'], 'scale': [1, 7]},
            'darkness':    {'variables': ['DA1', 'DA2', 'DA3'], 'scale': [1, 7]},
            'traffic':     {'variables': ['TA1', 'TA2'],        'scale': [1, 7]},
            'temperature': {'variables': ['TEA1','TEA2','TEA3'],'scale': [1, 7]},
        },
        'stressor_values': {'default': 0.0},
    },
}


# =============================================================================
# §B  MobilParser
# =============================================================================

class MobilParser:
    """
    Parser for MOBIL section of LimeSurvey data.

    Outputs:
        frequencies  — {mode: float [0,1]}  (usage frequency, used as soft targets)
        tolerances   — {stressor: float [0,1]}  (psychological resilience)
        stressors    — {stressor: float [0,1]}  (environmental context; usually 0)
        feasibility  — {mode: 0.0 or 1.0}  (access / ownership)
        metadata     — {agent_id, timestamp, age, gender}
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config         = config or _DEFAULT_CONFIG
        self.freq_config    = self.config.get('frequencies', _DEFAULT_CONFIG['frequencies'])
        self.stressor_config= self.config.get('stressors',   _DEFAULT_CONFIG['stressors'])
        # GRM tolerance model — calibrated externally by LimeSurveyParser
        self.grm_enabled    = False
        self.grm_params: Optional[Dict] = None   # stressor → {a, b, K}
        self.grm_norm:   Optional[Dict] = None   # stressor → (mu, sigma)

    # ------------------------------------------------------------------
    # Top-level parse
    # ------------------------------------------------------------------

    def parse(self, input_data: Union[str, Dict]) -> Dict:
        """Parse complete MOBIL section and return structured dict."""
        if isinstance(input_data, str):
            try:
                data = json.loads(input_data)
            except json.JSONDecodeError:
                logger.error("MobilParser: invalid JSON string")
                return self._empty_result()
        else:
            data = input_data
        if not data:
            return self._empty_result()
        return {
            'frequencies': self.parse_frequencies(data),
            'tolerances':  self.parse_tolerances(data),
            'stressors':   self.parse_stressors(data),
            'feasibility': self.parse_feasibility(data),
            'metadata':    self.extract_metadata(data),
        }

    # ------------------------------------------------------------------
    # Frequencies
    # ------------------------------------------------------------------

    def parse_frequencies(self, data: Dict) -> Dict[str, float]:
        """
        Parse mode usage frequencies.

        Aggregation rules (per config):
            CAR:  max(s{n}_car, s{n}_pass, s{n}_taxi)  → mean/max across S1+S2
            BIKE: max(s{n}_bike, s{n}_bikeshare)
            PT:   mean(s{n}_pt, s{n}_train)
            WALK: s{n}_walk

        Scale: 1=Never … 5=Daily → normalised to [0,1].
        """
        MODE_SUBMODES = self.freq_config.get('mode_submodes', {
            'car':  ['car', 'pass', 'taxi'],
            'bike': ['bike', 'bikeshare'],
            'pt':   ['pt', 'train'],
            'walk': ['walk'],
        })
        _agg_rules = self.config.get('modes', {}).get('aggregation_rules', {})
        MODE_INTRA_AGG = {
            m: _agg_rules.get(m, {}).get('aggregation', 'max')
            for m in ['car', 'bike', 'pt', 'walk', 'car_green']
        }
        MODE_INTRA_AGG.setdefault('pt', 'mean')

        scale_min   = self.freq_config.get('scale_min', 1)
        scale_max   = self.freq_config.get('scale_max', 5)
        aggregation = self.freq_config.get('aggregation', 'mean')

        # Discover scenario prefixes dynamically (s1, s2, …)
        scenario_prefixes = sorted(set(
            k.split('_')[0] for k in data
            if k.startswith('s') and '_' in k and k.split('_')[0][1:].isdigit()
        ))

        frequencies: Dict[str, float] = {}
        for mode in MODES_4:
            submodes = MODE_SUBMODES.get(mode, [mode])
            scenario_vals = []
            for prefix in scenario_prefixes:
                sub_vals = [
                    self._safe_float(data[f"{prefix}_{sm}"], scale_min)
                    for sm in submodes if f"{prefix}_{sm}" in data
                ]
                if sub_vals:
                    intra = MODE_INTRA_AGG.get(mode, 'max')
                    scenario_vals.append(
                        sum(sub_vals) / len(sub_vals) if intra == 'mean' else max(sub_vals)
                    )
            if scenario_vals:
                raw = (sum(scenario_vals)/len(scenario_vals) if aggregation == 'mean'
                       else max(scenario_vals))
            else:
                raw = self._fallback_frequency_lookup(data, mode, scale_min)
            frequencies[mode] = normalize_range(raw, scale_min, scale_max)
        return frequencies

    def _fallback_frequency_lookup(self, data: Dict, mode: str, default: float) -> float:
        """Try legacy key formats (use_car, freq_car, SQ001…) for backward compat."""
        legacy = {
            'car':  ['use_car',  'freq_car',  'SQ001'],
            'bike': ['use_bike', 'freq_bike', 'SQ002'],
            'pt':   ['use_pt',   'freq_pt',   'SQ003'],
            'walk': ['use_walk', 'freq_walk', 'SQ004'],
        }
        for key in legacy.get(mode, []):
            if key in data:
                return self._safe_float(data[key], default)
        for key in self.config.get('modes', {}).get(mode, []):
            if key in data:
                return self._safe_float(data[key], default)
        return default

    # ------------------------------------------------------------------
    # Tolerances
    # ------------------------------------------------------------------

    def parse_tolerances(self, data: Dict) -> Dict[str, float]:
        """
        Parse psychological tolerances for each stressor.

        Uses GRM (Graded Response Model) when calibrated by LimeSurveyParser;
        falls back to 1 − normalized(mean(annoyance_items)).
        """
        tol_config = self.stressor_config.get(
            'tolerances', _DEFAULT_CONFIG['stressors']['tolerances']
        )
        if self.grm_enabled and self.grm_params and self.grm_norm:
            return self._tolerances_grm(data, tol_config)
        return self._tolerances_fallback(data, tol_config)

    def _tolerances_grm(self, data: Dict, tol_config: Dict) -> Dict[str, float]:
        eps = 0.03
        tolerances: Dict[str, float] = {}
        for stressor in STRESSOR_NAMES:
            cfg  = tol_config.get(stressor, {})
            vars_ = cfg.get('variables', [])
            scale = cfg.get('scale', [1, 7])
            K    = int(scale[1])
            x    = []
            for v in vars_:
                if v in data and data[v] not in (None, ''):
                    y = self._safe_float(data[v], float('nan'))
                    if not math.isnan(y):
                        x.append(max(1, min(K, int(round(y)))))
            if not x or stressor not in self.grm_params:
                tolerances[stressor] = 0.5
                continue
            theta = self._grm_score_person_theta(stressor, x)
            mu, sigma = self.grm_norm.get(stressor, (0.0, 1.0))
            tol = 0.5 if sigma <= 1e-8 else 0.5 * (1.0 + math.erf((theta-mu)/(sigma*math.sqrt(2))))
            tolerances[stressor] = max(0.0, min(1.0, eps + (1.0 - 2*eps) * tol))
        return tolerances

    def _tolerances_fallback(self, data: Dict, tol_config: Dict) -> Dict[str, float]:
        tolerances: Dict[str, float] = {}
        for stressor in STRESSOR_NAMES:
            cfg = tol_config.get(stressor)
            if not cfg:
                tolerances[stressor] = 0.5
                continue
            variables = cfg.get('variables', [])
            scale     = cfg.get('scale', [1, 7])
            vals      = [self._safe_float(data[v]) for v in variables if v in data and data[v]]
            if not vals:
                tolerances[stressor] = 0.5
                continue
            norm_ann = normalize_range(sum(vals)/len(vals), scale[0], scale[1])
            tolerances[stressor] = max(0.0, min(1.0, 1.0 - norm_ann))
        return tolerances

    def _grm_score_person_theta(self, stressor: str, x_ordinal: List[int]) -> float:
        """MAP scoring of latent theta for one stressor given ordinal responses."""
        params = self.grm_params.get(stressor)
        if not params:
            return 0.0
        a = np.array(params['a'], dtype=float)
        b = np.array(params['b'], dtype=float)
        K = int(params.get('K', 7))
        grid    = np.linspace(-4, 4, 161)
        logpost = np.empty_like(grid)

        def sig(u: float) -> float:
            return (1.0/(1.0+math.exp(-u))) if u >= 0 else (ez:=math.exp(u), ez/(1+ez))[1]

        for idx, th in enumerate(grid):
            lp = -0.5 * th * th
            for j, y in enumerate(x_ordinal):
                y   = max(1, min(K, int(y)))
                aj  = float(a[j])
                bj  = b[j]
                S   = lambda k: (1.0 if k <= 1 else (0.0 if k >= K+1
                                 else sig(aj*(th - float(bj[k-2])))))
                pk  = max(1e-12, min(1.0, S(y) - S(y+1)))
                lp += math.log(pk)
            logpost[idx] = lp
        return float(grid[int(np.argmax(logpost))])

    # ------------------------------------------------------------------
    # Stressors
    # ------------------------------------------------------------------

    def parse_stressors(self, data: Dict) -> Dict[str, float]:
        """
        Parse environmental context stressors.

        In static surveys these are almost always 0 (neutral scenario).
        The training loop should override these with scenario-specific values.
        """
        scenario_keys = {
            'rain':        ['SC_weather',  'scenario_rain'],
            'crowding':    ['SC_crowd',    'scenario_crowd'],
            'darkness':    ['SC_time',     'scenario_time'],
            'traffic':     ['SC_traffic',  'scenario_traffic'],
            'temperature': ['SC_temp',     'scenario_temp'],
        }
        stressors: Dict[str, float] = {}
        for stressor in STRESSOR_NAMES:
            val = 0.0
            for key in scenario_keys.get(stressor, []):
                if key in data:
                    raw = str(data[key])
                    if stressor == 'rain'        and raw in ['2', 'rainy', 'wet']:  val = 0.8
                    elif stressor == 'darkness'  and raw in ['2', 'night', 'dusk']: val = 0.9
                    elif stressor == 'crowding'  and raw in ['3', 'high', 'full']:  val = 0.9
                    break
            stressors[stressor] = val
        return stressors

    # ------------------------------------------------------------------
    # Feasibility
    # ------------------------------------------------------------------

    def parse_feasibility(self, data: Dict) -> Dict[str, float]:
        """
        Parse mode feasibility (ownership / access).

        Priority: explicit keys → frequency inference → mode default.
        """
        explicit = {
            'car':  ['avail_car',  'd_car_avail',  'F_CAR',  'has_car'],
            'bike': ['avail_bike', 'd_bike_avail', 'F_BIKE', 'has_bike'],
            'pt':   ['avail_pt',   'd_pt_connect', 'F_PT',   'has_pt'],
            'walk': ['can_walk',   'physical_ability', 'F_WALK'],
        }
        defaults = {'car': 0.5, 'bike': 0.5, 'pt': 1.0, 'walk': 1.0}
        feasibility: Dict[str, float] = {}
        for mode in MODES_4:
            feas = None
            for key in explicit.get(mode, []):
                if key in data:
                    v = str(data[key]).lower()
                    feas = 0.0 if v in ('2','no','0','false') else 1.0 if v in ('1','yes','true') else None
                    if feas is not None:
                        break
            if feas is None:
                feas = self._infer_feasibility_from_freq(data, mode)
            if feas is None:
                feas = defaults.get(mode, 0.5)
            feasibility[mode] = feas
        return feasibility

    def _infer_feasibility_from_freq(self, data: Dict, mode: str) -> Optional[float]:
        """Infer feasibility from usage frequency keys."""
        submodes_map = {
            'car':  ['car','pass','taxi'],
            'bike': ['bike','bikeshare'],
            'pt':   ['pt','train'],
            'walk': ['walk'],
        }
        submodes = submodes_map.get(mode, [mode])
        has_data = False
        for key, val in data.items():
            if not (key.startswith('s') and '_' in key):
                continue
            parts = key.split('_', 1)
            if len(parts) == 2 and parts[1] in submodes:
                has_data = True
                if self._safe_float(val, 1) > 1:
                    return 1.0
        return 0.5 if has_data else None

    # ------------------------------------------------------------------
    # Car-green frequency (for prevalence check)
    # ------------------------------------------------------------------

    def parse_car_green_frequency(self, data: Dict) -> Optional[float]:
        """Extract car_green (EV/PHEV) frequency for prevalence check."""
        scale_min = self.freq_config.get('scale_min', 1)
        scale_max = self.freq_config.get('scale_max', 5)
        ev_submodes = ['bev', 'phev']
        ev_vals = [
            self._safe_float(data[k], scale_min)
            for k in data
            if (k.startswith('s') and '_' in k and k.split('_',1)[1] in ev_submodes)
        ]
        if ev_vals:
            mx = max(ev_vals)
            return normalize_range(mx, scale_min, scale_max) if mx > scale_min else 0.0
        legacy = ['use_bev','freq_bev','use_ev','freq_ev','use_phev','freq_phev',
                  'use_electric','freq_electric','SQ_BEV','SQ_PHEV','SQ_EV']
        for key in legacy:
            if key in data and data[key] not in (None, ''):
                val = self._safe_float(data[key], 0.0)
                if val > 0:
                    return normalize_range(val, scale_min, scale_max)
        return None

    # ------------------------------------------------------------------
    # Metadata and helpers
    # ------------------------------------------------------------------

    def extract_metadata(self, data: Dict) -> Dict:
        return {
            'agent_id':  data.get('id', 'unknown'),
            'timestamp': data.get('submitdate', ''),
            'age':       data.get('age', 0),
            'gender':    data.get('gender', 'unknown'),
        }

    def _empty_result(self) -> Dict:
        return {
            'frequencies': {m: 0.0 for m in MODES_4},
            'tolerances':  {s: 0.5 for s in STRESSOR_NAMES},
            'stressors':   {s: 0.0 for s in STRESSOR_NAMES},
            'feasibility': {m: 1.0 for m in MODES_4},
            'metadata':    {},
        }

    @staticmethod
    def _safe_float(value: Any, default: float = 0.0) -> float:
        if value is None or value == '':
            return default
        try:
            return float(value)
        except (ValueError, TypeError):
            return default


# =============================================================================
# §C  AppParser
# =============================================================================

class AppParser:
    """
    Parser for APP section (needs, beliefs, ranking).

    Parsing order within a single agent:
        1. parse_needs()             — need importance ratings [0,1]
        2. _parse_beliefs_with_meta() — full belief matrix + audit metadata
    """

    def __init__(self, config: Dict):
        self.config       = config
        self.needs_config = config.get('needs', {})

    def parse(
        self,
        app_json: Any,
        frequencies: Optional[Dict] = None,
        valences:    Optional[Dict] = None,
    ) -> Dict:
        """
        Parse complete APP section.

        Args:
            app_json    : Raw JSON string or dict from LimeSurvey 'APP' field.
            frequencies : Optional {mode: float [0,1]} for per-agent imputation.
            valences    : Optional {mode: float [-1,1]} for per-agent imputation.

        Returns:
            {'needs', 'beliefs', 'ranking', 'metadata'}
        """
        data = safe_json_load(app_json)
        if not data:
            return self._empty_result()
        answers     = data.get('answers', {})
        ratings     = answers.get('ratings', {})
        beliefs_raw = answers.get('beliefs', {})
        ranking     = answers.get('ranking', [])

        needs = self.parse_needs(ratings)
        parsed_beliefs, beliefs_meta = self._parse_beliefs_with_meta(
            beliefs_raw, ranking,
            need_importance=needs,
            frequencies=frequencies,
            valences=valences,
        )
        base_meta = self.extract_metadata(answers)
        base_meta['belief_imputation'] = beliefs_meta

        return {
            'needs':    needs,
            'beliefs':  parsed_beliefs,
            'ranking':  self._clean_ranking(ranking),
            'metadata': base_meta,
        }

    def parse_needs(self, ratings: Dict) -> Dict[str, float]:
        """
        Parse need importance ratings (1-7 Likert) → [0,1].

        Note on 'privacy' / 'crowding':
            The survey asks "How important is PRIVACY when using transport?"
            The JSON key happens to be 'crowding' (legacy naming), but NO
            inversion is applied — higher raw value = higher privacy importance.
        """
        needs: Dict[str, float] = {}
        for model_need in NEED_NAMES:
            json_key = NEED_MAPPING.get(model_need, model_need)
            raw = ratings.get(json_key)
            if raw is not None and raw != '':
                try:
                    needs[model_need] = normalize_range(float(raw), 1, 7, 0.0, 1.0)
                except ValueError:
                    needs[model_need] = 0.5
            else:
                needs[model_need] = 0.5
        return needs

    def _parse_beliefs_with_meta(
        self,
        beliefs_data: Dict,
        ranking: List[str],
        need_importance: Optional[Dict] = None,
        frequencies:     Optional[Dict] = None,
        valences:        Optional[Dict] = None,
    ) -> Tuple[Dict, Dict]:
        """
        Build full belief matrix [n_modes × n_needs] with imputation metadata.

        Returns:
            beliefs : {mode: {need: float [-1,+1]}}
            meta    : per-mode audit metadata dict
        """
        beliefs: Dict[str, Dict] = {}
        meta:    Dict[str, Dict] = {}
        clean_ranking = self._clean_ranking(ranking)

        for model_mode in MODES_5:
            found_data: Dict = {}
            for key in JSON_MODE_MAPPING.get(model_mode, [model_mode]):
                if key in beliefs_data:
                    found_data = beliefs_data[key]
                    break

            partial: Dict[str, float] = {}
            for model_need in NEED_NAMES:
                jk = NEED_MAPPING.get(model_need, model_need)
                if jk in found_data:
                    raw = found_data[jk]
                    if raw is not None and raw != '':
                        try:
                            partial[jk] = float(raw)
                        except ValueError:
                            pass

            mode_freq    = frequencies.get(model_mode) if frequencies else None
            mode_valence = valences.get(model_mode)    if valences    else None

            full_vec = impute_full_beliefs(
                model_mode, partial,
                need_ranking=clean_ranking,
                need_importance=need_importance,
                frequency=mode_freq,
                valence=mode_valence,
            )
            warnings = full_vec.pop('_constraint_warnings', [])

            observed = [n for n in NEED_NAMES if NEED_MAPPING.get(n, n) in partial]
            imputed  = [n for n in NEED_NAMES if n not in observed]
            pop      = EMPIRICAL_POP_MEANS.get(model_mode, {})
            residuals = [abs(float(full_vec.get(n,0.0)) - float(pop.get(n,0.0))) for n in imputed]
            residual  = sum(residuals)/len(residuals) if residuals else 0.0
            mean_f    = MEAN_FREQ.get(model_mode, 0.3)

            meta[model_mode] = {
                'n_observed':          len(observed),
                'n_imputed':           len(imputed),
                'observed_needs':      observed,
                'imputed_needs':       imputed,
                'constraint_warnings': warnings,
                'mode_residual':       round(residual, 4),
                'freq_dev':            round((mode_freq - mean_f) if mode_freq is not None else 0.0, 4),
                'valence':             round(float(mode_valence or 0.0), 4),
            }
            beliefs[model_mode] = full_vec

        return beliefs, {'per_mode': meta}

    def _clean_ranking(self, raw_ranking: List[str]) -> List[str]:
        """Map raw ranking strings to internal need names."""
        inv = {v: k for k, v in NEED_MAPPING.items()}
        return [inv.get(r, r) for r in raw_ranking if inv.get(r, r) in NEED_NAMES]

    def extract_metadata(self, answers: Dict) -> Dict:
        meta: Dict = {'custom_needs': answers.get('custom', []),
                      'skipped_modes': answers.get('skippedModes', [])}
        m = answers.get('meta', {})
        if m:
            meta.update({
                'display_order_needs':    m.get('displayOrderNeeds', []),
                'display_order_modes':    m.get('displayOrderModes', []),
                'response_time_seconds':  m.get('responseTimeSeconds'),
                'timestamp':              m.get('timestamp'),
                'user_agent':             m.get('userAgent'),
            })
        return meta

    def _empty_result(self) -> Dict:
        return {
            'needs':    {n: 0.5 for n in NEED_NAMES},
            'beliefs':  {},
            'ranking':  [],
            'metadata': {},
        }


def build_belief_matrix(
    beliefs_dict: Dict,
    ordered_modes: List[str],
    ordered_needs: List[str],
) -> np.ndarray:
    """
    Convert belief dict to numpy array [n_modes, n_needs].

    Clamps to [-1, +1] and warns if values outside that range are detected.
    """
    n_modes = len(ordered_modes)
    n_needs = len(ordered_needs)
    matrix  = np.zeros((n_modes, n_needs), dtype=np.float32)
    for i, mode in enumerate(ordered_modes):
        for j, need in enumerate(ordered_needs):
            matrix[i, j] = beliefs_dict.get(mode, {}).get(need, 0.0)
    if np.any(np.abs(matrix) > 1.5):
        logger.warning(
            f"Belief values outside [-1,+1] detected "
            f"(range [{matrix.min():.2f},{matrix.max():.2f}]). Clamping."
        )
        matrix = np.clip(matrix, -1.0, 1.0)
    return matrix


# =============================================================================
# §D  EmovalParser
# =============================================================================

class EmovalParser:
    """Parser for emotional valences (EMOVAL section)."""

    def __init__(self, config: Dict):
        self.config         = config
        self.valence_config = config.get('valences', {})

    def parse(
        self,
        emoval_json: str,
        beliefs: Optional[Dict] = None,
        values:  Optional[Dict] = None,
    ) -> Dict[str, float]:
        """
        Parse emotional valences for modes → [-1, +1].

        Falls back to estimate_valence_from_beliefs() when data is missing.
        """
        data        = safe_json_load(emoval_json)
        raw_answers = data.get('answers', data) if isinstance(data, dict) else {}
        valences:   Dict[str, float] = {}

        for model_mode in MODES_5:
            found = []
            for key in JSON_MODE_MAPPING.get(model_mode, [model_mode]):
                val = raw_answers.get(key)
                if val is not None and val not in ('NA', ''):
                    try:
                        found.append(float(val))
                    except ValueError:
                        pass
            if found:
                avg = sum(found) / len(found)
                # Auto-detect scale: -3…+3 vs 1…7
                valences[model_mode] = (
                    normalize_range(avg, 1, 7, -1, 1)
                    if (avg > 3.0 or avg < -3.0)
                    else normalize_range(avg, -3, 3, -1, 1)
                )
            else:
                if beliefs and values and model_mode in beliefs:
                    est = estimate_valence_from_beliefs(beliefs[model_mode], values)
                    valences[model_mode] = est * 2.0 - 1.0
                else:
                    valences[model_mode] = 0.0

        return valences


# =============================================================================
# §E  ValuesParser
# =============================================================================

class ValuesParser:
    """Parser for Schwartz value dimensions."""

    def __init__(self, config: Dict):
        self.config       = config
        self.values_cfg   = config.get('values', {})

    def parse(self, values_json: str) -> Dict:
        """Parse and aggregate Schwartz values → {dimension: float [0,1]}."""
        data = safe_json_load(values_json)
        if not data:
            return self._default()
        answers   = data.get('answers', data) if isinstance(data, dict) else {}
        agg       = self.values_cfg.get('aggregation', {})
        scale_min = self.values_cfg.get('scale_min', 1)
        scale_max = self.values_cfg.get('scale_max', 7)
        neutral   = (scale_min + scale_max) / 2.0

        values:     Dict[str, float] = {}
        values_raw: Dict[str, float] = {}

        for dim, items in agg.items():
            scores = []
            for item in items:
                raw = answers.get(item)
                score = neutral if (raw is None or raw in ('', 'NA')) else raw
                try:
                    s = float(score)
                    values_raw[item] = s
                    scores.append(normalize_range(s, scale_min, scale_max))
                except (ValueError, TypeError):
                    pass
            values[dim] = float(np.mean(scores)) if scores else 0.5

        for dim in VALUE_NAMES:
            values.setdefault(dim, 0.5)
        return {'values': values, 'values_raw': values_raw}

    def _default(self) -> Dict:
        return {'values': {d: 0.5 for d in VALUE_NAMES}, 'values_raw': {}}


# =============================================================================
# §F  ProfileParser
# =============================================================================

class ProfileParser:
    """Parser for demographic profile (PROFILE section)."""

    def __init__(self, config: Dict):
        self.config    = config
        self.demo_cfg  = config['metadata'].get('demographics', {})

    def parse(self, profile_json: str) -> Dict:
        """Parse demographics → encoded profile dict."""
        data = safe_json_load(profile_json)
        if not data:
            return {}
        answers = data.get('answers', {})
        profile: Dict[str, Any] = {}

        for var in self.demo_cfg.get('include', []):
            raw = answers.get(var)
            if var == 'age':
                try:
                    profile[var] = int(raw) if raw else 0
                except (ValueError, TypeError):
                    profile[var] = 0
            elif var == 'ovgu_active':
                profile[var] = str(raw).lower() == 'yes'
            elif var in ('gender', 'education', 'income'):
                profile[f'{var}_raw'] = raw
                profile[var]          = encode_demographic(raw, var, self.config)
            else:
                profile[var] = raw

        connection = answers.get('connection', [])
        if not isinstance(connection, list):
            connection = []
        profile['magdeburg_connection'] = connection
        profile['lives_in_magdeburg']   = any('live' in str(c) for c in connection)
        return profile


# =============================================================================
# §G  POIParser
# =============================================================================

class POIParser:
    """
    Parser for Points of Interest (POI section).
    Delegates to parse_poi_list / compute_poi_accessibility in parser_support.
    """

    def __init__(self, config: Dict):
        self.config   = config
        self.poi_cfg  = config['metadata'].get('poi', {})

    def parse(self, poi_json: Optional[str]) -> Dict:
        """Parse POI JSON and return Game Tab–compatible structure."""
        if not self.poi_cfg.get('include', False):
            return {'pois': [], 'n_pois': 0, 'mode_accessibility': {}}
        try:
            pois = parse_poi_list(poi_json)
            if not pois:
                return {'pois': [], 'n_pois': 0, 'mode_accessibility': {}}
            return {
                'pois':              pois,
                'n_pois':            len(pois),
                'mode_accessibility':compute_poi_accessibility(pois),
            }
        except Exception as e:
            logger.error(f"POI parsing error: {e}", exc_info=True)
            return {'pois': [], 'n_pois': 0, 'mode_accessibility': {}}


# =============================================================================
# §H  LimeSurveyParser  (Orchestrator)
# =============================================================================

class LimeSurveyParser:
    """
    Orchestrates all section parsers and converts survey responses to
    PyTorch tensors compatible with the HotCo-C / DeepHOTCO model.

    Parsing pipeline (per agent):
        PROFILE → MOBIL → VALUES → EMOVAL (preliminary) → APP (with imputation) → POI

    Outputs (from parse_json()):
        needs, beliefs, valences, tolerances, stressors, feasibility,
        initial_state, values, frequencies,
        soft_targets, hard_targets,            ← merged S1+S2
        soft_targets_s1/s2, hard_targets_s1/s2, ← scenario-split
        scenario_flip_mask, metadata, modes, n_modes,
        belief_audit, belief_centering
    """

    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            config_path = str(Path(__file__).parent / 'config.yaml')
        self.config = load_config(config_path)

        self.mobil_parser   = MobilParser(self.config)
        self.app_parser     = AppParser(self.config)
        self.emoval_parser  = EmovalParser(self.config)
        self.values_parser  = ValuesParser(self.config)
        self.profile_parser = ProfileParser(self.config)
        self.poi_parser     = POIParser(self.config)

        self.n_modes: Optional[int] = None
        self.modes:   Optional[List[str]] = None
        self.center_beliefs: bool = self._resolve_center_beliefs()
        self.print_belief_audit_per_agent: bool = bool(
            self.config.get('logging', {}).get('belief_audit_per_agent', False)
        )
        self.stats: Dict[str, Any] = {
            'n_total': 0, 'n_valid': 0, 'n_excluded': 0,
            'exclusion_reasons': {}, 'missing_data': {},
            'mode_prevalence': {}, 'icc_results': {}, 'cronbach_alpha': {},
            'outliers': {},
            'belief_audit': {
                'agents_with_beliefs': 0,
                'total_observed_cells': 0,
                'total_imputed_cells': 0,
                'per_mode': {},
                'per_need': {n: {'observed': 0, 'imputed': 0} for n in NEED_NAMES},
                'agent_summaries': [],
            },
        }

    def _resolve_center_beliefs(self) -> bool:
        beliefs_cfg = self.config.get('beliefs', {})
        if isinstance(beliefs_cfg, dict) and 'center_beliefs' in beliefs_cfg:
            return bool(beliefs_cfg['center_beliefs'])
        return bool(self.config.get('center_beliefs', False))

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def parse_json(self, json_path: str, device: str = 'cpu') -> Dict:
        """
        Parse a LimeSurvey JSON export and return all tensors for HotCo-C.
        """
        logger.info("=" * 80)
        logger.info(f"LimeSurvey parsing: {json_path}")
        logger.info("=" * 80)

        with open(json_path, 'r', encoding='utf-8') as f:
            raw = json.load(f)
        responses = raw if isinstance(raw, list) else raw.get('responses', [])
        self.stats['n_total'] = len(responses)
        logger.info(f"Total responses: {self.stats['n_total']}")
        logger.info(f"Belief representation: {'centered' if self.center_beliefs else 'raw'}")

        logger.info("\n[PHASE 1] Checking car_green prevalence…")
        self._check_car_green_prevalence(responses)

        logger.info("\n[PHASE 1b] Calibrating GRM tolerances…")
        self._calibrate_grm_tolerances(responses)

        logger.info("\n[PHASE 2] Parsing responses…")
        agents = self._parse_responses(responses)
        self.stats['n_valid']    = len(agents)
        self.stats['n_excluded'] = self.stats['n_total'] - len(agents)
        logger.info(f"Valid: {self.stats['n_valid']}  Excluded: {self.stats['n_excluded']}")

        if not agents:
            raise ValueError("No valid responses to parse!")

        self._print_belief_audit_summary()

        logger.info("\n[PHASE 3] Converting to tensors…")
        tensors = self._agents_to_tensors(agents, device)

        if self.config['frequencies']['compute_icc']:
            logger.info("\n[PHASE 4] Computing ICC…")
            self._compute_icc_stats(agents)

        if self.config['logging']['report']['enabled']:
            logger.info("\n[PHASE 5] Saving report…")
            save_report(generate_parsing_report(self.stats),
                        self.config['logging']['report']['output_file'])

        logger.info("=" * 80)
        logger.info("Parsing complete!")
        logger.info(f"  Modes: {self.n_modes}  Needs: {len(NEED_NAMES)}  N: {tensors['needs'].shape[0]}")
        logger.info("=" * 80)
        return tensors

    # ------------------------------------------------------------------
    # Phase 1: car_green prevalence
    # ------------------------------------------------------------------

    def _check_car_green_prevalence(self, responses: List[Dict]) -> None:
        cg_freqs = []
        for resp in responses:
            mobil_json = resp.get('MOBIL')
            if not mobil_json:
                continue
            try:
                mobil_data = json.loads(mobil_json)
                freq = self.mobil_parser.parse_car_green_frequency(mobil_data)
                if freq is not None:
                    cg_freqs.append(freq)
            except Exception:
                continue
        cg_arr    = np.array(cg_freqs)
        threshold = self.config['modes']['ev_threshold']
        has_prev  = check_mode_prevalence(
            {'car_green': cg_arr},
            'car_green',
            min_users=threshold['min_users'],
            min_percentage=threshold['min_percentage'],
        )
        self.n_modes = 5 if has_prev else 4
        self.modes   = MODES_5 if has_prev else MODES_4
        logger.info(f"{'✅' if has_prev else '❌'} car_green → using {self.n_modes} modes")
        self.stats['mode_prevalence']['car_green'] = {
            'n_users':  int(np.sum(cg_arr >= 0.2)),
            'pct':      float(np.mean(cg_arr >= 0.2)),
            'included': has_prev,
        }

    # ------------------------------------------------------------------
    # Phase 1b: GRM tolerance calibration
    # ------------------------------------------------------------------

    def _calibrate_grm_tolerances(self, responses: List[Dict]) -> None:
        tol_config = self.mobil_parser.stressor_config.get('tolerances', {})
        if not tol_config:
            self.mobil_parser.grm_enabled = False
            return
        X:     Dict[str, List] = {s: [] for s in STRESSOR_NAMES}
        K_map: Dict[str, int]  = {}
        for resp in responses:
            mobil_json = resp.get('MOBIL')
            if not mobil_json:
                continue
            try:
                data = json.loads(mobil_json)
            except Exception:
                continue
            answers = data.get('answers', data) if isinstance(data, dict) else {}
            if not isinstance(answers, dict):
                continue
            for s in STRESSOR_NAMES:
                cfg   = tol_config.get(s, {})
                vars_ = cfg.get('variables', [])
                scale = cfg.get('scale', [1, 7])
                K     = int(scale[1])
                row   = []
                ok    = True
                for v in vars_:
                    val = answers.get(v)
                    if val is None or val == '':
                        ok = False; break
                    try:
                        y = max(1, min(K, int(round(float(val)))))
                        row.append(y)
                    except Exception:
                        ok = False; break
                if ok:
                    X[s].append(row)
                    K_map[s] = K

        grm_params: Dict = {}
        grm_norm:   Dict = {}
        for s in STRESSOR_NAMES:
            xs = np.array(X[s], dtype=int)
            if xs.size == 0 or xs.shape[0] < max(50, xs.shape[1] * 30):
                if xs.size:
                    logger.warning(f"[GRM] '{s}': too few cases (N={xs.shape[0]}). Skipping.")
                continue
            K = int(K_map.get(s, 7))
            try:
                params, mu_sigma = self._fit_grm(xs, K=K)
                grm_params[s] = params
                grm_norm[s]   = mu_sigma
                logger.info(f"[GRM] Fitted '{s}': N={xs.shape[0]} items={xs.shape[1]} K={K}")
            except Exception as e:
                logger.warning(f"[GRM] Failed '{s}': {e}")

        if grm_params:
            self.mobil_parser.grm_enabled = True
            self.mobil_parser.grm_params  = grm_params
            self.mobil_parser.grm_norm    = grm_norm
            logger.info(f"[GRM] Enabled for: {list(grm_params)}")
        else:
            self.mobil_parser.grm_enabled = False
            logger.warning("[GRM] No stressor fitted; using fallback tolerances.")

    def _fit_grm(self, X: np.ndarray, K: int = 7) -> Tuple[Dict, Tuple[float, float]]:
        """Pragmatic GRM fit using proxy theta (standardised mean score) + per-item MLE."""
        N, m    = X.shape
        theta0  = X.mean(axis=1).astype(float)
        mu      = float(theta0.mean())
        sigma   = float(theta0.std(ddof=0) + 1e-8)
        theta   = (theta0 - mu) / sigma

        def sig(u: np.ndarray) -> np.ndarray:
            return 1.0 / (1.0 + np.exp(-u))

        a_list: List[float]       = []
        b_list: List[List[float]] = []

        for j in range(m):
            y = X[:, j].astype(int)
            init_bs = np.array([
                -math.log(max(0.01, min(0.99, np.mean(y >= k))) /
                          (1 - max(0.01, min(0.99, np.mean(y >= k)))))
                for k in range(2, K+1)
            ])

            def unpack(p):
                a = math.exp(p[0])
                b = [p[1]]
                for d in p[2:]:
                    b.append(b[-1] + math.exp(d))
                return a, np.array(b, dtype=float)

            def nll(p):
                a, b  = unpack(p)
                th    = theta.reshape(-1,1)
                S_mid = sig(a * (th - b.reshape(1,-1)))
                Sy    = np.ones(N);  Sy1 = np.zeros(N)
                m2    = y >= 2;      Sy[m2]  = S_mid[m2, y[m2]-2]
                mK    = y <= K-1;    Sy1[mK] = S_mid[mK, y[mK]-1]
                return -float(np.sum(np.log(np.clip(Sy - Sy1, 1e-12, 1.0))))

            ds0 = [math.log(max(0.1, float(init_bs[i]-init_bs[i-1]))) for i in range(1, K-1)]
            p0  = np.array([0.0, float(init_bs[0]) if len(init_bs) else 0.0] + ds0)
            res = minimize(nll, p0, method='L-BFGS-B')
            ah, bh = unpack(res.x)
            a_list.append(float(ah))
            b_list.append([float(v) for v in bh.tolist()])

        return {'a': a_list, 'b': b_list, 'K': int(K)}, (0.0, 1.0)

    # ------------------------------------------------------------------
    # Phase 2: parse responses
    # ------------------------------------------------------------------

    def _parse_responses(self, responses: List[Dict]) -> List[Dict]:
        agents = []
        for i, resp in enumerate(responses):
            valid, reason = validate_response(resp, self.config)
            if not valid:
                self.stats['exclusion_reasons'][reason or 'unknown'] = \
                    self.stats['exclusion_reasons'].get(reason or 'unknown', 0) + 1
                continue
            try:
                agents.append(self._parse_single_response(resp))
            except Exception as e:
                logger.error(f"Error parsing response {i}: {e}")
                self.stats['exclusion_reasons']['parse_error'] = \
                    self.stats['exclusion_reasons'].get('parse_error', 0) + 1
        return agents

    def _parse_single_response(self, resp: Dict) -> Dict:
        """
        Parse one response.

        MOBIL and EMOVAL are parsed BEFORE APP so per-agent frequency and
        valence signals are available for the 3-layer belief imputation.
        """
        agent = {'response_id': resp.get('id')}

        # 1. PROFILE
        agent['profile'] = self.profile_parser.parse(resp.get('PROFILE'))

        # 2. MOBIL
        mobil_data             = self.mobil_parser.parse(resp.get('MOBIL'))
        agent['frequencies']   = mobil_data['frequencies']
        agent['tolerances']    = mobil_data['tolerances']
        agent['stressors']     = mobil_data['stressors']
        agent['feasibility']   = mobil_data['feasibility']
        agent['mobil_metadata']= mobil_data['metadata']

        # Scenario split (R2 fix)
        try:
            _raw = json.loads(resp['MOBIL']) if isinstance(resp.get('MOBIL'), str) else (resp.get('MOBIL') or {})
        except Exception:
            _raw = {}
        agent['_mobil_raw']      = _raw
        agent['frequencies_s1']  = self._extract_scenario_freq(_raw, '1')
        agent['frequencies_s2']  = self._extract_scenario_freq(_raw, '2')

        # 3. VALUES
        values_data      = self.values_parser.parse(resp.get('values'))
        agent['values']  = values_data['values']
        agent['values_raw'] = values_data['values_raw']

        # 4. EMOVAL (before APP — needed for imputation)
        agent['valences'] = self.emoval_parser.parse(
            resp.get('emoval'), beliefs=None, values=agent['values']
        )

        # 5. APP (with per-agent imputation)
        app_data         = self.app_parser.parse(
            resp.get('APP'),
            frequencies=agent['frequencies'],
            valences=agent['valences'],
        )
        agent['needs']        = app_data['needs']
        agent['beliefs']      = app_data['beliefs']
        agent['ranking']      = app_data['ranking']
        agent['app_metadata'] = app_data['metadata']
        self._update_belief_audit(agent)

        # 6. POI
        agent['pois'] = self.poi_parser.parse(resp.get('POI'))

        return agent

    # ------------------------------------------------------------------
    # Belief audit helpers
    # ------------------------------------------------------------------

    def _update_belief_audit(self, agent: Dict) -> None:
        belief_meta = agent.get('app_metadata', {}).get('belief_imputation')
        if not belief_meta:
            return
        audit = self.stats['belief_audit']
        audit['agents_with_beliefs'] += 1
        summary: Dict[str, Any] = {
            'response_id': agent.get('response_id'),
            'per_mode': {}, 'total_observed': 0, 'total_imputed': 0,
        }
        for mode, meta in belief_meta.get('per_mode', {}).items():
            n_obs, n_imp = int(meta.get('n_observed',0)), int(meta.get('n_imputed',0))
            audit['total_observed_cells'] += n_obs
            audit['total_imputed_cells']  += n_imp
            summary['total_observed'] += n_obs
            summary['total_imputed']  += n_imp
            if mode not in audit['per_mode']:
                audit['per_mode'][mode] = {'observed':0,'imputed':0,'agents_touched':0}
            audit['per_mode'][mode]['observed']       += n_obs
            audit['per_mode'][mode]['imputed']        += n_imp
            audit['per_mode'][mode]['agents_touched'] += 1
            for need in meta.get('observed_needs', []):
                if need in audit['per_need']:
                    audit['per_need'][need]['observed'] += 1
            for need in meta.get('imputed_needs', []):
                if need in audit['per_need']:
                    audit['per_need'][need]['imputed'] += 1
            summary['per_mode'][mode] = {
                'observed':       n_obs, 'imputed': n_imp,
                'observed_needs': meta.get('observed_needs',[]),
                'imputed_needs':  meta.get('imputed_needs',[]),
                'mode_residual':  float(meta.get('mode_residual',0.0)),
                'freq_dev':       float(meta.get('freq_dev',0.0)),
                'valence':        float(meta.get('valence',0.0)),
            }
        audit['agent_summaries'].append(summary)
        if self.print_belief_audit_per_agent:
            logger.info(self._format_agent_belief_audit(summary))

    def _format_agent_belief_audit(self, s: Dict) -> str:
        lines = [f"\n[Belief audit] response_id={s['response_id']}",
                 f"  observed={s['total_observed']}  imputed={s['total_imputed']}"]
        for mode, m in s['per_mode'].items():
            lines.append(
                f"  - {mode}: obs={m['observed']} imp={m['imputed']} "
                f"resid={m['mode_residual']:.3f} freq_dev={m['freq_dev']:.3f} val={m['valence']:.3f}"
            )
        return "\n".join(lines)

    def _print_belief_audit_summary(self) -> None:
        audit = self.stats['belief_audit']
        if not audit['agents_with_beliefs']:
            return
        total = audit['total_observed_cells'] + audit['total_imputed_cells']
        obs_p = 100.0 * audit['total_observed_cells'] / max(total, 1)
        logger.info("\n" + "-"*78)
        logger.info("BELIEF IMPUTATION AUDIT")
        logger.info("-"*78)
        logger.info(f"Agents: {audit['agents_with_beliefs']}  "
                    f"Observed: {audit['total_observed_cells']} ({obs_p:.1f}%)  "
                    f"Imputed: {audit['total_imputed_cells']} ({100-obs_p:.1f}%)")
        for mode, m in audit['per_mode'].items():
            d = max(m['observed']+m['imputed'],1)
            logger.info(f"  {mode:<10} obs={m['observed']:<5} imp={m['imputed']:<5} "
                        f"obs%={100*m['observed']/d:5.1f}")
        logger.info("-"*78)

    # ------------------------------------------------------------------
    # Phase 3: tensors
    # ------------------------------------------------------------------

    def _agents_to_tensors(self, agents: List[Dict], device: str) -> Dict:
        N         = len(agents)
        n_modes   = self.n_modes
        n_needs   = len(NEED_NAMES)
        n_stress  = len(STRESSOR_NAMES)
        n_values  = len(VALUE_NAMES)

        needs_arr        = np.zeros((N, n_needs),         dtype=np.float32)
        beliefs_arr      = np.zeros((N, n_modes, n_needs), dtype=np.float32)
        valences_arr     = np.zeros((N, n_modes),          dtype=np.float32)
        tolerances_arr   = np.zeros((N, n_stress),         dtype=np.float32)
        stressors_arr    = np.zeros((N, n_stress),         dtype=np.float32)
        feasibility_arr  = np.ones( (N, n_modes),          dtype=np.float32)
        values_arr       = np.zeros((N, n_values),         dtype=np.float32)
        frequencies_arr  = np.zeros((N, n_modes),          dtype=np.float32)
        freq_s1_arr      = np.zeros((N, n_modes),          dtype=np.float32)
        freq_s2_arr      = np.zeros((N, n_modes),          dtype=np.float32)
        metadata_list: List[Dict] = []

        for i, agent in enumerate(agents):
            for j, need in enumerate(NEED_NAMES):
                needs_arr[i, j] = agent['needs'].get(need, 0.5)
            beliefs_arr[i] = build_belief_matrix(agent['beliefs'], self.modes, NEED_NAMES)
            for j, mode in enumerate(self.modes):
                valences_arr[i, j]    = agent['valences'].get(mode, 0.0)
                tolerances_arr[i, j] if j < n_stress else None
            for j, s in enumerate(STRESSOR_NAMES):
                tolerances_arr[i, j] = agent['tolerances'].get(s, 0.5)
                stressors_arr[i, j]  = agent['stressors'].get(s, 0.0)
            for j, mode in enumerate(self.modes):
                feas = agent.get('feasibility', {}).get(mode)
                feasibility_arr[i, j] = feas if feas is not None else (
                    1.0 if agent['frequencies'].get(mode, 0.0) > 0.01 else 0.5
                )
            for j, dim in enumerate(VALUE_NAMES):
                values_arr[i, j] = agent['values'].get(dim, 0.5)
            for j, mode in enumerate(self.modes):
                frequencies_arr[i, j] = agent['frequencies'].get(mode, 0.0)
                freq_s1_arr[i, j]     = agent.get('frequencies_s1', {}).get(mode, 0.0)
                freq_s2_arr[i, j]     = agent.get('frequencies_s2', {}).get(mode, 0.0)
            metadata_list.append({
                'response_id':   agent['response_id'],
                'profile':       agent['profile'],
                'mobil_metadata':agent['mobil_metadata'],
                'app_metadata':  agent['app_metadata'],
                'pois':          agent['pois'],
                'ranking':       agent['ranking'],
                'values_raw':    agent['values_raw'],
                '_mobil_raw':    agent.get('_mobil_raw', {}),
            })

        # Optional belief centering
        if self.center_beliefs:
            mode_means   = beliefs_arr.mean(axis=0, keepdims=True)
            beliefs_arr  = np.clip(beliefs_arr - mode_means, -1.0, 1.0)
            self.stats['belief_centering'] = {'applied': True, 'representation': 'centered'}
            logger.info(f"[BeliefRepresentation] Centered. Mean→{beliefs_arr.mean():.3f}")
        else:
            self.stats['belief_centering'] = {'applied': False, 'representation': 'raw'}
            logger.info(f"[BeliefRepresentation] Raw. Mean={beliefs_arr.mean():.3f}")

        dev = torch.device(device)
        needs       = torch.tensor(needs_arr,       dtype=torch.float32, device=dev)
        beliefs     = torch.tensor(beliefs_arr,     dtype=torch.float32, device=dev)
        valences    = torch.tensor(valences_arr,    dtype=torch.float32, device=dev)
        tolerances  = torch.tensor(tolerances_arr,  dtype=torch.float32, device=dev)
        stressors   = torch.tensor(stressors_arr,   dtype=torch.float32, device=dev)
        feasibility = torch.tensor(feasibility_arr, dtype=torch.float32, device=dev)
        values      = torch.tensor(values_arr,      dtype=torch.float32, device=dev)
        frequencies = torch.tensor(frequencies_arr, dtype=torch.float32, device=dev)

        n_nodes      = n_needs + n_modes + n_modes
        init_state   = torch.zeros(N, n_nodes, dtype=torch.float32, device=dev)
        init_state[:, :n_needs]                     = needs
        init_state[:, n_needs+n_modes:]              = valences
        init_state[:, n_needs:n_needs+n_modes]       = torch.sigmoid(valences) * 0.2 * feasibility

        # Pipeline assertions
        assert valences.min() >= -1.0 and valences.max() <= 1.0
        b_bound = 1.05 if self.center_beliefs else 1.55
        assert beliefs.min() >= -b_bound and beliefs.max() <= b_bound, \
            f"Beliefs out of [{-b_bound:.2f},{b_bound:.2f}]: [{beliefs.min():.3f},{beliefs.max():.3f}]"
        assert init_state.shape == (N, n_nodes)

        # Targets
        soft_targets = frequencies / (frequencies.sum(dim=1, keepdim=True) + 1e-8)
        hard_targets = soft_targets.argmax(dim=1)

        freq_s1      = torch.tensor(freq_s1_arr, dtype=torch.float32, device=dev)
        freq_s2      = torch.tensor(freq_s2_arr, dtype=torch.float32, device=dev)
        soft_s1      = freq_s1 / freq_s1.sum(dim=1, keepdim=True).clamp(min=1e-8)
        soft_s2      = freq_s2 / freq_s2.sum(dim=1, keepdim=True).clamp(min=1e-8)
        hard_s1      = soft_s1.argmax(dim=1)
        hard_s2      = soft_s2.argmax(dim=1)

        return {
            'needs': needs, 'beliefs': beliefs, 'valences': valences,
            'tolerances': tolerances, 'stressors': stressors,
            'feasibility': feasibility, 'initial_state': init_state,
            'values': values, 'frequencies': frequencies,
            'soft_targets': soft_targets, 'hard_targets': hard_targets,
            'soft_targets_s1': soft_s1,   'hard_targets_s1': hard_s1,
            'soft_targets_s2': soft_s2,   'hard_targets_s2': hard_s2,
            'scenario_flip_mask': (hard_s1 != hard_s2),
            'metadata': metadata_list,
            'modes': self.modes, 'n_modes': self.n_modes,
            'belief_audit':    self.stats['belief_audit'],
            'belief_centering':self.stats.get('belief_centering', {'applied': False}),
        }

    # ------------------------------------------------------------------
    # Scenario frequency extraction
    # ------------------------------------------------------------------

    def _extract_scenario_freq(self, raw_mobil: Dict, scenario: str) -> Dict[str, float]:
        """Extract per-scenario frequencies (S1 or S2)."""
        MODE_SUBMODES = {
            'car':  ['car','pass','taxi'],
            'bike': ['bike','bikeshare'],
            'pt':   ['pt','train'],
            'walk': ['walk'],
        }
        scale_min = self.mobil_parser.freq_config.get('scale_min', 1)
        scale_max = self.mobil_parser.freq_config.get('scale_max', 5)
        prefix    = f's{scenario}_'
        result:   Dict[str, float] = {}
        for mode in self.modes:
            submodes = MODE_SUBMODES.get(mode, [mode])
            vals = []
            for k, rv in raw_mobil.items():
                if not k.startswith(prefix):
                    continue
                sm = k[len(prefix):]
                if sm not in submodes:
                    continue
                try:
                    v = float(rv) if rv not in (None,'') else scale_min
                    vals.append(max(0.0, min(1.0, (v-scale_min)/(scale_max-scale_min+1e-8))))
                except (ValueError, TypeError):
                    pass
            result[mode] = max(vals) if vals else 0.0
        return result

    def _compute_icc_stats(self, agents: List[Dict]) -> None:
        logger.info("  ICC computation not implemented (need separate S1/S2 arrays).")
        self.stats['icc_results'] = {'not_implemented': True}


# =============================================================================
# §I  PUBLIC CONVENIENCE FUNCTION
# =============================================================================

def load_data_from_limesurvey(
    json_path:   str,
    config_path: Optional[str] = None,
    device:      str = 'cpu',
) -> Dict:
    """
    Convenience entry point.  Equivalent to:
        LimeSurveyParser(config_path).parse_json(json_path, device)
    """
    return LimeSurveyParser(config_path).parse_json(json_path, device)
