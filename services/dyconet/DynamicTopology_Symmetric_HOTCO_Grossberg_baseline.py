"""
===============================================================================
SYMMETRIC HOTCO-GROSSBERG BASELINE v1.0 — Dynamic Coherent Network, No Environmental Perturbations
===============================================================================
Proof-of-concept: Pure structural deliberation without P(t) stressor term.

Based on: DeepHOTCO v4.2.1 (C4 Full Hot Coherence)
Version:  1.2 "Baseline — Fix B + Fix E + Extended Process Diagnostics"
Author:   DYCONET Research Team
Date:     2026-03-18

PURPOSE:
  Establish the structural baseline of the DYCONET cognitive model.
  This version removes ALL stressor-related perturbations and demonstrates
  the pure belief→action mapping through Grossberg shunting dynamics.

WHAT WAS REMOVED vs v4.2.1:
  - StressorPerturbation module (log_stressor_scale[5] + log_global_magnitude[1])
  - build_stressor_targets() mapping function
  - GrossbergDynamics.perturbation context
  - Phase 3 (stressor perturbation) from forward()
  - Phase 4 (environmental dissonance D_env) from forward()
  - Magdeburg ecological scenario generator (in train script)
  - Seasonal evaluation (S1/S2 stressor comparison)
  - _compute_environmental_dissonance() method

WHAT IS PRESERVED / REFORMULATED:
  - Full Grossberg shunting ODE (τ dS/dt = -A·S + (B-S)·E - (C+S)·I)
  - Symmetric HOTCO-style signed coherence topology (W = Wᵀ for typed links)
  - Valence as an explicit node family, NOT as an excitation/inhibition gate
  - Symmetric lateral inhibition among action nodes (λ·sum-of-others)
  - Bidirectional Valence↔Action coupling as a symmetric hot-coherence link
  - C_structural: emergent structural conflict metric
  - D_behavioral: deliberative dissonance (base vs final preference)
  - Cognitive Passport generation with tolerance profile metadata
  - Dynamic N-node topology (n_needs + n_modes + n_modes valences)
  - All load_data() parsers (JSON/CSV)

LEARNABLE PARAMETERS: 0 (pure theory-constrained simulation)

DISSONANCE TRIAD IN BASELINE:
  - C_structural:   Active — emergent from ODE deliberation dynamics
  - D_environmental: Inactive — always 0.0 (no stressor context)
  - D_behavioral:   Active — base preference vs. final choice divergence

TOLERANCE DATA:
  Tolerances ARE still loaded from parser (they are a real survey variable).
  They are stored in the Cognitive Passport as agent profile metadata
  and drive contextual_flags in the routing section.
  They do NOT enter the ODE dynamics in this version.

ARCHITECTURE:
  - Dynamic N-node topology: n_needs + n_modes + n_modes valences
  - 19-node default (11 needs + 4 actions + 4 valences)
  - Valence-as-node hot coherence with symmetric Action↔Valence coupling κ
    configurable as fixed/global or as an agent-specific bounded function of
    initial affective polarization across active modes.
  - Fixed Grossberg dynamics (τ=0.8, A=0.15, B=1.0, C=1.0, λ=2.0)
  - Grossberg-clean shunting range: B=1 and C=1 define the canonical [-1,+1] bounded activation interval

IMPORT COMPATIBILITY:
  Class name DeepHOTCO_v4 is preserved for existing import chains.
  Use alias DYCONET_Baseline for new code.

===============================================================================
"""
import math
import os
import sys
import json
import logging
import importlib
import importlib.util
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any, Union, Callable

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
for _p in (BASE_DIR, PROJECT_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

try:
    from torchdiffeq import odeint  # type: ignore
    # Dynamic expanded topologies are faster and more reproducible with the
    # internal fixed-step Euler integrator. Use torchdiffeq only if explicitly
    # requested via HOTCO_USE_TORCHDIFFEQ=1.
    TORCHDIFFEQ_AVAILABLE = os.environ.get('HOTCO_USE_TORCHDIFFEQ', '0') == '1'
    if not TORCHDIFFEQ_AVAILABLE:
        print("ℹ️  torchdiffeq available but disabled for dynamic-topology runs. Using Euler fallback.")
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False
    print("⚠️  torchdiffeq not found. Using Euler fallback.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DYCONET_Baseline_v1")

try:
    from hotcoc_process_diagnostics import build_full_process_diagnostics
except Exception:
    build_full_process_diagnostics = None


# =============================================================================
# OUTPUT TEMPERATURE — single source of truth for softmax β
# =============================================================================
# Applied consistently in:
#   - DeliberationTrace probabilities (Phase 8 of forward())
#   - _build_routing_section() in Cognitive Passport
#   - config['temperature'] in train_DYCONET_baseline.py (must match)
#
# Rationale: β=10 produces winner-runner-up separation >15pp for agents
# with clear preferences while preserving uncertainty for conflicted agents.
# READ-OUT parameter — does NOT enter ODE dynamics.
# =============================================================================
OUTPUT_TEMPERATURE: float = 10.0


# =============================================================================
# CONSTANTS
# =============================================================================

NEED_NAMES = [
    'pro_env', 'physical', 'privacy', 'autonomy', 'cost', 'speed',
    'safety_accident', 'safety_crime', 'comfort', 'reliable', 'health_infection'
]

MODES_4 = ['car', 'bike', 'pt', 'walk']
MODES_5 = ['car', 'bike', 'pt', 'walk', 'car_green']

# Expanded canonical mode universe for the dynamic-topology parser.
# The model can still run with 4/5 modes, but for the new parser it accepts
# arbitrary mode_names and builds n_needs + n_modes + n_modes nodes.
CANONICAL_MODES = [
    'walk', 'bike', 'bikeshare',
    'pt_bus_tram', 'train',
    'car_driver', 'car_passenger', 'taxi',
    'ev', 'hybrid', 'carsharing',
    'escooter', 'motorcycle',
]

# Tolerance dimension names — used for Cognitive Passport profile metadata.
# Tolerances come from parser survey data and are stored as agent profile,
# but do NOT enter ODE dynamics in this baseline version.
STRESSOR_NAMES = ['rain', 'crowding', 'darkness', 'traffic', 'temperature']
N_STRESSORS = len(STRESSOR_NAMES)  # kept for load_data() tolerance shape reference

VALUE_NAMES = ['biospheric', 'altruistic', 'egoistic', 'hedonic']
VALUE_DESCRIPTIONS = {
    'biospheric': "Values nature and environmental protection",
    'altruistic': "Values welfare of others and social justice",
    'egoistic': "Values personal success, wealth, and status",
    'hedonic': "Values pleasure, comfort, and enjoyment"
}


# =============================================================================
# TOPOLOGY BUILDER (Dynamic)
# =============================================================================

def resolve_mode_names(n_modes: int = 4, mode_names: Optional[List[str]] = None) -> List[str]:
    """Resolve the mode names for fixed or expanded dynamic topologies.

    Priority:
      1. explicit mode_names from the parser,
      2. legacy 4/5-mode defaults,
      3. expanded CANONICAL_MODES prefix.
    """
    if mode_names is not None:
        names = [str(m) for m in mode_names]
        if len(names) != int(n_modes):
            raise ValueError(f"mode_names length ({len(names)}) must match n_modes ({n_modes}).")
        if len(set(names)) != len(names):
            raise ValueError(f"mode_names must be unique, got duplicates: {names}")
        return names
    if int(n_modes) == 4:
        return MODES_4.copy()
    if int(n_modes) == 5:
        return MODES_5.copy()
    if int(n_modes) <= len(CANONICAL_MODES):
        return CANONICAL_MODES[:int(n_modes)].copy()
    return [f'mode_{i}' for i in range(int(n_modes))]


def build_topology(n_needs=11, n_modes=4, mode_names: Optional[List[str]] = None):
    """
    Build dynamic topology.

    Returns:
        nodes: Dict mapping node name -> index
        idx_needs: List of need indices
        idx_acts: List of action indices
        idx_valence: List of valence indices
        n_nodes: Total number of nodes
    """
    mode_names = resolve_mode_names(n_modes=n_modes, mode_names=mode_names)
    nodes = {}
    idx = 0

    idx_needs = []
    for need in NEED_NAMES[:n_needs]:
        nodes[f'need_{need}'] = idx
        idx_needs.append(idx)
        idx += 1

    idx_acts = []
    for mode in mode_names:
        nodes[mode] = idx
        idx_acts.append(idx)
        idx += 1

    idx_valence = []
    for mode in mode_names:
        nodes[f'valence_{mode}'] = idx
        idx_valence.append(idx)
        idx += 1

    return nodes, idx_needs, idx_acts, idx_valence, idx


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DeliberationTrace:
    """
    Complete record of deliberation process for XAI.

    DISSONANCE TRIAD (BASELINE):
    - C_structural:      Emergent structural conflict before environmental context.
                         Metric: 1 - (max - 2nd_max) / max of converged action activations.
    - D_environmental:   Forced to 0.0 in baseline (no stressor context active).
    - D_behavioral:      1 if final choice differs from base preference, else 0.
                         Still meaningful: captures deliberative ambivalence from
                         structural conflict alone.
    """
    agent_id: int
    timestamps: List[float] = field(default_factory=list)
    states: List[torch.Tensor] = field(default_factory=list)

    # Decision outcomes
    reaction_time: float = 0.0
    convergence_achieved: bool = False
    final_choice: str = ""
    choice_confidence: float = 0.0
    probabilities: Dict[str, float] = field(default_factory=dict)
    integration_backend: str = ""
    base_action_activations: Dict[str, float] = field(default_factory=dict)
    final_action_activations: Dict[str, float] = field(default_factory=dict)

    # Dissonance Triad
    structural_conflict: float = 0.0      # C_structural ∈ [0, 1]  — active
    environmental_pressure: float = 0.0   # D_environmental — always 0.0 in baseline
    behavioral_dissonance: float = 0.0    # D_behavioral ∈ {0, 1}  — active (binary)
    behavioral_dissonance_continuous: float = 0.0  # JS-div base↔final ∈ [0, 1] — FIX E

    # Context
    base_preference: str = ""
    stress_level: float = 0.0             # always 0.0 in baseline
    conflict_intensity: List[float] = field(default_factory=list)

    def is_dissonant(self) -> bool:
        """Returns True if final choice differs from base preference."""
        return self.behavioral_dissonance > 0.5

    def get_dissonance_type(self) -> str:
        """
        Classify dissonance type.

        In baseline: only 'structural' is possible (D_env = 0 always).
        """
        if not self.is_dissonant():
            return "none"
        high_struct = self.structural_conflict > 0.5
        # D_environmental = 0 in baseline, so "environmental" and "compound" cannot occur
        if high_struct:
            return "structural"
        return "marginal"

    def to_dict(self) -> Dict:
        """Convert trace to dictionary for JSON serialization."""
        return {
            'agent_id': self.agent_id,
            'reaction_time': self.reaction_time,
            'convergence_achieved': self.convergence_achieved,
            'final_choice': self.final_choice,
            'choice_confidence': self.choice_confidence,
            'probabilities': self.probabilities,
            'integration_backend': self.integration_backend,
            'base_action_activations': self.base_action_activations,
            'final_action_activations': self.final_action_activations,
            'structural_conflict': self.structural_conflict,
            'environmental_pressure': self.environmental_pressure,  # always 0.0
            'behavioral_dissonance': self.behavioral_dissonance,
            'behavioral_dissonance_continuous': self.behavioral_dissonance_continuous,
            'base_preference': self.base_preference,
            'stress_level': self.stress_level,  # always 0.0
            'dissonance_type': self.get_dissonance_type()
        }


@dataclass
class ChoicePrediction:
    """Structured prediction output with Dissonance Triad."""
    choice: str
    probabilities: Dict[str, float]
    confidence: float
    reaction_time: float
    structural_conflict: float
    environmental_pressure: float   # always 0.0 in baseline
    behavioral_dissonance: float
    base_preference: str
    stress_level: float             # always 0.0 in baseline
    trace: Optional[DeliberationTrace] = None

    def is_dissonant(self) -> bool:
        return self.behavioral_dissonance > 0.5


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def safe_entropy(logits: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute normalized entropy in [0, 1] with numerical stability.

    Entropy = 0 means certainty (one option dominates)
    Entropy = 1 means maximum uncertainty (all options equal)
    """
    log_probs = F.log_softmax(logits, dim=dim)
    probs = torch.exp(log_probs)
    entropy = -torch.sum(probs * log_probs, dim=dim)
    max_entropy = torch.log(torch.tensor(logits.size(dim), device=logits.device, dtype=logits.dtype))
    return torch.clamp(entropy / (max_entropy + eps), 0.0, 1.0)


def _js_divergence_batch(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Jensen-Shannon divergence between batched probability vectors, normalised to [0, 1].

    FIX E — Continuous D_behavioral:
    Replaces the binary (final_choice != base_preference) indicator with a
    continuous measure of deliberative shift in probability space.

    JS(P || Q) = 0.5 * KL(P || M) + 0.5 * KL(Q || M)   where M = 0.5 * (P + Q)

    Normalised by log(2) so the output lies in [0, 1]:
      0.0 = base and final distributions are identical (no deliberative shift)
      1.0 = maximally different distributions (complete preference reversal)

    Unlike D_behavioral ∈ {0, 1}, this metric:
      - Captures partial shifts (e.g. confidence drop without mode flip)
      - Is smooth and symmetric
      - Remains meaningful even when base_preference == final_choice

    Args:
        p: [batch, n_modes] — base preference probability vector (softmax)
        q: [batch, n_modes] — final preference probability vector (softmax)
        eps: numerical stability floor

    Returns:
        js: [batch] — continuous deliberative shift ∈ [0, 1]
    """
    m = 0.5 * (p + q)
    kl_pm = (p * torch.log((p + eps) / (m + eps))).sum(dim=-1)
    kl_qm = (q * torch.log((q + eps) / (m + eps))).sum(dim=-1)
    js = 0.5 * (kl_pm + kl_qm)
    return torch.clamp(js / math.log(2.0), 0.0, 1.0)


def euler_integrate(
    dynamics_fn,
    initial_state: torch.Tensor,
    t_eval: torch.Tensor,
    n_needs: int = 11,
    n_modes: int = 4,
) -> torch.Tensor:
    """Euler fallback integrator for when torchdiffeq is unavailable."""
    t_eval = t_eval.to(device=initial_state.device, dtype=initial_state.dtype)

    if t_eval.numel() < 2:
        return initial_state.unsqueeze(0)

    states = [initial_state]
    n_nodes = n_needs + 2 * n_modes

    for i in range(1, len(t_eval)):
        t_prev = t_eval[i - 1]
        t = t_eval[i]
        dt = t - t_prev

        dS = dynamics_fn(t_prev, states[-1])
        raw_state = states[-1] + dt * dS

        needs = raw_state[:, :n_needs]
        actions = raw_state[:, n_needs:n_needs + n_modes]
        valences = raw_state[:, n_needs + n_modes:n_nodes]

        needs_clamped = torch.clamp(needs, 0.0, 1.0)
        actions_clamped = torch.clamp(actions, 0.0, 1.0)
        valences_clamped = torch.clamp(valences, -1.0, 1.0)

        new_state = torch.cat([needs_clamped, actions_clamped, valences_clamped], dim=1)
        states.append(new_state)

    return torch.stack(states)


# =============================================================================
# GROSSBERG DYNAMICS MODULE — Baseline (No Perturbation)
# =============================================================================

class GrossbergDynamics(nn.Module):
    """
    Symmetric Grossberg shunting dynamics for a HOTCO-style coherence topology.

    BASELINE VERSION:
        τ dS/dt = -A·S + (B-S)·E - (C+S)·I

    Design change relative to the prior DYCONET/DeepHOTCO baseline:
    - NO affective gating.
    - Valence is represented as an ordinary typed node family.
    - Excitation and inhibition are computed directly from W_pos and W_neg.
    - Lateral inhibition is a symmetric competitive field among action nodes.

    Theoretical role:
    HOTCO provides the signed coherence topology W.
    Grossberg provides bounded shunting activation dynamics.

    Where:
    - S: State activation
    - A: Passive decay
    - B: Excitatory cap
    - C: Inhibitory floor
    - E: Excitation = W_pos @ state
    - I: Inhibition = W_neg @ state + symmetric action competition

    FIXED PARAMETERS (buffers, not learned):
    - τ (tau): 0.8 — Time constant
    - A (decay): 0.15 — Passive decay rate
    - B (excitatory cap): 1.0 — Maximum activation bound
    - C (inhibitory floor): 1.0 — Grossberg lower-bound magnitude / inhibitory shunting offset
    - λ (lateral_inhib): 2.0 — Symmetric action-competition strength
    """

    def __init__(
        self,
        n_nodes: int,
        idx_needs: List[int],
        idx_acts: List[int],
        idx_valence: List[int],
        freeze_needs: bool = False,
        need_leak_to_baseline: bool = False,
        inhibitory_floor: float = 1.0,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.idx_needs = idx_needs
        self.idx_acts = idx_acts
        self.idx_valence = idx_valence
        self.n_modes = len(idx_acts)

        self.freeze_needs = freeze_needs
        self.need_leak_to_baseline = need_leak_to_baseline
        self.need_baseline = None  # set per integration

        # Fixed parameters (registered as buffers — not Parameters, not learned)
        self.register_buffer('tau',           torch.tensor(0.8))
        self.register_buffer('decay',         torch.tensor(0.15))
        self.register_buffer('B',             torch.tensor(1.0))
        self.register_buffer('C',             torch.tensor(float(inhibitory_floor)))
        self.register_buffer('lateral_inhib', torch.tensor(2.0))

        # Context tensors (set before ODE integration, no perturbation in baseline)
        self.W_pos = None       # Excitatory weights [batch, n_nodes, n_nodes]
        self.W_neg = None       # Inhibitory magnitudes [batch, n_nodes, n_nodes]
        self.availability = None  # Mode availability [batch, n_modes]

    def set_integration_context(
        self,
        W_pos: torch.Tensor,
        W_neg: torch.Tensor,
        availability: torch.Tensor,
        need_baseline: Optional[torch.Tensor] = None,
    ):
        """
        Set context tensors before ODE integration.

        BASELINE NOTE: No perturbation parameter and no affective gating.
        Only symmetric HOTCO-style weight matrices and availability are set.
        """
        self.W_pos = W_pos
        self.W_neg = W_neg
        self.availability = availability
        self.need_baseline = need_baseline

    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Symmetric HOTCO-Grossberg dynamics.

        Computes dS/dt for the ODE integrator.

        FEATURES:
        - No affective gating.
        - Valence contributes only through symmetric Action↔Valence links in W.
        - Symmetric lateral inhibition among action nodes:
              I_lat,m = λ * Σ_{k≠m} s_act,k
        - No environmental perturbation term.
        """
        batch_size = state.shape[0]
        device = state.device
        dtype = state.dtype

        idx_act_start = self.idx_acts[0]
        idx_act_end = self.idx_acts[-1] + 1

        # -----------------------------------------------------------------
        # HOTCO-style signed coherence propagation without gating
        # -----------------------------------------------------------------
        excitation = torch.bmm(self.W_pos, state.unsqueeze(2)).squeeze(2)
        inhibition_from_W = torch.bmm(self.W_neg, state.unsqueeze(2)).squeeze(2)

        # -----------------------------------------------------------------
        # Symmetric lateral inhibition among actions.
        # Each action node receives inhibition from the sum of all other
        # actions using the same λ for every pair. Self-inhibition is excluded.
        # -----------------------------------------------------------------
        actions = state[:, idx_act_start:idx_act_end]
        total_action = torch.sum(actions, dim=1, keepdim=True)
        other_action = total_action - actions
        lateral_per_mode = self.lateral_inhib * other_action

        inhibition_lateral = torch.zeros_like(state)
        inhibition_lateral[:, idx_act_start:idx_act_end] = lateral_per_mode

        total_inhibition = inhibition_from_W + inhibition_lateral

        # -----------------------------------------------------------------
        # Leak term
        # -----------------------------------------------------------------
        leak = -self.decay * state

        need_start = self.idx_needs[0]
        need_end = self.idx_needs[-1] + 1

        # Optional: need nodes relax toward their survey baseline instead of 0
        if self.need_leak_to_baseline and self.need_baseline is not None:
            leak[:, need_start:need_end] = -self.decay * (
                state[:, need_start:need_end] - self.need_baseline[:, need_start:need_end]
            )

        # -----------------------------------------------------------------
        # Grossberg shunting equation
        # -----------------------------------------------------------------
        # With the Grossberg-clean default C=1.0, the shunting term is
        # consistent with the canonical lower saturation magnitude -C=-1.
        # Needs/actions remain explicitly clamped to [0,1], while valences
        # use the full [-1,1] affective range.
        dS_dt = (
            leak
            + (self.B - state) * F.relu(excitation)
            - (self.C + state) * F.relu(total_inhibition)
        ) / self.tau

        # Optional: freeze need nodes completely
        if self.freeze_needs:
            dS_dt[:, need_start:need_end] = 0.0

        # Availability mask — zeroes gradient for unavailable action modes
        if self.availability is not None:
            mask = torch.ones(batch_size, self.n_nodes, device=device, dtype=dtype)
            mask[:, idx_act_start:idx_act_end] = self.availability
            dS_dt = dS_dt * mask

        return dS_dt


# =============================================================================
# MAIN MODEL: SYMMETRIC HOTCO-GROSSBERG BASELINE v1.0
# =============================================================================

class DeepHOTCO_v4(nn.Module):
    """
    Symmetric HOTCO-Grossberg Baseline v1.0 — Dynamic Coherent Network, No Environmental Perturbations.

    Previously: DeepHOTCO v4.2.1 (C4 Full Hot Coherence).
    Class name preserved for import compatibility. Use alias DYCONET_Baseline.

    ARCHITECTURE:
    - 0 learnable parameters (pure theory-constrained simulation)
    - Dynamic N-node topology: n_needs + n_modes + n_modes valences
    - Symmetric HOTCO-style signed coherence topology
    - Symmetric lateral inhibition among action nodes (λ=2.0)
    - Valence as explicit node family via symmetric Action↔Valence coupling
    - Pure Grossberg shunting without P(t) stressor term

    DISSONANCE TRIAD (BASELINE):
    - C_structural:    Active — emergent from deliberation ODE trajectory
    - D_environmental: Always 0.0 (no stressor context)
    - D_behavioral:    Active — structural ambivalence can still flip choice

    TOLERANCES:
    Loaded from parser survey data and stored in Cognitive Passport profile.
    NOT passed to forward() — they are metadata only in this version.

    Args:
        n_modes: Number of transport modes (4 or 5)
        n_needs: Number of psychological needs (default 11)
        t_max: Maximum ODE integration time
        dt_eval: Time step for evaluation points
        rtol: Relative tolerance for ODE solver
        atol: Absolute tolerance for ODE solver
        convergence_threshold: Max change threshold for convergence
        min_integration_time: Minimum time before convergence can be declared
    """

    def __init__(self, n_modes: int = 4, n_needs: int = 11,
                 mode_names: Optional[List[str]] = None,
                 t_max: float = 10.0,
                 dt_eval: float = 0.02,
                 rtol: float = 1e-3,
                 atol: float = 1e-4,
                 convergence_threshold: float = 0.005,
                 min_integration_time: float = 0.3,
                 weighted_forward: bool = False,
                 forward_weight_floor: float = 0.25,
                 weighted_feedback: bool = False,
                 freeze_needs: bool = False,
                 need_leak_to_baseline: bool = False,
                 inhibitory_floor: float = 1.0,
                 kappa_valence: float = 0.5,
                 kappa_mode: str = 'fixed',
                 kappa_min: float = 0.20,
                 kappa_max: float = 0.80,
                 kappa_polarization_center: float = 0.35,
                 kappa_polarization_slope: float = 8.0):
        super().__init__()

        self.mode_names = resolve_mode_names(n_modes=n_modes, mode_names=mode_names)
        self.n_modes = len(self.mode_names)
        self.n_needs = n_needs
        self.t_max = t_max
        self.dt_eval = dt_eval
        self.rtol = rtol
        self.atol = atol
        self.convergence_threshold = convergence_threshold
        self.min_integration_time = min_integration_time
        self.weighted_forward = weighted_forward
        self.forward_weight_floor = forward_weight_floor
        self.weighted_feedback = weighted_feedback
        self.freeze_needs = freeze_needs
        self.need_leak_to_baseline = need_leak_to_baseline
        self.inhibitory_floor = float(inhibitory_floor)

        # ------------------------------------------------------------------
        # Action↔Valence coupling κ
        # ------------------------------------------------------------------
        # Scientific change in this version:
        # κ is no longer hard-coded inside _build_W_matrices(). The model can
        # run as a fixed theory-constrained baseline (κ=0.5), as a globally
        # calibrated variant, or as an agent-specific rule based only on the
        # agent's INITIAL affective polarization across active modes.
        #
        # The agent_polarization rule intentionally does NOT use soft_targets,
        # hard_targets, final choices, or any post-ODE quantity. This avoids
        # target leakage and keeps the parameterization interpretable: agents
        # with clearly differentiated affective evaluations receive stronger
        # action-valence coupling; agents with flat affective profiles receive
        # weaker coupling.
        # ------------------------------------------------------------------
        self.kappa_valence = float(kappa_valence)
        self.kappa_mode = str(kappa_mode)
        self.kappa_min = float(kappa_min)
        self.kappa_max = float(kappa_max)
        self.kappa_polarization_center = float(kappa_polarization_center)
        self.kappa_polarization_slope = float(kappa_polarization_slope)
        self.last_kappa_valence = None  # [batch] tensor stored after W construction

        if self.kappa_mode not in {'fixed', 'global', 'agent_polarization'}:
            raise ValueError(
                f"Unknown kappa_mode={self.kappa_mode}. Use 'fixed', 'global', or 'agent_polarization'."
            )
        if not (0.0 <= self.kappa_min <= self.kappa_max):
            raise ValueError('Require 0 <= kappa_min <= kappa_max.')

        # Build dynamic topology. The parser may provide an expanded,
        # agent-specific canonical universe (e.g. 13 modes); this model builds
        # a fixed global tensor layout and applies mode_active_mask per agent.
        self.nodes, self.idx_needs, self.idx_acts, self.idx_valence, self.n_nodes = \
            build_topology(n_needs, self.n_modes, mode_names=self.mode_names)

        logger.info(
            f"Initialized Symmetric HOTCO-Grossberg Baseline v1.0: {n_needs} needs, {n_modes} modes, "
            f"{self.n_nodes} nodes, mode_names={self.mode_names}, weighted_forward={weighted_forward}, "
            f"freeze_needs={freeze_needs}, need_leak_to_baseline={need_leak_to_baseline}, "
            f"C={self.inhibitory_floor}, kappa_mode={self.kappa_mode}, kappa={self.kappa_valence}"
        )

        # Grossberg dynamics module (only module — no StressorPerturbation)
        self.dynamics = GrossbergDynamics(
            n_nodes=self.n_nodes,
            idx_needs=self.idx_needs,
            idx_acts=self.idx_acts,
            idx_valence=self.idx_valence,
            freeze_needs=freeze_needs,
            need_leak_to_baseline=need_leak_to_baseline,
            inhibitory_floor=self.inhibitory_floor
        )

        # State tracking (set during forward pass for dashboard export)
        self.current_W_pos  = None
        self.current_W_neg  = None
        self.last_traces    = None

    def get_state_bounds(self, device: Optional[torch.device] = None,
                         dtype: Optional[torch.dtype] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return lower and upper bounds for each node block."""
        if device is None:
            device = self.dynamics.tau.device
        if dtype is None:
            dtype = self.dynamics.tau.dtype
        low = torch.full((self.n_nodes,), -1.0, device=device, dtype=dtype)
        high = torch.ones((self.n_nodes,), device=device, dtype=dtype)
        low[self.idx_needs[0]:self.idx_needs[-1] + 1] = 0.0
        low[self.idx_acts[0]:self.idx_acts[-1] + 1] = 0.0
        return low, high

    def integration_backend_name(self, initial_state: Optional[torch.Tensor] = None) -> str:
        """Human-readable backend name used during integration.

        Dynamic-topology runs can contain 37+ nodes and hundreds of agents.
        Fixed-step RK4 is much faster and more reproducible than adaptive dopri5
        for this solver-style dashboard, so it is used whenever torchdiffeq is
        available, on CPU and CUDA alike.
        """
        if TORCHDIFFEQ_AVAILABLE:
            return 'torchdiffeq_rk4_fixed_step'
        return 'euler_fallback'

    def _compute_agent_kappa(
        self,
        initial_state: torch.Tensor,
        availability: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute the symmetric Action↔Valence coupling κ for each agent.

        Modes
        -----
        fixed/global:
            Returns the same κ for every agent. This preserves the zero-learned
            theory-constrained baseline and is the recommended reference model.

        agent_polarization:
            Returns κ_i as a bounded monotonic function of INITIAL affective
            polarization across active modes:

                κ_i = κ_min + (κ_max - κ_min) * sigmoid(s * (pol_i - c))

            where pol_i is the active-mode valence range divided by 2, therefore
            scaled to [0,1] because valence nodes live in [-1,+1].

        Scientific rationale
        --------------------
        This rule is deliberately not fitted per agent and does not use any
        behavioral target. It only says that affect should couple more strongly
        to action when the agent's affective evaluations are differentiated
        enough to carry information. Flat or largely imputed affective profiles
        should not dominate the HOTCO-Grossberg topology.
        """
        device = initial_state.device
        dtype = initial_state.dtype
        batch_size = initial_state.shape[0]

        if self.kappa_mode in {'fixed', 'global'}:
            return torch.full(
                (batch_size,),
                self.kappa_valence,
                device=device,
                dtype=dtype,
            )

        if self.kappa_mode != 'agent_polarization':
            raise ValueError(
                f"Unknown kappa_mode={self.kappa_mode}. "
                "Use 'fixed', 'global', or 'agent_polarization'."
            )

        valences = initial_state[:, self.idx_valence[0]:self.idx_valence[-1] + 1]
        if availability is not None:
            active = availability.to(device=device, dtype=dtype)
        else:
            active = torch.ones_like(valences)

        active_bool = active > 0
        active_count = active_bool.float().sum(dim=1)

        # Mask inactive modes before computing the range. If an agent has fewer
        # than two active modes, affective polarization is not identifiable;
        # assigning zero keeps κ_i close to κ_min instead of inventing evidence.
        very_low = torch.full_like(valences, -1e9)
        very_high = torch.full_like(valences, 1e9)
        val_max = torch.where(active_bool, valences, very_low).max(dim=1).values
        val_min = torch.where(active_bool, valences, very_high).min(dim=1).values
        affective_polarization = ((val_max - val_min) / 2.0).clamp(0.0, 1.0)
        affective_polarization = torch.where(
            active_count >= 2,
            affective_polarization,
            torch.zeros_like(affective_polarization),
        )

        z = self.kappa_polarization_slope * (
            affective_polarization - self.kappa_polarization_center
        )
        kappa = self.kappa_min + (self.kappa_max - self.kappa_min) * torch.sigmoid(z)
        return kappa.clamp(self.kappa_min, self.kappa_max)

    def _build_W_matrices(self, beliefs: torch.Tensor,
                          initial_state: torch.Tensor,
                          availability: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build symmetric HOTCO-style signed coherence matrices.

        The matrix W encodes how well each transport mode satisfies each
        psychological need. In this symmetric baseline, need-action links are
        reciprocal:

            W[action_m, need_n] = W[need_n, action_m] = b_mn

        Valence is also represented as an explicit node, not as a gate:

            W[action_m, valence_m] = W[valence_m, action_m] = κ_i

        κ_i is either fixed/global or derived from initial affective polarization,
        depending on kappa_mode. No affective gating is applied. Positive entries in W become W_pos;
        negative entries become W_neg. Lateral inhibition is handled separately
        in GrossbergDynamics as a symmetric action-competition field.

        Args:
            beliefs: [batch, n_modes, n_needs] with values in [-1, +1].
            initial_state: [batch, n_nodes]. Used for weighted_forward and,
                if kappa_mode='agent_polarization', for the initial valence profile.
            availability: Optional [batch, n_modes] active-mode mask used to
                compute κ_i only from active alternatives.

        Returns:
            W_pos: Positive excitatory weights [batch, n_nodes, n_nodes]
            W_neg: Positive inhibitory magnitudes [batch, n_nodes, n_nodes]
        """
        batch_size = beliefs.shape[0]
        device = beliefs.device
        dtype = beliefs.dtype

        W = torch.zeros(batch_size, self.n_nodes, self.n_nodes, device=device, dtype=dtype)

        # Optional symmetric need-importance modulation.
        # If disabled, link strength is exactly the signed belief b_mn.
        need_importance = initial_state[:, self.idx_needs[0]:self.idx_needs[-1] + 1]
        if self.weighted_forward:
            symmetric_gain = self.forward_weight_floor + (1.0 - self.forward_weight_floor) * need_importance
        else:
            symmetric_gain = torch.ones_like(need_importance)

        # Symmetric Need ↔ Action links.
        for m_idx in range(self.n_modes):
            action_idx = self.idx_acts[m_idx]
            for n_idx in range(self.n_needs):
                need_idx = self.idx_needs[n_idx]
                w_mn = beliefs[:, m_idx, n_idx] * symmetric_gain[:, n_idx]
                W[:, action_idx, need_idx] = w_mn
                W[:, need_idx, action_idx] = w_mn

        # Symmetric Action ↔ Valence links.
        # Valence is a node-level affective state, not an excitation/inhibition gate.
        # κ is now configurable: fixed/global for the reference baseline or
        # agent-specific via the non-leaky initial affective-polarization rule.
        kappa_by_agent = self._compute_agent_kappa(
            initial_state=initial_state,
            availability=availability,
        )
        self.last_kappa_valence = kappa_by_agent.detach()

        for m_idx in range(self.n_modes):
            action_idx = self.idx_acts[m_idx]
            valence_idx = self.idx_valence[m_idx]
            W[:, action_idx, valence_idx] = kappa_by_agent
            W[:, valence_idx, action_idx] = kappa_by_agent

        # Numerical symmetry check in development/debug use:
        # the effective typed coherence matrix should be symmetric before
        # splitting into positive and negative parts.
        # assert torch.allclose(W, W.transpose(1, 2), atol=1e-6)

        W_pos = F.relu(W)
        W_neg = F.relu(-W)
        return W_pos, W_neg


    def _compute_base_preference(self, initial_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute base preference via mini-ODE, delegating to GrossbergDynamics.

        FIX B — eliminates the hand-rolled Euler loop that previously duplicated
        GrossbergDynamics.forward() verbatim.  The mini-ODE now calls self.dynamics
        directly through euler_integrate(), so every subsequent change to the ODE
        physics (symmetric coherence, lateral inhibition, leak target, freeze_needs, etc.) is
        automatically reflected here without a second edit point.

        PRE-CONDITION:
            self.dynamics.set_integration_context() must already have been called
            with the current (W_pos, W_neg, availability) tensors before this
            method is invoked.  forward() ensures this ordering.

        BEHAVIOURAL DIFFERENCE vs v1.0 mini-ODE:
            The availability mask (dS_dt * mask inside GrossbergDynamics.forward)
            is now active during the mini-ODE.  Unavailable modes therefore do not
            accumulate gradient during pre-deliberation, which is theoretically
            correct: if car is structurally unavailable, the agent should not form
            a base preference for car.

        Parameters
        ----------
        initial_state : [batch, n_nodes]
            Initial cognitive state.  W_pos, W_neg and availability are taken
            from self.dynamics (set by the caller via set_integration_context).

        Returns
        -------
        base_preference       : [batch]   argmax of converged action activations
        action_excitation     : [batch, n_modes]  converged action activations
        structural_conflict   : [batch]   C_structural margin metric ∈ [0, 1]
        """
        device = initial_state.device
        dtype  = initial_state.dtype

        # Short integration: same dt as original mini-ODE (0.1) over t ∈ [0, 2.0]
        t_base    = 2.0
        dt_base   = 0.1
        n_steps_b = int(t_base / dt_base) + 1
        t_eval_base = torch.linspace(0.0, t_base, n_steps_b, device=device, dtype=dtype)

        # Run through GrossbergDynamics using the shared Euler integrator.
        # Context (W_pos, W_neg, availability, need_baseline) is already set.
        states_base = euler_integrate(
            self.dynamics, initial_state, t_eval_base,
            n_needs=self.n_needs, n_modes=self.n_modes
        )
        # states_base: [n_steps_b, batch, n_nodes]

        idx_act_start = self.idx_acts[0]
        idx_act_end   = self.idx_acts[-1] + 1

        action_conv     = states_base[-1, :, idx_act_start:idx_act_end]   # [batch, n_modes]
        base_preference = action_conv.argmax(dim=1)                        # [batch]

        sorted_actions, _ = torch.sort(action_conv, dim=1, descending=True)
        max_val = sorted_actions[:, 0]
        sec_val = sorted_actions[:, 1]
        structural_conflict = 1.0 - (max_val - sec_val) / (max_val + 1e-8)
        structural_conflict = torch.clamp(structural_conflict, 0.0, 1.0)

        return base_preference, action_conv, structural_conflict

    def _check_convergence(self, states: torch.Tensor,
                           t_eval: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Check if ODE has converged for each agent.

        Convergence = max change in action nodes < threshold AND t > min_time.

        Returns:
            converged: Boolean tensor [batch]
            reaction_times: Time of convergence [batch]
        """
        batch_size = states.shape[1]
        n_times    = states.shape[0]
        device     = states.device

        if n_times < 2:
            return (torch.zeros(batch_size, dtype=torch.bool, device=device),
                    torch.full((batch_size,), self.t_max, device=device))

        idx_act_start = self.idx_acts[0]
        idx_act_end   = self.idx_acts[-1] + 1

        actions    = states[:, :, idx_act_start:idx_act_end]  # [time, batch, n_modes]
        max_change = torch.abs(actions[1:] - actions[:-1]).max(dim=2)[0]  # [time-1, batch]

        min_idx   = int(self.min_integration_time / self.dt_eval)
        converged = torch.zeros(batch_size, dtype=torch.bool, device=device)
        reaction_times = torch.full((batch_size,), self.t_max, device=device)

        for b in range(batch_size):
            for t_idx in range(min_idx, n_times - 1):
                if max_change[t_idx, b] < self.convergence_threshold:
                    converged[b] = True
                    reaction_times[b] = t_eval[t_idx + 1]
                    break

        return converged, reaction_times

    def _compute_conflict_over_time(self, states: torch.Tensor) -> torch.Tensor:
        """
        Compute conflict intensity (entropy) at each timestep.

        Returns:
            conflict: Conflict trajectory [time, batch]
        """
        idx_act_start = self.idx_acts[0]
        idx_act_end   = self.idx_acts[-1] + 1
        actions       = states[:, :, idx_act_start:idx_act_end]
        n_times       = actions.shape[0]

        conflict = torch.stack([
            safe_entropy(actions[t] * OUTPUT_TEMPERATURE, dim=1)
            for t in range(n_times)
        ])
        return conflict

    def forward(self, initial_state: torch.Tensor,
                beliefs: torch.Tensor,
                availability: torch.Tensor,
                return_trace: bool = False) -> Tuple[torch.Tensor, Optional[List[DeliberationTrace]]]:
        """
        Full forward pass — BASELINE (no stressor perturbation).

        REMOVED from v4.2.1 signature:
          - tolerances: torch.Tensor  (now profile-only metadata)
          - stressors:  torch.Tensor  (fully removed from this version)

        Args:
            initial_state: Initial cognitive state [batch, n_nodes]
                          Needs ∈ [0, 1], Actions ≈ 0, Valences ∈ [-1, +1]
            beliefs: Belief matrix [batch, n_modes, n_needs]
            availability: Mode availability [batch, n_modes] binary
            return_trace: Whether to return deliberation traces

        Returns:
            final_state: Converged cognitive state [batch, n_nodes]
            traces: List of DeliberationTrace (if return_trace=True, else None)
        """
        batch_size = initial_state.shape[0]
        device     = initial_state.device

        # =================================================================
        # PHASE 1: Build Weight Matrices from beliefs
        # =================================================================
        W_pos, W_neg = self._build_W_matrices(
            beliefs=beliefs,
            initial_state=initial_state,
            availability=availability,
        )
        self.current_W_pos = W_pos.detach()
        self.current_W_neg = W_neg.detach()

        # =================================================================
        # PHASE 2: Set integration context (must precede base preference)
        # =================================================================
        # FIX B: set_integration_context moved here so _compute_base_preference
        # can reuse self.dynamics instead of duplicating the ODE physics.
        # The main ODE (Phase 5) reuses the same already-set context.
        self.dynamics.set_integration_context(
            W_pos, W_neg, availability,
            need_baseline=initial_state[:, :self.n_needs]
        )

        # =================================================================
        # PHASE 3: Compute Base Preference (mini-ODE via GrossbergDynamics)
        # =================================================================
        with torch.no_grad():
            base_preference, action_excitation, structural_conflict = \
                self._compute_base_preference(initial_state)

        # PHASES 4a & 4b REMOVED IN BASELINE:
        # Phase 4a: Stressor Perturbation → eliminated
        # Phase 4b: Environmental Dissonance → D_environmental = 0.0 always
        environmental_dissonance = torch.zeros(batch_size, device=device)

        # =================================================================
        # PHASE 5: ODE Integration (full dynamics, no P term)
        # Context already set in Phase 2 — no second call needed.
        # =================================================================

        n_steps = int(self.t_max / self.dt_eval) + 1
        t_eval  = torch.linspace(
            0.0, float(self.t_max), n_steps,
            device=initial_state.device,
            dtype=initial_state.dtype
        )

        backend_name = self.integration_backend_name(initial_state)

        if TORCHDIFFEQ_AVAILABLE:
            states = odeint(
                self.dynamics, initial_state, t_eval,
                method='rk4',
                options={'step_size': float(self.dt_eval)}
            )
        else:
            states = euler_integrate(
                self.dynamics, initial_state, t_eval,
                n_needs=self.n_needs, n_modes=self.n_modes
            )

        # =================================================================
        # PHASE 6: Check Convergence
        # =================================================================
        converged, reaction_times = self._check_convergence(states, t_eval)

        final_states = []
        for b in range(batch_size):
            if converged[b]:
                t_idx = min(int(reaction_times[b].item() / self.dt_eval), n_steps - 1)
            else:
                t_idx = n_steps - 1
            final_states.append(states[t_idx, b])

        final_state = torch.stack(final_states)

        # =================================================================
        # PHASE 7: Final Choice and Behavioral Dissonance
        # =================================================================
        idx_act_start = self.idx_acts[0]
        idx_act_end   = self.idx_acts[-1] + 1
        final_actions = final_state[:, idx_act_start:idx_act_end]
        final_choice  = final_actions.argmax(dim=1)

        # D_behavioral: did structural deliberation flip the base preference?
        # In baseline this captures deliberative ambivalence from C_structural alone.
        behavioral_dissonance = (final_choice != base_preference).float()

        # =================================================================
        # PHASE 8: Build Traces
        # =================================================================
        traces = None
        if return_trace:
            conflict_trajectory = self._compute_conflict_over_time(states)

            # FIX E — Continuous D_behavioral: JS divergence between base and
            # final probability distributions, computed once over the whole batch.
            # d_beh_continuous ∈ [0, 1]:
            #   0.0 → deliberation left the distribution unchanged
            #   1.0 → maximum shift (mode reversal + certainty collapse)
            base_probs_batch  = F.softmax(action_excitation.detach() * OUTPUT_TEMPERATURE, dim=1)
            final_probs_batch = F.softmax(final_actions.detach()     * OUTPUT_TEMPERATURE, dim=1)
            d_beh_continuous  = _js_divergence_batch(base_probs_batch, final_probs_batch)  # [batch]

            traces = []

            for b in range(batch_size):
                sorted_actions, _ = torch.sort(final_actions[b], descending=True)
                confidence = float((sorted_actions[0] - sorted_actions[1]).detach())

                probs = F.softmax(final_actions[b].detach() * OUTPUT_TEMPERATURE, dim=0)
                prob_dict = {self.mode_names[i]: float(probs[i]) for i in range(self.n_modes)}
                base_action_dict = {self.mode_names[i]: float(action_excitation[b, i].detach()) for i in range(self.n_modes)}
                final_action_dict = {self.mode_names[i]: float(final_actions[b, i].detach()) for i in range(self.n_modes)}

                trace = DeliberationTrace(
                    agent_id=b,
                    timestamps=t_eval.cpu().tolist(),
                    states=[states[t, b].detach().cpu() for t in range(n_steps)],
                    reaction_time=float(reaction_times[b]),
                    convergence_achieved=bool(converged[b]),
                    final_choice=self.mode_names[int(final_choice[b])],
                    choice_confidence=confidence,
                    probabilities=prob_dict,
                    integration_backend=backend_name,
                    base_action_activations=base_action_dict,
                    final_action_activations=final_action_dict,
                    structural_conflict=float(structural_conflict[b]),
                    environmental_pressure=0.0,                          # always 0.0 in baseline
                    behavioral_dissonance=float(behavioral_dissonance[b]),
                    behavioral_dissonance_continuous=float(d_beh_continuous[b]),  # FIX E
                    base_preference=self.mode_names[int(base_preference[b])],
                    stress_level=0.0,                                    # always 0.0 in baseline
                    conflict_intensity=conflict_trajectory[:, b].cpu().tolist()
                )
                traces.append(trace)

            self.last_traces = traces

        return final_state, traces

    def predict_choice(self, initial_state: torch.Tensor,
                       beliefs: torch.Tensor,
                       availability: torch.Tensor) -> List[ChoicePrediction]:
        """
        High-level prediction API — BASELINE (no stressor arguments).
        """
        self.eval()
        with torch.no_grad():
            final_state, traces = self.forward(
                initial_state, beliefs, availability,
                return_trace=True
            )

        return [
            ChoicePrediction(
                choice=t.final_choice,
                probabilities=t.probabilities,
                confidence=t.choice_confidence,
                reaction_time=t.reaction_time,
                structural_conflict=t.structural_conflict,
                environmental_pressure=0.0,
                behavioral_dissonance=t.behavioral_dissonance,
                base_preference=t.base_preference,
                stress_level=0.0,
                trace=t
            )
            for t in traces
        ]

    # =========================================================================
    # COGNITIVE PASSPORT GENERATION
    # =========================================================================

    def generate_cognitive_passport(self,
                                    agent_id: Union[int, str],
                                    initial_state: torch.Tensor,
                                    beliefs: torch.Tensor,
                                    tolerances: torch.Tensor,
                                    availability: torch.Tensor,
                                    values: Optional[torch.Tensor] = None,
                                    metadata: Optional[Dict] = None,
                                    trace: Optional[DeliberationTrace] = None,
                                    final_state: Optional[torch.Tensor] = None) -> str:
        """
        Generate the compact Cognitive Passport JSON.

        This summary passport preserves the human-readable / app-facing view of
        the agent. When ``trace`` and ``final_state`` are supplied, the method
        reuses already computed simulation artifacts instead of re-running the
        ODE.
        """
        device = self.dynamics.tau.device
        dtype = initial_state.dtype if torch.is_tensor(initial_state) else self.dynamics.tau.dtype

        if tolerances is None:
            tolerances = torch.full((N_STRESSORS,), 0.5, device=device, dtype=dtype)

        # Ensure batch dimension
        if initial_state.dim() == 1: initial_state = initial_state.unsqueeze(0)
        if beliefs.dim() == 2:       beliefs       = beliefs.unsqueeze(0)
        if tolerances.dim() == 1:    tolerances    = tolerances.unsqueeze(0)
        if availability.dim() == 1:  availability  = availability.unsqueeze(0)
        if values is not None and values.dim() == 1: values = values.unsqueeze(0)

        initial_state = initial_state.to(device)
        beliefs       = beliefs.to(device)
        availability  = availability.to(device)
        tolerances    = tolerances.to(device)
        if values is not None:
            values = values.to(device)

        if trace is None or final_state is None:
            self.eval()
            with torch.no_grad():
                final_state_batch, traces = self.forward(
                    initial_state, beliefs, availability,
                    return_trace=True
                )
            trace = traces[0]
            state = final_state_batch[0]
        else:
            state = final_state[0] if final_state.dim() > 1 else final_state

        passport = {
            "cognitive_passport": {
                "version": "baseline_1.0",
                "model": "Symmetric HOTCO-Grossberg Baseline — No Environmental Perturbations",
                "agent_id": str(agent_id),
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "topology": {
                    "n_needs": self.n_needs,
                    "n_modes": self.n_modes,
                    "n_nodes": self.n_nodes,
                    "mode_names": self.mode_names,
                    "need_names": NEED_NAMES[:self.n_needs]
                },
                "profile": self._build_profile_section(initial_state[0], tolerances[0], values),
                "deliberation": self._build_deliberation_section(trace),
                "dissonance_triad": self._build_dissonance_section(trace, beliefs[0]),
                "routing_parameters": self._build_routing_section(state, tolerances[0]),
                "xai_summary": self._build_xai_section(trace, state, beliefs[0])
            }
        }

        if metadata:
            passport['cognitive_passport']['agent_profile'] = {
                'age': metadata.get('profile', {}).get('age'),
                'gender': metadata.get('profile', {}).get('gender_raw'),
                'ovgu_affiliation': metadata.get('profile', {}).get('ovgu_active'),
                'magdeburg_connection': metadata.get('profile', {}).get('magdeburg_connection', [])
            }
            passport['cognitive_passport']['spatial_context'] = {
                'n_pois': metadata.get('pois', {}).get('n_pois', 0),
                'mode_accessibility': metadata.get('pois', {}).get('mode_accessibility', {})
            }
            passport['cognitive_passport']['top_needs_ranking'] = metadata.get('ranking', [])

        return json.dumps(passport, indent=2, ensure_ascii=False)

    def _build_profile_section(self, initial_state: torch.Tensor,
                                tolerances: torch.Tensor,
                                values: Optional[torch.Tensor]) -> Dict:
        """Build agent profile section."""
        needs      = initial_state[self.idx_needs].cpu().numpy()
        need_names = NEED_NAMES[:self.n_needs]
        need_dict  = {need_names[i]: round(float(needs[i]), 4) for i in range(self.n_needs)}

        # Tolerances: profile metadata (do not enter dynamics)
        tol_dict = {STRESSOR_NAMES[i]: round(float(tolerances[i]), 4)
                    for i in range(N_STRESSORS)}

        value_dict = None
        if values is not None:
            values_np = values.cpu().numpy() if values.dim() == 1 else values[0].cpu().numpy()
            n_values  = min(len(VALUE_NAMES), len(values_np))
            value_dict = {VALUE_NAMES[i]: round(float(values_np[i]), 4)
                          for i in range(n_values)}

        return {
            "needs": need_dict,
            "environmental_tolerances": tol_dict,  # metadata only in baseline
            "tolerance_note": "Profile metadata only — tolerances do not enter ODE dynamics in Baseline v1.0",
            "values": value_dict
        }

    def _build_deliberation_section(self, trace: DeliberationTrace) -> Dict:
        """Build deliberation dynamics section."""
        c_struct = trace.structural_conflict
        if c_struct < 0.2:
            difficulty = "EASY"
        elif c_struct < 0.4:
            difficulty = "MODERATE"
        elif c_struct < 0.6:
            difficulty = "DIFFICULT"
        else:
            difficulty = "CONFLICTED"

        return {
            "final_choice": trace.final_choice.upper(),
            "probabilities": trace.probabilities,
            "confidence": round(trace.choice_confidence, 4),
            "reaction_time_seconds": round(trace.reaction_time, 3),
            "convergence_achieved": trace.convergence_achieved,
            "decision_difficulty": difficulty
        }

    def _build_dissonance_section(self, trace: DeliberationTrace, beliefs: Optional[torch.Tensor] = None) -> Dict:
        """
        Build Dissonance Triad section — BASELINE.

        Preserves the legacy summary fields and, when available, appends the
        richer HotCo-C extended process diagnostics as a nested block so the
        passport stays backward-compatible while exposing the new metrics.
        """
        c_struct = trace.structural_conflict
        if c_struct < 0.3:
            c_interp = "LOW - Clear structural preference"
        elif c_struct < 0.5:
            c_interp = "MODERATE - Some internal value conflict"
        else:
            c_interp = "HIGH - Significant structural ambivalence"

        d_behav = trace.behavioral_dissonance
        d_behav_cont = trace.behavioral_dissonance_continuous
        if d_behav > 0.5:
            d_behav_interp = "STRUCTURAL DISSONANCE - Deliberation flipped base preference"
        else:
            d_behav_interp = "ALIGNED - Final choice matches structural base preference"

        if d_behav_cont < 0.05:
            d_cont_interp = "STABLE — distribution unchanged by deliberation"
        elif d_behav_cont < 0.20:
            d_cont_interp = "MILD SHIFT — minor reweighting, mode unchanged"
        elif d_behav_cont < 0.40:
            d_cont_interp = "MODERATE SHIFT — preferences redistributed"
        else:
            d_cont_interp = "STRONG SHIFT — significant deliberative reweighting"

        shift_explanation = None
        if trace.base_preference != trace.final_choice:
            shift_explanation = (
                f"Agent shifted from {trace.base_preference.upper()} to "
                f"{trace.final_choice.upper()} due to structural ambivalence "
                f"(C_structural={trace.structural_conflict:.3f}). "
                f"No environmental stressor active in this baseline run."
            )

        out = {
            "C_structural": round(c_struct, 4),
            "C_interpretation": c_interp,
            "D_environmental": 0.0,
            "D_environmental_note": "Inactive in Baseline v1.0 (no stressor context)",
            "D_behavioral": int(d_behav),
            "D_behavioral_interpretation": d_behav_interp,
            "D_behavioral_continuous": round(float(d_behav_cont), 4),
            "D_behavioral_continuous_interpretation": d_cont_interp,
            "dissonance_type": trace.get_dissonance_type(),
            "base_preference": trace.base_preference.upper(),
            "final_choice": trace.final_choice.upper(),
            "preference_shifted": trace.base_preference != trace.final_choice,
            "shift_explanation": shift_explanation
        }

        if build_full_process_diagnostics is not None and beliefs is not None:
            try:
                # External legacy diagnostics may contain affective-gating fields.
                # The symmetric baseline has no gating, so we pass neutral values
                # only for backward compatibility and label the result accordingly.
                out["extended_process_diagnostics"] = build_full_process_diagnostics(
                    trace=trace,
                    beliefs=beliefs,
                    idx_needs=self.idx_needs,
                    idx_acts=self.idx_acts,
                    idx_valence=self.idx_valence,
                    mode_names=self.mode_names,
                    need_names=NEED_NAMES[:self.n_needs],
                    alpha=1.0,
                    beta=1.0,
                )
                out["extended_process_diagnostics"]["model_note"] = (
                    "Symmetric baseline: valence is a node, not a gate. "
                    "Any legacy gating fields should be ignored."
                )
            except Exception as exc:
                out["extended_process_diagnostics"] = {
                    "error": f"Extended process diagnostics failed: {type(exc).__name__}: {exc}"
                }
        else:
            out["extended_process_diagnostics"] = {
                "note": "Extended process diagnostics unavailable."
            }

        return out

    def _build_routing_section(self, state: torch.Tensor,
                                tolerances: torch.Tensor) -> Dict:
        """
        Build routing parameters for downstream optimizer.

        Tolerances still drive contextual_flags even though they don't
        affect ODE dynamics — they represent agent preferences/sensitivities
        that are relevant for route recommendations.
        """
        idx_act_start = self.idx_acts[0]
        idx_act_end   = self.idx_acts[-1] + 1

        actions = state[idx_act_start:idx_act_end]
        probs   = F.softmax(actions * OUTPUT_TEMPERATURE, dim=0)
        mode_weights = {self.mode_names[i]: round(float(probs[i]), 4)
                        for i in range(self.n_modes)}

        needs      = state[self.idx_needs].cpu().numpy()
        need_names = NEED_NAMES[:self.n_needs]

        utility_coefficients = {}
        need_to_utility = {
            'speed':            'time_penalty',
            'cost':             'cost_penalty',
            'safety_accident':  'safety_accident_bonus',
            'safety_crime':     'safety_crime_bonus',
            'pro_env':          'eco_bonus',
            'comfort':          'comfort_penalty',
            'physical':         'exercise_bonus',
            'privacy':          'privacy_bonus',
            'autonomy':         'autonomy_bonus',
            'reliable':         'reliability_bonus',
            'health_infection': 'health_infection_penalty'
        }
        for i, name in enumerate(need_names):
            if name in need_to_utility:
                utility_coefficients[need_to_utility[name]] = round(float(needs[i]), 4)

        # Contextual flags from tolerance profile (agent sensitivity preferences)
        contextual_flags = {
            "avoid_unlit_paths":      float(tolerances[2]) < 0.4,
            "prefer_covered_paths":   float(tolerances[0]) < 0.4,
            "tolerate_crowding":      float(tolerances[1]) > 0.6,
            "avoid_traffic":          float(tolerances[3]) < 0.4,
            "prefer_climate_control": float(tolerances[4]) < 0.4
        }

        return {
            "mode_weights": mode_weights,
            "utility_coefficients": utility_coefficients,
            "contextual_flags": contextual_flags
        }

    def _build_xai_section(self, trace: DeliberationTrace,
                           state: torch.Tensor,
                           beliefs: torch.Tensor) -> Dict:
        """Build human-readable XAI summary — BASELINE."""
        choice     = trace.final_choice.upper()
        choice_idx = self.mode_names.index(trace.final_choice)

        mode_beliefs = beliefs[choice_idx].cpu().numpy()
        needs_state  = state[self.idx_needs].cpu().numpy()
        need_names   = NEED_NAMES[:self.n_needs]

        driver_scores  = mode_beliefs * needs_state
        driver_ranking = np.argsort(-driver_scores)

        key_drivers   = []
        key_inhibitors = []
        for idx in driver_ranking:
            if driver_scores[idx] > 0.1:
                key_drivers.append(need_names[idx])
            elif driver_scores[idx] < -0.1:
                key_inhibitors.append(need_names[idx])

        parts = [f"Agent chose {choice}"]
        if key_drivers:
            parts.append(f"driven by {', '.join(key_drivers[:3])} needs")
        if trace.base_preference != trace.final_choice:
            parts.append(
                f"despite initial preference for {trace.base_preference.upper()}"
            )
            parts.append(
                f"Structural ambivalence (C_structural={trace.structural_conflict:.2f}) "
                f"caused the deliberative shift"
            )
        if trace.structural_conflict > 0.5:
            parts.append("The decision involved significant internal value conflict")

        narrative = ". ".join(parts) + "."

        return {
            "decision_narrative": narrative,
            "key_drivers": key_drivers[:5],
            "key_inhibitors": key_inhibitors[:5],
            "confidence_level": round(trace.choice_confidence, 4)
        }


    def _compress_winner_path(self, winners: List[int], timestamps: List[float]) -> List[Dict[str, Any]]:
        """Compress consecutive winner runs into interpretable path segments."""
        if not winners:
            return []
        if not timestamps:
            timestamps = [float(i) for i in range(len(winners))]

        segments: List[Dict[str, Any]] = []
        start_idx = 0
        current = winners[0]
        for i in range(1, len(winners) + 1):
            boundary = i == len(winners) or winners[i] != current
            if not boundary:
                continue
            end_idx = i - 1
            segments.append({
                'winner': self.mode_names[int(current)],
                'start_time': round(float(timestamps[start_idx]), 6),
                'end_time': round(float(timestamps[end_idx]), 6),
                'n_steps': int(i - start_idx),
            })
            if i < len(winners):
                start_idx = i
                current = winners[i]
        return segments

    def _build_trajectory_summary(self, trace: DeliberationTrace) -> Dict[str, Any]:
        """Compact process summary suitable for clustering and reporting."""
        if not trace.states:
            return {}

        states = torch.stack([s.float() for s in trace.states], dim=0)
        acts = states[:, self.idx_acts]
        probs = F.softmax(acts * OUTPUT_TEMPERATURE, dim=1)
        winners = acts.argmax(dim=1)
        timestamps = trace.timestamps if getattr(trace, 'timestamps', None) else [float(i) for i in range(states.shape[0])]
        times_t = torch.tensor(timestamps, dtype=torch.float32)

        switches = (winners[1:] != winners[:-1]).nonzero(as_tuple=False).flatten()
        n_switches = int(switches.numel())
        last_switch_time = float(times_t[int(switches[-1].item()) + 1].item()) if n_switches > 0 else 0.0

        final_winner = int(winners[-1].item())
        first_commit_idx = 0
        for idx in range(len(winners)):
            if bool((winners[idx:] == final_winner).all()):
                first_commit_idx = idx
                break
        first_commit_time = float(times_t[first_commit_idx].item())

        tail_start_time = 0.8 * float(times_t[-1].item()) if len(times_t) > 0 else 0.0
        tail_mask = times_t >= tail_start_time
        stable_tail = bool((winners[tail_mask] == winners[-1]).all().item()) if bool(tail_mask.any()) else True

        sorted_final, _ = torch.sort(acts[-1], descending=True)
        final_margin = float((sorted_final[0] - sorted_final[1]).item()) if acts.shape[1] >= 2 else 0.0
        entropies = (-(probs * torch.log(probs + 1e-8)).sum(dim=1))
        conflicts = torch.tensor(trace.conflict_intensity, dtype=torch.float32) if trace.conflict_intensity else entropies.clone()

        return {
            'n_switches': n_switches,
            'last_switch_time': round(last_switch_time, 6),
            'time_to_first_commitment': round(first_commit_time, 6),
            'stable_tail': stable_tail,
            'peak_conflict': round(float(conflicts.max().item()), 6),
            'final_conflict': round(float(conflicts[-1].item()), 6),
            'conflict_drop': round(float((conflicts.max() - conflicts[-1]).item()), 6),
            'peak_action_entropy': round(float(entropies.max().item()), 6),
            'final_action_entropy': round(float(entropies[-1].item()), 6),
            'final_action_margin': round(final_margin, 6),
            'winner_path_compact': self._compress_winner_path(winners.cpu().tolist(), timestamps),
        }

    def _build_trajectory_snapshots(self, trace: DeliberationTrace, max_snapshot_points: int = 10) -> List[Dict[str, Any]]:
        """Sample interpretable trajectory snapshots across the full deliberation."""
        if not trace.states:
            return []

        states = torch.stack([s.float() for s in trace.states], dim=0)
        acts = states[:, self.idx_acts]
        vals = states[:, self.idx_valence]
        needs = states[:, self.idx_needs]
        probs = F.softmax(acts * OUTPUT_TEMPERATURE, dim=1)
        winners = acts.argmax(dim=1)
        conflicts = torch.tensor(trace.conflict_intensity, dtype=torch.float32) if trace.conflict_intensity else (-(probs * torch.log(probs + 1e-8)).sum(dim=1))
        timestamps = trace.timestamps if getattr(trace, 'timestamps', None) else [float(i) for i in range(states.shape[0])]

        n_points = states.shape[0]
        if max_snapshot_points <= 0 or max_snapshot_points >= n_points:
            indices = list(range(n_points))
        else:
            indices = sorted(set(int(round(i)) for i in np.linspace(0, n_points - 1, max_snapshot_points)))

        snapshots: List[Dict[str, Any]] = []
        for idx in indices:
            snapshots.append({
                't': round(float(timestamps[idx]), 6),
                'winner': self.mode_names[int(winners[idx].item())],
                'conflict': round(float(conflicts[idx].item()), 6),
                'actions': {self.mode_names[m]: round(float(acts[idx, m].item()), 6) for m in range(self.n_modes)},
                'probabilities': {self.mode_names[m]: round(float(probs[idx, m].item()), 6) for m in range(self.n_modes)},
                'valences': {self.mode_names[m]: round(float(vals[idx, m].item()), 6) for m in range(self.n_modes)},
                'needs': {NEED_NAMES[n]: round(float(needs[idx, n].item()), 6) for n in range(self.n_needs)},
            })
        return snapshots

    def _build_full_trajectory_block(self, trace: DeliberationTrace) -> Dict[str, Any]:
        """Full per-time-step trajectory block for downstream research use."""
        if not trace.states:
            return {}

        states = torch.stack([s.float() for s in trace.states], dim=0)
        acts = states[:, self.idx_acts]
        vals = states[:, self.idx_valence]
        needs = states[:, self.idx_needs]
        probs = F.softmax(acts * OUTPUT_TEMPERATURE, dim=1)
        winners = acts.argmax(dim=1)
        conflicts = torch.tensor(trace.conflict_intensity, dtype=torch.float32) if trace.conflict_intensity else (-(probs * torch.log(probs + 1e-8)).sum(dim=1))
        timestamps = trace.timestamps if getattr(trace, 'timestamps', None) else [float(i) for i in range(states.shape[0])]

        return {
            'timestamps': [round(float(t), 6) for t in timestamps],
            'winners': [self.mode_names[int(w.item())] for w in winners],
            'conflict_intensity': [round(float(x), 6) for x in conflicts.tolist()],
            'action_activations': {self.mode_names[m]: [round(float(x), 6) for x in acts[:, m].tolist()] for m in range(self.n_modes)},
            'action_probabilities': {self.mode_names[m]: [round(float(x), 6) for x in probs[:, m].tolist()] for m in range(self.n_modes)},
            'valence_activations': {self.mode_names[m]: [round(float(x), 6) for x in vals[:, m].tolist()] for m in range(self.n_modes)},
            'need_activations': {NEED_NAMES[n]: [round(float(x), 6) for x in needs[:, n].tolist()] for n in range(self.n_needs)},
        }

    def _build_ode_core_block(self,
                              initial_state: torch.Tensor,
                              final_state: torch.Tensor,
                              beliefs: torch.Tensor,
                              trace: DeliberationTrace) -> Dict[str, Any]:
        """Explicit ODE-core state block for downstream feature-matrix export."""
        init_needs = initial_state[self.idx_needs].detach().cpu()
        final_needs = final_state[self.idx_needs].detach().cpu()
        init_actions = initial_state[self.idx_acts].detach().cpu()
        final_actions_state = final_state[self.idx_acts].detach().cpu()
        init_valences = initial_state[self.idx_valence].detach().cpu()
        final_valences = final_state[self.idx_valence].detach().cpu()
        beliefs_cpu = beliefs.detach().cpu()

        block: Dict[str, Any] = {
            'needs_init': {NEED_NAMES[n]: round(float(init_needs[n].item()), 6) for n in range(self.n_needs)},
            'needs_final': {NEED_NAMES[n]: round(float(final_needs[n].item()), 6) for n in range(self.n_needs)},
            'valence_init': {self.mode_names[m]: round(float(init_valences[m].item()), 6) for m in range(self.n_modes)},
            'valence_final': {self.mode_names[m]: round(float(final_valences[m].item()), 6) for m in range(self.n_modes)},
            'action_init': {self.mode_names[m]: round(float(init_actions[m].item()), 6) for m in range(self.n_modes)},
            'action_base': {mode: round(float(value), 6) for mode, value in trace.base_action_activations.items()},
            'action_final': {mode: round(float(value), 6) for mode, value in trace.final_action_activations.items()},
            'beliefs': {
                self.mode_names[m]: {
                    NEED_NAMES[n]: round(float(beliefs_cpu[m, n].item()), 6)
                    for n in range(self.n_needs)
                }
                for m in range(self.n_modes)
            },
            'action_final_state': {
                self.mode_names[m]: round(float(final_actions_state[m].item()), 6)
                for m in range(self.n_modes)
            },
        }
        return block

    def build_full_deliberation_style_passport(self,
                                               agent_id: Union[int, str],
                                               initial_state: torch.Tensor,
                                               beliefs: torch.Tensor,
                                               tolerances: torch.Tensor,
                                               availability: torch.Tensor,
                                               values: Optional[torch.Tensor] = None,
                                               metadata: Optional[Dict] = None,
                                               trace: Optional[DeliberationTrace] = None,
                                               final_state: Optional[torch.Tensor] = None,
                                               max_snapshot_points: int = 10,
                                               include_full_trajectory: bool = True) -> Dict[str, Any]:
        """
        Build the richer deliberation-style passport used for clustering and
        process-oriented downstream analyses.
        """
        device = self.dynamics.tau.device
        dtype = initial_state.dtype if torch.is_tensor(initial_state) else self.dynamics.tau.dtype

        if tolerances is None:
            tolerances = torch.full((N_STRESSORS,), 0.5, device=device, dtype=dtype)

        if initial_state.dim() == 1: initial_state = initial_state.unsqueeze(0)
        if beliefs.dim() == 2:       beliefs       = beliefs.unsqueeze(0)
        if tolerances.dim() == 1:    tolerances    = tolerances.unsqueeze(0)
        if availability.dim() == 1:  availability  = availability.unsqueeze(0)
        if values is not None and values.dim() == 1: values = values.unsqueeze(0)

        initial_state = initial_state.to(device)
        beliefs = beliefs.to(device)
        tolerances = tolerances.to(device)
        availability = availability.to(device)
        if values is not None:
            values = values.to(device)

        if trace is None or final_state is None:
            self.eval()
            with torch.no_grad():
                final_state_batch, traces = self.forward(
                    initial_state, beliefs, availability,
                    return_trace=True
                )
            trace = traces[0]
            state = final_state_batch[0]
        else:
            state = final_state[0] if final_state.dim() > 1 else final_state

        payload: Dict[str, Any] = {
            'full_deliberation_style_passport': {
                'version': 'baseline_1.0_full_deliberation',
                'model': 'Symmetric HOTCO-Grossberg Baseline — No Environmental Perturbations',
                'agent_id': str(agent_id),
                'timestamp': datetime.utcnow().isoformat() + 'Z',
                'topology': {
                    'n_needs': self.n_needs,
                    'n_modes': self.n_modes,
                    'n_nodes': self.n_nodes,
                    'mode_names': self.mode_names,
                    'need_names': NEED_NAMES[:self.n_needs],
                },
                'profile': self._build_profile_section(initial_state[0], tolerances[0], values),
                'deliberation': self._build_deliberation_section(trace),
                'dissonance_triad': self._build_dissonance_section(trace, beliefs[0]),
                'routing_parameters': self._build_routing_section(state, tolerances[0]),
                'xai_summary': self._build_xai_section(trace, state, beliefs[0]),
                'ode_core': self._build_ode_core_block(initial_state[0], state, beliefs[0], trace),
                'trajectory_summary': self._build_trajectory_summary(trace),
                'trajectory_snapshots': self._build_trajectory_snapshots(trace, max_snapshot_points=max_snapshot_points),
            }
        }

        if include_full_trajectory:
            payload['full_deliberation_style_passport']['trajectory_full'] = self._build_full_trajectory_block(trace)

        if metadata:
            payload['full_deliberation_style_passport']['agent_profile'] = {
                'age': metadata.get('profile', {}).get('age'),
                'gender': metadata.get('profile', {}).get('gender_raw'),
                'ovgu_affiliation': metadata.get('profile', {}).get('ovgu_active'),
                'magdeburg_connection': metadata.get('profile', {}).get('magdeburg_connection', []),
            }
            payload['full_deliberation_style_passport']['spatial_context'] = {
                'n_pois': metadata.get('pois', {}).get('n_pois', 0),
                'mode_accessibility': metadata.get('pois', {}).get('mode_accessibility', {}),
            }
            payload['full_deliberation_style_passport']['top_needs_ranking'] = metadata.get('ranking', [])

        return payload


# Backward-compatible alias for new code
DYCONET_Baseline = DeepHOTCO_v4


# =============================================================================
# DATA LOADING — BASELINE (JSON + CSV backward compatibility)
# =============================================================================


def _load_symbol_from_module_name(module_name: str, symbol_name: str) -> Optional[Callable[..., Any]]:
    try:
        module = importlib.import_module(module_name)
    except Exception:
        return None
    return getattr(module, symbol_name, None)


def _load_symbol_from_file(file_path: Path, symbol_name: str) -> Optional[Callable[..., Any]]:
    if not file_path.exists() or not file_path.is_file():
        return None
    try:
        spec = importlib.util.spec_from_file_location(f"_dyn_{file_path.stem}", file_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    except Exception:
        return None
    return getattr(module, symbol_name, None)


def _candidate_loader_specs() -> List[Tuple[str, str, Optional[Path]]]:
    specs: List[Tuple[str, str, Optional[Path]]] = [
        ("module", "limesurvey_parser_dynamic_topology_matrix_completion", None),
        ("module", "parsers.limesurvey_dynamic_topology_matrix_completion", None),
        ("module", "parsers.limesurvey", None),
        ("module", "parsers.limesurvey.loader", None),
        ("module", "limesurvey", None),
        ("module", "mobil_parser", None),
        ("module", "parsers.mobil_parser", None),
    ]
    roots = [BASE_DIR, PROJECT_ROOT]
    rels = [
        ("load_data_from_limesurvey", Path("parsers/limesurvey.py")),
        ("load_data_from_limesurvey", Path("parsers/limesurvey/__init__.py")),
        ("load_data_from_limesurvey", Path("parsers/limesurvey/loader.py")),
        ("load_mobility_data", Path("mobil_parser.py")),
        ("load_mobility_data", Path("parsers/mobil_parser.py")),
    ]
    for root in roots:
        for symbol, rel in rels:
            specs.append((symbol, str((root / rel).resolve()), root / rel))
    return specs



def _resolve_json_loader() -> Tuple[Callable[..., Any], str]:
    """
    Resolve a JSON loader function for mobility_data.json across several parser
    layouts used in the DYCONET project.

    Returns
    -------
    tuple
        (loader_fn, loader_source)
    """
    import importlib
    import importlib.util

    search_roots = [BASE_DIR, PROJECT_ROOT]
    attempts: List[str] = []

    def _try_module(module_name: str, fn_name: str):
        try:
            mod = importlib.import_module(module_name)
            fn = getattr(mod, fn_name, None)
            if callable(fn):
                return fn, f"{module_name}.{fn_name}"
            attempts.append(f"module found but function missing: {module_name}.{fn_name}")
        except Exception as e:
            attempts.append(f"module import failed: {module_name}.{fn_name} ({type(e).__name__}: {e})")
        return None

    module_candidates = [
        ("limesurvey_parser_dynamic_topology_matrix_completion", "load_data_from_limesurvey"),
        ("parsers.limesurvey_dynamic_topology_matrix_completion", "load_data_from_limesurvey"),
        ("parsers.limesurvey", "load_data_from_limesurvey"),
        ("parsers.limesurvey.loader", "load_data_from_limesurvey"),
        ("parsers.limesurvey.main_parser", "load_data_from_limesurvey"),
        ("parsers.limesurvey.mobil_parser", "load_mobility_data"),
        ("limesurvey", "load_data_from_limesurvey"),
        ("mobil_parser", "load_mobility_data"),
        ("parsers.mobil_parser", "load_mobility_data"),
    ]

    for module_name, fn_name in module_candidates:
        out = _try_module(module_name, fn_name)
        if out is not None:
            return out

    file_candidates = []
    for root in search_roots:
        file_candidates.extend([
            (root / "limesurvey_parser_dynamic_topology_matrix_completion.py", "load_data_from_limesurvey"),
            (root / "parsers" / "limesurvey_dynamic_topology_matrix_completion.py", "load_data_from_limesurvey"),
            (root / "parsers" / "limesurvey.py", "load_data_from_limesurvey"),
            (root / "parsers" / "limesurvey" / "__init__.py", "load_data_from_limesurvey"),
            (root / "parsers" / "limesurvey" / "loader.py", "load_data_from_limesurvey"),
            (root / "parsers" / "limesurvey" / "main_parser.py", "load_data_from_limesurvey"),
            (root / "parsers" / "limesurvey" / "mobil_parser.py", "load_mobility_data"),
            (root / "mobil_parser.py", "load_mobility_data"),
            (root / "parsers" / "mobil_parser.py", "load_mobility_data"),
        ])

    for file_path, fn_name in file_candidates:
        try:
            if not file_path.exists():
                attempts.append(f"file missing: {file_path}::{fn_name}")
                continue

            spec = importlib.util.spec_from_file_location(
                f"_dyconet_loader_{file_path.stem}",
                str(file_path),
            )
            if spec is None or spec.loader is None:
                attempts.append(f"spec creation failed: {file_path}::{fn_name}")
                continue

            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            fn = getattr(mod, fn_name, None)
            if callable(fn):
                return fn, f"{file_path}::{fn_name}"

            attempts.append(f"file imported but function missing: {file_path}::{fn_name}")
        except Exception as e:
            attempts.append(f"file import failed: {file_path}::{fn_name} ({type(e).__name__}: {e})")

    searched = "\n".join(f"  - {str(r)}" for r in search_roots)
    attempt_txt = "\n".join(f"  - {a}" for a in attempts)

    raise ImportError(
        "No JSON loader found for mobility_data.json.\n"
        f"Searched around these roots:\n{searched}\n"
        f"Attempts:\n{attempt_txt}\n"
        "Expected one of:\n"
        "  - parsers.limesurvey.main_parser.load_data_from_limesurvey\n"
        "  - parsers.limesurvey.mobil_parser.load_mobility_data\n"
        "  - or equivalent importable modules."
    )


def _resolve_default_config_path(config_path: Optional[str]) -> Optional[str]:
    if config_path:
        return config_path
    candidates = [
        BASE_DIR / "parsers" / "limesurvey" / "config.yaml",
        PROJECT_ROOT / "parsers" / "limesurvey" / "config.yaml",
        Path("parsers/limesurvey/config.yaml"),
        BASE_DIR / "config.yaml",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None

def load_data(data_path: str, device: str = 'cuda', config_path: str = None) -> Dict:
    """
    Load data from LimeSurvey JSON (real data) or CSV (legacy synthetic).

    BASELINE VERSION:
    - Tolerances ARE loaded and returned (they are a survey variable used as
      agent profile metadata in Cognitive Passport).
    - Stressors are NOT loaded and NOT returned. The data dict does not
      include a 'stressors' key to prevent accidental use in dynamics.
    - forward() does not accept stressors or tolerances as arguments.
      Tolerances are passed separately to generate_cognitive_passport().

    Args:
        data_path:   Path to .json or .csv file
        device:      'cpu' or 'cuda'
        config_path: Parser config (for JSON only)

    Returns:
        dict with tensors for Symmetric HOTCO-Grossberg Baseline.
        Keys: needs, beliefs, valences, tolerances, availability,
              initial_state, values, soft_targets, hard_targets,
              frequencies (if available), metadata, modes, n_modes, dataframe
        NOTE: 'stressors' key is NOT present.
    """
    if data_path.endswith('.json'):
        logger.info(f"Loading real data from JSON: {data_path}")

        loader_fn, loader_source = _resolve_json_loader()
        logger.info(f"   Using JSON loader: {loader_source}")

        config_path = _resolve_default_config_path(config_path)

        import inspect
        sig = inspect.signature(loader_fn)
        kwargs = {}

        if 'device' in sig.parameters:
            kwargs['device'] = device
        if 'config_path' in sig.parameters and config_path is not None:
            kwargs['config_path'] = config_path

        data = loader_fn(data_path, **kwargs)

        # Remove stressors from dict to prevent accidental use
        data.pop('stressors', None)

        # Dynamic-topology compatibility: prefer the parser's agent-specific
        # mask as effective availability. This prevents inactive modes from
        # receiving gradients or winning the final readout.
        if 'mode_active_mask' in data:
            data['effective_availability'] = data['mode_active_mask'].float()
            data['availability'] = data['effective_availability']
            data['feasibility'] = data['effective_availability']

        # Make mode names explicit for the model/solver.
        if 'mode_names' not in data and 'modes' in data:
            data['mode_names'] = list(data['modes'])
        if 'modes' not in data and 'mode_names' in data:
            data['modes'] = list(data['mode_names'])
        data['n_modes'] = int(data.get('n_modes', data['beliefs'].shape[1]))

        logger.info(f"✅ Loaded {data['needs'].shape[0]} agents, "
                    f"{data.get('n_modes', data['beliefs'].shape[1])} modes, "
                    f"{data['needs'].shape[1]} needs")
        if 'mode_names' in data:
            logger.info(f"   Mode names: {data['mode_names']}")
        return data

    elif data_path.endswith('.csv'):
        logger.warning("⚠️  CSV loading deprecated. Use JSON for real data.")
        return _load_csv_legacy(data_path, device)
    else:
        raise ValueError(f"Unsupported file format: {data_path}. Use .json or .csv")


load_and_prep_data_v4 = load_data  # backward-compatible alias


def _load_csv_legacy(csv_path: str, device: str = 'cpu') -> Dict:
    """
    Legacy CSV loader (backward compatible with v4.0 synthetic data).
    DEPRECATED: Use JSON with LimeSurvey parser for real data.

    BASELINE CHANGES:
    - Stressor columns removed — 'stressors' key not in returned dict.
    - Tolerances still loaded and returned as profile metadata.
    """
    import pandas as pd

    logger.info(f"📂 Loading legacy CSV: {csv_path}")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")

    df   = pd.read_csv(csv_path)
    N    = len(df)
    logger.info(f"   Loaded {N} agents (legacy CSV)")

    n_modes = 4
    n_needs = 11

    # =========================================================================
    # NEEDS [N, 11]
    # =========================================================================
    old_need_cols = ['need_pro_env', 'need_physical_activity', 'need_privacy',
                     'need_autonomy', 'need_hedonism', 'need_cost',
                     'need_speed', 'need_safety', 'need_comfort']
    old_needs_data = []
    for col in old_need_cols:
        if col in df.columns:
            old_needs_data.append(df[col].values)
        elif col.replace('_activity', '') in df.columns:
            old_needs_data.append(df[col.replace('_activity', '')].values)
        else:
            logger.warning(f"   Need column {col} not found, using default 4.0")
            old_needs_data.append(np.full(N, 4.0))

    old_needs  = np.stack(old_needs_data, axis=1)
    old_needs  = (old_needs - 1.0) / 6.0

    needs_11        = np.full((N, 11), 0.5, dtype=np.float32)
    needs_11[:, 0]  = old_needs[:, 0]  # pro_env
    needs_11[:, 1]  = old_needs[:, 1]  # physical
    needs_11[:, 2]  = old_needs[:, 2]  # privacy
    needs_11[:, 3]  = old_needs[:, 3]  # autonomy
    needs_11[:, 4]  = old_needs[:, 5]  # cost
    needs_11[:, 5]  = old_needs[:, 6]  # speed
    needs_11[:, 6]  = old_needs[:, 7]  # safety → safety_accident
    needs_11[:, 7]  = old_needs[:, 7]  # safety → safety_crime (proxy)
    needs_11[:, 8]  = old_needs[:, 8]  # comfort
    needs = torch.tensor(needs_11, dtype=torch.float32).to(device)

    # =========================================================================
    # BELIEFS [N, 4, 11]
    # =========================================================================
    old_beliefs   = np.zeros((N, 4, 9), dtype=np.float32)
    mode_names    = ['car', 'bike', 'pt', 'walk']
    need_suffixes = ['pro_env', 'physical_activity', 'privacy', 'autonomy',
                     'hedonism', 'cost', 'speed', 'safety', 'comfort']

    for m_idx, mode in enumerate(mode_names):
        for n_idx, need_suffix in enumerate(need_suffixes):
            col = f"belief_{mode}_{need_suffix}"
            if col in df.columns:
                old_beliefs[:, m_idx, n_idx] = (df[col].values - 3.0) / 2.0
            else:
                alt_col = col.replace('_activity', '')
                if alt_col in df.columns:
                    old_beliefs[:, m_idx, n_idx] = (df[alt_col].values - 3.0) / 2.0

    beliefs_11 = np.zeros((N, 4, 11), dtype=np.float32)
    beliefs_11[:, :, 0] = old_beliefs[:, :, 0]
    beliefs_11[:, :, 1] = old_beliefs[:, :, 1]
    beliefs_11[:, :, 2] = old_beliefs[:, :, 2]
    beliefs_11[:, :, 3] = old_beliefs[:, :, 3]
    beliefs_11[:, :, 4] = old_beliefs[:, :, 5]
    beliefs_11[:, :, 5] = old_beliefs[:, :, 6]
    beliefs_11[:, :, 6] = old_beliefs[:, :, 7]
    beliefs_11[:, :, 7] = old_beliefs[:, :, 7]
    beliefs_11[:, :, 8] = old_beliefs[:, :, 8]
    beliefs = torch.tensor(beliefs_11, dtype=torch.float32).to(device)

    # =========================================================================
    # VALENCES [N, 4]
    # =========================================================================
    valence_cols = ['valence_car', 'valence_bike', 'valence_pt', 'valence_walk']
    valence_data = []
    for col in valence_cols:
        if col in df.columns:
            valence_data.append(df[col].values)
        else:
            logger.warning(f"   Valence column {col} not found, using default 0.0")
            valence_data.append(np.zeros(N))

    valences = np.stack(valence_data, axis=1) / 2.0  # -2 to +2 → [-1, +1]
    valences = np.clip(valences, -1, 1)
    valences = torch.tensor(valences, dtype=torch.float32).to(device)

    # =========================================================================
    # TOLERANCES [N, 5] — profile metadata only (not used in ODE dynamics)
    # =========================================================================
    tol_cols = ['tol_rain', 'tol_crowding', 'tol_darkness',
                'tol_traffic', 'tol_temperature']
    tol_data = []
    for col in tol_cols:
        if col in df.columns:
            tol_data.append(df[col].values)
        else:
            logger.warning(f"   Tolerance column {col} not found, using default 0.5")
            tol_data.append(np.full(N, 4.0))  # Middle of 1-7 scale

    tolerances = np.stack(tol_data, axis=1)
    tolerances = (tolerances - 1.0) / 6.0  # 1-7 → 0-1
    tolerances = torch.tensor(tolerances, dtype=torch.float32).to(device)

    # STRESSORS: NOT LOADED in baseline version.

    # =========================================================================
    # AVAILABILITY [N, 4]
    # =========================================================================
    feas_cols = ['has_car', 'has_bike', 'has_pt', 'can_walk']
    feas_data = []
    for col in feas_cols:
        if col in df.columns:
            feas_data.append(df[col].values)
        else:
            feas_data.append(np.ones(N))

    availability = np.stack(feas_data, axis=1)
    availability = torch.tensor(availability, dtype=torch.float32).to(device)

    # =========================================================================
    # INITIAL STATE [N, n_nodes]
    # =========================================================================
    n_nodes = n_needs + 2 * n_modes
    nodes, idx_needs, idx_acts, idx_valence, _ = build_topology(n_needs, n_modes)

    initial_state = torch.zeros(N, n_nodes, dtype=torch.float32).to(device)
    initial_state[:, idx_needs[0]:idx_needs[-1]+1] = needs
    action_prior  = torch.sigmoid(valences) * 0.2 * availability
    initial_state[:, idx_acts[0]:idx_acts[-1]+1]   = action_prior
    initial_state[:, idx_valence[0]:idx_valence[-1]+1] = valences

    # =========================================================================
    # VALUES [N, 4] — metadata only
    # =========================================================================
    val_cols = ['val_bio', 'val_alt', 'val_ego', 'val_hed']
    val_data = []
    for col in val_cols:
        if col in df.columns:
            val_data.append(df[col].values)
        else:
            val_data.append(np.full(N, 4.0))

    values = np.stack(val_data, axis=1)
    values = (values - 1.0) / 6.0
    values = torch.tensor(values, dtype=torch.float32).to(device)

    # =========================================================================
    # TARGETS [N, 4]
    # =========================================================================
    freq_cols = ['freq_car', 'freq_bike', 'freq_pt', 'freq_walk']
    if all(col in df.columns for col in freq_cols):
        freqs     = torch.tensor(df[freq_cols].values, dtype=torch.float32)
        row_sums  = freqs.sum(dim=1, keepdim=True)
        soft_targets = torch.where(
            row_sums > 0,
            freqs / row_sums,
            torch.ones_like(freqs) / 4.0
        ).to(device)
    else:
        logger.warning("   Frequency columns not found, using uniform targets")
        soft_targets = torch.ones(N, 4, dtype=torch.float32).to(device) / 4.0

    hard_targets = soft_targets.argmax(dim=1)

    return {
        'needs':         needs,
        'beliefs':       beliefs,
        'valences':      valences,
        'tolerances':    tolerances,   # profile metadata — not in ODE
        # 'stressors' is intentionally absent in baseline
        'availability':  availability,
        'feasibility':   availability,
        'initial_state': initial_state,
        'values':        values,
        'soft_targets':  soft_targets,
        'hard_targets':  hard_targets,
        'metadata':      [{}] * N,
        'modes':         MODES_4,
        'n_modes':       n_modes,
        'dataframe':     df
    }


# =============================================================================
# SELF-TEST — Symmetric HOTCO-Grossberg Baseline v1.0
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🧠 Symmetric HOTCO-Grossberg Baseline v1.0 — Self Test")
    print("   No Environmental Perturbations | 0 Learnable Parameters")
    print("=" * 70)

    DEVICE   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    N_TEST   = 50
    N_NEEDS  = 11
    N_MODES  = 4
    N_NODES  = N_NEEDS + 2 * N_MODES  # 19

    print(f"🖥️  Device: {DEVICE}")
    print(f"📦 torchdiffeq: {TORCHDIFFEQ_AVAILABLE}")

    # =========================================================================
    # TEST 1: Model Creation
    # =========================================================================
    print("\n📋 Test 1: Model Creation (4 modes, 11 needs)")
    model = DeepHOTCO_v4(n_modes=N_MODES, n_needs=N_NEEDS, t_max=4.0).to(DEVICE)

    n_params    = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"   Total parameters: {n_params}")
    print(f"   Trainable parameters: {n_trainable} (expected: 0)")
    print(f"   Topology: {model.n_nodes} nodes = {model.n_needs} needs + "
          f"{model.n_modes} actions + {model.n_modes} valences")
    print(f"   Mode names: {model.mode_names}")
    assert n_trainable == 0, f"Expected 0 learnable params, got {n_trainable}"

    # =========================================================================
    # TEST 2: Synthetic Data (no stressors)
    # =========================================================================
    print("\n📋 Test 2: Synthetic Data (no stressors)")
    initial_state = torch.zeros(N_TEST, N_NODES, device=DEVICE)
    initial_state[:, model.idx_needs[0]:model.idx_needs[-1]+1] = \
        torch.rand(N_TEST, N_NEEDS, device=DEVICE)
    initial_state[:, model.idx_valence[0]:model.idx_valence[-1]+1] = \
        torch.rand(N_TEST, N_MODES, device=DEVICE) * 2 - 1
    valences_init = initial_state[:, model.idx_valence[0]:model.idx_valence[-1]+1]
    initial_state[:, model.idx_acts[0]:model.idx_acts[-1]+1] = \
        torch.sigmoid(valences_init) * 0.2

    beliefs      = torch.randn(N_TEST, N_MODES, N_NEEDS, device=DEVICE) * 0.5
    tolerances   = torch.rand(N_TEST, N_STRESSORS, device=DEVICE) * 0.5 + 0.25  # profile only
    availability = torch.ones(N_TEST, N_MODES, device=DEVICE)
    availability[:5, 0] = 0  # First 5 agents don't have car

    print(f"   Initial state shape: {initial_state.shape}")
    print(f"   Beliefs shape:       {beliefs.shape}")
    print(f"   Tolerances shape:    {tolerances.shape}  (profile metadata)")
    print(f"   Availability shape:  {availability.shape}")
    print("   ✅ No stressors tensor — baseline mode confirmed")

    # =========================================================================
    # TEST 3: Forward Pass (no stressors, no tolerances in signature)
    # =========================================================================
    print("\n📋 Test 3: Forward Pass")
    with torch.no_grad():
        final_state, traces = model(
            initial_state, beliefs, availability,
            return_trace=True
        )

    print(f"   Final state shape: {final_state.shape}")
    print(f"   Traces generated: {len(traces)}")
    print(f"   W_pos stored: {model.current_W_pos is not None}")

    # =========================================================================
    # TEST 4: Trace Analysis
    # =========================================================================
    print("\n📋 Test 4: Trace Analysis")
    conv_rate      = sum(1 for t in traces if t.convergence_achieved) / len(traces)
    diss_rate      = sum(1 for t in traces if t.is_dissonant()) / len(traces)
    mean_rt        = np.mean([t.reaction_time for t in traces])
    mean_c_struct  = np.mean([t.structural_conflict for t in traces])
    env_pressures  = [t.environmental_pressure for t in traces]

    print(f"   Convergence rate:      {conv_rate*100:.1f}%")
    print(f"   Dissonance rate:       {diss_rate*100:.1f}%  (structural only)")
    print(f"   Mean reaction time:    {mean_rt:.2f}s")
    print(f"   Mean C_structural:     {mean_c_struct:.3f}")
    print(f"   D_environmental mean:  {np.mean(env_pressures):.3f}  (expected: 0.0)")
    assert all(p == 0.0 for p in env_pressures), "D_env should be 0.0 in baseline"

    mode_counts = {}
    for t in traces:
        mode_counts[t.final_choice] = mode_counts.get(t.final_choice, 0) + 1
    print(f"   Mode distribution:")
    for mode in model.mode_names:
        count = mode_counts.get(mode, 0)
        print(f"      {mode.upper()}: {count/len(traces)*100:.1f}%")

    # =========================================================================
    # TEST 5: Sample Trace (D_env = 0 always)
    # =========================================================================
    print("\n📋 Test 5: Sample Trace (Agent 0)")
    t0 = traces[0]
    print(f"   Final choice:      {t0.final_choice.upper()}")
    print(f"   Base preference:   {t0.base_preference.upper()}")
    print(f"   C_structural:      {t0.structural_conflict:.3f}")
    print(f"   D_environmental:   {t0.environmental_pressure:.3f}  (fixed at 0.0)")
    print(f"   D_behavioral:      {int(t0.behavioral_dissonance)}  (binary — mode flip)")
    print(f"   D_beh_continuous:  {t0.behavioral_dissonance_continuous:.4f}  (JS-divergence)")
    print(f"   Dissonance type:   {t0.get_dissonance_type()}")

    # =========================================================================
    # TEST 6: Cognitive Passport (tolerances as profile metadata)
    # =========================================================================
    print("\n📋 Test 6: Cognitive Passport")
    values_meta   = torch.rand(1, len(VALUE_NAMES), device=DEVICE)
    test_metadata = {
        'profile': {'age': 28, 'gender_raw': 'female', 'ovgu_active': True},
        'pois': {'n_pois': 5, 'mode_accessibility': {'car': True, 'bike': True}},
        'ranking': ['comfort', 'speed', 'cost']
    }

    passport_json = model.generate_cognitive_passport(
        agent_id="BASELINE-001",
        initial_state=initial_state[0],
        beliefs=beliefs[0],
        tolerances=tolerances[0],  # profile metadata — not in ODE
        availability=availability[0],
        values=values_meta[0],
        metadata=test_metadata
    )

    passport = json.loads(passport_json)
    cp       = passport['cognitive_passport']
    print(f"   Version:           {cp['version']}")
    print(f"   Model:             {cp['model']}")
    print(f"   Agent ID:          {cp['agent_id']}")
    print(f"   Final choice:      {cp['deliberation']['final_choice']}")
    print(f"   D_environmental:   {cp['dissonance_triad']['D_environmental']}  (0.0 confirmed)")
    print(f"   Tolerance note:    {cp['profile']['tolerance_note'][:50]}...")
    print(f"   Narrative:         {cp['xai_summary']['decision_narrative'][:80]}...")

    # =========================================================================
    # TEST 7: 5-mode topology
    # =========================================================================
    print("\n📋 Test 7: 5-Mode Topology (car_green)")
    model5 = DeepHOTCO_v4(n_modes=5, n_needs=11, t_max=4.0).to(DEVICE)
    print(f"   Topology: {model5.n_nodes} nodes = {model5.n_needs} needs + "
          f"{model5.n_modes} actions + {model5.n_modes} valences")

    N5    = 10
    init5 = torch.zeros(N5, model5.n_nodes, device=DEVICE)
    init5[:, :11]   = torch.rand(N5, 11, device=DEVICE)
    init5[:, 16:21] = torch.rand(N5, 5, device=DEVICE) * 2 - 1
    init5[:, 11:16] = torch.sigmoid(init5[:, 16:21]) * 0.2
    bel5  = torch.randn(N5, 5, 11, device=DEVICE) * 0.5
    fea5  = torch.ones(N5, 5, device=DEVICE)

    with torch.no_grad():
        fs5, tr5 = model5(init5, bel5, fea5, return_trace=True)
    print(f"   Final state shape: {fs5.shape}")
    print(f"   Sample choice:     {tr5[0].final_choice.upper()}")
    print(f"   ✅ 5-mode topology works without stressors")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("✅ Symmetric HOTCO-Grossberg Baseline v1.1 — All tests passed")
    print("   Fix B: mini-ODE reuses GrossbergDynamics via euler_integrate")
    print("   Fix E: D_behavioral_continuous (JS-divergence) added to traces")
    print("=" * 70)
    print(f"\n📊 Architecture Summary:")
    print(f"   Topology:           {model.n_nodes} nodes  ({model.n_needs}N + {model.n_modes}A + {model.n_modes}V)")
    print(f"   Learnable params:   {n_trainable}  (pure theory-constrained)")
    print(f"   Fixed dynamics:     τ=0.8, A=0.15, B=1.0, C={model.inhibitory_floor:.3f}, λ={float(model.dynamics.lateral_inhib):.3f}")
    print(f"   Affective gating:   DISABLED — valence is modeled as a node")
    print(f"   ODE solver:         {'dopri5 / rk4 (torchdiffeq)' if TORCHDIFFEQ_AVAILABLE else 'Euler fallback'}")
    print(f"   Stressor P(t):      DISABLED  (Baseline v1.0)")
    print(f"   D_environmental:    Always 0.0")
    print(f"   D_behavioral:       Active (structural deliberation)")
    print(f"   Tolerances:         Profile metadata only (not in ODE)")
