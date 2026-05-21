"""
===============================================================================
DEEP HOTCO v4.1 : Affective-Cognitive Transport Simulation (Fixed Dynamics)
===============================================================================
A Simplified Hybrid Neuro-Symbolic Architecture for Transport Mode Choice

Version: 4.1 "Full Hot Coherence Edition"
Author: Deep HOTCO Research Team

CHANGES from v4.0:
- NON-ZERO action init: sigmoid(valence)*0.2*feasibility (prevents dead neurons)
- DUAL AFFECTIVE GATING: gate_exc + gate_inh for full hot coherence
- SELECTIVE LATERAL INHIBITION: each action inhibited by sum of OTHERS
- REDUCED gating_alpha from 3.0 to 1.5 (wider deliberative zone)
- Added gating_beta = 0.75 (alpha/2, asymmetric confirmation bias)

ARCHITECTURE:
- 17-node topology (9 needs + 4 actions + 4 valences)
- Values are METADATA ONLY (not in ODE, only in Cognitive Passport)
- Dual Affective Gating (excitatory + inhibitory)
- 6 learnable parameters: stressor_scale[5] + global_magnitude[1]
- Fixed Grossberg dynamics (tau, A, B, C, lambda)

===============================================================================
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import os
import json
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any, Union
from datetime import datetime
from flask import Flask, jsonify, request

try:
    from torchdiffeq import odeint # type: ignore
    TORCHDIFFEQ_AVAILABLE = True
except ImportError:
    TORCHDIFFEQ_AVAILABLE = False
    print("torchdiffeq not found. Using Euler fallback.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DeepHOTCO_v4")


# =============================================================================
# COGNITIVE MAP TOPOLOGY (17 Nodes)
# =============================================================================
"""
State Vector Layout [17 nodes]:
    IDX 0-8:   NEEDS [9]     - Psychological requirements (from survey)
    IDX 9-12:  ACTIONS [4]   - Transport modes (compete via inhibition)
    IDX 13-16: VALENCES [4]  - Affective disposition (from survey)

Values are NOT in the state - they are metadata for Cognitive Passport only.
"""

NODES = {
    # Needs (0-8)
    'need_pro_env': 0, 'need_physical': 1, 'need_privacy': 2,
    'need_autonomy': 3, 'need_hedonism': 4, 'need_cost': 5,
    'need_speed': 6, 'need_safety': 7, 'need_comfort': 8,
    # Actions (9-12)
    'car': 9, 'bike': 10, 'pt': 11, 'walk': 12,
    # Valences (13-16)
    'valence_car': 13, 'valence_bike': 14, 'valence_pt': 15, 'valence_walk': 16
}

# Index mappings
IDX_NEEDS = list(range(0, 9))      # [0, 1, 2, 3, 4, 5, 6, 7, 8]
IDX_ACTS = list(range(9, 13))      # [9, 10, 11, 12]
IDX_VALENCE = list(range(13, 17))  # [13, 14, 15, 16]

MODES = ['car', 'bike', 'pt', 'walk']
NEED_NAMES = ['pro_env', 'physical', 'privacy', 'autonomy', 'hedonism', 
              'cost', 'speed', 'safety', 'comfort']
N_NODES = 17
N_NEEDS = 9
N_ACTIONS = 4
N_VALENCES = 4

IDX_MAP = {
    'NEEDS': IDX_NEEDS,
    'ACTS': IDX_ACTS,
    'VALENCE': IDX_VALENCE
}


# =============================================================================
# STRESSOR CONFIGURATION
# =============================================================================
"""
Stressors are environmental factors that perturb the cognitive system.
Each stressor affects specific nodes based on psychological theory.
"""

STRESSOR_NAMES = ['rain', 'crowding', 'darkness', 'traffic', 'temperature']
N_STRESSORS = len(STRESSOR_NAMES)

# Stressor Ã¢â€ â€™ Node perturbation mapping (effect strengths)
# Positive = excitation, Negative = inhibition
STRESSOR_NODE_TARGETS = {
    'rain': {
        'need_comfort': -0.5,      # Rain reduces comfort satisfaction
        'need_physical': -0.3,     # Rain reduces physical activity desire
        'valence_bike': -0.8,      # Strong negative affect toward bike
        'valence_walk': -0.6,      # Negative affect toward walk
        'valence_car': +0.4,       # Positive affect toward car
        'valence_pt': +0.2         # Slight positive toward PT
    },
    'crowding': {
        'need_privacy': -0.6,      # Crowding threatens privacy
        'need_comfort': -0.5,      # Crowding reduces comfort
        'valence_pt': -0.7,        # Strong negative toward PT
        'valence_walk': -0.3       # Negative toward walking in crowds
    },
    'darkness': {
        'need_safety': +0.5,       # Darkness activates safety concerns
        'valence_walk': -0.6,      # Negative toward walking at night
        'valence_bike': -0.5,      # Negative toward biking at night
        'valence_car': +0.2        # Car feels safer
    },
    'traffic': {
        'need_speed': +0.4,        # Traffic activates speed frustration
        'need_autonomy': -0.3,     # Traffic reduces sense of control
        'valence_car': -0.6,       # Car stuck in traffic
        'valence_bike': +0.4,      # Bike can bypass
        'valence_pt': +0.2         # PT has dedicated lanes
    },
    'temperature': {
        'need_comfort': -0.6,      # Extreme temp reduces comfort
        'need_physical': -0.4,     # Less desire for physical activity
        'valence_bike': -0.5,      # Negative toward biking
        'valence_walk': -0.5,      # Negative toward walking
        'valence_car': +0.3        # Car has climate control
    }
}


# =============================================================================
# VALUE NAMES (Metadata for Cognitive Passport)
# =============================================================================
VALUE_NAMES = ['biospheric', 'altruistic', 'egoistic', 'hedonic', 'security']
VALUE_DESCRIPTIONS = {
    'biospheric': "Values nature and environmental protection",
    'altruistic': "Values welfare of others and social justice",
    'egoistic': "Values personal success, wealth, and status",
    'hedonic': "Values pleasure, comfort, and enjoyment",
    'security': "Values safety, stability, and order"
}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DeliberationTrace:
    """
    Complete record of deliberation process for XAI.
    
    DISSONANCE TRIAD:
    - C_structural: Internal conflict before context (margin: 1 - (max-2nd)/max of converged actions)
    - D_environmental: How much stressors oppose base preference
    - D_behavioral: 1 if final choice differs from base preference, else 0
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
    
    # Dissonance Triad
    structural_conflict: float = 0.0       # C_structural Ã¢Ë†Ë† [0, 1]
    environmental_pressure: float = 0.0    # D_environmental Ã¢Ë†Ë† Ã¢â€žÂ
    behavioral_dissonance: float = 0.0     # D_behavioral Ã¢Ë†Ë† {0, 1}
    
    # Context
    base_preference: str = ""
    stress_level: float = 0.0
    conflict_intensity: List[float] = field(default_factory=list)
    
    def is_dissonant(self) -> bool:
        """Returns True if final choice differs from base preference."""
        return self.behavioral_dissonance > 0.5
    
    def get_dissonance_type(self) -> str:
        """Classify dissonance type based on triad values."""
        if not self.is_dissonant():
            return "none"
        
        high_struct = self.structural_conflict > 0.5
        high_env = abs(self.environmental_pressure) > 0.3
        
        if high_struct and high_env:
            return "compound"
        elif high_struct:
            return "structural"
        elif high_env:
            return "environmental"
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
            'structural_conflict': self.structural_conflict,
            'environmental_pressure': self.environmental_pressure,
            'behavioral_dissonance': self.behavioral_dissonance,
            'base_preference': self.base_preference,
            'stress_level': self.stress_level,
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
    environmental_pressure: float
    behavioral_dissonance: float
    base_preference: str
    stress_level: float
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
    max_entropy = np.log(logits.shape[dim])
    return torch.clamp(entropy / (max_entropy + eps), 0.0, 1.0)


def euler_integrate(dynamics_fn, initial_state: torch.Tensor, 
                    t_eval: torch.Tensor) -> torch.Tensor:
    """
    Simple Euler integration fallback (Gradient-Safe Version).
    Evita operaciones in-place ([:]=) para no romper el autograd.
    """
    states = [initial_state]
    dt = t_eval[1] - t_eval[0]
    
    for t in t_eval[1:]:
        dS = dynamics_fn(t, states[-1])
        
        # 1. Calcular el estado crudo
        raw_state = states[-1] + dt * dS
        
        # 2. Separar las partes (slicing es seguro, devuelve vistas)
        needs    = raw_state[:, 0:9]
        actions  = raw_state[:, 9:13]
        valences = raw_state[:, 13:17]
        
        # 3. Clampear creando NUEVOS tensores (sin modificar raw_state in-place)
        needs_clamped    = torch.clamp(needs, 0.0, 1.0)
        actions_clamped  = torch.clamp(actions, 0.0, 1.0)
        valences_clamped = torch.clamp(valences, -1.0, 1.0)
        
        # 4. Concatenar para formar el estado final limpio
        new_state = torch.cat([needs_clamped, actions_clamped, valences_clamped], dim=1)
        
        states.append(new_state)
    
    return torch.stack(states)


# =============================================================================
# STRESSOR PERTURBATION MODULE
# =============================================================================

class StressorPerturbation(nn.Module):
    """
    Converts environmental stressors into cognitive perturbations.
    
    PARTIALLY LEARNABLE perturbation matrix:
    
        perturbation_matrix = sign_mask * softplus(raw_magnitudes)
    
    Architecture:
      - sign_mask [N_STRESSORS x N_NODES]:       BUFFER (frozen).
                                                  +1 / -1 / 0 from VBN theory.
                                                  Defines topology AND direction.
                                                  Actions [9-12] are structurally 0.
      - raw_magnitudes [N_STRESSORS x N_NODES]:  PARAMETER.
                                                  Initialized via inv_softplus so that
                                                  softplus(raw) == |original theory values|.
                                                  softplus guarantees output > 0 always
                                                  (no dead-gradient risk unlike relu).
    
    Design rationale:
      - Theory defines WHO affects WHOM and in WHAT direction (sign_mask).
      - Data learns HOW MUCH (magnitudes) for the specific population.
      - softplus over relu: gradient = sigmoid(raw) > 0 everywhere.
        A magnitude can shrink but never fully die during training.
      - Positions where sign_mask == 0 output 0 regardless of magnitude.
        Their gradient through the matmul is also 0 â†’ they don't learn.
    
    Learnable parameters (30 total):
      - raw_magnitudes [5x17] : 24 positions with active gradient
      - stressor_scale [5]    : per-stressor global scaling
      - global_magnitude [1]  : overall perturbation amplitude
    """
    
    def __init__(self, n_nodes: int = N_NODES):
        super().__init__()
        self.n_nodes = n_nodes
        
        # â”€â”€ Build sign_mask and target magnitudes from theory â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        sign_mask      = torch.zeros(N_STRESSORS, n_nodes)
        mag_target     = torch.zeros(N_STRESSORS, n_nodes)   # |effect| values
        
        for s_idx, stressor_name in enumerate(STRESSOR_NAMES):
            if stressor_name in STRESSOR_NODE_TARGETS:
                for node_name, effect in STRESSOR_NODE_TARGETS[stressor_name].items():
                    if node_name in NODES:
                        node_idx = NODES[node_name]
                        sign_mask[s_idx, node_idx] = 1.0 if effect > 0 else -1.0
                        mag_target[s_idx, node_idx] = abs(effect)
        
        # sign_mask: frozen buffer. Topology + direction from theory.
        self.register_buffer('sign_mask', sign_mask)
        
        # raw_magnitudes: initialized so softplus(raw) == mag_target
        # inv_softplus(y) = log(exp(y) - 1).  Only applied where mag_target > 0.
        # Where mag_target == 0 (dead positions), raw = 0 (doesn't matter,
        # sign_mask will zero the output).
        raw_init = torch.zeros_like(mag_target)
        active   = mag_target > 0
        raw_init[active] = torch.log(torch.exp(mag_target[active]) - 1.0)
        
        # FROZEN: magnitudes from theory, not learned
        self.register_buffer('raw_magnitudes', raw_init)
        
        # OPTION A: log-space reparametrization (6 learnable parameters)
        # Gradients: ∂L/∂log(s) = s·∂L/∂s → normalizes signal across scales
        self.log_stressor_scale   = nn.Parameter(torch.log(torch.ones(N_STRESSORS) * 1.5))
        self.log_global_magnitude = nn.Parameter(torch.log(torch.tensor(2.0)))

    @property
    def stressor_scale(self) -> torch.Tensor:
        """Stressor scales in linear space (for logging/visualization)."""
        return torch.exp(self.log_stressor_scale)
    
    @property
    def global_magnitude(self) -> torch.Tensor:
        """Global magnitude in linear space (for logging/visualization)."""
        return torch.exp(self.log_global_magnitude)
    
    @property
    def perturbation_matrix(self) -> torch.Tensor:
        """Reconstruct perturbation matrix on the fly.
        
        sign_mask * softplus(raw_magnitudes):
          - Active positions: sign * positive_magnitude
          - Dead positions (sign==0): exactly 0
        """
        return self.sign_mask * F.softplus(self.raw_magnitudes)
    
    def forward(self, stressors: torch.Tensor, 
                tolerances: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute perturbation vector.
        
        Pipeline:
            vulnerability       = (1 - tolerance)^1.5
            effective_intensity = stressor * vulnerability * stressor_scale
            P = effective_intensity @ perturbation_matrix * global_magnitude
        
        Args:
            stressors:   [batch, N_STRESSORS] in [0, 1]
            tolerances:  [batch, N_STRESSORS] in [0, 1]
            
        Returns:
            perturbation:     [batch, N_NODES]
            effective_stress: [batch]
        """
        vulnerability       = torch.pow(1.0 - tolerances, 1.5)
        
        # Option A: convert from log-space to linear for computation
        scale      = torch.exp(self.log_stressor_scale)
        global_mag = torch.exp(self.log_global_magnitude)
        
        effective_intensity = stressors * vulnerability * scale
        
        # perturbation_matrix is the @property: sign_mask * softplus(raw)
        perturbation = torch.matmul(effective_intensity, self.perturbation_matrix)
        perturbation = perturbation * global_mag
        
        effective_stress = effective_intensity.mean(dim=1)
        
        return perturbation, effective_stress

# =============================================================================
# GROSSBERG DYNAMICS MODULE
# =============================================================================

class GrossbergDynamics(nn.Module):
    """
    Grossberg Shunting Equation with Simple Affective Gating.
    
    Ãâ€ž dS/dt = -AÃ‚Â·S + (B-S)Ã‚Â·E - (C+S)Ã‚Â·I + P(t)
    
    Where:
    - S: State activation
    - A: Passive decay
    - B: Excitatory cap
    - C: Inhibitory floor
    - E: Excitation (from W_pos @ state)
    - I: Inhibition (lateral inhibition + W_neg)
    - P: Stressor perturbation
    
    FIXED PARAMETERS (not learned):
    - Ãâ€ž (tau): 0.8 - Time constant
    - A (decay): 0.15 - Passive decay rate
    - B (excitatory_cap): 1.0 - Maximum activation
    - C (inhibitory_floor): 0.1 - Minimum activation
    - ÃŽÂ» (lateral_inhib): 3.0 - Lateral inhibition strength
    
    AFFECTIVE GATING:
    - Simple sigmoid gate based on valences
    - High valence for mode Ã¢â€ â€™ amplify excitation to that mode
    - Gate = ÃÆ’(ÃŽÂ± Ãƒâ€” valence) where ÃŽÂ± = 0.3 (fixed)
    
    BIDIRECTIONAL COHERENCE:
    - Actions and valences are bidirectionally coupled through W_pos,
      implementing true hot coherence within Grossberg dynamics.
    - Coupling strength is per-agent, derived from emotional intensity
      (mean absolute valence from survey data), requiring no additional
      learnable parameters (Domarchi et al. 2024, Thagard 2006).
    - This respects Grossberg gain control: feedback flows through
      (B-S)*relu(E) terms, not as an external additive force.
    """
    
    def __init__(self, n_nodes: int = N_NODES):
        super().__init__()
        self.n_nodes = n_nodes
        
        # FIXED parameters (registered as buffers, not Parameters)
        self.register_buffer('tau', torch.tensor(0.8))
        self.register_buffer('decay', torch.tensor(0.15))
        self.register_buffer('B', torch.tensor(1.0))
        self.register_buffer('C', torch.tensor(0.1))
        self.register_buffer('lateral_inhib', torch.tensor(3.0))
        self.register_buffer('divisive_sigma', torch.tensor(0.3))
        self.register_buffer('gating_alpha', torch.tensor(1.5))
        self.register_buffer('gating_beta', torch.tensor(0.75))  # α/2 for inhibitory gate
        
        # Context (set before integration)
        self.W_pos = None  # Excitatory weights [batch, N_NODES, N_NODES]
        self.W_neg = None  # Inhibitory weights [batch, N_NODES, N_NODES]
        self.feasibility = None  # Mode feasibility [batch, 4]
        self.perturbation = None  # Stressor perturbation [batch, N_NODES]
        
    def set_integration_context(self, W_pos: torch.Tensor, W_neg: torch.Tensor,
                                 feasibility: torch.Tensor, 
                                 perturbation: torch.Tensor):
        """Set context tensors before ODE integration."""
        self.W_pos = W_pos
        self.W_neg = W_neg
        self.feasibility = feasibility
        self.perturbation = perturbation
    
    def forward(self, t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """
        Grossberg Shunting con P divergente por tipo de nodo + Dual Affective Gating.
        
        CHANGES from v4.0:
        - Dual gating (excitatory alpha + inhibitory beta) for full hot coherence
        - Selective lateral inhibition: each action inhibited by sum of OTHER actions
        - Inhibitory gate: negative valence amplifies inhibition (confirmation bias)
        """
        batch_size = state.shape[0]
        device     = state.device
        dtype      = state.dtype

        # =====================================================================
        # DIVERGENCIA DE P: separar por tipo de nodo
        # =====================================================================
        valences = state[:, IDX_VALENCE]  # [batch, 4]

        if self.perturbation is not None:
            P_valences = self.perturbation[:, IDX_VALENCE]
            P_needs    = self.perturbation[:, IDX_NEEDS]
            valences_eval = valences + P_valences
            E_env_needs = F.relu(P_needs)
            I_env_needs = F.relu(-P_needs)
        else:
            valences_eval = valences
            E_env_needs   = torch.zeros(batch_size, len(IDX_NEEDS), device=device, dtype=dtype)
            I_env_needs   = torch.zeros(batch_size, len(IDX_NEEDS), device=device, dtype=dtype)

        # =====================================================================
        # DUAL AFFECTIVE GATING
        # gate_exc: positive valence -> amplify excitation
        # gate_inh: negative valence -> amplify inhibition (full hot coherence)
        # =====================================================================
        gate_exc = torch.sigmoid(self.gating_alpha * valences_eval)   # [batch, 4]
        gate_inh = torch.sigmoid(-self.gating_beta * valences_eval)   # [batch, 4]

        gate_exc_expanded = torch.ones(batch_size, self.n_nodes, device=device, dtype=dtype)
        gate_exc_expanded[:, IDX_ACTS[0]:IDX_ACTS[-1]+1] = gate_exc

        gate_inh_expanded = torch.ones(batch_size, self.n_nodes, device=device, dtype=dtype)
        gate_inh_expanded[:, IDX_ACTS[0]:IDX_ACTS[-1]+1] = gate_inh

        # =====================================================================
        # BELIEF PROPAGATION — dual-gated
        # =====================================================================
        W_exc_gated = self.W_pos * gate_exc_expanded.unsqueeze(2)
        excitation = torch.bmm(W_exc_gated, state.unsqueeze(2)).squeeze(2)

        W_inh_gated = self.W_neg * gate_inh_expanded.unsqueeze(2)
        inhibition_from_W = torch.bmm(W_inh_gated, state.unsqueeze(2)).squeeze(2)

        # =====================================================================
        # FUERZA EXTERNA EN NEEDS
        # =====================================================================
        E_env_full = torch.zeros_like(state)
        E_env_full[:, IDX_NEEDS[0]:IDX_NEEDS[-1]+1] = E_env_needs

        I_env_full = torch.zeros_like(state)
        I_env_full[:, IDX_NEEDS[0]:IDX_NEEDS[-1]+1] = I_env_needs

        total_excitation = excitation + E_env_full

        # =====================================================================
        # SELECTIVE LATERAL INHIBITION — each action inhibited by OTHERS only
        # (Grossberg 1988: competitive field uses sum of competitors)
        # =====================================================================
        actions      = state[:, IDX_ACTS]                           # [batch, 4]
        total_action = torch.sum(actions, dim=1, keepdim=True)      # [batch, 1]
        other_action = total_action - actions                        # [batch, 4]

        lateral_per_mode = (self.lateral_inhib * other_action) / (
            self.divisive_sigma + other_action + 1e-6
        )

        inhibition_lateral = torch.zeros_like(state)
        inhibition_lateral[:, IDX_ACTS[0]:IDX_ACTS[-1]+1] = lateral_per_mode

        total_inhibition = inhibition_from_W + inhibition_lateral + I_env_full

        # =====================================================================
        # GROSSBERG SHUNTING
        # =====================================================================
        dS_dt = (
            -self.decay * state +
            (self.B - state) * F.relu(total_excitation) -
            (self.C + state) * F.relu(total_inhibition)
        ) / self.tau

        # =====================================================================
        # FEASIBILITY MASK
        # =====================================================================
        if self.feasibility is not None:
            mask = torch.ones(batch_size, self.n_nodes, device=device, dtype=dtype)
            mask[:, IDX_ACTS[0]:IDX_ACTS[-1]+1] = self.feasibility
            dS_dt = dS_dt * mask

        return dS_dt



# =============================================================================
# MAIN MODEL: DEEP HOTCO v4
# =============================================================================

class DeepHOTCO_v4(nn.Module):
    """
    Deep HOTCO v4.1: Full Hot Coherence Transport Simulation
    
    6 learnable parameters in log-space (5 stressor scales + 1 global amplitude).
    Perturbation magnitudes frozen from VBN theory.
    Valences in [-1, +1] with dual gating (alpha=1.5, beta=0.75) for full hot coherence
    
    TOPOLOGY (17 nodes):
    - Needs [9]: Psychological requirements (indices 0-8)
    - Actions [4]: Transport modes (indices 9-12)
    - Valences [4]: Affective dispositions (indices 13-16)
    
    VALUES: Stored as metadata only (not in ODE state)
    
    KEY FEATURES:
    - Fixed Grossberg dynamics (Ãâ€ž, A, B, C, ÃŽÂ»)
    - Simple affective gating via valences
    - Stressor perturbation with learnable scales
    - Dissonance Triad computation
    - Cognitive Passport generation
    """
    
    def __init__(self, n_nodes: int = N_NODES,
                 t_max: float = 10.0,
                 dt_eval: float = 0.1,
                 rtol: float = 1e-3,
                 atol: float = 1e-4,
                 convergence_threshold: float = 0.005,
                 min_integration_time: float = 0.3):
        super().__init__()
        
        self.n_nodes = n_nodes
        self.t_max = t_max
        self.dt_eval = dt_eval
        self.rtol = rtol
        self.atol = atol
        self.convergence_threshold = convergence_threshold
        self.min_integration_time = min_integration_time
        
        # Modules
        self.stressor_perturbation = StressorPerturbation(n_nodes)
        self.dynamics = GrossbergDynamics(n_nodes)
        
        # State tracking (set during forward pass)
        self.current_stress_level = None
        self.current_W_pos = None
        self.current_W_neg = None
        self.last_traces = None
    
    def _build_W_matrices(self, beliefs: torch.Tensor, initial_state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Build excitatory (W_pos) and inhibitory (W_neg) weight matrices from beliefs.
        
        Args:
            beliefs: Belief matrix [batch, 4, 9] where beliefs[b, m, n] is
                     how well mode m satisfies need n (in [-1, +1])
                     
            initial_state: Initial cognitive state [batch, N_NODES] from survey.
                          Valences at IDX_VALENCE used to compute per-agent
                          emotional intensity for bidirectional coupling.
                     
        Returns:
            W_pos: Positive (excitatory) weights [batch, N_NODES, N_NODES]
            W_neg: Negative (inhibitory) weights [batch, N_NODES, N_NODES]
        """
        batch_size = beliefs.shape[0]
        device = beliefs.device
        dtype = beliefs.dtype
        
        # Initialize full weight matrix
        W = torch.zeros(batch_size, self.n_nodes, self.n_nodes, device=device, dtype=dtype)
        
        # Fill in belief connections: Action Ã¢â€ Â Need
        # beliefs[b, mode_idx, need_idx] Ã¢â€ â€™ W[b, action, need]
        for m_idx in range(N_ACTIONS):
            action_idx = IDX_ACTS[m_idx]
            for n_idx in range(N_NEEDS):
                need_idx = IDX_NEEDS[n_idx]
                W[:, action_idx, need_idx] = beliefs[:, m_idx, n_idx]
        
        # Bidirectional: Need Ã¢â€ Â Action (feedback, weaker)
        for m_idx in range(N_ACTIONS):
            action_idx = IDX_ACTS[m_idx]
            for n_idx in range(N_NEEDS):
                need_idx = IDX_NEEDS[n_idx]
                W[:, need_idx, action_idx] = beliefs[:, m_idx, n_idx] * 0.5
        
        # Valence Ã¢â€ â€™ Action (direct affective boost)
        # valence_car Ã¢â€ â€™ car, valence_bike Ã¢â€ â€™ bike, etc.
        for m_idx in range(N_ACTIONS):
            action_idx = IDX_ACTS[m_idx]
            valence_idx = IDX_VALENCE[m_idx]
            W[:, action_idx, valence_idx] = 1.0
        
        # Split into positive and negative parts
        W_pos = F.relu(W)
        W_neg = -F.relu(-W)
        
        # Bidirectional connections: actions <-> valences (Thagard 2006)
        # Coupling strength derived from per-agent emotional intensity:
        # agents with extreme valences (high emotional reactivity) couple
        # more strongly between actions and valences, consistent with
        # affect-as-information theory (Schwarz 2012) and individual
        # differences in coherence-seeking (Thagard & Aubie 2008).
        # Symmetric coupling: no learnable parameters added.
        agent_valences = initial_state[:, IDX_VALENCE]          # [batch, 4]
        emotional_intensity = agent_valences.abs().mean(dim=1, keepdim=True)  # [batch, 1]
        coupling = 0.3 + 0.2 * emotional_intensity              # [batch, 1] in [0.3, 0.5]
        for i in range(N_ACTIONS):
            W_pos[:, IDX_VALENCE[i], IDX_ACTS[i]] = coupling.squeeze(1)  # Action -> Valence
            W_pos[:, IDX_ACTS[i], IDX_VALENCE[i]] = coupling.squeeze(1)  # Valence -> Action
        
        return W_pos, W_neg
    
    def _compute_base_preference(self, initial_state: torch.Tensor,
                                  W_pos: torch.Tensor,
                                  W_neg: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute base preference via mini-ODE WITHOUT stressor perturbation.
        
        Updated to match main dynamics: dual gating + selective lateral inhibition.
        
        Returns:
            base_preference: Index of preferred mode [batch]
            action_excitation: Converged action activations [batch, 4]
            structural_conflict: C_structural margin metric [batch]
        """
        batch_size = initial_state.shape[0]
        device = initial_state.device
        dtype = initial_state.dtype
        
        decay = self.dynamics.decay
        B     = self.dynamics.B
        C     = self.dynamics.C
        tau   = self.dynamics.tau
        lat_inhib = self.dynamics.lateral_inhib
        div_sigma = self.dynamics.divisive_sigma
        alpha     = self.dynamics.gating_alpha
        beta      = self.dynamics.gating_beta
        
        state = initial_state.clone()
        dt = 0.1
        t_base = 2.0
        n_steps = int(t_base / dt)
        
        for _ in range(n_steps):
            valences = state[:, IDX_VALENCE]
            
            # Dual gating (matching main dynamics)
            gate_exc = torch.sigmoid(alpha * valences)
            gate_inh = torch.sigmoid(-beta * valences)
            
            gate_exc_exp = torch.ones(batch_size, N_NODES, device=device, dtype=dtype)
            gate_exc_exp[:, IDX_ACTS[0]:IDX_ACTS[-1]+1] = gate_exc
            
            gate_inh_exp = torch.ones(batch_size, N_NODES, device=device, dtype=dtype)
            gate_inh_exp[:, IDX_ACTS[0]:IDX_ACTS[-1]+1] = gate_inh
            
            # Excitation gated by positive affect
            W_exc_gated = W_pos * gate_exc_exp.unsqueeze(2)
            exc = torch.bmm(W_exc_gated, state.unsqueeze(2)).squeeze(2)
            
            # Inhibition gated by negative affect
            W_inh_gated = W_neg * gate_inh_exp.unsqueeze(2)
            inh_W = torch.bmm(W_inh_gated, state.unsqueeze(2)).squeeze(2)
            
            # Selective lateral inhibition (sum of OTHERS, not total)
            actions = state[:, IDX_ACTS]
            total_act = actions.sum(dim=1, keepdim=True)
            other_act = total_act - actions
            lateral_per_mode = (lat_inhib * other_act) / (div_sigma + other_act + 1e-6)
            inh_lat = torch.zeros_like(state)
            inh_lat[:, IDX_ACTS[0]:IDX_ACTS[-1]+1] = lateral_per_mode
            
            total_inh = inh_W + inh_lat
            
            dS = (
                -decay * state +
                (B - state) * F.relu(exc) -
                (C + state) * F.relu(total_inh)
            ) / tau
            
            # Gradient-safe clamp (no in-place)
            raw_state = state + dt * dS
            needs_c    = torch.clamp(raw_state[:, :9], 0.0, 1.0)
            actions_c  = torch.clamp(raw_state[:, 9:13], 0.0, 1.0)
            valences_c = torch.clamp(raw_state[:, 13:], -1.0, 1.0)
            state = torch.cat([needs_c, actions_c, valences_c], dim=1)
        
        # Extract results
        action_conv = state[:, IDX_ACTS]
        base_preference = action_conv.argmax(dim=1)
        
        sorted_actions, _ = torch.sort(action_conv, dim=1, descending=True)
        max_val  = sorted_actions[:, 0]
        sec_val  = sorted_actions[:, 1]
        structural_conflict = 1.0 - (max_val - sec_val) / (max_val + 1e-8)
        structural_conflict = torch.clamp(structural_conflict, 0.0, 1.0)
        
        return base_preference, action_conv, structural_conflict

    
    def _compute_environmental_dissonance(self, perturbation: torch.Tensor,
                                           base_preference: torch.Tensor) -> torch.Tensor:
        """
        Compute D_environmental: how much stressors oppose the base preference.
        
        Positive D_env means context OPPOSES the preference.
        Negative D_env means context REINFORCES the preference.
        """
        # Get perturbation to actions
        pert_actions = perturbation[:, IDX_ACTS]  # [batch, 4]
        
        # Get perturbation to valences
        pert_valences = perturbation[:, IDX_VALENCE]  # [batch, 4]
        
        # Combined perturbation effect
        total_pert = pert_actions + pert_valences * 0.5
        
        # Perturbation on preferred mode
        batch_size = base_preference.shape[0]
        pert_on_preferred = total_pert.gather(1, base_preference.unsqueeze(1)).squeeze(1)
        
        # Average perturbation on alternatives
        mask = torch.ones_like(total_pert)
        mask.scatter_(1, base_preference.unsqueeze(1), 0)
        pert_on_alternatives = (total_pert * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-6)
        
        # D_env = -pert_on_preferred + 0.5 Ãƒâ€” pert_on_alternatives
        # Negative perturbation on preferred = context opposes = positive D_env
        d_env = -pert_on_preferred + 0.5 * pert_on_alternatives
        
        return d_env
    
    def _check_convergence(self, states: torch.Tensor,
                           t_eval: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Check if ODE has converged for each agent.
        
        Convergence = max change in action nodes < threshold AND t > min_time
        
        Returns:
            converged: Boolean tensor [batch]
            reaction_times: Time of convergence [batch]
        """
        batch_size = states.shape[1]
        n_times = states.shape[0]
        device = states.device
        
        if n_times < 2:
            return (torch.zeros(batch_size, dtype=torch.bool, device=device),
                    torch.full((batch_size,), self.t_max, device=device))
        
        # Extract action trajectories
        actions = states[:, :, IDX_ACTS]  # [time, batch, 4]
        
        # Compute max change between consecutive timesteps
        max_change = torch.abs(actions[1:] - actions[:-1]).max(dim=2)[0]  # [time-1, batch]
        
        # Minimum timesteps before convergence can be declared
        min_idx = int(self.min_integration_time / self.dt_eval)
        
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
        actions = states[:, :, IDX_ACTS]  # [time, batch, 4]
        n_times = actions.shape[0]
        
        conflict = torch.stack([
            safe_entropy(actions[t] * 5.0, dim=1)
            for t in range(n_times)
        ])
        
        return conflict
    
    def forward(self, initial_state: torch.Tensor,
                beliefs: torch.Tensor,
                tolerances: torch.Tensor,
                stressors: torch.Tensor,
                feasibility: torch.Tensor,
                return_trace: bool = False) -> Tuple[torch.Tensor, Optional[List[DeliberationTrace]]]:
        """
        Full forward pass with Dissonance Triad computation.
        
        Args:
            initial_state: Initial cognitive state [batch, N_NODES]
                          Needs in [0, 1], Actions at 0, Valences in [0, 1]
            beliefs: Belief matrix [batch, 4, 9] where [m, n] is how well mode m
                     satisfies need n, in [-1, +1]
            tolerances: Stressor tolerances [batch, N_STRESSORS] in [0, 1]
            stressors: Current stressor levels [batch, N_STRESSORS] in [0, 1]
            feasibility: Mode feasibility [batch, 4] binary
            return_trace: Whether to return deliberation traces
            
        Returns:
            final_state: Converged state [batch, N_NODES]
            traces: List of DeliberationTrace (if return_trace=True)
        """
        batch_size = initial_state.shape[0]
        device = initial_state.device
        
        # =====================================================================
        # PHASE 1: Build Weight Matrices
        # =====================================================================
        W_pos, W_neg = self._build_W_matrices(beliefs, initial_state)
        self.current_W_pos = W_pos.detach()
        self.current_W_neg = W_neg.detach()
        
        # =====================================================================
        # PHASE 2: Compute Base Preference (BEFORE stressors)
        # =====================================================================
        with torch.no_grad():
            base_preference, action_excitation, structural_conflict = \
                self._compute_base_preference(initial_state, W_pos, W_neg)
        
        # =====================================================================
        # PHASE 3: Stressor Perturbation
        # =====================================================================
        perturbation, effective_stress = self.stressor_perturbation(stressors, tolerances)
        self.current_stress_level = effective_stress
        
        # =====================================================================
        # PHASE 4: Compute Environmental Dissonance
        # =====================================================================
        with torch.no_grad():
            environmental_dissonance = self._compute_environmental_dissonance(
                perturbation, base_preference
            )
        
        # =====================================================================
        # PHASE 5: ODE Integration
        # =====================================================================
        # Set dynamics context
        self.dynamics.set_integration_context(W_pos, W_neg, feasibility, perturbation)
        
        # Time points for evaluation
        n_steps = int(self.t_max / self.dt_eval) + 1
        t_eval = torch.linspace(0, self.t_max, n_steps, device=device)
        
        # Integrate
        if TORCHDIFFEQ_AVAILABLE:
            states = odeint(
                self.dynamics, initial_state, t_eval,
                method='dopri5', rtol=self.rtol, atol=self.atol
            )  # [time, batch, N_NODES]
        else:
            states = euler_integrate(self.dynamics, initial_state, t_eval)
        
        # =====================================================================
        # PHASE 6: Check Convergence
        # =====================================================================
        converged, reaction_times = self._check_convergence(states, t_eval)
        
        # Extract final state at convergence time (or end)
        final_states = []
        for b in range(batch_size):
            if converged[b]:
                t_idx = min(int(reaction_times[b].item() / self.dt_eval), n_steps - 1)
            else:
                t_idx = n_steps - 1
            final_states.append(states[t_idx, b])
        
        final_state = torch.stack(final_states)
        
        # =====================================================================
        # PHASE 7: Compute Final Choice and Behavioral Dissonance
        # =====================================================================
        final_actions = final_state[:, IDX_ACTS]
        final_choice = final_actions.argmax(dim=1)
        
        # Behavioral dissonance: did we choose differently from base preference?
        behavioral_dissonance = (final_choice != base_preference).float()
        
        # =====================================================================
        # PHASE 8: Build Traces
        # =====================================================================
        traces = None
        if return_trace:
            conflict_trajectory = self._compute_conflict_over_time(states)
            traces = []
            
            for b in range(batch_size):
                # Compute confidence (gap between top 2)
                sorted_actions, _ = torch.sort(final_actions[b], descending=True)
                confidence = float((sorted_actions[0] - sorted_actions[1]).detach())
                # Compute probabilities
                probs = F.softmax(final_actions[b].detach() * 5.0, dim=0)
                prob_dict = {MODES[i]: float(probs[i]) for i in range(N_ACTIONS)}
                
                trace = DeliberationTrace(
                    agent_id=b,
                    timestamps=t_eval.cpu().tolist(),
                    states=[states[t, b].detach().cpu() for t in range(n_steps)],
                    reaction_time=float(reaction_times[b]),
                    convergence_achieved=bool(converged[b]),
                    final_choice=MODES[int(final_choice[b])],
                    choice_confidence=confidence,
                    probabilities=prob_dict,
                    structural_conflict=float(structural_conflict[b]),
                    environmental_pressure=float(environmental_dissonance[b]),
                    behavioral_dissonance=float(behavioral_dissonance[b]),
                    base_preference=MODES[int(base_preference[b])],
                    stress_level=float(effective_stress[b]),
                    conflict_intensity=conflict_trajectory[:, b].cpu().tolist()
                )
                traces.append(trace)
            
            self.last_traces = traces
        
        return final_state, traces
    
    def predict_choice(self, initial_state: torch.Tensor,
                       beliefs: torch.Tensor,
                       tolerances: torch.Tensor,
                       stressors: torch.Tensor,
                       feasibility: torch.Tensor) -> List[ChoicePrediction]:
        """
        High-level prediction API with Dissonance Triad.
        """
        self.eval()
        with torch.no_grad():
            final_state, traces = self.forward(
                initial_state, beliefs, tolerances, stressors, feasibility,
                return_trace=True
            )
        
        return [
            ChoicePrediction(
                choice=t.final_choice,
                probabilities=t.probabilities,
                confidence=t.choice_confidence,
                reaction_time=t.reaction_time,
                structural_conflict=t.structural_conflict,
                environmental_pressure=t.environmental_pressure,
                behavioral_dissonance=t.behavioral_dissonance,
                base_preference=t.base_preference,
                stress_level=t.stress_level,
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
                                    stressors: torch.Tensor,
                                    feasibility: torch.Tensor,
                                    values: Optional[torch.Tensor] = None) -> str:
        """
        Generate complete Cognitive Passport JSON for routing engine integration.
        
        Args:
            agent_id: Unique identifier
            initial_state: Initial state [N_NODES] or [1, N_NODES]
            beliefs: Belief matrix [4, 9] or [1, 4, 9]
            tolerances: Tolerances [N_STRESSORS] or [1, N_STRESSORS]
            stressors: Stressors [N_STRESSORS] or [1, N_STRESSORS]
            feasibility: Feasibility [4] or [1, 4]
            values: Values metadata [5] or [1, 5] (optional, for description)
            
        Returns:
            JSON string of Cognitive Passport
        """
        device = next(self.parameters()).device
        
        # Ensure batch dimension
        if initial_state.dim() == 1:
            initial_state = initial_state.unsqueeze(0)
        if beliefs.dim() == 2:
            beliefs = beliefs.unsqueeze(0)
        if tolerances.dim() == 1:
            tolerances = tolerances.unsqueeze(0)
        if stressors.dim() == 1:
            stressors = stressors.unsqueeze(0)
        if feasibility.dim() == 1:
            feasibility = feasibility.unsqueeze(0)
        if values is not None and values.dim() == 1:
            values = values.unsqueeze(0)
        
        # Move to device
        initial_state = initial_state.to(device)
        beliefs = beliefs.to(device)
        tolerances = tolerances.to(device)
        stressors = stressors.to(device)
        feasibility = feasibility.to(device)
        
        # Run forward pass
        self.eval()
        with torch.no_grad():
            final_state, traces = self.forward(
                initial_state, beliefs, tolerances, stressors, feasibility,
                return_trace=True
            )
        
        trace = traces[0]
        state = final_state[0]
        
        # Build passport sections
        passport = {
            "cognitive_passport": {
                "version": "4.0.0",
                "agent_id": str(agent_id),
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "profile": self._build_profile_section(initial_state[0], tolerances[0], values),
                "context": self._build_context_section(stressors[0], trace),
                "deliberation": self._build_deliberation_section(trace),
                "dissonance_triad": self._build_dissonance_section(trace),
                "routing_parameters": self._build_routing_section(state, tolerances[0]),
                "xai_summary": self._build_xai_section(trace, state, beliefs[0])
            }
        }
        
        return json.dumps(passport, indent=2, ensure_ascii=False)
    
    def _build_profile_section(self, initial_state: torch.Tensor,
                                tolerances: torch.Tensor,
                                values: Optional[torch.Tensor]) -> Dict:
        """Build agent profile section."""
        # Extract needs from initial state
        needs = initial_state[IDX_NEEDS].cpu().numpy()
        need_dict = {NEED_NAMES[i]: round(float(needs[i]), 4) for i in range(N_NEEDS)}
        
        # Tolerances
        tol_dict = {STRESSOR_NAMES[i]: round(float(tolerances[i]), 4) 
                    for i in range(N_STRESSORS)}
        
        # Values (metadata)
        value_dict = None
        if values is not None:
            values_np = values.cpu().numpy() if values.dim() == 1 else values[0].cpu().numpy()
            value_dict = {VALUE_NAMES[i]: round(float(values_np[i]), 4) 
                          for i in range(len(VALUE_NAMES))}
        
        return {
            "needs": need_dict,
            "tolerances": tol_dict,
            "values": value_dict
        }
    
    def _build_context_section(self, stressors: torch.Tensor,
                                trace: DeliberationTrace) -> Dict:
        """Build environmental context section."""
        stressor_dict = {STRESSOR_NAMES[i]: round(float(stressors[i]), 4)
                         for i in range(N_STRESSORS)}
        
        # Interpret stress level
        stress = trace.stress_level
        if stress < 0.2:
            interpretation = "LOW - Agent is resilient to current conditions"
        elif stress < 0.4:
            interpretation = "MODERATE - Some environmental pressure present"
        elif stress < 0.6:
            interpretation = "ELEVATED - Significant stressor impact on decision"
        else:
            interpretation = "HIGH - Strong environmental pressure, potential for dissonance"
        
        return {
            "stressors": stressor_dict,
            "effective_stress": round(stress, 4),
            "stress_interpretation": interpretation
        }
    
    def _build_deliberation_section(self, trace: DeliberationTrace) -> Dict:
        """Build deliberation dynamics section."""
        # Decision difficulty based on structural conflict
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
    
    def _build_dissonance_section(self, trace: DeliberationTrace) -> Dict:
        """Build Dissonance Triad section."""
        # C_structural interpretation
        c_struct = trace.structural_conflict
        if c_struct < 0.3:
            c_interp = "LOW - Clear preference before context"
        elif c_struct < 0.5:
            c_interp = "MODERATE - Some internal conflict"
        else:
            c_interp = "HIGH - Significant value conflict"
        
        # D_environmental interpretation
        d_env = trace.environmental_pressure
        if d_env > 0.3:
            d_env_interp = "Context OPPOSES base preference (high friction)"
        elif d_env < -0.3:
            d_env_interp = "Context REINFORCES base preference (supportive)"
        else:
            d_env_interp = "Context has MINIMAL effect"
        
        # D_behavioral interpretation
        d_behav = trace.behavioral_dissonance
        if d_behav > 0.5:
            d_behav_interp = "DISSONANCE - Final choice conflicts with base preference"
        else:
            d_behav_interp = "ALIGNED - Final choice matches base preference"
        
        # Shift explanation
        shift_explanation = None
        if trace.base_preference != trace.final_choice:
            shift_explanation = (
                f"Agent shifted from {trace.base_preference.upper()} to "
                f"{trace.final_choice.upper()} due to environmental pressure "
                f"(D_env={d_env:.3f}) and stress level ({trace.stress_level:.3f})"
            )
        
        return {
            "C_structural": round(c_struct, 4),
            "C_interpretation": c_interp,
            "D_environmental": round(d_env, 4),
            "D_interpretation": d_env_interp,
            "D_behavioral": int(d_behav),
            "D_behavioral_interpretation": d_behav_interp,
            "dissonance_type": trace.get_dissonance_type(),
            "base_preference": trace.base_preference.upper(),
            "final_choice": trace.final_choice.upper(),
            "preference_shifted": trace.base_preference != trace.final_choice,
            "shift_explanation": shift_explanation
        }
    
    def _build_routing_section(self, state: torch.Tensor,
                                tolerances: torch.Tensor) -> Dict:
        """Build routing parameters for TOPTW optimizer."""
        # Mode weights from final action activations
        actions = state[IDX_ACTS]
        probs = F.softmax(actions * 5.0, dim=0)
        mode_weights = {MODES[i]: round(float(probs[i]), 4) for i in range(N_ACTIONS)}
        
        # Utility coefficients from needs
        needs = state[IDX_NEEDS]
        utility_coefficients = {
            "time_penalty": round(float(needs[6]), 4),      # need_speed
            "cost_penalty": round(float(needs[5]), 4),      # need_cost
            "safety_bonus": round(float(needs[7]), 4),      # need_safety
            "eco_bonus": round(float(needs[0]), 4),         # need_pro_env
            "comfort_penalty": round(float(needs[8]), 4),   # need_comfort
            "exercise_bonus": round(float(needs[1]), 4),    # need_physical
            "privacy_bonus": round(float(needs[2]), 4),     # need_privacy
            "autonomy_bonus": round(float(needs[3]), 4)     # need_autonomy
        }
        
        # Contextual flags from tolerances
        contextual_flags = {
            "avoid_unlit_paths": float(tolerances[2]) < 0.4,     # low darkness tolerance
            "prefer_covered_paths": float(tolerances[0]) < 0.4,  # low rain tolerance
            "tolerate_crowding": float(tolerances[1]) > 0.6,     # high crowding tolerance
            "avoid_traffic": float(tolerances[3]) < 0.4,         # low traffic tolerance
            "prefer_climate_control": float(tolerances[4]) < 0.4 # low temperature tolerance
        }
        
        return {
            "mode_weights": mode_weights,
            "utility_coefficients": utility_coefficients,
            "contextual_flags": contextual_flags
        }
    
    def _build_xai_section(self, trace: DeliberationTrace,
                           state: torch.Tensor,
                           beliefs: torch.Tensor) -> Dict:
        """Build human-readable XAI summary."""
        choice = trace.final_choice.upper()
        choice_idx = MODES.index(trace.final_choice)
        
        # Find key drivers (needs with positive beliefs for chosen mode)
        mode_beliefs = beliefs[choice_idx].cpu().numpy()  # [9]
        needs_state = state[IDX_NEEDS].cpu().numpy()
        
        # Driver = high need Ãƒâ€” positive belief
        driver_scores = mode_beliefs * needs_state
        driver_ranking = np.argsort(-driver_scores)
        
        key_drivers = []
        key_inhibitors = []
        for idx in driver_ranking:
            if driver_scores[idx] > 0.1:
                key_drivers.append(NEED_NAMES[idx])
            elif driver_scores[idx] < -0.1:
                key_inhibitors.append(NEED_NAMES[idx])
        
        # Generate narrative
        parts = [f"Agent chose {choice}"]
        
        if key_drivers:
            parts.append(f"driven by {', '.join(key_drivers[:3])} needs")
        
        if trace.base_preference != trace.final_choice:
            parts.append(
                f"despite base preference for {trace.base_preference.upper()}"
            )
            parts.append(
                f"Environmental pressure (D_env={trace.environmental_pressure:.2f}) "
                f"caused the shift"
            )
        
        if trace.structural_conflict > 0.5:
            parts.append("The decision involved significant internal conflict")
        
        narrative = ". ".join(parts) + "."
        
        return {
            "decision_narrative": narrative,
            "key_drivers": key_drivers[:5],
            "key_inhibitors": key_inhibitors[:5],
            "confidence_level": round(trace.choice_confidence, 4)
        }


# =============================================================================
# DATA LOADING
# =============================================================================

def load_and_prep_data_v4(csv_path: str, device: str = 'cpu') -> Dict:
    """
    Load and prepare data for DeepHOTCO v4.
    
    Expected CSV columns:
    - Needs: need_pro_env, need_physical, need_privacy, need_autonomy,
             need_hedonism, need_cost, need_speed, need_safety, need_comfort
             (scale 1-7, normalized to [0,1])
    - Beliefs: belief_{mode}_{need} for all 4Ãƒâ€”9 combinations
             (scale 1-5, centered to [-1,+1])
    - Valences: valence_car, valence_bike, valence_pt, valence_walk
             (scale -2 to +2, normalized to [0,1])
    - Tolerances: tol_rain, tol_crowding, tol_darkness, tol_traffic, tol_temperature
             (scale 1-7, normalized to [0,1])
    - Stressors: stressor_rain, stressor_crowding, stressor_darkness,
                 stressor_traffic, stressor_temperature
             (already [0,1])
    - Feasibility: has_car, has_bike, has_pt, can_walk (binary)
    - Targets: freq_car, freq_bike, freq_pt, freq_walk (scale 0-5)
    - Values (metadata): val_bio, val_alt, val_ego, val_hed, val_sec (scale 1-7)
    
    Returns:
        Dictionary with all tensors ready for model
    """
    logger.info(f"Ã°Å¸â€œâ€š Loading data from: {csv_path}")
    
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Data file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    N = len(df)
    logger.info(f"   Loaded {N} agents")
    
    # =========================================================================
    # NEEDS [N, 9] - Scale 1-7 Ã¢â€ â€™ [0, 1]
    # =========================================================================
    need_cols = ['need_pro_env', 'need_physical_activity', 'need_privacy',
                 'need_autonomy', 'need_hedonism', 'need_cost',
                 'need_speed', 'need_safety', 'need_comfort']
    
    # Handle column name variations
    need_data = []
    for col in need_cols:
        if col in df.columns:
            need_data.append(df[col].values)
        elif col.replace('_activity', '') in df.columns:
            need_data.append(df[col.replace('_activity', '')].values)
        else:
            logger.warning(f"   Need column {col} not found, using default 4.0")
            need_data.append(np.full(N, 4.0))
    
    needs = np.stack(need_data, axis=1)
    needs = (needs - 1.0) / 6.0  # Normalize 1-7 Ã¢â€ â€™ 0-1
    needs = torch.tensor(needs, dtype=torch.float32).to(device)
    
    # =========================================================================
    # BELIEFS [N, 4, 9] - Scale 1-5 Ã¢â€ â€™ [-1, +1]
    # =========================================================================
    beliefs = np.zeros((N, 4, 9), dtype=np.float32)
    
    mode_names = ['car', 'bike', 'pt', 'walk']
    need_suffixes = ['pro_env', 'physical_activity', 'privacy', 'autonomy',
                     'hedonism', 'cost', 'speed', 'safety', 'comfort']
    
    for m_idx, mode in enumerate(mode_names):
        for n_idx, need_suffix in enumerate(need_suffixes):
            col = f"belief_{mode}_{need_suffix}"
            if col in df.columns:
                raw_val = df[col].values
                centered = (raw_val - 3.0) / 2.0  # 1-5 Ã¢â€ â€™ [-1, +1]
                beliefs[:, m_idx, n_idx] = centered
            else:
                # Try alternative column names
                alt_col = col.replace('_activity', '')
                if alt_col in df.columns:
                    raw_val = df[alt_col].values
                    centered = (raw_val - 3.0) / 2.0
                    beliefs[:, m_idx, n_idx] = centered
    
    beliefs = torch.tensor(beliefs, dtype=torch.float32).to(device)
    
    # =========================================================================
    # VALENCES [N, 4] - Scale -2 to +2 → [-1, +1]
    # Preserves emotional polarity: negative = aversion, positive = attraction
    # =========================================================================
    valence_cols = ['valence_car', 'valence_bike', 'valence_pt', 'valence_walk']
    valence_data = []
    
    for col in valence_cols:
        if col in df.columns:
            valence_data.append(df[col].values)
        else:
            logger.warning(f"   Valence column {col} not found, using default 0.0")
            valence_data.append(np.zeros(N))
    
    valences = np.stack(valence_data, axis=1)
    valences = valences / 2.0  # -2 to +2 → [-1, +1]
    valences = np.clip(valences, -1, 1)
    valences = torch.tensor(valences, dtype=torch.float32).to(device)
    
    # =========================================================================
    # TOLERANCES [N, 5] - Scale 1-7 Ã¢â€ â€™ [0, 1]
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
    tolerances = (tolerances - 1.0) / 6.0  # 1-7 Ã¢â€ â€™ 0-1
    tolerances = torch.tensor(tolerances, dtype=torch.float32).to(device)
    
    # =========================================================================
    # STRESSORS [N, 5] - Already [0, 1]
    # =========================================================================
    stressor_cols = ['stressor_rain', 'stressor_crowding', 'stressor_darkness',
                     'stressor_traffic', 'stressor_temperature']
    stressor_data = []
    
    for col in stressor_cols:
        if col in df.columns:
            stressor_data.append(df[col].values)
        else:
            # Try alternative names
            alt_name = col.replace('stressor_', '')
            if alt_name in df.columns:
                stressor_data.append(df[alt_name].values)
            else:
                logger.warning(f"   Stressor column {col} not found, using default 0.0")
                stressor_data.append(np.zeros(N))
    
    stressors = np.stack(stressor_data, axis=1)
    stressors = np.clip(stressors, 0, 1)
    stressors = torch.tensor(stressors, dtype=torch.float32).to(device)
    
    # =========================================================================
    # FEASIBILITY [N, 4] - Binary
    # =========================================================================
    feas_cols = ['has_car', 'has_bike', 'has_pt', 'can_walk']
    feas_data = []
    
    for col in feas_cols:
        if col in df.columns:
            feas_data.append(df[col].values)
        else:
            # Default to available
            feas_data.append(np.ones(N))
    
    feasibility = np.stack(feas_data, axis=1)
    feasibility = torch.tensor(feasibility, dtype=torch.float32).to(device)
    
    # =========================================================================
    # INITIAL STATE [N, 17]
    # =========================================================================
    initial_state = torch.zeros(N, N_NODES, dtype=torch.float32).to(device)
    
    # Needs go into indices 0-8
    initial_state[:, IDX_NEEDS] = needs
    
    # Actions: valence-derived prior (prevents dead-neuron collapse in Grossberg)
    # Cognitive justification: agents have a pre-deliberative disposition toward
    # each mode based on their emotional valence (Thagard 2006, affect-as-information).
    action_prior = torch.sigmoid(valences) * 0.2 * feasibility  # [N, 4] in [0.05, 0.15]
    initial_state[:, IDX_ACTS] = action_prior
    
    # Valences go into indices 13-16
    initial_state[:, IDX_VALENCE] = valences
    
    # =========================================================================
    # VALUES [N, 5] - Metadata only
    # =========================================================================
    val_cols = ['val_bio', 'val_alt', 'val_ego', 'val_hed', 'val_sec']
    val_data = []
    
    for col in val_cols:
        if col in df.columns:
            val_data.append(df[col].values)
        else:
            val_data.append(np.full(N, 4.0))
    
    values = np.stack(val_data, axis=1)
    values = (values - 1.0) / 6.0  # 1-7 Ã¢â€ â€™ 0-1
    values = torch.tensor(values, dtype=torch.float32).to(device)
    
    # =========================================================================
    # TARGETS [N, 4] - Soft targets from frequencies
    # =========================================================================
    freq_cols = ['freq_car', 'freq_bike', 'freq_pt', 'freq_walk']
    
    if all(col in df.columns for col in freq_cols):
        freqs = torch.tensor(df[freq_cols].values, dtype=torch.float32)
        row_sums = freqs.sum(dim=1, keepdim=True)
        soft_targets = torch.where(
            row_sums > 0,
            freqs / row_sums,
            torch.ones_like(freqs) / 4.0
        ).to(device)
    else:
        logger.warning("   Frequency columns not found, using uniform targets")
        soft_targets = torch.ones(N, 4, dtype=torch.float32).to(device) / 4.0
    
    hard_targets = soft_targets.argmax(dim=1)
    
    # =========================================================================
    # RETURN
    # =========================================================================
    return {
        'needs': needs,
        'beliefs': beliefs,
        'valences': valences,
        'tolerances': tolerances,
        'stressors': stressors,
        'feasibility': feasibility,
        'initial_state': initial_state,
        'values': values,
        'soft_targets': soft_targets,
        'hard_targets': hard_targets,
        'dataframe': df
    }


# =============================================================================
# SELF-TEST
# =============================================================================

def run_hotoco():
    print("=" * 70)
    print("Ã°Å¸Â§Â  DEEP HOTCO v4.0 : Streamlined Edition - Self Test")
    print("=" * 70)
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Ã°Å¸â€“Â¥Ã¯Â¸Â Device: {DEVICE}")
    print(f"Ã°Å¸â€œÂ¦ torchdiffeq: {TORCHDIFFEQ_AVAILABLE}")
    
    # =========================================================================
    # TEST 1: Model Creation
    # =========================================================================
    print("\nÃ°Å¸â€œâ€¹ Test 1: Model Creation")
    model = DeepHOTCO_v4(t_max=4.0).to(DEVICE)
    
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total parameters: {n_params}")
    print(f"   Trainable parameters: {n_trainable}")
    print(f"   Expected: 6 (stressor_scale[5] + global_magnitude)")
    
    # =========================================================================
    # TEST 2: Generate Synthetic Data
    # =========================================================================
    print("\nÃ°Å¸â€œâ€¹ Test 2: Synthetic Data Generation")
    N_TEST = 50
    
    # Generate random data
    initial_state = torch.zeros(N_TEST, N_NODES, device=DEVICE)
    initial_state[:, IDX_NEEDS] = torch.rand(N_TEST, N_NEEDS, device=DEVICE)
    initial_state[:, IDX_VALENCE] = torch.rand(N_TEST, N_VALENCES, device=DEVICE) * 2 - 1  # [-1, +1]
 
    beliefs = torch.randn(N_TEST, N_ACTIONS, N_NEEDS, device=DEVICE) * 0.5
    tolerances = torch.rand(N_TEST, N_STRESSORS, device=DEVICE) * 0.5 + 0.25
    stressors = torch.rand(N_TEST, N_STRESSORS, device=DEVICE) * 0.7
    feasibility = torch.ones(N_TEST, N_ACTIONS, device=DEVICE)
    feasibility[:10, 0] = 0  # First 10 agents don't have car
    
    print(f"   Initial state shape: {initial_state.shape}")
    print(f"   Beliefs shape: {beliefs.shape}")
    print(f"   Tolerances shape: {tolerances.shape}")
    print(f"   Stressors shape: {stressors.shape}")
    print(f"   Feasibility shape: {feasibility.shape}")
    
    # =========================================================================
    # TEST 3: Forward Pass
    # =========================================================================
    print("\nÃ°Å¸â€œâ€¹ Test 3: Forward Pass")
    
    with torch.no_grad():
        final_state, traces = model(
            initial_state, beliefs, tolerances, stressors, feasibility,
            return_trace=True
        )
    
    print(f"   Final state shape: {final_state.shape}")
    print(f"   Traces generated: {len(traces)}")
    print(f"   W_pos stored: {model.current_W_pos is not None}")
    print(f"   W_neg stored: {model.current_W_neg is not None}")
    
    # =========================================================================
    # TEST 4: Trace Analysis
    # =========================================================================
    print("\nÃ°Å¸â€œâ€¹ Test 4: Trace Analysis")
    
    conv_rate = sum(1 for t in traces if t.convergence_achieved) / len(traces)
    diss_rate = sum(1 for t in traces if t.is_dissonant()) / len(traces)
    mean_rt = np.mean([t.reaction_time for t in traces])
    mean_c_struct = np.mean([t.structural_conflict for t in traces])
    mean_d_env = np.mean([t.environmental_pressure for t in traces])
    
    print(f"   Convergence rate: {conv_rate*100:.1f}%")
    print(f"   Dissonance rate: {diss_rate*100:.1f}%")
    print(f"   Mean reaction time: {mean_rt:.2f}s")
    print(f"   Mean C_structural: {mean_c_struct:.3f}")
    print(f"   Mean D_environmental: {mean_d_env:.3f}")
    
    # Mode distribution
    mode_counts = {}
    for t in traces:
        mode_counts[t.final_choice] = mode_counts.get(t.final_choice, 0) + 1
    
    print(f"   Mode distribution:")
    for mode in MODES:
        count = mode_counts.get(mode, 0)
        pct = count / len(traces) * 100
        print(f"      {mode.upper()}: {pct:.1f}%")
    
    # =========================================================================
    # TEST 5: Sample Trace
    # =========================================================================
    print("\nÃ°Å¸â€œâ€¹ Test 5: Sample Trace (Agent 0)")
    t = traces[0]
    print(f"   Final choice: {t.final_choice.upper()}")
    print(f"   Base preference: {t.base_preference.upper()}")
    print(f"   Probabilities: {t.probabilities}")
    print(f"   Confidence: {t.choice_confidence:.3f}")
    print(f"   Reaction time: {t.reaction_time:.2f}s")
    print(f"   Converged: {t.convergence_achieved}")
    print(f"   Dissonance Triad:")
    print(f"      C_structural: {t.structural_conflict:.3f}")
    print(f"      D_environmental: {t.environmental_pressure:.3f}")
    print(f"      D_behavioral: {int(t.behavioral_dissonance)}")
    print(f"      Type: {t.get_dissonance_type()}")
    
    # =========================================================================
    # TEST 6: Cognitive Passport
    # =========================================================================
    print("\nÃ°Å¸â€œâ€¹ Test 6: Cognitive Passport Generation")
    
    values = torch.rand(1, 5, device=DEVICE)
    
    passport_json = model.generate_cognitive_passport(
        agent_id="TEST-001",
        initial_state=initial_state[0],
        beliefs=beliefs[0],
        tolerances=tolerances[0],
        stressors=stressors[0],
        feasibility=feasibility[0],
        values=values[0]
    )
    
    passport = json.loads(passport_json)
    cp = passport['cognitive_passport']
    
    print(f"   Version: {cp['version']}")
    print(f"   Agent ID: {cp['agent_id']}")
    print(f"   Final choice: {cp['deliberation']['final_choice']}")
    print(f"   Dissonance type: {cp['dissonance_triad']['dissonance_type']}")
    print(f"   Narrative: {cp['xai_summary']['decision_narrative'][:80]}...")
    
    # =========================================================================
    # TEST 7: Gradient Flow (if training)
    # =========================================================================
    print("\nÃ°Å¸â€œâ€¹ Test 7: Gradient Flow")
    
    model.train()
    
    # Forward pass with gradients
    final_state, _ = model(
        initial_state, beliefs, tolerances, stressors, feasibility,
        return_trace=False
    )
    
    # Dummy loss
    target = torch.softmax(torch.randn(N_TEST, N_ACTIONS, device=DEVICE), dim=1)
    action_logits = final_state[:, IDX_ACTS]
    loss = F.mse_loss(F.softmax(action_logits, dim=1), target)
    
    loss.backward()
    
    grad_info = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_info[name] = param.grad.norm().item()
        else:
            grad_info[name] = None
    
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Gradients:")
    for name, norm in grad_info.items():
        if norm is not None:
            print(f"      {name}: {norm:.6f}")
        else:
            print(f"      {name}: None")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("Ã¢Å“â€¦ DEEP HOTCO v4.0 - All tests passed")
    print("=" * 70)
    
    print("\nÃ°Å¸â€œÅ  Architecture Summary:")
    print(f"   Topology: {N_NODES} nodes ({N_NEEDS} needs + {N_ACTIONS} actions + {N_VALENCES} valences)")
    print(f"   Learnable parameters: {n_trainable}")
    print(f"   Fixed dynamics: Ãâ€ž=0.8, A=0.15, B=1.0, C=0.1, ÃŽÂ»=3.0")
    print(f"   ODE solver: {'dopri5 (torchdiffeq)' if TORCHDIFFEQ_AVAILABLE else 'Euler fallback'}")
    print(f"   Outputs: Dissonance Triad + Cognitive Passport")

def launch_server():
    app = Flask(__name__)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepHOTCO_v4(t_max=4.0).to(DEVICE)

    def get_as_array(data, key, keys):
        values = data[key]
        return torch.tensor([values[i] for i in keys], device=DEVICE, dtype=torch.float32)
   
    @app.route('/api/dyconet', methods=['POST'])
    def invoke():
        data = request.get_json()

        N_TEST = 1
        initial_state = torch.zeros(N_TEST, N_NODES, device=DEVICE)
        beliefs = torch.ones(N_TEST, N_ACTIONS, N_NEEDS, device=DEVICE) * 0.5 # TODO: where to get from?
        tolerances = torch.ones(N_TEST, N_STRESSORS, device=DEVICE)
        stressors = torch.ones(N_TEST, N_STRESSORS, device=DEVICE)
        feasibility = torch.ones(N_TEST, N_ACTIONS, device=DEVICE) # zero means does not own e.g car

        try:
            initial_state[:, IDX_NEEDS] = get_as_array(data, 'needs', NEED_NAMES)
            initial_state[:, IDX_VALENCE] = get_as_array(data, 'valences', ['car', 'bike', 'pt', 'walk'])
            stressors[:] = get_as_array(data, 'stressors', STRESSOR_NAMES)
            tolerances[:] = get_as_array(data, 'tolerances', STRESSOR_NAMES)
        except KeyError as e:
            return jsonify({'error': f"Missing key {e}"}, 400)

        with torch.no_grad():
            final_state, traces = model(
                initial_state, beliefs, tolerances, stressors, feasibility,
                return_trace=True
            )
        t = traces[0]
        return jsonify({
            'probabilities': t.probabilities,
            'confidence': t.choice_confidence,
        }), 200
       
    app.run(host="0.0.0.0")

if __name__ == "__main__":
    launch_server()