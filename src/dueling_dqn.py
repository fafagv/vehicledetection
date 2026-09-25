"""
src/rl/dueling_dqn.py

Roadmap step 3 (model half): the 6-Stage Dueling DQN policy network.

NOT RUNTIME-VERIFIED in this sandbox (no network access to install
`torch` here -- see README's "What's been verified" section for this
platform's existing honesty convention on that point). Reviewed against
the documented `torch.nn` API; shapes are asserted in
`__init__`/`forward` rather than assumed, so a shape mismatch fails loud
at construction/first-call time instead of silently producing garbage
Q-values.

Stage breakdown (each stage is one `nn.Sequential` block so a stage can
be swapped/widened independently, matching this platform's general
"narrow, composable interfaces" convention -- see `src/core/base_model.py`):

    1. Input normalization/embedding -- raw StateVector features -> a
       fixed-width embedding via LayerNorm + Linear (stabilizes training
       against wildly different feature scales: position in meters,
       heading in radians, normalized LiDAR ranges in [0, 1]).
    2. Vision/kinematic feature stage -- position, velocity, heading,
       vehicle extents.
    3. LiDAR fusion stage -- the 9 normalized ray ranges, processed
       separately before merging, since they're a structurally different
       signal (spatial sweep) from stage 2's per-vehicle kinematics.
    4. Cross-modal fusion stage -- concatenates stage 2 + stage 3 outputs
       and mixes them (this is where "vision says X, LiDAR confirms/
       contradicts X" gets resolved into one representation).
    5. Multi-agent context stage -- a GRUCell that folds in a summary of
       *other* nearby agents' fused representations (mean-pooled), so a
       single vehicle's Q-values account for surrounding traffic, not
       just its own state -- the "Multi-Agent Decision" part of the
       roadmap.
    6. Shared trunk -- final shared representation feeding both dueling
       heads.

Dueling combination: Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a)), the
standard Wang et al. (2016) identity that makes V and A separately
identifiable during training.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from src.rl.types import Action

NUM_ACTIONS = len(Action)
# Must match StateVector.to_feature_tuple()'s output width in
# src/rl/types.py: 10 scalar kinematic/geometry features + 9 LiDAR rays.
NUM_KINEMATIC_FEATURES = 10
NUM_LIDAR_FEATURES = 9
STATE_FEATURE_DIM = NUM_KINEMATIC_FEATURES + NUM_LIDAR_FEATURES


class SixStageDuelingDQN(nn.Module):
    """See module docstring for the 6-stage breakdown.

    Args:
        embedding_dim: width of stage 1's output (and stages 2/3's input).
        hidden_dim: width used throughout stages 2-6.
        context_agents: expected number of *other* agents summarized by
            stage 5's context vector; only affects the input width of the
            GRUCell (the summary itself is mean-pooled down to
            `hidden_dim` regardless of how many agents are actually
            present at inference time -- see `forward`'s `context`
            argument).
    """

    def __init__(self, embedding_dim: int = 64, hidden_dim: int = 128) -> None:
        super().__init__()

        # Stage 1: input normalization/embedding.
        self.stage1_embed = nn.Sequential(
            nn.LayerNorm(STATE_FEATURE_DIM),
            nn.Linear(STATE_FEATURE_DIM, embedding_dim),
            nn.ReLU(inplace=True),
        )

        # Stage 2: vision/kinematic features (first 10 embedding dims'
        # worth of signal, but operating on the full stage-1 embedding
        # since stage 1 already mixed everything -- this stage's job is
        # depth, not selecting a feature subset).
        self.stage2_kinematic = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Stage 3: LiDAR fusion stage -- operates on the raw normalized
        # ranges directly (not the shared stage-1 embedding) so the
        # spatial ray structure isn't blurred together with kinematics
        # before this stage gets to specialize on it.
        self.stage3_lidar = nn.Sequential(
            nn.Linear(NUM_LIDAR_FEATURES, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Stage 4: cross-modal fusion of stage 2 + stage 3.
        self.stage4_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
        )

        # Stage 5: multi-agent context aggregation. GRUCell's hidden
        # state carries the fused representation forward while being
        # updated by a mean-pooled summary of nearby agents -- a cheap
        # stand-in for full attention-based multi-agent fusion that's
        # still permutation-invariant to neighbor ordering (mean pooling)
        # and O(agents) rather than O(agents^2) per step.
        self.stage5_context = nn.GRUCell(input_size=hidden_dim, hidden_size=hidden_dim)

        # Stage 6: shared trunk before the dueling split.
        self.stage6_shared = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.value_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(inplace=True), nn.Linear(hidden_dim // 2, 1))
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(inplace=True), nn.Linear(hidden_dim // 2, NUM_ACTIONS)
        )

        self.hidden_dim = hidden_dim

    def forward(
        self,
        state_features: torch.Tensor,
        context_summary: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            state_features: FloatTensor[B, STATE_FEATURE_DIM], each row a
                `StateVector.to_feature_tuple()`.
            context_summary: optional FloatTensor[B, hidden_dim] --
                mean-pooled stage-4 fusion output of *other* agents in the
                scene (caller's responsibility to compute; see
                `policy_engine.py::MultiAgentPolicyEngine`). Zeros if
                omitted (e.g. a lone vehicle with no neighbors).

        Returns:
            FloatTensor[B, NUM_ACTIONS] Q-values.
        """
        if state_features.dim() != 2 or state_features.size(-1) != STATE_FEATURE_DIM:
            raise ValueError(
                f"Expected state_features shape [B, {STATE_FEATURE_DIM}], got {tuple(state_features.shape)}."
            )
        batch_size = state_features.size(0)
        lidar_raw = state_features[:, NUM_KINEMATIC_FEATURES:]

        embedded = self.stage1_embed(state_features)
        kinematic_feat = self.stage2_kinematic(embedded)
        lidar_feat = self.stage3_lidar(lidar_raw)
        fused = self.stage4_fusion(torch.cat([kinematic_feat, lidar_feat], dim=-1))

        if context_summary is None:
            context_summary = torch.zeros(batch_size, self.hidden_dim, dtype=fused.dtype, device=fused.device)
        contextualized = self.stage5_context(fused, context_summary)

        shared = self.stage6_shared(contextualized)

        value = self.value_head(shared)  # [B, 1]
        advantage = self.advantage_head(shared)  # [B, NUM_ACTIONS]
        q_values = value + (advantage - advantage.mean(dim=-1, keepdim=True))
        return q_values

    def fusion_embedding(self, state_features: torch.Tensor) -> torch.Tensor:
        """Exposes stage 4's output on its own -- this is the vector
        `MultiAgentPolicyEngine` mean-pools across agents to build the
        `context_summary` each agent's own `forward` call consumes,
        without duplicating stages 1-4's logic there."""
        lidar_raw = state_features[:, NUM_KINEMATIC_FEATURES:]
        embedded = self.stage1_embed(state_features)
        kinematic_feat = self.stage2_kinematic(embedded)
        lidar_feat = self.stage3_lidar(lidar_raw)
        return self.stage4_fusion(torch.cat([kinematic_feat, lidar_feat], dim=-1))
