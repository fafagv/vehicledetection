"""
src/rl/config.py

Strictly-typed config for `src/rl`, loaded via
`src.config_loader.load_yaml_settings` against `configs/rl.yaml` --
same Pydantic Settings pattern `detect.py`/`export.py` use (see those
modules' docstrings for the full rationale). Chosen over Hydra here for
the same reason as `detect.py`: this configures deterministic thresholds
and hyperparameters for a single training/inference wiring, not a
sweep -- `python train_policy.py --multirun stop_line_approach_zone_m=...`
isn't a workflow this subsystem needs. If large-scale hyperparameter
sweeps over the DQN/trainer settings become a real need later, that's
the point to switch `train_policy.py` to Hydra the same way `train.py`
already is for detector training.

`VEHICLE_CV_RL_*` env vars override any field, same convention as
`src/api/dependencies.py::Settings`.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class RLSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="VEHICLE_CV_RL_", env_file=".env", extra="ignore")

    # -- state_bridge.py --
    fps_hint: float = 25.0
    pixels_per_meter: Optional[float] = None
    homography: Optional[List[List[float]]] = None
    heavy_class_names: List[str] = ["bus", "truck"]

    # -- dueling_dqn.py --
    embedding_dim: int = 64
    hidden_dim: int = 128

    # -- lidar_sim.py --
    lidar_num_rays: int = 9
    lidar_fov_deg: float = 180.0
    lidar_max_range_m: float = 60.0
    lidar_confirmation_tolerance_m: float = 1.5

    # -- rules.py --
    stop_line_approach_zone_m: float = 15.0
    stop_line_max_approach_speed_kmh: float = 15.0
    corridor_max_deviation_deg: float = 45.0
    separation_base_gap_m: float = 2.0
    separation_time_headway_s: float = 1.5
    separation_heavy_multiplier: float = 1.8

    # -- trainer.py --
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gamma: float = 0.99
    target_update_every_steps: int = 500
    huber_delta: float = 1.0
    grad_clip_norm: float = 10.0
    loss_ema_alpha: float = 0.05
    scheduler_factor: float = 0.5
    scheduler_patience: int = 20
    scheduler_min_lr: float = 1e-6
    replay_buffer_capacity: int = 100_000
    batch_size: int = 64
    warmup_transitions: int = 1_000  # min buffer size before training starts
    num_episodes: int = 500
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_episodes: int = 300


def build_rule_enforcer(settings: RLSettings):
    """Factory mirroring `src/pipelines/use_cases.py`'s `build_*_use_case`
    convention: the only place that turns flat settings fields into the
    nested `rules.py` objects."""
    import math

    from src.rl.rules import ElasticSATSeparation, HeadingCorridorRule, RuleEnforcer, StopLineRule

    return RuleEnforcer(
        stop_line_rule=StopLineRule(
            approach_zone_m=settings.stop_line_approach_zone_m,
            max_approach_speed_kmh=settings.stop_line_max_approach_speed_kmh,
        ),
        corridor_rule=HeadingCorridorRule(max_deviation_rad=math.radians(settings.corridor_max_deviation_deg)),
        separation_rule=ElasticSATSeparation(
            base_gap_m=settings.separation_base_gap_m,
            time_headway_s=settings.separation_time_headway_s,
            heavy_vehicle_multiplier=settings.separation_heavy_multiplier,
        ),
    )
