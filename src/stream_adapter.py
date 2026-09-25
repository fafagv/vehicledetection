"""
src/rl/stream_adapter.py

Wires the RL decision subsystem into a *live* `StreamTracker` feed via
composition, not modification: `StreamTracker.__init__` already accepts
an `on_result: Callable[[FrameResult], None]` callback (see
`src/pipelines/stream_tracker.py`) precisely so a consumer like this one
can hook the perception pipeline without either module depending on the
other. This keeps `src/rl` entirely optional -- omit `RLStreamAdapter`
and `StreamTracker` behaves exactly as it did before this subsystem
existed, and none of its existing tests needed to change.

NOT RUNTIME-VERIFIED end-to-end here, since the default policy engine
(`MultiAgentPolicyEngine`) needs torch (see `dueling_dqn.py`'s
docstring). The glue logic itself (state building, optional LiDAR
fusion, exception isolation, callback dispatch) is torch-independent and
IS covered, against a fake duck-typed policy engine, in
`tests/test_stream_adapter.py`.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, List, Optional, Protocol, Sequence, Tuple

from src.core.types import FrameResult
from src.rl.lidar_sim import FusedTrack, LidarRaycaster, VisionLidarFusion
from src.rl.rules import StopLine
from src.rl.state_bridge import VisionStateBridge
from src.rl.types import OrientedBoundingBox, StateVector

logger = logging.getLogger(__name__)


class DecidesActions(Protocol):
    """Structural contract for whatever this adapter hands states to --
    deliberately a `Protocol` (not importing `MultiAgentPolicyEngine`
    directly as a type) so this module stays importable/testable without
    torch, matching this platform's existing `Trainable`/`Exportable`
    Protocol convention in `src.core.base_model`."""

    def decide(
        self,
        states: Sequence[StateVector],
        stop_lines: Sequence[StopLine] = (),
        corridor_heading_rad: Optional[float] = None,
    ) -> List[object]: ...


class RLStreamAdapter:
    """Per frame: `FrameResult` -> `StateVector`s -> (optional simulated
    LiDAR fusion, for diagnostics) -> policy decisions, dispatched via an
    `on_decisions` callback.

    Usage:
        bridge = VisionStateBridge(speed_estimator, fps=25.0, pixels_per_meter=10.0)
        engine = MultiAgentPolicyEngine(network)  # from src.rl.policy_engine
        adapter = RLStreamAdapter(bridge, engine)
        adapter.on_decisions = lambda frame, decisions: ...
        tracker = StreamTracker(..., on_result=adapter.on_frame_result)
    """

    def __init__(
        self,
        state_bridge: VisionStateBridge,
        policy_engine: DecidesActions,
        ego_position_m: Tuple[float, float] = (0.0, 0.0),
        ego_heading_rad: float = 0.0,
        lidar: Optional[LidarRaycaster] = None,
        fusion: Optional[VisionLidarFusion] = None,
        stop_lines: Sequence[StopLine] = (),
        corridor_heading_rad: Optional[float] = None,
        on_decisions: Optional[Callable[[FrameResult, List[object]], None]] = None,
    ) -> None:
        self.state_bridge = state_bridge
        self.policy_engine = policy_engine
        self.ego_position_m = ego_position_m
        self.ego_heading_rad = ego_heading_rad
        self.lidar = lidar
        self.fusion = fusion
        self.stop_lines = stop_lines
        self.corridor_heading_rad = corridor_heading_rad
        self.on_decisions = on_decisions
        self.last_decisions: List[object] = []
        self.last_fusion: Dict[int, FusedTrack] = {}

    def on_frame_result(self, frame: FrameResult) -> None:
        """Pass this bound method directly as `StreamTracker(...,
        on_result=adapter.on_frame_result)`. Exceptions are caught and
        logged rather than propagated, matching `StreamTracker`'s own
        inference-loop policy of never letting one bad frame kill a live
        pipeline (see `_inference_loop` in `stream_tracker.py`)."""
        try:
            decisions = self.process(frame)
        except Exception:  # noqa: BLE001 - keep the stream alive on a bad frame
            logger.exception("RL decision step failed on frame %d; skipping.", frame.frame_index)
            return
        self.last_decisions = decisions
        if self.on_decisions is not None:
            self.on_decisions(frame, decisions)

    def process(self, frame: FrameResult) -> List[object]:
        states = self.state_bridge.build_states(frame)

        if self.lidar is not None and self.fusion is not None and states:
            obstacles = [OrientedBoundingBox.from_state(s) for s in states]
            track_ids = [s.track_id for s in states]
            scan = self.lidar.cast(self.ego_position_m, self.ego_heading_rad, obstacles, track_ids)
            # Fusion result is diagnostic here (exposed via
            # `last_fusion` for logging/monitoring) rather than fed back
            # into `states`: the simulated LiDAR models the EGO
            # platform's own sensor, and this platform's vision states
            # already describe every other vehicle directly, so there's
            # no missing "actual sensor reading" for fusion to fill in
            # here the way there would be on a real AV stack.
            self.last_fusion = self.fusion.fuse(self.ego_position_m, states, scan)
        else:
            self.last_fusion = {}

        if not states:
            return []
        return self.policy_engine.decide(
            states, stop_lines=self.stop_lines, corridor_heading_rad=self.corridor_heading_rad
        )
