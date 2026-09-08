"""RL 阶段 1 · T1 包：冻结跟踪策略 ONNX->flax 精确移植 + 观测/动作管线 jax 化。

- FrozenPolicy / load_policy / load_metadata / load_extras：
  从 scripts/export_onnx_weights.py 导出的 npz 重建冻结 MLP（推理用）。
- get_nn_state / update_coord_cmd / nn2motor_action：
  tracking/infer_utils.py 对应函数的 jax 纯函数版（支持批量）。
- sample_realistic_obs：真实量级随机观测合成（对齐验证用）。
"""

from rl.frozen_policy import (
    DEFAULT_NPZ,
    DEFAULT_ONNX,
    FORWARD_CHUNK_ROWS,
    FrozenPolicy,
    batch_forward,
    get_nn_state,
    load_extras,
    load_metadata,
    load_policy,
    nn2motor_action,
    quat2yaw,
    sample_obs_near_frames,
    sample_realistic_obs,
    update_coord_cmd,
    warp_pi,
)

__all__ = [
    "DEFAULT_NPZ",
    "DEFAULT_ONNX",
    "FORWARD_CHUNK_ROWS",
    "FrozenPolicy",
    "batch_forward",
    "get_nn_state",
    "load_extras",
    "load_metadata",
    "load_policy",
    "nn2motor_action",
    "quat2yaw",
    "sample_obs_near_frames",
    "sample_realistic_obs",
    "update_coord_cmd",
    "warp_pi",
]
