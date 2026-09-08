"""录制冻结策略真实轨迹回放帧（tests/test_rl_policy.py 对齐验证用）。

用法: uv run python scripts/record_policy_traj.py [--force]
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from tests.test_rl_policy import REC_PATH, record_frames  # noqa: E402

if __name__ == "__main__":
    if "--force" in sys.argv and REC_PATH.exists():
        REC_PATH.unlink()
    frames = record_frames()
    print(f"recorded {len(frames['obs'])} frames -> {REC_PATH}")
