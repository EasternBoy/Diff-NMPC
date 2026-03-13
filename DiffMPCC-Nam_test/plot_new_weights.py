import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

LOG_NAME = "friction0.9_outer_steps40_pg_iters1_lr0.2_clipped"
SAVE_CLIPPED = True
CLIPPED_SUFFIX = "_clipped"

base_dir = Path(__file__).resolve().parent
log_path = base_dir / LOG_NAME

with log_path.open("r", encoding="utf-8") as f:
    log = json.load(f)

q_theta_cur = np.asarray(log["q_theta_cur"], dtype=float)
q_theta_next_raw = np.asarray(log["q_contour_next"], dtype=float)
if q_theta_next_raw.shape != q_theta_cur.shape:
    raise ValueError(
        f"Shape mismatch: q_theta_next={q_theta_next_raw.shape}, "
        f"q_theta_cur={q_theta_cur.shape}"
    )

# # Element-wise clipping: q_theta_next_raw[i] is clipped by bounds from q_theta_cur[i].
# q_theta_clipped = np.clip(q_theta_next_raw, q_theta_cur - 50.0, q_theta_cur + 200.0)
# q_theta_clipped = np.clip(q_theta_clipped, 0.0, 500.0)

print("keys:", log.keys())

plt.plot(q_theta_next_raw, label="q_theta_next_raw", alpha=0.6)
# plt.plot(q_theta_clipped, label="q_theta_next_clipped", linewidth=2)
plt.legend()
plt.tight_layout()
plt.show()

# if SAVE_CLIPPED:
#     out = dict(log)
#     out["q_theta_next"] = q_theta_clipped.tolist()
#     out_path = base_dir / f"{LOG_NAME}{CLIPPED_SUFFIX}"
#     with out_path.open("w", encoding="utf-8") as f:
#         json.dump(out, f, indent=2)
#     print(f"saved clipped log: {out_path}")
