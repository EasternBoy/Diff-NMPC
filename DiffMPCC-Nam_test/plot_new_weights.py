import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


LOG_PATH = "merged_data_no_adapt_n8788_outer_steps40_pg_iters1_lr0.1"
LF = 0.88392
LR = 1.50876

with open(LOG_PATH, "r") as f:
    log = json.load(f)



# vy = np.asarray(log["vy"], dtype=float)
# keep = vy < 0.2
# for k, v in log.items():
#     arr = np.asarray(v)
#     if arr.ndim > 0 and arr.shape[0] == vy.shape[0]:
#         log[k] = arr[keep].tolist()

# LOG_PATH_merge = "Vinit_6.0_c30.0_l3000.0_p100.0_friction1.2_weight1.0_non_n1041_outer_steps30_pg_iters1_lr5"
# with open(LOG_PATH_merge, "r") as f:
#     log_merge = json.load(f)

# # Merge log_merge into log by concatenating values for each key.
# for k in log:
#     a = np.asarray(log[k])
#     b = np.asarray(log_merge[k])
#     if a.ndim > 0 and b.ndim > 0 and a.shape[1:] == b.shape[1:]:
#         log[k] = np.concatenate([a, b], axis=0).tolist()
#     else:
#         raise ValueError(f"Cannot merge key '{k}' with shapes {a.shape} and {b.shape}")
    
print("keys:", log.keys())


q_contour = np.asarray(log["q_contour_next"], dtype=float)
q_lag = np.asarray(log["q_lag_next"], dtype=float)
q_theta = np.asarray(log["q_theta_next"], dtype=float)
theta = np.asarray(log["theta"], dtype=float)
plt.plot(q_theta[:2000], label="q_theta")
plt.legend()
plt.show()
