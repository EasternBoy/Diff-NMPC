import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection


LOG_PATH = "friction0.6_n1122_outer_steps40_pg_iters1_lr0.1"
LF = 0.88392
LR = 1.50876

with open(LOG_PATH, "r") as f:
    log = json.load(f)

q_theta_next = np.asarray(log["q_theta_next"], dtype=float)
# print("min", min(q_theta_next))
q_theta_next = q_theta_next - 20
q_theta_next = np.clip(q_theta_next, 0.0, 200.0)

# Save clipped values back to the same log file.
log["q_theta_next"] = q_theta_next.tolist()
with open(LOG_PATH, "w") as f:
    json.dump(log, f, indent=2)

print("keys:", log.keys())

q_contour = np.asarray(log["q_contour_next"], dtype=float)
q_lag = np.asarray(log["q_lag_next"], dtype=float)
q_theta = q_theta_next
theta = np.asarray(log["theta"], dtype=float)

plt.plot(q_theta, label="q_theta")
plt.legend()
plt.show()
