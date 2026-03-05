import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection

LOG_PATH = "friction0.5_outer_steps40_pg_iters5_lr0.2"
LF = 0.88392
LR = 1.50876

with open(LOG_PATH, "r") as f:
    log = json.load(f)
# q_theta_next = np.asarray(log["q_theta_next"], dtype=float)

# # q_theta_next = np.clip(q_theta_next, 0.0, 700.0)
# q_contour = np.asarray(log["q_contour_next"], dtype=float)
# q_lag = np.asarray(log["q_lag_next"], dtype=float)

# # Save clipped values back to the same log file.
# log["q_theta_next"] = q_theta_next.tolist()
# log["q_contour_next"] = q_contour.tolist()
# log["q_lag_next"] = q_lag.tolist()
# with open(LOG_PATH, "w") as f:
#     json.dump(log, f, indent=2)

# print("keys:", log.keys())

q_contour = np.asarray(log["q_contour_next"], dtype=float)
q_lag = np.asarray(log["q_lag_next"], dtype=float)
q_theta =np.asarray(log["q_theta_next"], dtype=float)
theta = np.asarray(log["theta"], dtype=float)

plt.plot(q_theta, label="q_theta")
plt.legend()
plt.show()
