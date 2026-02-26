import json

ec = 1.0
el = 1000.0
ep = 200.0

# filename = "scale0.25_TK20_log_Oschersleben_full_Vinit_8.0friction1.2"

# 1. Load existing dict
with open(filename, "r") as f:
    data = json.load(f)

length = len(data["time"])  # use an existing key as reference

# 2. Add new "columns"
data["q_contour"] = [ec] * length
data["q_lag"] = [el] * length
data["q_theta"] = [ep] * length

# 3. Write back
with open(filename, "w") as f:
    json.dump(data, f, indent=2)