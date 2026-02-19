import json

ec = 30.0
el = 3000.0
ep = 100.0

filename = f"data/scale0.25_log_Oschersleben_full_Vinit_6.0_c30.0_l3000.0_p100.0_friction1.2_weight1.0_slip_100_150_350_450_800_900"

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