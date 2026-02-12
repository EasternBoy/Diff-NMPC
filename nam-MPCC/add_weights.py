import json

ec = 20.0
el = 3000.0
ep = 100.0

filename = f"data/log_full_Vinit_8.0_c{ec}_l{el}_p{ep}_weightslip0.5_thetaslip_100_150_290_310_non"

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