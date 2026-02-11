import json

with open('data/data_for_diff/log_ca_c100.0_l400.0_p10.0', 'r') as f:
    log = json.load(f)
print(log.keys())
# dict_keys(['time', 'x', 'y', 'lap_n', 'vx', 'v_ref', 'tracking_error'])

print(len(log['time']))
print(len(log['x']))
print(len(log['tracking_error']))


import matplotlib.pyplot as plt

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# 1) Trajectory (x-y)
axs[0, 0].plot(log['x'], log['y'])
axs[0, 0].set_aspect('equal', adjustable='box')
axs[0, 0].set_xlabel('x [m]')
axs[0, 0].set_ylabel('y [m]')
axs[0, 0].set_title('Vehicle trajectory')
axs[0, 0].grid(True)

# 2) Tracking error vs time
axs[0, 1].plot(log['time'], log['BR'])
axs[0, 1].plot(log['time'], log['BF'])
axs[0, 1].set_xlabel('Time [s]')
# axs[0, 1].set_ylabel('Tracking error [m]')
axs[0, 1].set_title('Parameters')
axs[0, 1].grid(True)

# 3) Speed tracking
axs[1, 0].plot(log['time'], log['vx'], label='vx')
# axs[1, 0].plot(log['time'], log['v_ref'], '--', label='v_ref')
axs[1, 0].set_xlabel('Time [s]')
axs[1, 0].set_ylabel('Speed [m/s]')
axs[1, 0].set_title('Speed tracking')
axs[1, 0].legend()
axs[1, 0].grid(True)

# 4) Lap progression
axs[1, 1].plot(log['time'], log['lap_n'])
axs[1, 1].set_xlabel('Time [s]')
axs[1, 1].set_ylabel('Lap count')
axs[1, 1].set_title('Lap progression')
axs[1, 1].grid(True)

plt.tight_layout()
plt.show()