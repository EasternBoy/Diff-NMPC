from pathlib import Path
import json
import numpy as np


def load_log(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def main() -> None:
    log_path_1 = "scale0.25_TK30_log_Oschersleben_Vinit_6.0_c30.0_l3000.0_p100.0_friction1.2"
    log_path_2 = "scale0.25_TK30_log_Oschersleben_Vinit_6.0_c30.0_l3000.0_p100.0_friction1.1"
    log_path_3 = "scale0.25_TK30_log_Oschersleben_Vinit_6.0_c30.0_l3000.0_p100.0_friction1.0"

    log_1 = load_log(log_path_1)
    log_2 = load_log(log_path_2)
    log_3 = load_log(log_path_3)

    merged_log = {}

    for k in log_1:
        if k not in log_2 or k not in log_3:
            raise KeyError(f"Key '{k}' missing in one of the logs")

        a = np.asarray(log_1[k])
        b = np.asarray(log_2[k])
        c = np.asarray(log_3[k])

        if a.ndim == 0 or b.ndim == 0 or c.ndim == 0:
            raise ValueError(f"Key '{k}' contains scalar data and cannot be merged")

        if not (a.shape[1:] == b.shape[1:] == c.shape[1:]):
            raise ValueError(
                f"Shape mismatch for key '{k}': "
                f"{a.shape}, {b.shape}, {c.shape}"
            )

        merged_log[k] = np.concatenate([a, b, c], axis=0).tolist()

    output_path = Path(log_path_1).with_name(
        "scale0.25_TK30_log_Oschersleben_merged_friction.json"
    )

    with open(output_path, "w") as f:
        json.dump(merged_log, f, indent=2)

    print(f"done! merged log saved to {output_path}")


if __name__ == "__main__":
    main()