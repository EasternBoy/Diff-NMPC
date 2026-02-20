from pathlib import Path
import json


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    output_path = base_dir / "merged_data_no_adapt.json"

    merged = {}
    merged["source_file"] = []

    # Keep only data files in this folder (skip scripts and previous merged output).
    data_files = sorted(
        f
        for f in base_dir.iterdir()
        if f.is_file()
        and f.suffix != ".py"
        and f.name != output_path.name
    )

    if not data_files:
        print("No data files found to merge.")
        return

    for data_file in data_files:
        with data_file.open("r", encoding="utf-8") as f:
            data = json.load(f)

        # Use the longest list length in this file to tag rows by source filename.
        row_count = max((len(v) for v in data.values() if isinstance(v, list)), default=0)
        merged["source_file"].extend([data_file.name] * row_count)

        for key, value in data.items():
            if not isinstance(value, list):
                # Keep scalar metadata once; ignore later conflicting values.
                merged.setdefault(key, value)
                continue

            merged.setdefault(key, [])
            merged[key].extend(value)

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    print(f"Merged {len(data_files)} files -> {output_path}")


if __name__ == "__main__":
    main()
