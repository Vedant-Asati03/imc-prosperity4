import json
import argparse
import sys
from pathlib import Path


def extract_csv(json_log_file: Path) -> None:
    if not json_log_file.exists():
        print(f"Error: JSON file '{json_log_file}' not found.", file=sys.stderr)
        sys.exit(1)

    try:
        with open(json_log_file, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from '{json_log_file}': {e}", file=sys.stderr)
        sys.exit(1)

    if "activitiesLog" not in data:
        print(
            f"Error: 'activitiesLog' key not found in JSON '{json_log_file}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    activities_log: str = data["activitiesLog"]

    round = data.get("round", "unknown")
    day = activities_log.splitlines()[1].split(";")[0] if activities_log else "unknown"

    identifier: str = json_log_file.name.split(".")[0]

    dump_path = json_log_file.parent.parent.absolute() / "dump" / f"round{round}"
    dump_path.mkdir(parents=True, exist_ok=True)

    csv_path = dump_path / f"{identifier}_prices_round_{round}_day_{day}.csv"

    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        f.write(activities_log)
        if not activities_log.endswith("\n"):
            f.write("\n")

    print(f"Successfully extracted activitiesLog to '{csv_path}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract a CSV file from IMC Prosperity JSON logs."
    )
    parser.add_argument(
        "json_log_file", help="Path to the input JSON file (e.g., data/289560.json)"
    )
    args = parser.parse_args()

    extract_csv(Path(args.json_log_file))
