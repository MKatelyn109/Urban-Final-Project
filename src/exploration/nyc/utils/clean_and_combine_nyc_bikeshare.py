import os
import re
import pandas as pd

# Directory containing the CSV files
DATA_DIR = "local/Data/nyc_bikeshare"
OUT_DIR = "local/Data/PROCESSED/nyc/ped/"

# Target year-months (YYYYMM)
TARGET_PERIODS = {
    "202205",  # May 2022
    "202210",  # Oct 2022
    "202305",  # May 2023
    "202310",  # Oct 2023
    "202406",  # Jun 2024
    "202410",  # Oct 2024
}

# Regex to match file pattern: YYYYMM-citibike-tripdata_N.csv
pattern = re.compile(r"^(20\d{4})-citibike-tripdata_(\d)\.csv$")

def collect_files(directory):
    """
    Returns a dict mapping YYYYMM -> {part: full_path}
    """
    collected = {}

    for filename in os.listdir(directory):
        match = pattern.match(filename)
        if not match:
            continue

        yyyymm, part = match.groups()

        # Only keep needed months
        if yyyymm not in TARGET_PERIODS:
            continue

        part = int(part)
        collected.setdefault(yyyymm, {})[part] = os.path.join(directory, filename)

    return collected


def combine_all(collected):
    """
    Reads all available parts for each selected YYYYMM
    and concatenates everything into ONE combined CSV.
    """
    all_dfs = []

    for yyyymm, parts in sorted(collected.items()):
        print(f"\nProcessing {yyyymm}...")

        for part in sorted(parts):
            filepath = parts[part]
            print(f"  Reading part {part}: {filepath}")
            df = pd.read_csv(filepath)
            df["source_month"] = yyyymm  # Optional: tag source month
            all_dfs.append(df)

    if all_dfs:
        final_df = pd.concat(all_dfs, ignore_index=True)
        out_to = OUT_DIR + "citibike_selected_months_combined.csv"
        final_df.to_csv(out_to, index=False)
        print("\nWritten final file:", out_to)
    else:
        print("No matching files found.")


if __name__ == "__main__":
    files = collect_files(DATA_DIR)
    combine_all(files)
