import csv
from pathlib import Path


BASE_DIR = Path(".")

TARGETS = {
    "Wednesday-14-02-2018_TrafficForML_CICFlowMeter.csv": {
        410956,
        410957,
        410958,
        410959,
        412184,
    },
    "Friday-16-02-2018_TrafficForML_CICFlowMeter.csv": {
        999999,
    },
}


for filename, target_indices in TARGETS.items():

    matches = list(BASE_DIR.rglob(filename))

    if not matches:
        print(f"\nFILE NOT FOUND: {filename}")
        continue

    path = matches[0]

    print("\n" + "=" * 100)
    print(f"FILE: {filename}")
    print("=" * 100)

    with open(path, "r", encoding="utf-8", errors="replace", newline="") as f:

        reader = csv.reader(f)

        header = next(reader)

        rows_to_show = set()

        for idx in target_indices:
            rows_to_show.add(idx - 1)
            rows_to_show.add(idx)
            rows_to_show.add(idx + 1)

        for data_index, row in enumerate(reader):

            if data_index not in rows_to_show:
                continue

            print(f"\nDATA ROW INDEX: {data_index}")
            print(f"Timestamp:       {row[2] if len(row) > 2 else '<missing>'}")
            print(f"Dst Port:        {row[0] if len(row) > 0 else '<missing>'}")
            print(f"Protocol:        {row[1] if len(row) > 1 else '<missing>'}")
            print(f"Flow Duration:   {row[3] if len(row) > 3 else '<missing>'}")
            print(f"Tot Fwd Pkts:    {row[4] if len(row) > 4 else '<missing>'}")
            print(f"Tot Bwd Pkts:    {row[5] if len(row) > 5 else '<missing>'}")
            print(f"Label:           {row[79] if len(row) > 79 else '<missing>'}")
            print(f"Column count:    {len(row)}")

            if data_index in target_indices:
                print("*** TARGET ANOMALOUS ROW ***")