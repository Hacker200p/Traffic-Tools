import argparse
import numpy as np
import pandas as pd


def pick_density_col(df: pd.DataFrame) -> str:
    candidates = [
        "density_veh_per_km_lane_smoothed",  # single-lane script
        "total_density_smoothed",            # multi-lane script
        "density_veh_per_km",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"No density column found. Columns: {list(df.columns)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv1", required=True, help="Video1 density CSV")
    ap.add_argument("--csv2", required=True, help="Video2 density CSV")
    ap.add_argument("--out", required=True, help="Output comparison CSV")
    ap.add_argument("--name1", default="video1")
    ap.add_argument("--name2", default="video2")
    ap.add_argument("--tolerance", type=float, default=0.10, help="Time match tolerance in seconds")
    ap.add_argument("--offset2", type=float, default=0.0, help="Shift video2 time_sec by this many seconds")
    args = ap.parse_args()

    df1 = pd.read_csv(args.csv1)
    df2 = pd.read_csv(args.csv2)

    c1 = pick_density_col(df1)
    c2 = pick_density_col(df2)

    if "time_sec" not in df1.columns or "time_sec" not in df2.columns:
        raise ValueError("Both CSVs must contain 'time_sec' column.")

    a = df1[["time_sec", c1]].copy().rename(columns={c1: "density_1"})
    b = df2[["time_sec", c2]].copy().rename(columns={c2: "density_2"})

    a["time_sec"] = pd.to_numeric(a["time_sec"], errors="coerce")
    b["time_sec"] = pd.to_numeric(b["time_sec"], errors="coerce") + args.offset2
    a["density_1"] = pd.to_numeric(a["density_1"], errors="coerce")
    b["density_2"] = pd.to_numeric(b["density_2"], errors="coerce")

    a = a.dropna().sort_values("time_sec")
    b = b.dropna().sort_values("time_sec")

    merged = pd.merge_asof(
        a, b,
        on="time_sec",
        direction="nearest",
        tolerance=args.tolerance
    ).dropna(subset=["density_2"])

    d1 = merged["density_1"].to_numpy()
    d2 = merged["density_2"].to_numpy()

    merged[f"{args.name1}_signal"] = np.where(d1 > d2, "RED", np.where(d1 < d2, "GREEN", "GREEN"))
    merged[f"{args.name2}_signal"] = np.where(d2 > d1, "RED", np.where(d2 < d1, "GREEN", "GREEN"))
    merged["winner_higher_density"] = np.where(d1 > d2, args.name1, np.where(d2 > d1, args.name2, "TIE"))

    merged.to_csv(args.out, index=False)

    red1 = (merged[f"{args.name1}_signal"] == "RED").sum()
    red2 = (merged[f"{args.name2}_signal"] == "RED").sum()

    print(f"Saved: {args.out}")
    print(f"{args.name1} RED frames: {red1}")
    print(f"{args.name2} RED frames: {red2}")


if __name__ == "__main__":
    main()