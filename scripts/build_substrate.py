"""Build the longitudinal substrate for a SEEDED RANDOM subset of patients.

Reads the one-time full scans (cached in the scratchpad):
  labs_long_all.csv   patient,code,date,value
Writes (small, into artifacts/):
  substrate_labs.parquet   patient,date,metric,value
  substrate_wellness.parquet  patient,date

Random subset drawn from the FULL roster (not high-risk only) so the cohort includes
never-enroll and would-graduate patients. Re-sampling is instant; no rescans.
Usage: python scripts/build_substrate.py [N]
"""
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from neh import config

SCRATCH = Path(
    r"C:\Users\alexw\AppData\Local\Temp\claude"
    r"\c--Users-alexw-OneDrive-Documents-Projects-Silent-Risk-Final"
    r"\b8151d35-1ac5-482a-918b-20020155be01\scratchpad"
)
# re-extracted from observations.csv with 9 codes (the original had only 6); meds unchanged
LABS_RAW = Path(
    r"C:\Users\alexw\AppData\Local\Temp\claude"
    r"\c--Users-alexw-OneDrive-Documents-Projects-Silent-Risk-Final"
    r"\3b45d32d-d260-419b-be15-aadc2264dc37\scratchpad\labs_long_9.csv"
)
# wellness (checkup) encounter dates, extracted from encounters.csv (ENCOUNTERCLASS==wellness)
WELLNESS_RAW = Path(
    r"C:\Users\alexw\AppData\Local\Temp\claude"
    r"\c--Users-alexw-OneDrive-Documents-Projects-Silent-Risk-Final"
    r"\3b45d32d-d260-419b-be15-aadc2264dc37\scratchpad\wellness_all.csv"
)

CODE2METRIC = {"8480-6": "sbp", "8462-4": "dbp", "4548-4": "a1c",
               "18262-6": "ldl", "2093-3": "total_chol", "2085-9": "hdl",
               "39156-5": "bmi", "2571-8": "trig", "33914-3": "egfr"}

N = int(sys.argv[1]) if len(sys.argv) > 1 else 15000


def main():
    config.ARTIFACTS.mkdir(exist_ok=True)
    roster = pd.read_parquet(config.ENRICHED, columns=["patient"])["patient"].drop_duplicates()
    subset = set(roster.sample(n=min(N, roster.nunique()), random_state=config.SEED))
    print(f"[subset] {len(subset)} / {roster.nunique()} patients (seed={config.SEED})")

    # --- labs ---
    chunks = []
    for ch in pd.read_csv(LABS_RAW, chunksize=3_000_000,
                          dtype={"patient": str, "code": str, "date": str, "value": str}):
        ch = ch[ch["patient"].isin(subset)]
        if len(ch):
            chunks.append(ch)
    labs = pd.concat(chunks, ignore_index=True)
    labs["metric"] = labs["code"].map(CODE2METRIC)
    labs["value"] = pd.to_numeric(labs["value"], errors="coerce")
    # raw observation dates are ISO with a 'Z' (UTC) and a time-of-day; store tz-naive and
    # floored to the DAY so date-only lookups (as_of, claim payloads) match exactly
    labs["date"] = (pd.to_datetime(labs["date"], errors="coerce", utc=True)
                    .dt.tz_localize(None).dt.normalize())
    labs = (labs.dropna(subset=["value", "date", "metric"])
                .drop_duplicates(["patient", "date", "metric"])
                [["patient", "date", "metric", "value"]]
                .sort_values(["patient", "metric", "date"]))
    labs.to_parquet(config.SUBSTRATE_LABS, index=False)

    # --- wellness checkup dates (the real periodic visits, not every reading) ---
    well_chunks = []
    for ch in pd.read_csv(WELLNESS_RAW, chunksize=2_000_000, dtype={"patient": str, "date": str}):
        ch = ch[ch["patient"].isin(subset)]
        if len(ch):
            well_chunks.append(ch)
    well = pd.concat(well_chunks, ignore_index=True)
    well["date"] = (pd.to_datetime(well["date"], errors="coerce", utc=True)
                    .dt.tz_localize(None).dt.normalize())
    well = (well.dropna(subset=["date"]).drop_duplicates(["patient", "date"])
                [["patient", "date"]].sort_values(["patient", "date"]))
    well.to_parquet(config.SUBSTRATE_WELLNESS, index=False)

    # --- coverage report (random subset, not high-risk only) ---
    print(f"[labs] {len(labs):,} rows | "
          f"[wellness] {len(well):,} visit-dates, {well['patient'].nunique():,} patients")
    wc = well.groupby("patient")["date"].nunique()
    print(f"[wellness] median checkups/patient: {int(wc.median())}, "
          f">=3 checkups: {(wc >= 3).sum()}")
    print("[coverage] patients with >=3 dated readings (of the random subset):")
    for m in ["sbp", "dbp", "ldl", "a1c", "bmi", "trig", "egfr"]:
        d = labs[labs["metric"] == m].groupby("patient")["date"].nunique()
        print(f"   {m:11s} {(d >= 3).sum():6d}   (any: {len(d):6d})")
    print(f"wrote {config.SUBSTRATE_LABS.name} + {config.SUBSTRATE_WELLNESS.name}")


if __name__ == "__main__":
    main()
