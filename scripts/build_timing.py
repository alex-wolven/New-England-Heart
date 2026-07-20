"""Per-patient event timing + follow-up, for a horizon-specific (e.g. 5-year) label.

For each cohort patient: first post-cutoff acute-CV event date (if any) and last-observed
encounter date (censoring time). Writes data/neh_timing.parquet with:
  event (0/1), event_date, last_seen, time_days, time_years, cutoff_date, patient
No birthdate needed (life-years target dropped).
"""
import sys
import duckdb
import pandas as pd
sys.stdout.reconfigure(encoding="utf-8")
DIR = "C:/Users/alexw/OneDrive/Documents/Projects/Silent_Risk_Final"
DATA = "C:/Users/alexw/OneDrive/Documents/Globus/datathon_dataset"
con = duckdb.connect()
con.execute("PRAGMA threads=6"); con.execute("PRAGMA memory_limit='9GB'")
con.execute("PRAGMA temp_directory='C:/Users/alexw/AppData/Local/Temp/claude/duck_timing'")
def csv(n): return f"read_csv('{DATA}/{n}.csv',header=true,all_varchar=true,sample_size=-1)"

coh = pd.read_parquet(f"{DIR}/data/neh_cohort_enriched.parquet")[
    ["patient", "cutoff_date", "cv_event"]].copy()
coh["cutoff_date"] = pd.to_datetime(coh["cutoff_date"])
con.register("cohort", coh)
label_pat = "myocardial infarction|cardiac arrest|cerebral infarction|stroke"

print("first post-cutoff acute-CV event date per patient ...", flush=True)
ev = con.execute(f"""
    SELECT co.patient, MIN(CAST(substr(c.START,1,10) AS DATE)) AS event_date
    FROM cohort co JOIN {csv('conditions')} c ON c.PATIENT = co.patient
    WHERE CAST(substr(c.START,1,10) AS DATE) > co.cutoff_date
      AND regexp_matches(lower(c.DESCRIPTION),'{label_pat}')
      AND NOT lower(c.DESCRIPTION) LIKE '%history%'
    GROUP BY co.patient""").df()
print(f"  event dates for {len(ev):,} patients", flush=True)

print("last-observed (max encounter date) per patient ...", flush=True)
last = con.execute(f"""
    SELECT co.patient, MAX(CAST(substr(e.START,1,10) AS DATE)) AS last_seen
    FROM cohort co JOIN {csv('encounters')} e ON e.PATIENT = co.patient
    GROUP BY co.patient""").df()
print(f"  last-seen for {len(last):,} patients", flush=True)

t = coh.merge(ev, on="patient", how="left").merge(last, on="patient", how="left")
for c in ["cutoff_date", "event_date", "last_seen"]:
    t[c] = pd.to_datetime(t[c])
t["event"] = t["event_date"].notna().astype(int)
end = t["event_date"].where(t["event"] == 1, t["last_seen"])
t["time_days"] = (end - t["cutoff_date"]).dt.days
t["time_years"] = t["time_days"] / 365.25
t = t[t["time_days"] > 0]
t.to_parquet(f"{DIR}/data/neh_timing.parquet", index=False)
print(f"\nwrote timing.parquet: {len(t):,} rows, {int(t.event.sum()):,} events", flush=True)
print("DONE", flush=True)
