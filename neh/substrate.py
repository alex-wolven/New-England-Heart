"""Longitudinal substrate access: per-patient dated lab series + CV-med start dates,
plus as-of lookups used for trend claims and score-at-date risk deltas."""
from __future__ import annotations
from functools import lru_cache
from typing import Optional
import pandas as pd
from . import config


@lru_cache(maxsize=1)
def _labs() -> pd.DataFrame:
    df = pd.read_parquet(config.SUBSTRATE_LABS)
    # floor to the day: raw readings carry a time-of-day, but we match on date only, so a reading
    # and a same-day checkup line up exactly
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    return df


@lru_cache(maxsize=1)
def _labs_by_patient() -> dict:
    """patient -> {metric -> (dates ndarray, values ndarray)} sorted by date.
    Built in one vectorized pass over sorted arrays (fast: slices views, no per-group objects)."""
    import numpy as np
    df = _labs().sort_values(["patient", "metric", "date"])
    pat = df["patient"].to_numpy()
    met = df["metric"].to_numpy()
    dts = df["date"].to_numpy()
    vls = df["value"].to_numpy(dtype=float)
    if len(df) == 0:
        return {}
    keys = np.char.add(np.char.add(pat.astype(str), "|"), met.astype(str))
    change = np.empty(len(keys), bool)
    change[0] = True
    change[1:] = keys[1:] != keys[:-1]
    starts = np.flatnonzero(change)
    ends = np.append(starts[1:], len(keys))
    out: dict = {}
    for s, e in zip(starts, ends):
        out.setdefault(pat[s], {})[met[s]] = (dts[s:e], vls[s:e])
    return out


@lru_cache(maxsize=1)
def cohort_patients() -> tuple:
    """Patients present in the substrate (the random subset)."""
    return tuple(sorted(_labs()["patient"].unique()))


_EMPTY = pd.DataFrame({"date": pd.Series([], dtype="datetime64[ns]"), "value": pd.Series([], dtype=float)})


def series(patient: str, metric: str) -> pd.DataFrame:
    """Dated readings for one patient/metric, sorted by date (columns: date, value)."""
    arrs = _labs_by_patient().get(patient, {}).get(metric)
    if arrs is None:
        return _EMPTY
    return pd.DataFrame({"date": arrs[0], "value": arrs[1]})


def as_of(patient: str, metric: str, when: pd.Timestamp) -> Optional[dict]:
    """Most recent reading at or before `when`. Returns {'date','value'} or None."""
    s = series(patient, metric)
    s = s[s["date"] <= pd.Timestamp(when)]
    if len(s) == 0:
        return None
    row = s.iloc[-1]
    return {"date": row["date"], "value": float(row["value"])}


@lru_cache(maxsize=1)
def _wellness_by_patient() -> dict:
    """patient -> sorted DatetimeIndex of wellness (checkup) visit dates."""
    df = pd.read_parquet(config.SUBSTRATE_WELLNESS)
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()  # match on date only
    return {p: pd.DatetimeIndex(sorted(g["date"].unique()))
            for p, g in df.groupby("patient")}


def wellness_dates(patient: str) -> pd.DatetimeIndex:
    """The patient's actual wellness-checkup dates (the periodic visits we anchor outreach to,
    NOT every reading). Empty index if the patient has no recorded wellness visits."""
    return _wellness_by_patient().get(patient, pd.DatetimeIndex([]))
