# New England Heart

A grounding-gated, trajectory-grounded, bilingual (EN/ES) cardiovascular-risk outreach
pipeline with a clinician review layer. Built for the RI-AI4H Datathon (Brown, July 2026) on
SyntheticRI (Synthea) data.

> Every message is reviewed and approved by a clinician before a (simulated) send. Nothing
> auto-sends. Grounding pre-blocks ungrounded drafts (every number is re-derived from the record);
> it never grants the system autonomy.

## The idea

A CV-risk model finds high-risk patients. For each, we generate a warm, plain-language message
that reports the patient's own measurements that are outside the clinical healthy range (top 3 by
model contribution), and encourages them, over time, as their numbers move. The core property is
trajectory grounding: every clinical fact in a message must be re-derivable from the patient's own
record, or it is blocked and regenerated. Recommendations are grounded by RAG retrieval over a
single clinical guideline (Chroma), guarded by an allow-set so only clinically valid passages
attach. Messages are written by Claude (no offline template mode) from those verified facts.

## Model

Race-blind, registry-first LightGBM predicting a 5-year CV event. Five years is the longest
horizon this data can validate: about 65 percent of patients have at least 5 years of observed
follow-up and none have 10, so only patients with a 5-year event OR at least 5 event-free years
are used (the observability filter). Probabilities are sigmoid-calibrated. Age is a driver at
enrollment (it is part of who we reach out to) but is HELD FIXED at the enrollment date once a
patient is enrolled, so progress, setback, and graduation move only with the modifiable measures,
never merely because the patient got older. Outreach threshold = 80th percentile of calibrated risk
(about 3.9 percent 5-year risk); at that cut the messaged ~20 percent of patients contain about half
(52 percent) of all 5-year events, and the messaged group's event rate is about 2.6 times the
2.7 percent population rate. (We use the 80th, not the 90th, because the intervention is low-cost,
low-harm lifestyle outreach, so a wider net captures far more events for a costless message; the
90th would reach 10 percent and capture 35 percent.) Race-blind, primary-prevention (untreated, no
prior CVD) LightGBM, AUC about 0.73 held-out. Reproduce with `python scripts/threshold_analysis.py`.

## Claim types (trajectory grounding)

- `snapshot`: latest value ("your most recent blood pressure was 132/84"), verified vs the record
- `change`: a measure's value at a prior checkup (dated) vs now, reported only when it (a) exceeds
  the metric's change-beyond-variation threshold (Reference Change Value for labs, ADA 0.5 pts for
  blood sugar, minimal detectable change for BP; see DEFENSE.md), (b) is confirmed over two
  consecutive checkups, and (c) is out of range at one end (a healthy-range wiggle is never
  messaged); progress, setback, and graduation all report this movement
- `actionable`: a modifiable driver plus a guideline recommendation (never age or sex)
- `maintenance`: a general keep-up-your-habits suggestion with a guideline citation (graduation)

Messaged measures: LDL, total cholesterol, HDL, blood sugar (A1c), blood pressure (systolic +
diastolic), and BMI (coached only at the clinical obesity line, >=30, and framed around behaviors
like healthy eating and activity, never the number). Triglycerides and eGFR are used by the risk
model and shown to the clinician but not coached to the patient (trig is too labile to read
visit-to-visit; eGFR is a kidney marker, not a lifestyle target).

Message lifecycle, walked over each patient's actual wellness checkups (the real periodic visits,
roughly annual, not every reading, so we never spam): enrollment (all out-of-range measures,
ranked by model contribution; confirmed over two consecutive above-threshold checkups so the full
panel is captured) then any number of check-ins: a newly out-of-range measure is flagged
immediately as a state fact (this closes the gap where a measure that is out of range the first
time it is drawn would otherwise wait for a second reading), and progress / setback are called on
the modifiable trajectory (a change is progress if a flagged measure improved, none worsened, and
modifiable risk did not rise; setback if a flagged measure worsened, none improved, and modifiable
risk did not fall; because age is held at enrollment, aging alone never triggers a setback and
never denies progress). Then graduation (modifiable risk fell below the outreach threshold; stated
as leaving the elevated range, never as zero risk), and a re-enrollment if risk later climbs back.
Every message ends with the same reply-to-a-health-professional line and a STOP opt-out. A stall becomes a
clinician flag, never a nag.

Triage: `red` (grounding failed, routed to a clinician with the failing claim flagged) or `normal`
(grounded, ready for review). Nothing auto-sends regardless.

## Run it

```bash
pip install -r requirements.txt

# one-time data prep + model (writes artifacts/)
python scripts/build_timing.py              # per-patient event/censoring dates (5-year label)
python scripts/build_substrate.py 15000     # seeded random subset: longitudinal labs + med starts
python scripts/train_model.py               # registry-first, 5-year LightGBM + calibration
python scripts/precompute_queue.py          # review queue + the scripted demo patient

# launch the clinician review app
python -m streamlit run app/streamlit_app.py
```

`ANTHROPIC_API_KEY` is required (offline template mode was removed): Claude Haiku 4.5 writes the
messages from pre-verified facts, and every draft is re-checked against the record and blocked or
regenerated if a number drifts. The precomputed queue stores the results, so viewing the app needs
no key.

## Review queue

The app is a single queue: pick a patient, then a stage (Enrollment / Progress / Setback /
Graduation) to see the bilingual draft, the diagnostic panel, guideline citations, and
Approve / Reject. Each patient is walked across their own timeline, so one patient can carry
several stages. (A grounding-catch row, where an injected false number is blocked and
regenerated, is still precomputed in the queue parquet but never shown; the filter in
app/streamlit_app.py restores it if wanted.)

## Layout

```
neh/                 config, substrate, risk, guidelines, llm, claims, draft, grounding, lifecycle, pipeline
scripts/             build_timing.py, build_substrate.py, train_model.py, threshold_analysis.py, precompute_queue.py
app/                 streamlit_app.py   (clinician review queue)
data/                enriched cohort + timing (5-year label inputs)
docs/                Arnett 2019 primary-prevention guideline (the single guideline source)
artifacts/           model, substrate, RAG index, precomputed queue   (git-ignored; regenerate)
```

Guideline source (single doc): 2019 ACC/AHA Guideline on the Primary Prevention of Cardiovascular
Disease (Arnett DK et al., Circulation 2019;140:e596-e646), PDF in docs/.

See [DEFENSE.md](DEFENSE.md) for the full story: positioning and novelty (where this fits on top of
Epic/Emmie and Rhode Island's programs), the justification behind every non-obvious design choice
(elevated-risk threshold, 5-year horizon, calibration, age-held trajectory, RAG scope, trend
thresholds), and the honest limitations. The honesty is part of the pitch.
