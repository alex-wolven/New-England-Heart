# New England Heart — positioning, design rationale, and honest limitations

One reference doc: what the project is and where it fits (novelty), why each non-obvious design
choice is defensible (anticipated judge questions), and the limitations we volunteer. Theme
throughout: we chose the honest option and can say exactly why.

---

## Part 1 — What it is and where it fits

**What it is.** A cardiovascular primary-prevention outreach pipeline: a risk model selects an
elevated-risk, untreated cohort; we walk each patient's own longitudinal record across their
wellness checkups; when a measure moves beyond normal variation (or is newly out of range) we draft
a warm, plain-language, bilingual (EN/ES) message that reports the patient's *own* numbers and
encourages them over a staged lifecycle (enrollment → progress → setback → graduation →
re-enrollment); every clinical fact is re-verified against the record and every message is approved
by a clinician before a simulated send.

**The AI is deliberately minimal — that is the safety design, not a weakness.** The AI never
generates facts. Content is computed deterministically from the record into a fixed message
skeleton; the LLM only rephrases it into warm, natural, 6th-grade-level EN/ES; then a verification
gate re-derives every number from the record and, if the LLM altered a single value, blocks it and
falls back to the exact skeleton. So the LLM is bounded to the one thing it is safe at — tone — and
the facts are guaranteed. Pure templates would lose the natural, warm, bilingual phrasing that drives
engagement in underserved communities (the whole point); an unconstrained chatbot would lose the
safety. This is the middle, on purpose.

**The novelty is the combination, not any single part.** Every ingredient exists somewhere:
EHR-driven CVD risk identification with individualized outreach (2012 RCT), CVD-prevention texting
(many RCTs), LLM-drafted patient messaging grounded/fact-checked against the EHR with clinician
review (an active 2025–26 research direction), and staged engagement frameworks. What no deployed
product or published system assembles is all of: (1) model-selected primary-prevention cohort,
(2) tracking each patient's own clinical-measure *trajectory* across visits, (3) auto-drafted
messages reporting the patient's own numbers with record-grounded verification, (4) a staged
clinical lifecycle judged on *modifiable* risk, (5) a clinician gate on every message, bilingually.
Novelty of integration is real novelty, and it scores on innovation, feasibility, and impact at once.

**Where it fits: a safe layer on top of Epic.** Epic already finds high-risk patients (Healthy
Planet registries, care-gap outreach), runs campaign messaging (Cheers), drafts inbound-message
replies (In-Basket AI, clinician-reviewed but *not* record-grounded), and now explains a single
result in plain language on request (Emmie). None of that is proactive, trajectory-staged
encouragement off the patient's own measures, with fact-level verification. We are that layer, and
Brown University Health already runs Epic ("LifeChart" + MyChart), so the substrate we'd plug into
is deployed statewide.

**Rhode Island validates the need and supplies the channel, not the mechanism.** RIDOH funds
community-driven hypertension/diabetes outreach (FY2026), trains Certified Community Health Workers
for CVD/diabetes, and runs the Healthy Heart Ambassador home-BP coaching program; CTC-RI coordinates
registry-based primary care for ~900k Rhode Islanders. This outreach is human-labor-driven and
bottlenecked on hours; the health-system automation is the generic Epic patterns above. We are the
engine that lets the trusted people RI already funds reach far more patients, safely.

**What NOT to claim** (so a judge cannot puncture it): not "first to text CV patients," not "first
LLM-with-a-clinician-gate" (Epic In-Basket, 1M+ drafts/month), not "we invented EHR grounding of LLM
output" (actively published), not "RI has nothing automated" (Brown Health runs Epic; Emmie exists).
We also cannot see inside Brown Health's private Epic config, so every landscape claim is scoped to
public documentation.

---

## Part 2 — Design rationale (anticipated judge questions)

**Elevated risk on an 80th-percentile cut, and why 80th not 90th.** *Empirical (primary
justification):* the messaged group (top 20% of calibrated 5-year risk, threshold ~3.9%) has ~2.6×
the population 5-year event rate and contains ~52% of all 5-year events, so "elevated relative to
population" is true by construction (`python scripts/threshold_analysis.py`). *Why 80th:* the
intervention is low-cost, low-harm lifestyle outreach, and lifestyle counseling is universally
guideline-recommended, so the acting threshold should scale with how cheap/safe the action is — a
wider net is defensible here in a way a statin recommendation would not be. Concretely, moving from
the 90th to the 80th roughly doubles both reach (10%→20%) and event-capture (35%→52%) for a message
that costs nothing; we stop at 80th because below ~75th the group's risk multiple falls toward the
population average and "elevated" weakens. *Clinical anchor (secondary, approximate):* ~3.9% at 5
years maps under the constant-hazard doubling approximation to ~7.8% at 10 years, right at the entry
to the ACC/AHA **intermediate** band. Honest caveat: our synthetic acute-CV label ≠ the Pooled
Cohort ASCVD endpoint and the doubling is an approximation — and unlike the 90th cut, the 80th does
**not** clear the 5% borderline line without the doubling assumption — so we lead with the empirical
justification and treat the clinical anchor as *directionally consistent*, not equal. Patient
messages never assert a clinical band, only risk relative to our own threshold.

**5-year horizon.** The data can't validate 10 years (~65% of patients have ≥5y follow-up, ~none
have 10). An observability filter keeps only patients with a 5-year event OR ≥5 event-free years
(~54,000 censored-before-5y excluded), so we never train or score on an unknowable outcome.

**AUC ~0.73.** Expected, not a bug — capped by the synthetic signal ceiling, and the model is *not*
the contribution. It's a race-blind, registry-first selector; any comparable CV model drops in.

**Race-blind, registry-first.** Race is excluded so the model can't encode group disparities; risk
is driven by modifiable clinical measures. We train only on untreated, no-prior-CVD patients (drop
`on_statin`, `prior_ascvd`), which avoids the treatment-paradox confound and matches the deployment
population exactly. The model estimates untreated baseline risk — what an outreach tool should act on.

**Sigmoid (Platt) calibration.** Isotonic collapsed to ~3 probability steps on this rare-event data,
zeroing the risk deltas the progress/graduation stages need. Sigmoid gives smooth, monotone
probabilities.

**Trend thresholds (a change must be a real shift, not noise).** LDL 20% / total chol 17% / HDL 18%
(Reference Change Value from lipid biological variation); A1c ≥0.5 pts (ADA/NICE); BP ≥8 mmHg
systolic or ≥12 diastolic (Minimal Detectable Change), evaluated and reported as the systolic/
diastolic pair; BMI ≥5% (ACC/AHA clinically-meaningful weight change), flagged only at the clinical
obesity line (≥30) and framed around behaviors (healthy eating, activity), never the number, to keep
it supportive and avoid messaging the merely-overweight. Triglycerides and eGFR stay in the model
but are never messaged (trig too labile at RCV ~60%; eGFR is a kidney marker, not a lifestyle
target). A change is messaged only when it's **out of range at one end**, and a value **equal** to a
threshold counts as healthy.

**Age at enrollment, held for the journey (the confound we took most seriously).** A single
age-inclusive score was selecting patients (right) *and* narrating their journey (wrong): since age
only rises, a patient could fix every modifiable measure and still see their score drift up,
suppressing progress and blocking graduation. So enrollment/selection uses the full score (age
belongs in *who* we reach), but progress, setback, and graduation use the score with **age held
fixed at the enrollment date**, so it moves only with the modifiable measures. Consequently we claim
a patient's *modifiable* risk moved, never that total risk did; the clinician panel still shows true
current risk and real age.

- **Progress:** a flagged measure improved, none worsened, modifiable risk didn't rise.
- **Setback:** a flagged measure worsened, none improved, modifiable risk didn't fall — so aging
  alone never triggers one.
- **Newly out-of-range measure:** a measure outside the healthy range we've never flagged is reported
  immediately as a state fact (single reading, like enrollment) — this closes the gap where a measure
  out of range the first time it's drawn would otherwise wait for a second reading to form a change.
- **Confirmation:** change-driven progress/setback requires the risk direction confirmed over two
  consecutive checkups; enrollment and graduation likewise require two consecutive above/below-
  threshold checkups. The newly-out-of-range flag is the one exception (an out-of-range state isn't
  noise). The "none moved the other way" veto is scoped to flagged measures, so an unflagged wiggle
  never silences real news.

**What grounding verifies (and doesn't).** Every claim is independently re-derived from the record,
bounded by its own as-of date (snapshots re-read, change claims recompute both endpoints, in-range/
actionable must map to an allowed guideline chunk); the realized text must contain the exact numbers
and not invert direction. Failures are blocked, regenerated with the reason fed back (≤2×), then
routed to a clinician flagged red. Grounding checks *computability from the record*, not clinical
appropriateness — which is exactly why a clinician reviews every message.

**RAG over one guideline.** Retrieval touches only the suggestion line and the citation on
actionable/in-range claims; it never generates or selects a number. An allow-set keyed by
(claim kind, metric) vetoes clinically wrong passages and falls back to a deterministic pick. Honest:
with one guideline and eight passages this is closer to a governed lookup than heavy retrieval — but
the architecture cleanly separates record-grounded facts from guideline-grounded advice, so adding
conditions means adding guideline docs, not rewriting logic. The passages are faithful summaries of
the 2019 ACC/AHA Primary Prevention guideline (Arnett et al.; PDF in docs/), cited as the source,
not quoted verbatim.

**Message cadence.** Outreach anchors to actual **wellness checkups** (roughly annual), not every
lab reading — the synthetic data stamps repeated values at many non-checkup encounters, so walking
every reading would spam. The date of the patient's last contact is stated once in the message
intro (e.g. "since 2021-07-31"), not repeated in every bullet.

**Demo queue selection.** The review queue is a random sample of eligible patients (seeded, so it's
reproducible), which spreads across the population instead of skewing toward any one risk band. The
live system would triage by risk; the demo just needs varied lifecycles to review.

**Nothing auto-sends.** Central. Every message is clinician-approved before a simulated send;
grounding pre-blocks ungrounded drafts; the system never has autonomy. Outreach is one-directional,
not a two-way conversation.

---

## Part 3 — Honest limitations we volunteer

1. **No outcome claim.** SMS-for-CV trials (TXT2HEART, TEXT ME, TextMe2; 2025 meta-analysis) show
   texting can improve adherence, LDL, and BP; we cite them as the plausibility basis, not as
   evidence this system changes outcomes. We ran no trial.
2. **One-directional.** No automated two-way conversation — deliberately scoped out as too risky for
   a first build. Patients reply to reach a health professional; a future gated version could draft
   that reply, still clinician-approved.
3. **Grounding ≠ clinical safety.** It certifies a claim is re-derivable from the record and catches
   hallucinated/patient-false numbers; it does not certify clinical appropriateness — hence the
   clinician gate.
4. **Synthetic data cuts both ways.** SyntheticRI (Synthea) means no PHI risk, but message quality
   can't be validated against real patients; implausible values are filtered with physiologic bounds.
   Real-world biological-variation thresholds on synthetic trajectories are a reasonable proxy — the
   numbers demonstrate behavior, not real-world yield.
5. **Calibration is sigmoid (Platt), not isotonic**, which over-discretized here. A reasonable,
   documented choice for smooth risk deltas, not a theoretical optimum.
6. **Trajectory is judged on modifiable risk, not total risk** (age held at enrollment). We claim the
   coachable part of risk moved, which is the part outreach can move — not that total risk fell.
7. **Score-at-date is faithful:** all nine time-varying features (BP, LDL, total chol, HDL, A1c, BMI,
   triglycerides, eGFR) are pulled longitudinally; only static features (sex, comorbidity flags) stay
   constant.
8. **Language is not inferred from ethnicity.** Both EN and ES are generated for every patient; the
   clinician picks in the UI.

---

## References

- Lipid biological variation / Reference Change Value (LDL ~16–25%, total chol ~17%, HDL ~18%):
  intra-individual lipid variation studies; myADLM biological-variation pearls; clinlabnavigator.
- Triglyceride within-subject variation ~22% (RCV ~60%): lipid biological-variation studies.
- ADA / NICE: 0.5 percentage points is the clinically significant HbA1c change.
- Minimal Detectable Change for BP (SBP 7.82, DBP 12.45 mmHg): resting-BP MDC study (PMC10573284).
- Arnett DK et al. 2019 ACC/AHA Guideline on the Primary Prevention of CVD (≥5% clinically
  meaningful weight change). Circulation 2019;140:e596-e646.

**Landscape sources:** EHR-identified individualized CVD outreach RCT (PubMed 23143672); CVD texting
(TextMe2, PMC7204915; meta-analysis, s12889-025-21818-0); LLM+EHR grounding/fact-checking+clinician
review (MedEduChat, PMC12714722; arXiv 2512.16189); Epic Emmie (epic.com/software/emmie); Epic
In-Basket AI + hallucination concern (mhaonline.com/blog/ai-messages-from-doctors); Epic Healthy
Planet (epic.com/software/population-health); Brown University Health on Epic
(brownhealth.org/providers/providers/lifechart); RIDOH FY2026 community outreach
(health.ri.gov/requests-proposals); RIDOH CHW CVD/diabetes program (jphmpdirect.com); CTC-RI
(ctc-ri.org/about-us/what-ctc-ri).
