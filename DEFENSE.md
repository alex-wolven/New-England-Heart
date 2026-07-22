# New England Heart, defense notes

What it is, why each choice is defensible, and the limits we volunteer. Theme: we chose the honest option and can say why.

---

## Part 1, what it is and where it fits

**What it is.** A cardiovascular primary-prevention outreach pipeline. A risk model selects an elevated-risk, untreated cohort; we walk each patient's own longitudinal record across wellness checkups; when a measure moves beyond normal variation (or is newly out of range) we draft a warm, plain-language, bilingual (EN/ES) message that reports the patient's *own* numbers over a staged lifecycle (enrollment, progress, setback, graduation, re-enrollment); every clinical fact is re-verified against the record and a clinician approves every message before send.

**The AI is deliberately minimal, that is the safety design.** The AI never generates facts. Content is computed deterministically from the record into a fixed skeleton; the LLM only rephrases it into warm, 6th-grade-level EN/ES; a verification gate then re-derives every number and, if the LLM changed one value, blocks it and falls back to the skeleton. So the LLM is bounded to tone, the one thing it is safe at, and the facts are guaranteed. Pure templates would lose the natural bilingual phrasing that drives engagement; an unconstrained chatbot would lose the safety. This is the middle, on purpose.

**Novelty is the combination.** Every ingredient exists somewhere: EHR-driven CVD identification with outreach (2012 RCT), CVD texting (many RCTs), LLM-drafted patient messaging grounded against the EHR with clinician review (active 2025-26 research), staged engagement. What nothing deployed assembles is all of: (1) model-selected primary-prevention cohort, (2) tracking each patient's own measure *trajectory* across visits, (3) auto-drafted messages reporting the patient's own numbers with record-grounded verification, (4) a staged lifecycle judged on *modifiable* risk, (5) a clinician gate on every message, bilingually.

**Where it fits: a safe layer on top of Epic.** Epic in RI already does care-gap outreach (MyChart campaigns: "you are overdue, schedule here") and bulk population-health outreach (Healthy Planet, for ACO/value-based care). It does *not* reach out when a patient's own risk *trajectory* changes, grounded in their own numbers, fact-verified per value, staged over a lifecycle. That is the layer we add. Epic is the dominant EHR substrate in RI primary care: Brown University Health runs it ("LifeChart," since 2015), Coastal Medical (largest independent PC group) migrated onto it, and Providence Community Health Centers was the first RI community health center to adopt it.

**What NOT to claim.** Not "first to text CV patients," not "first LLM-with-clinician-gate" (Epic In-Basket, 1M+ drafts/month), not "we invented EHR grounding of LLM output," not "RI has nothing automated." We cannot see Brown Health's private Epic config, so every landscape claim is scoped to public documentation. (Care New England's Epic status is unconfirmed, do not name it.)

---

## Part 2, design rationale

**Reach and cost.** The 0.80 quantile is a *score* cut, not a headcount. Funnel on the enriched cohort:

| Stage | Count | Share |
|---|---|---|
| Full cohort | 167,892 | 100% |
| Eligible (untreated, no prior CVD) | 103,128 | 61% |
| Above the 0.80-quantile threshold (~0.039) | 23,712 | 23% of eligible, **14% of cohort** |
| Also has ≥1 out-of-range vital (of those with lab coverage) | ~91.5% | **~13% of cohort** |

Honest reach is **~1 in 7 patients**. The out-of-range-vital gate is a *safety* check (never message with nothing concrete), not a volume filter, 91.5% of high-risk patients with labs clear it. **Cost:** long (~500-char) bilingual messages run ~4 SMS segments each (accented Spanish can double it); at ~$0.01/segment all-in and ~2.3 messages/patient, ~55k messages/yr is **$2k to $4k/yr** in-cohort, **~$12k to $25k/yr** RI-statewide, plus one-time 10DLC registration and a small monthly fee. A rounding error against one avoided ~$20k+ CVD admission.

**Why the 80th percentile, not the 90th.** The messaged group has ~2.6x the population 5-year event rate and holds ~52% of all 5-year events, so "elevated" is true by construction. We pick 80th because the action (lifestyle outreach) is low-cost and low-harm, and lifestyle counseling is universally guideline-recommended, so the threshold should scale with how cheap/safe the action is. Moving 90th to 80th roughly doubles reach (10% to 20%) and event-capture (35% to 52%) for a message that costs nothing; below ~75th the risk multiple falls toward average and "elevated" weakens. Clinical anchor (secondary): ~3.9% at 5y maps under constant-hazard doubling to ~7.8% at 10y, the ACC/AHA intermediate entry, but our synthetic label is not the Pooled Cohort endpoint, so we lead with the empirical justification.

**5-year horizon + observability filter.** The data cannot validate 10 years (~65% have ≥5y follow-up, ~none have 10). We train/score only on patients whose 5-year outcome is *observable*: a CV event within 5y, OR ≥5 event-free years. Patients censored before 5y have an unknowable outcome, so we exclude them (~54,000) rather than label them "no event." Standard survival reasoning; the tradeoff is a mild selection toward longer follow-up.

**AUC ~0.73.** Expected, capped by the synthetic signal ceiling. The model is a race-blind, registry-first *selector*, not the contribution; any comparable CV model drops in.

**Race-blind, registry-first, and why sex stays.** Race is excluded because in these models it proxies structural disparity; encoding it bakes in inequity. Sex stays because it is a biological CV risk factor in every guideline (Pooled Cohort, Framingham, SCORE2) and carries real signal here (2nd-highest feature). Dropping it would make the model worse-calibrated for women, not fairer. Both age and sex are non-modifiable: they inform *who* we reach, never an actionable claim. We train only on untreated, no-prior-CVD patients (drop `on_statin`, `prior_ascvd`), avoiding the treatment-paradox confound and matching the deployment population.

**Feature contributions (mean |SHAP|, be honest if asked).** age 0.67 > sex 0.42 > HDL 0.23 > SBP 0.22 > LDL 0.20 > eGFR/total-chol/trig ~0.15 > BMI/DBP ~0.13 > A1c 0.06 > **hypertension 0.01, CKD 0.008, diabetes 0.006, smoking 0.000**. The comorbidity flags contribute little because the continuous labs already capture that physiology (a diabetic shows it in A1c, a hypertensive in BP); the model correctly prefers the measured value over the label. So on slide 4, lead with demographics/vitals/labs as drivers; the flags are included but largely subsumed. (Smoking = 0 exactly warrants a data check before claiming it.)

**Sigmoid (Platt) calibration.** Isotonic collapsed to ~3 probability steps on this rare-event data, zeroing the risk deltas the lifecycle stages need. Sigmoid gives smooth, monotone probabilities.

**Trend thresholds (a change must be real, not noise).** LDL 20% / total chol 17% / HDL 18% (Reference Change Value from lipid biological variation); A1c ≥0.5 pts (ADA/NICE); BP ≥8 systolic or ≥12 diastolic (Minimal Detectable Change), reported as the pair; BMI ≥5% (ACC/AHA), flagged only at obesity (≥30) and framed around behaviors, never the number. Triglycerides and eGFR stay in the model but are never messaged (trig too labile; eGFR is a kidney marker, not a lifestyle target). A change is messaged only when out of range at one end; a value equal to a threshold counts as healthy.

**Age-adjusted trajectory (the confound we took most seriously).** A single age-inclusive score selected patients (right) *and* narrated their journey (wrong): since age only rises, a patient could fix every modifiable measure and still see their score drift up, blocking progress and graduation. So selection uses the full score (age belongs in *who* we reach), but progress/setback/graduation use the score with **age held at the enrollment date**, so it moves only with modifiable measures. We claim a patient's *modifiable* risk moved, never total risk; the clinician panel still shows true current risk and real age. On slides this is worded "age-adjusted," precisely: age anchored at enrollment.

- **Progress:** a flagged measure improved, none worsened, modifiable risk did not rise.
- **Setback:** a flagged measure worsened, none improved, modifiable risk did not fall, so aging alone never triggers one.
- **Newly out-of-range:** a measure outside range we have never flagged is reported immediately as a state fact (single reading, like enrollment).
- **Confirmation:** change-driven progress/setback needs the direction confirmed over two consecutive checkups; enrollment/graduation need two consecutive above/below-threshold checkups. Newly-out-of-range is the one exception. The "nothing moved the other way" veto is scoped to flagged measures.

**What grounding verifies (and doesn't).** Every claim is independently re-derived from the record, bounded by its as-of date; the realized text must contain the exact numbers and not invert direction. Failures are blocked, regenerated with the reason fed back (≤2x), then routed to a clinician flagged red. Grounding checks *computability from the record*, not clinical appropriateness, which is why a clinician reviews every message.

**RAG over one guideline.** Retrieval touches only the suggestion line and citations on actionable/in-range claims; it never generates or selects a number. An allow-set keyed by (claim kind, metric) vetoes wrong passages. With one guideline and eight passages this is closer to a governed lookup than heavy retrieval, but the architecture cleanly separates record-grounded facts from guideline-grounded advice, so adding conditions means adding docs, not rewriting logic. Passages are faithful summaries of the 2019 ACC/AHA Primary Prevention guideline (Arnett et al.), cited not quoted.

**Cadence.** Outreach anchors to wellness checkups (~annual), not every lab reading, so we do not spam. Last-contact date is stated once in the intro.

**Nothing auto-sends.** Every message is clinician-approved before a simulated send; grounding pre-blocks ungrounded drafts. Outreach is one-directional. (A future automated path is credible *because* the messages are conservative and clinically approved, but that is a claim about the future, not today.)

---

## Part 3, limitations we volunteer

1. **No outcome claim.** SMS-for-CV trials (TXT2HEART, TEXT ME, TextMe2; 2025 meta-analysis) show texting can improve adherence, LDL, BP; we cite them as plausibility, not evidence this system changes outcomes. We ran no trial.
2. **One-directional.** No automated two-way conversation, scoped out as too risky for a first build.
3. **Grounding ≠ clinical safety.** It certifies a claim is re-derivable and catches false numbers; it does not certify appropriateness. Hence the clinician gate.
4. **Synthetic data cuts both ways.** SyntheticRI (Synthea) means no PHI risk, but message quality cannot be validated against real patients. Real-world variation thresholds on synthetic trajectories are a reasonable proxy; the numbers demonstrate behavior, not real-world yield.
5. **Calibration is sigmoid, not isotonic**, which over-discretized here. Documented choice, not a theoretical optimum.
6. **Trajectory judged on modifiable risk, not total risk** (age anchored). We claim the coachable part moved.
7. **Comorbidity flags are weak predictors** (see SHAP), subsumed by the continuous labs.

---

## References

- Lipid biological variation / RCV (LDL ~16-25%, total chol ~17%, HDL ~18%): intra-individual lipid variation studies; myADLM biological-variation pearls.
- ADA / NICE: 0.5 pts is the clinically significant HbA1c change.
- Minimal Detectable Change for BP (SBP 7.82, DBP 12.45 mmHg): resting-BP MDC study (PMC10573284).
- Arnett DK et al. 2019 ACC/AHA Guideline on Primary Prevention of CVD (≥5% meaningful weight change). Circulation 2019;140:e596-e646.
- **Landscape:** EHR-identified CVD outreach RCT (PubMed 23143672); CVD texting (TextMe2, PMC7204915; meta-analysis s12889-025-21818-0); LLM+EHR grounding+clinician review (MedEduChat, PMC12714722; arXiv 2512.16189); Epic Emmie, In-Basket AI, Healthy Planet (epic.com); Brown University Health "LifeChart" on Epic (brownhealth.org); Coastal Medical + PCHC Epic adoption; RIDOH FY2026 community outreach (health.ri.gov); CTC-RI (ctc-ri.org).
