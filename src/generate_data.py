"""
Synthetic Sepsis Dataset Generator (v2 — realistic overlap)
=============================================================
Generates a physiologically plausible dataset of 2,000 patients with NEWS2
vital signs and laboratory markers. Targets classification difficulty comparable to real-world sepsis
   studies (AUROC ~0.80–0.88); in practice the cleaner synthetic
   distributions push this to ~0.91 (see the README note on this gap).

Key design choices for realism:
- Latent severity factor drives all vitals → realistic inter-feature correlation
- Significant distributional overlap between sepsis and non-sepsis groups
- ~35% of sepsis patients are normothermic (matching literature)
- Correlated noise simulates comorbidity effects and measurement variability
- Sepsis prevalence at ~20% (high-acuity inpatient cohort)

Reference ranges: NEWS2 (Royal College of Physicians, 2017),
Sepsis-3 (Singer et al., JAMA 2016).

Usage:
    python src/generate_data.py
    python src/generate_data.py --seed 0
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path


def generate_sepsis_cohort(
    n_patients: int = 2000,
    sepsis_prevalence: float = 0.20,
    seed: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_sepsis = int(n_patients * sepsis_prevalence)
    n_healthy = n_patients - n_sepsis
    labels = np.array([0] * n_healthy + [1] * n_sepsis)
    rng.shuffle(labels)

    # --- Latent severity factor ---
    # Each patient has an underlying "sickness severity" that influences all vitals.
    # This creates realistic within-group heterogeneity and cross-group overlap.
    severity = np.where(
        labels == 1,
        rng.beta(2.5, 2.0, n_patients),   # sepsis: right-skewed, some mild cases
        rng.beta(1.8, 4.0, n_patients),    # non-sepsis: left-skewed, some sick ones
    )

    # --- Demographics ---
    age = (40 + severity * 40 + rng.normal(0, 8, n_patients)).clip(18, 100).round(0).astype(int)
    sex = rng.choice(["M", "F"], size=n_patients, p=[0.52, 0.48])

    # --- NEWS2 Vital Signs (driven by severity + noise) ---
    resp_rate = (14 + severity * 16 + rng.normal(0, 3, n_patients)).clip(8, 45).round(0).astype(int)
    spo2 = (99 - severity * 10 + rng.normal(0, 2, n_patients)).clip(75, 100).round(0).astype(int)
    heart_rate = (65 + severity * 55 + (resp_rate - 16) * 0.5 + rng.normal(0, 10, n_patients)).clip(40, 175).round(0).astype(int)
    systolic_bp = (135 - severity * 50 + rng.normal(0, 14, n_patients)).clip(60, 220).round(0).astype(int)

    # Temperature — mix of febrile, hypothermic, and normothermic even in sepsis
    temp_vals = np.zeros(n_patients)
    for i in range(n_patients):
        if labels[i] == 1:
            r = rng.random()
            if r < 0.35:  # normothermic sepsis
                temp_vals[i] = 36.8 + rng.normal(0, 0.4)
            elif r < 0.90:  # febrile
                temp_vals[i] = 37.5 + severity[i] * 2.0 + rng.normal(0, 0.5)
            else:  # hypothermic
                temp_vals[i] = 35.8 - severity[i] * 1.5 + rng.normal(0, 0.4)
        else:
            temp_vals[i] = 36.8 + rng.normal(0, 0.4) + severity[i] * 0.3
    temperature = np.clip(temp_vals, 33.5, 41.5).round(1)

    # AVPU — consciousness declines with severity
    avpu = np.full(n_patients, "A", dtype="U1")
    for i in range(n_patients):
        if severity[i] > 0.85 and labels[i] == 1:
            avpu[i] = rng.choice(["V", "P", "U"], p=[0.5, 0.35, 0.15])
        elif severity[i] > 0.7 and labels[i] == 1:
            avpu[i] = rng.choice(["A", "V", "P"], p=[0.55, 0.30, 0.15])
        elif severity[i] > 0.6:
            avpu[i] = rng.choice(["A", "V"], p=[0.80, 0.20])

    # --- NEWS2 Score ---
    def calc_news2(rr, sp, hr, sbp, temp, consciousness):
        score = 0
        if rr <= 8: score += 3
        elif rr <= 11: score += 1
        elif rr <= 20: score += 0
        elif rr <= 24: score += 2
        else: score += 3
        if sp <= 91: score += 3
        elif sp <= 93: score += 2
        elif sp <= 95: score += 1
        if hr <= 40: score += 3
        elif hr <= 50: score += 1
        elif hr <= 90: score += 0
        elif hr <= 110: score += 1
        elif hr <= 130: score += 2
        else: score += 3
        if sbp <= 90: score += 3
        elif sbp <= 100: score += 2
        elif sbp <= 110: score += 1
        elif sbp <= 219: score += 0
        else: score += 3
        if temp <= 35.0: score += 3
        elif temp <= 36.0: score += 1
        elif temp <= 38.0: score += 0
        elif temp <= 39.0: score += 1
        else: score += 2
        if consciousness != "A": score += 3
        return score

    news2_score = np.array([
        calc_news2(resp_rate[i], spo2[i], heart_rate[i], systolic_bp[i],
                   temperature[i], avpu[i])
        for i in range(n_patients)
    ])

    # --- Laboratory Markers ---
    wcc_vals = np.zeros(n_patients)
    for i in range(n_patients):
        if labels[i] == 1 and rng.random() < 0.15:
            wcc_vals[i] = 3.0 - severity[i] * 1.5 + rng.normal(0, 0.8)
        else:
            wcc_vals[i] = 6.5 + severity[i] * 12 + rng.normal(0, 2.5)
    wcc = np.clip(wcc_vals, 0.5, 40).round(1)

    lactate_mean = 0.8 + severity * 3.5
    lactate = (lactate_mean * np.exp(rng.normal(0, 0.35, n_patients))).clip(0.3, 15.0).round(1)

    # CRP — driven by severity (like every other lab), not directly by label.
    # Sepsis patients still tend to have higher CRP on average because sepsis
    # correlates with higher severity — but there is meaningful overlap:
    # some septic patients present with near-normal CRP (early sepsis,
    # immunocompromised), and some non-septic patients have raised CRP
    # (post-operative, inflammatory conditions, malignancy).
    crp_base = 5 + severity * 80
    crp_noise = rng.lognormal(mean=0, sigma=0.6, size=n_patients)
    crp = (crp_base * crp_noise).clip(0.5, 500).round(1)

    on_oxygen = (rng.random(n_patients) < (0.1 + severity * 0.6)).astype(int)

    df = pd.DataFrame({
        "patient_id": [f"P{str(i).zfill(4)}" for i in range(n_patients)],
        "age": age, "sex": sex,
        "resp_rate": resp_rate, "spo2": spo2, "heart_rate": heart_rate,
        "systolic_bp": systolic_bp, "temperature": temperature,
        "avpu": avpu, "on_supplemental_o2": on_oxygen,
        "news2_score": news2_score,
        "wcc": wcc, "lactate": lactate, "crp": crp,
        "sepsis_onset": labels,
    })
    return df


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic sepsis cohort")
    parser.add_argument("--n", type=int, default=2000)
    parser.add_argument("--prevalence", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(__file__).resolve().parent.parent / "data" / "raw"
    output_dir.mkdir(parents=True, exist_ok=True)

    df = generate_sepsis_cohort(args.n, args.prevalence, args.seed)
    out = output_dir / "sepsis_cohort.csv"
    df.to_csv(out, index=False)

    print(f"Generated {len(df)} patients -> {out}")
    print(f"  Sepsis cases: {df['sepsis_onset'].sum()} ({df['sepsis_onset'].mean():.1%})")
    print(f"  Mean NEWS2 (sepsis):     {df.loc[df['sepsis_onset']==1, 'news2_score'].mean():.1f}")
    print(f"  Mean NEWS2 (non-sepsis): {df.loc[df['sepsis_onset']==0, 'news2_score'].mean():.1f}")


if __name__ == "__main__":
    main()
