import numpy as np
import pandas as pd

rng = np.random.default_rng(42)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sample_from_probs(labels, probs):
    probs = np.array(probs, dtype=float)
    probs = probs / probs.sum()
    return rng.choice(labels, p=probs)

def make_synthetic_rrp(n=1000):
    rows = []
    for i in range(1, n + 1):
        patient_id = f"P{i:04d}"

        # --- Latent severity (0..1) ---
        # Mixture: many mild/moderate, some severe
        severity = np.clip(rng.beta(2.2, 2.8), 0, 1)

        # --- Age distribution (juvenile + adult) ---
        # severity mildly higher in juvenile onset patterns
        if rng.random() < 0.35:
            age = int(np.clip(rng.normal(8, 5), 0, 25))
            severity = np.clip(severity + 0.08, 0, 1)
        else:
            age = int(np.clip(rng.normal(35, 12), 18, 70))

        sex = rng.choice(["F", "M"], p=[0.48, 0.52])

        immune_compromised = int(rng.random() < (0.08 + 0.18 * severity))  # more likely if severe

        # --- HPV type ---
        # More HPV-11 among severe cases
        hpv_roll = rng.random()
        if hpv_roll < (0.50 - 0.15 * severity):
            hpv_type = "6"
        elif hpv_roll < (0.85 + 0.10 * severity):
            hpv_type = "11"
        else:
            hpv_type = "unknown"

        # --- Anatomic extent (1=localized,2=multi-site,3=diffuse) ---
        if severity < 0.35:
            anatomic_extent = sample_from_probs([1, 2, 3], [0.75, 0.22, 0.03])
        elif severity < 0.70:
            anatomic_extent = sample_from_probs([1, 2, 3], [0.35, 0.50, 0.15])
        else:
            anatomic_extent = sample_from_probs([1, 2, 3], [0.15, 0.45, 0.40])

        # --- HPO flags (probabilities depend on severity) ---
        # labels from your mapping:
        # HP_0001609 hoarse voice is common even in mild
        HP_0001609 = int(rng.random() < (0.75 + 0.20 * severity))

        # airway-related increase sharply with severity
        HP_0010307 = int(rng.random() < sigmoid(-1.2 + 3.2 * severity))  # stridor
        HP_0002094 = int(rng.random() < sigmoid(-1.0 + 3.0 * severity))  # dyspnea
        HP_0006536 = int(rng.random() < sigmoid(-1.4 + 3.6 * severity))  # airway obstruction

        # cough common; infections increase with immune compromise + severity
        HP_0012735 = int(rng.random() < (0.35 + 0.45 * severity))
        HP_0002205 = int(rng.random() < sigmoid(-1.3 + 2.4 * severity + 1.2 * immune_compromised))

        # --- Surgeries last 12m (depends on severity + airway flags) ---
        airway_burden = HP_0010307 + HP_0002094 + HP_0006536
        expected_surgeries = 1 + 10 * severity + 1.2 * airway_burden + 1.2 * (hpv_type == "11") + 1.0 * immune_compromised
        surgeries_last_12m = int(np.clip(rng.poisson(lam=max(0.2, expected_surgeries)), 0, 15))

        # --- Avg months between surgeries (inverse-ish, add noise) ---
        if surgeries_last_12m == 0:
            avg_months_between_surgeries = 12.0
        else:
            avg_months_between_surgeries = float(np.clip(12.0 / surgeries_last_12m + rng.normal(0, 0.4), 0.5, 12.0))

        # --- Treatments ---
        # Surgery almost always if recurrent, but new pts may be 0
        surgical_treatment = int(surgeries_last_12m > 0 or rng.random() < 0.60)

        # Medical treatment used more in severe/recurrent
        medical_treatment = int(rng.random() < (0.20 + 0.65 * severity + 0.05 * surgeries_last_12m))

        if medical_treatment == 0:
            medical_treatment_type = ""
            medical_response = ""
        else:
            # Choose a therapy type biased by severity
            # mild/moderate: cidofovir-like; severe: bevacizumab-like; some steroid/interferon
            if severity < 0.45:
                medical_treatment_type = sample_from_probs(
                    ["cidofovir", "steroid", "interferon", "bevacizumab"],
                    [0.55, 0.25, 0.12, 0.08]
                )
            else:
                medical_treatment_type = sample_from_probs(
                    ["bevacizumab", "cidofovir", "steroid", "interferon"],
                    [0.55, 0.20, 0.15, 0.10]
                )

            # --- Responses derived from severity + treatment type ---
            # Lower severity → better response; anti-VEGF improves odds in high severity
            base_good = 0.60 - 0.55 * severity
            if medical_treatment_type == "bevacizumab":
                base_good += 0.18
            if immune_compromised:
                base_good -= 0.08

            p_good = np.clip(base_good, 0.05, 0.90)
            p_partial = np.clip(0.25 + 0.15 * severity, 0.05, 0.70)
            p_poor = np.clip(1 - (p_good + p_partial), 0.05, 0.80)

            medical_response = sample_from_probs(["Good", "Partial", "Poor"], [p_good, p_partial, p_poor])

        # Surgical response: generally good in mild/moderate, worse when diffuse + high severity
        base_surg_good = 0.75 - 0.45 * severity
        if anatomic_extent == 3:
            base_surg_good -= 0.10
        if airway_burden >= 2:
            base_surg_good -= 0.06

        p_surg_good = np.clip(base_surg_good, 0.10, 0.90)
        p_surg_partial = np.clip(0.18 + 0.12 * severity, 0.05, 0.70)
        p_surg_poor = np.clip(1 - (p_surg_good + p_surg_partial), 0.05, 0.80)

        surgical_response = sample_from_probs(["Good", "Partial", "Poor"], [p_surg_good, p_surg_partial, p_surg_poor])

        rows.append({
            "patient_id": patient_id,
            "age": age,
            "sex": sex,
            "immune_compromised": immune_compromised,
            "hpv_type": hpv_type,
            "surgeries_last_12m": surgeries_last_12m,
            "avg_months_between_surgeries": round(avg_months_between_surgeries, 2),
            "anatomic_extent": anatomic_extent,
            "medical_treatment": medical_treatment,
            "medical_treatment_type": medical_treatment_type,
            "medical_response": medical_response,
            "surgical_treatment": surgical_treatment,
            "surgical_response": surgical_response,
            "HP_0001609": HP_0001609,
            "HP_0010307": HP_0010307,
            "HP_0002094": HP_0002094,
            "HP_0006536": HP_0006536,
            "HP_0012735": HP_0012735,
            "HP_0002205": HP_0002205,
        })

    return pd.DataFrame(rows)

if __name__ == "__main__":
    df = make_synthetic_rrp(n=1000)
    df.to_csv("rrp_synthetic.csv", index=False)
    print("Wrote rrp_synthetic.csv", df.shape)
    print(df.head())