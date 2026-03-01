# Dual-mode RRP therapy strategy ranking + criticality
# - NEW patients: no surgery history required (phenotype-driven)
# - RECURRENT patients: uses surgery burden + interval + phenotype

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

STRATEGIES = [
    "Surgery only",
    "Surgery + intralesional adjunct (cidofovir-like)",
    "Surgery + anti-VEGF (bevacizumab-like)",
    "Immunotherapy / immune modulation",
    "Clinical trial / specialty referral",
]


@dataclass
class PatientInput:
    age: int
    hpv_type: str  # "6" | "11" | "other" | "unknown"
    hpo_flags: Dict[str, int]
    # Optional / may be missing for new patients
    immune_compromised: Optional[int] = None  # 0/1/None
    surgeries_last_12m: Optional[int] = None
    avg_months_between_surgeries: Optional[float] = None
    anatomic_extent: Optional[int] = None  # 1 localized, 2 multi-site, 3 diffuse/unknown


def _clip(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def is_new_patient(p: PatientInput) -> bool:
    return p.surgeries_last_12m is None and p.avg_months_between_surgeries is None


def _airway_flags(hpo: Dict[str, int]) -> Tuple[int, int, int, int]:
    airway_obstruction = int(hpo.get("HP_0006536", 0))
    dyspnea = int(hpo.get("HP_0002094", 0))
    stridor = int(hpo.get("HP_0010307", 0))
    airway_count = airway_obstruction + dyspnea + stridor
    return airway_obstruction, dyspnea, stridor, airway_count


def compute_criticality(p: PatientInput) -> Tuple[str, List[str]]:
    """
    Clinically-inspired airway triage (works with/without surgery history):
    - 🟥 Critical: airway obstruction + (dyspnea or stridor)
    - 🟧 High: >=2 airway symptoms OR (recurrent + high burden + >=1 airway symptom)
    - 🟨 Moderate: 1 airway symptom OR hoarse voice alone (new) OR high burden alone (recurrent)
    - 🟩 Low: non-airway symptoms only
    """
    hpo = p.hpo_flags or {}
    airway_obstruction, dyspnea, stridor, airway_count = _airway_flags(hpo)
    reasons: List[str] = []

    if airway_obstruction == 1 and (dyspnea == 1 or stridor == 1):
        reasons.append("Airway obstruction with dyspnea/stridor suggests urgent airway risk")
        return "🟥 Critical", reasons

    if is_new_patient(p):
        if airway_count >= 2:
            reasons.append("Multiple airway compromise symptoms present")
            return "🟧 High", reasons
        if airway_count == 1:
            reasons.append("Single airway symptom present")
            return "🟨 Moderate", reasons
        if int(hpo.get("HP_0001609", 0)) == 1:
            reasons.append("Hoarse voice without airway compromise")
            return "🟨 Moderate", reasons
        if int(hpo.get("HP_0012735", 0)) == 1 or int(hpo.get("HP_0002205", 0)) == 1:
            reasons.append("Respiratory symptoms present without airway compromise")
            return "🟩 Low", reasons
        return "🟩 Low", ["No high-risk airway signals detected"]

    # Recurrent mode
    surg = int(p.surgeries_last_12m or 0)
    interval = float(p.avg_months_between_surgeries or 12.0)

    high_burden = (surg >= 6) or (interval <= 2.0)

    if airway_count >= 2:
        reasons.append("Multiple airway compromise symptoms present")
    if surg >= 6:
        reasons.append("High recurrence burden (≥6 surgeries/12m)")
    if interval <= 2.0:
        reasons.append("Very short interval between surgeries (≤2 months)")

    if airway_count >= 2 and high_burden:
        return "🟧 High", reasons

    if airway_count >= 1 and high_burden:
        if "Multiple airway compromise symptoms present" not in reasons:
            reasons.append("Airway symptom present")
        return "🟧 High", reasons

    if airway_count >= 1:
        if "Airway symptom present" not in reasons:
            reasons.append("Airway symptom present")
        return "🟨 Moderate", reasons

    if high_burden:
        return "🟨 Moderate", reasons or ["High recurrence burden"]

    return "🟩 Low", ["No high-risk airway or high-burden signals detected"]


def compute_severity(p: PatientInput) -> Tuple[float, str, List[str]]:
    """
    Severity score (0-10-ish).
    New: phenotype + small HPV/immune modifiers.
    Recurrent: adds recurrence burden + interval.
    """
    hpo = p.hpo_flags or {}
    airway_obstruction, dyspnea, stridor, airway_count = _airway_flags(hpo)

    reasons: List[str] = []
    score = 0.0

    # phenotype contribution
    score += 1.3 * airway_obstruction + 0.8 * dyspnea + 0.8 * stridor
    if airway_count >= 2:
        reasons.append("Multiple airway compromise symptoms")
        score += 0.7
    if int(hpo.get("HP_0001609", 0)) == 1:
        reasons.append("Hoarse voice present")
        score += 0.3
    if int(hpo.get("HP_0012735", 0)) == 1:
        reasons.append("Cough present")
        score += 0.2
    if int(hpo.get("HP_0002205", 0)) == 1:
        reasons.append("Recurrent respiratory infections present")
        score += 0.2

    # small modifiers
    if str(p.hpv_type).strip() == "11":
        reasons.append("HPV-11 (potentially more aggressive course, if confirmed)")
        score += 0.5
    if p.immune_compromised == 1:
        reasons.append("Immunocompromised status")
        score += 0.4

    # recurrence burden if available
    if not is_new_patient(p):
        surg = int(p.surgeries_last_12m or 0)
        interval = float(p.avg_months_between_surgeries or 12.0)

        if surg >= 10:
            reasons.append("Very high surgery burden (≥10/12m)")
        elif surg >= 5:
            reasons.append("High surgery burden (5–9/12m)")
        elif surg >= 1:
            reasons.append("Some surgery burden (1–4/12m)")

        score += min(4.0, surg / 3.0)  # up to ~4

        if interval <= 1.0:
            reasons.append("Very short recurrence interval (≤1 month)")
            score += 1.8
        elif interval <= 3.0:
            reasons.append("Short recurrence interval (1–3 months)")
            score += 1.0
        elif interval <= 6.0:
            reasons.append("Moderate recurrence interval (3–6 months)")
            score += 0.5

    # anatomy 
    if p.anatomic_extent in (2, 3):
        reasons.append("Multi-site/diffuse involvement (if confirmed)")
        score += 0.3 if p.anatomic_extent == 2 else 0.6

    if score >= 7.0:
        band = "High"
    elif score >= 4.0:
        band = "Medium"
    else:
        band = "Low"

    return score, band, reasons[:10]


def _strategy_boosts(p: PatientInput) -> Dict[str, float]:
    hpo = p.hpo_flags or {}
    airway_obstruction, dyspnea, stridor, airway_count = _airway_flags(hpo)

    boosts = {s: 0.0 for s in STRATEGIES}

    # anti-VEGF: prefer when airway compromise or high burden (if known)
    if airway_obstruction == 1 or airway_count >= 2:
        boosts["Surgery + anti-VEGF (bevacizumab-like)"] += 0.10

    if not is_new_patient(p):
        surg = int(p.surgeries_last_12m or 0)
        interval = float(p.avg_months_between_surgeries or 12.0)
        if surg >= 6 or interval <= 2.0:
            boosts["Surgery + anti-VEGF (bevacizumab-like)"] += 0.10

   
    if str(p.hpv_type).strip() in ("6", "11"):
        boosts["Surgery + intralesional adjunct (cidofovir-like)"] += 0.05
    if not is_new_patient(p):
        surg = int(p.surgeries_last_12m or 0)
        if surg >= 3:
            boosts["Surgery + intralesional adjunct (cidofovir-like)"] += 0.05

    if not is_new_patient(p):
        surg = int(p.surgeries_last_12m or 0)
        interval = float(p.avg_months_between_surgeries or 12.0)
        if surg >= 8 or interval <= 1.5:
            boosts["Immunotherapy / immune modulation"] += 0.07
            boosts["Clinical trial / specialty referral"] += 0.10

    return boosts


def _base_response(p: PatientInput, strategy: str, severity_band: str) -> Tuple[float, List[str]]:
    reasons: List[str] = []
    base = 0.58

    if severity_band == "High":
        base -= 0.08
        reasons.append("High severity phenotype")
    elif severity_band == "Low":
        base += 0.04

    hpo = p.hpo_flags or {}
    _, _, _, airway_count = _airway_flags(hpo)

  
    if is_new_patient(p):
        if strategy == "Surgery only":
            base += 0.05
            reasons.append("New patient: surgery-first baseline")
        if "anti-VEGF" in strategy and airway_count >= 2:
            base += 0.06
            reasons.append("Airway compromise supports early escalation consideration")
        return _clip(base), reasons

 
    surg = int(p.surgeries_last_12m or 0)
    interval = float(p.avg_months_between_surgeries or 12.0)
    airway_obstruction = int(hpo.get("HP_0006536", 0))

    if strategy == "Surgery only" and surg >= 6:
        base -= 0.10
        reasons.append("High recurrence suggests surgery-only may be insufficient")

    if "intralesional adjunct" in strategy:
        if str(p.hpv_type).strip() in ("6", "11"):
            base += 0.03
            reasons.append("HPV-driven etiology supports adjunct rationale (proxy)")
        if surg >= 3:
            base += 0.03
            reasons.append("Moderate recurrence may benefit from adjunct")

    if "anti-VEGF" in strategy:
        if surg >= 6:
            base += 0.07
            reasons.append("High recurrence may benefit from escalation (proxy)")
        if airway_obstruction == 1:
            base += 0.05
            reasons.append("Airway obstruction supports escalation strategy (proxy)")

    if "Immunotherapy" in strategy:
        if surg >= 8 or interval <= 2.0:
            base += 0.05
            reasons.append("Refractory pattern supports escalation (proxy)")
        if p.immune_compromised == 1:
            base -= 0.05
            reasons.append("Immunocompromised status may reduce immune strategy response")

    if "Clinical trial" in strategy:
        if surg >= 8 or interval <= 1.5:
            base += 0.10
            reasons.append("Refractory/high-burden disease supports trial/referral consideration")

    return _clip(base), reasons


def _burden_reduction_proxy(p: PatientInput, strategy: str) -> float:
    if is_new_patient(p):
        if strategy == "Surgery only":
            return 0.20
        if "intralesional adjunct" in strategy:
            return 0.25
        if "anti-VEGF" in strategy:
            return 0.30
        if "Immunotherapy" in strategy:
            return 0.28
        return 0.30

    surg = int(p.surgeries_last_12m or 0)
    burden = _clip(surg / 12.0)
    base = 0.15 + 0.35 * burden

    if strategy == "Surgery only":
        return _clip(base - 0.05)
    if "intralesional adjunct" in strategy:
        return _clip(base + 0.05)
    if "anti-VEGF" in strategy:
        return _clip(base + 0.12)
    if "Immunotherapy" in strategy:
        return _clip(base + 0.08)
    if "Clinical trial" in strategy:
        return _clip(base + 0.10)
    return _clip(base)


def rank_strategies(p: PatientInput) -> Dict:
    sev_score, sev_band, sev_reasons = compute_severity(p)
    crit, crit_reasons = compute_criticality(p)
    boosts = _strategy_boosts(p)

    rows = []
    for s in STRATEGIES:
        resp, why = _base_response(p, s, sev_band)
        resp = _clip(resp + boosts.get(s, 0.0))
        burden = _burden_reduction_proxy(p, s)
        score = 0.6 * resp + 0.4 * burden

        rows.append({
            "strategy": s,
            "response_likelihood": round(resp, 3),
            "burden_reduction_proxy": round(burden, 3),
            "rank_score": round(score, 3),
            "symptom_boost": round(boosts.get(s, 0.0), 3),
            "why": why[:3],
        })

    rows.sort(key=lambda x: x["rank_score"], reverse=True)

    filled = 0
    filled += 1 if p.hpv_type else 0
    filled += 1 if any(int(v) == 1 for v in (p.hpo_flags or {}).values()) else 0
    filled += 1 if p.immune_compromised in (0, 1) else 0
    if not is_new_patient(p):
        filled += 1 if p.surgeries_last_12m is not None else 0
        filled += 1 if p.avg_months_between_surgeries is not None else 0

    confidence = "High" if filled >= 4 else ("Medium" if filled >= 2 else "Low")

    return {
        "mode": "New patient" if is_new_patient(p) else "Recurrent patient",
        "severity_score": round(sev_score, 2),
        "severity_band": sev_band,
        "severity_reasons": sev_reasons,
        "criticality": crit,
        "criticality_reasons": crit_reasons,
        "confidence": confidence,
        "ranking": rows,
    }
