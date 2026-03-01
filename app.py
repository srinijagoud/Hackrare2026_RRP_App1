
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
from src.scoring import PatientInput, rank_strategies

st.logo(image="images/streamlit-logo-primary-colormark-lighttext.png",
        icon_image="images/streamlit-mark-color.png")

MODELS_DIR = Path("models")
MED_MODEL_PATH = MODELS_DIR / "medical_response_model.pkl"
SURG_MODEL_PATH = MODELS_DIR / "surgical_response_model.pkl"

DATA_CANDIDATES = [
    Path("data") / "rrp_synthetic.csv",
]

HPO_JSON_CANDIDATES = [Path("hpo_mapping.json"), Path("data") / "hpo_mapping.json"]


st.set_page_config(page_title="RRP Therapy Matching and Response Map", layout="wide")
st.markdown(
    """
<style>
.title{font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial; font-weight: 850; font-size: 2.15rem; letter-spacing: -0.02em; margin-bottom: .25rem;}
.title span{background: linear-gradient(90deg, #06B6D4, #7C3AED); -webkit-background-clip: text; -webkit-text-fill-color: transparent;}
.sub{ color:#94a3b8; margin-top:-0.2rem; margin-bottom:1rem; font-size:.95rem; }
.card{ background: rgba(2,6,23,0.35); border: 1px solid rgba(148,163,184,0.15); border-radius: 14px; padding: 14px 14px; }
.card h4{ margin:0; font-size:0.95rem; color:#cbd5e1; font-weight:700; }
.status{ margin-top:6px; font-size: 2.1rem; font-weight: 850; letter-spacing: -0.02em; }
.pill{ display:inline-block; margin-top:8px; padding: 6px 10px; border-radius: 999px; font-weight: 700; font-size: 0.9rem; }
.good{ background: rgba(34,197,94,0.18); color: #86efac; border: 1px solid rgba(34,197,94,0.35); }
.bad{ background: rgba(239,68,68,0.16); color: #fca5a5; border: 1px solid rgba(239,68,68,0.35); }
.unk{ background: rgba(148,163,184,0.16); color: #cbd5e1; border: 1px solid rgba(148,163,184,0.28); }
.note{ color:#94a3b8; font-size:0.9rem; }
</style>
""",
    unsafe_allow_html=True,
)

st.markdown('<div class="title"><span>RRP Therapy Matching and Treatment Response Map </span></div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Clinician-facing prototype (hackathon). Not medical advice.</div>', unsafe_allow_html=True)


def load_hpo_mapping() -> dict:
    path = next((p for p in HPO_JSON_CANDIDATES if p.exists()), None)
    if path is None:
        raise FileNotFoundError("Missing hpo_mapping.json (place next to app file).")
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {k.replace("HP:", "HP_"): v for k, v in raw.items()}


def normalize_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"[^a-z0-9\s\-]", " ", text)
    #text = re.sub(r"[^A-zZ0-9\s\-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def map_text_to_hpo_flags(text: str, hpo_map: dict) -> dict:
    text = normalize_text(text)
    flags = {k: 0 for k in hpo_map.keys()}
    for hpo_col, meta in hpo_map.items():
        terms = []
        label = (meta.get("label") or "").strip()
        if label:
            terms.append(label.lower())
        for s in meta.get("synonyms", []) or []:
            s = (s or "").strip()
            if s:
                terms.append(s.lower())
        for t in terms:
            if t and re.search(rf"\b{re.escape(t)}\b", text):
                flags[hpo_col] = 1
                break
    return flags


def safe_load_model(path: Path):
    if not path.exists():
        return None
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        return None


def find_data_path() -> Path | None:
    return next((p for p in DATA_CANDIDATES if p.exists()), None)


def prob_to_status(prob: float | None, threshold: float = 0.5):
    if prob is None:
        return "Unknown", "unk"
    if prob >= threshold:
        return "Good", "good"
    return "Not Good", "bad"


def _encode_hpv(series: pd.Series) -> pd.DataFrame:
    s = series.fillna("unknown").astype(str).str.strip().replace({"": "unknown", "nan": "unknown", "NaN": "unknown"})
    return pd.get_dummies(s, prefix="hpv")


def phenotype_label_from_hpo(hpo: dict) -> str:
    obstruction = int(hpo.get("HP_0006536", 0))
    dyspnea = int(hpo.get("HP_0002094", 0))
    stridor = int(hpo.get("HP_0010307", 0))
    hoarse = int(hpo.get("HP_0001609", 0))
    cough = int(hpo.get("HP_0012735", 0))
    rti = int(hpo.get("HP_0002205", 0))

    airway = obstruction + dyspnea + stridor
    if obstruction == 1 and (dyspnea == 1 or stridor == 1):
        return "Airway severe"
    if airway >= 2:
        return "Airway severe"
    if airway == 1:
        return "Moderate airway"
    if hoarse == 1 or cough == 1 or rti == 1:
        return "Mild symptomatic"
    return "Minimal symptoms"


def build_row_for_models(df: pd.DataFrame, hpo_cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for col in ["age","immune_compromised","hpv_type","sex","surgeries_last_12m","avg_months_between_surgeries","anatomic_extent",
                "medical_treatment","medical_treatment_type","surgical_treatment"]:
        if col not in out.columns:
            out[col] = np.nan

    out["sex"] = out["sex"].fillna("Unknown").astype(str)
    out["age"] = pd.to_numeric(out["age"], errors="coerce").fillna(out["age"].median() if out["age"].notna().any() else 30)
    out["immune_compromised"] = pd.to_numeric(out["immune_compromised"], errors="coerce").fillna(0).astype(int)
    out["surgeries_last_12m"] = pd.to_numeric(out["surgeries_last_12m"], errors="coerce").fillna(0).astype(int)
    out["avg_months_between_surgeries"] = pd.to_numeric(out["avg_months_between_surgeries"], errors="coerce").fillna(12.0)
    out["anatomic_extent"] = pd.to_numeric(out["anatomic_extent"], errors="coerce").fillna(1).astype(int)
    out["medical_treatment"] = pd.to_numeric(out["medical_treatment"], errors="coerce").fillna(0).astype(int)
    out["medical_treatment_type"] = out["medical_treatment_type"].fillna("").astype(str)
    out["surgical_treatment"] = pd.to_numeric(out["surgical_treatment"], errors="coerce").fillna(0).astype(int)
    out["hpv_type"] = out["hpv_type"].fillna("unknown").astype(str)
    for c in hpo_cols:
        if c not in out.columns:
            out[c] = 0
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0).astype(int)
    return out


def make_patient_feature_row(
    *,
    age: int,
    immune_compromised,
    hpv_type: str,
    hpo_flags: dict,
    new_patient: bool,
    surgeries_last_12m=None,
    avg_months_between=None,
    anatomic_extent=None,
) -> pd.DataFrame:
    if new_patient:
        surgeries_last_12m = 0
        avg_months_between = 12.0
        surgical_treatment = 0
    else:
        surgeries_last_12m = int(surgeries_last_12m or 0)
        avg_months_between = float(avg_months_between or 12.0)
        surgical_treatment = 1 if surgeries_last_12m > 0 else 0

    immune_val = 0 if immune_compromised is None else int(immune_compromised)
    anatomic_val = 1 if anatomic_extent is None else int(anatomic_extent)

    row = {
        "age": int(age),
        "sex": "Unknown",
        "immune_compromised": immune_val,
        "hpv_type": str(hpv_type),
        "surgeries_last_12m": surgeries_last_12m,
        "avg_months_between_surgeries": avg_months_between,
        "anatomic_extent": anatomic_val,
        "medical_treatment": 0,
        "medical_treatment_type": "",
        "surgical_treatment": surgical_treatment,
    }
    for k, v in hpo_flags.items():
        row[k] = int(v)
    return pd.DataFrame([row])

hpo_map = load_hpo_mapping()
hpo_cols = list(hpo_map.keys())

med_model = safe_load_model(MED_MODEL_PATH)
surg_model = safe_load_model(SURG_MODEL_PATH)
data_path = find_data_path()

with st.sidebar:
    st.header("Required inputs")
    age = st.number_input("Age", min_value=0, max_value=120, value=25)
    hpv_type = st.selectbox("HPV type", options=["6", "11", "other", "unknown"], index=0)

    st.divider()
    st.subheader("Symptoms (HPO)")
    checkbox_flags = {}
    for hpo_id, meta in hpo_map.items():
        checkbox_flags[hpo_id] = 1 if st.checkbox(meta.get("label", hpo_id), value=False) else 0

    symptom_text = st.text_area("Free text symptoms (optional)", height=70, placeholder="Example: noisy breathing, voice change, shortness of breath")

    st.divider()
    st.subheader("Optional details")
    new_patient = st.checkbox("New patient (no surgery history)", value=True)

    immune_opt = st.selectbox("Immune compromised?", options=["Unknown", "No", "Yes"], index=0)
    immune_compromised = None if immune_opt == "Unknown" else (1 if immune_opt == "Yes" else 0)

    surgeries_last_12m = None
    avg_months_between = None
    if not new_patient:
        surgeries_last_12m = st.number_input("Surgeries in last 12 months", min_value=0, max_value=50, value=3)
        avg_months_between = st.number_input("Avg months between surgeries", min_value=0.1, max_value=24.0, value=3.0 if surgeries_last_12m > 0 else 12.0, step=0.5)

    extent_opt = st.selectbox("Anatomic extent", options=["Unknown", "Localized", "Multi-site", "Diffuse"], index=0)
    anatomic_extent = {"Unknown": None, "Localized": 1, "Multi-site": 2, "Diffuse": 3}[extent_opt]

    run_btn = st.button("Generate output", type="primary")

if not run_btn:
    st.info("Select symptoms and click **Generate output**.")
    st.stop()

text_flags = map_text_to_hpo_flags(symptom_text, hpo_map)
hpo_flags = {k: int(checkbox_flags.get(k, 0) or text_flags.get(k, 0)) for k in hpo_map.keys()}
active = [hpo_map[k].get("label", k) for k, v in hpo_flags.items() if int(v) == 1]
patient_pheno = phenotype_label_from_hpo(hpo_flags)

p = PatientInput(
    age=int(age),
    hpv_type=str(hpv_type),
    hpo_flags=hpo_flags,
    immune_compromised=immune_compromised,
    surgeries_last_12m=(None if new_patient else int(surgeries_last_12m)),
    avg_months_between_surgeries=(None if new_patient else float(avg_months_between)),
    anatomic_extent=anatomic_extent,
)
out = rank_strategies(p)

X_patient = make_patient_feature_row(
    age=int(age),
    immune_compromised=immune_compromised,
    hpv_type=str(hpv_type),
    hpo_flags=hpo_flags,
    new_patient=new_patient,
    surgeries_last_12m=surgeries_last_12m,
    avg_months_between=avg_months_between,
    anatomic_extent=anatomic_extent,
)

p_med = None
p_surg = None
try:
    if med_model is not None and hasattr(med_model, "predict_proba"):
        p_med = float(med_model.predict_proba(X_patient)[0, 1])
except Exception:
    p_med = None
try:
    if surg_model is not None and hasattr(surg_model, "predict_proba"):
        p_surg = float(surg_model.predict_proba(X_patient)[0, 1])
except Exception:
    p_surg = None

med_status, med_cls = prob_to_status(p_med)
surg_status, surg_cls = prob_to_status(p_surg)

left, right = st.columns([1, 1], vertical_alignment="top")

with left:
    st.subheader("Clinical triage")
    st.write(f"**Phenotype tier:** **{patient_pheno}**")
    st.write(f"**Mode:** {out['mode']}  |  **Confidence:** {out['confidence']}")
    st.markdown(f"### {out['criticality']}")
    for r in out.get("criticality_reasons", []):
        st.write(f"- {r}")
    st.write(f"**Severity score:** {out['severity_score']}  |  **Band:** {out['severity_band']}")

    st.divider()
    st.subheader("Active symptoms")
    st.write(active if active else ["(none selected)"])

with right:
    st.subheader("Therapy strategy ranking")
    df_rank = pd.DataFrame(out["ranking"]).rename(
        columns={
            "strategy": "Strategy",
            "rank_score": "Rank score",
            "response_likelihood": "Clinical rules",
            "burden_reduction_proxy": "Burden reduction (proxy)",
            "symptom_boost": "Symptom boost",
            "why": "Why (top points)",
        }
    )
    #df_rank.insert(1, "Criticality", out["criticality"])
    st.dataframe(df_rank, use_container_width=True)

st.divider()

st.subheader("Predicted response status")

def response_card(title: str, status: str, css_cls: str, prob: float | None):
    pill = "P(Good)=N/A" if prob is None else f"P(Good)={prob:.2f}"
    st.markdown(
        f"""
<div class="card">
  <h4>{title}</h4>
  <div class="status">{status}</div>
  <div class="pill {css_cls}">{pill}</div>
</div>
""",
        unsafe_allow_html=True,
    )

c1, c2 = st.columns(2)
with c1:
    response_card("Medical response (final)", med_status, med_cls, p_med)
with c2:
    response_card("Surgical response (final)", surg_status, surg_cls, p_surg)

st.markdown('<div class="note">Color semantics fixed: green = Good, red = Not Good, gray = Unknown.</div>', unsafe_allow_html=True)

st.divider()

st.subheader("Treatment Response Map")

if data_path is None:
    st.warning("No dataset found- 'rrp_synthetic.csv'.")
else:
    df = pd.read_csv(data_path)
    df = build_row_for_models(df, hpo_cols)

    df["phenotype"] = df.apply(lambda r: phenotype_label_from_hpo({c: int(r.get(c, 0)) for c in hpo_cols}), axis=1)
    df["symptom_burden"] = df[hpo_cols].sum(axis=1).astype(int)

    if med_model is not None and hasattr(med_model, "predict_proba"):
        try:
            df["p_med_good"] = med_model.predict_proba(df)[:, 1]
        except Exception:
            df["p_med_good"] = np.nan
    else:
        if "medical_response" in df.columns:
            s = df["medical_response"].fillna("").astype(str).str.lower()
            df["p_med_good"] = np.where(s.eq("good"), 1.0, np.where(s.eq("not good"), 0.0, np.nan))
        else:
            df["p_med_good"] = np.nan

    if surg_model is not None and hasattr(surg_model, "predict_proba"):
        try:
            df["p_surg_good"] = surg_model.predict_proba(df)[:, 1]
        except Exception:
            df["p_surg_good"] = np.nan
    else:
        if "surgical_response" in df.columns:
            s = df["surgical_response"].fillna("").astype(str).str.lower()
            df["p_surg_good"] = np.where(s.eq("good"), 1.0, np.where(s.eq("not good"), 0.0, np.nan))
        else:
            df["p_surg_good"] = np.nan

    plot_df = df.copy()
    plot_df["p_med_good"] = pd.to_numeric(plot_df["p_med_good"], errors="coerce")
    plot_df["p_surg_good"] = pd.to_numeric(plot_df["p_surg_good"], errors="coerce")
    plot_df = plot_df.dropna(subset=["p_med_good","p_surg_good"])

    st.caption("X: P(Medical Good), Y: P(Surgical Good). Color: phenotype tier. Bubble size: symptom burden.")
    try:
        import plotly.express as px
        import plotly.graph_objects as go

        fig = px.scatter(
            plot_df,
            x="p_med_good",
            y="p_surg_good",
            color="phenotype",
            size="symptom_burden",
            hover_data={"symptom_burden": True},
            labels={"p_med_good": "P(Medical response = Good)", "p_surg_good": "P(Surgical response = Good)"},
            title="Treatment Response Map",
        )

        fig.add_shape(type="line", x0=0.5, x1=0.5, y0=0, y1=1, line=dict(width=1, dash="dot"))
        fig.add_shape(type="line", x0=0, x1=1, y0=0.5, y1=0.5, line=dict(width=1, dash="dot"))

        if p_med is not None and p_surg is not None:
            fig.add_trace(
                go.Scatter(
                    x=[p_med], y=[p_surg],
                    mode="markers+text",
                    name="You",
                    text=["You"],
                    textposition="top center",
                    marker=dict(size=20, symbol="circle", line=dict(width=4)),
                )
            )

        fig.update_xaxes(range=[0,1])
        fig.update_yaxes(range=[0,1])

        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<div class="note">Quadrants: top-right = both respond well; bottom-left = both low.</div>', unsafe_allow_html=True)
    except Exception:
        st.info("Install plotly to view the response map: pip install plotly")

st.caption("This is a decision-support visualization for a hackathon prototype, not clinical guidance.")
