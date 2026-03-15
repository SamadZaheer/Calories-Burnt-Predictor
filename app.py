import streamlit as st
import numpy as np
import pandas as pd
import pickle
import plotly.graph_objects as go

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Calories Burnt Predictor",
    page_icon="🔥",
    layout="centered",
)

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Primary button: teal fill ── */
    div.stButton > button[kind="primary"] {
        background-color: #0d9488 !important;
        color: white !important;
        border: none !important;
    }
    div.stButton > button[kind="primary"]:hover {
        background-color: #0a7c72 !important;
        color: white !important;
        border: none !important;
    }
    div.stButton > button[kind="primary"]:focus {
        background-color: #0d9488 !important;
        color: white !important;
        border: none !important;
        box-shadow: none !important;
    }

    /* ── Slider colours ── */
    .stSlider [data-baseweb="slider"] [role="slider"] {
        background-color: #0d9488 !important;
        border-color: #0d9488 !important;
    }
    div[data-testid="stThumbValue"] {
        color: #0d9488 !important;
    }
    .stSlider [data-baseweb="slider"] div[data-baseweb="slider-track-fill"] {
        background-color: #0d9488 !important;
    }

    /* Input group cards */
    .card {
        background-color: #1e1e2e;
        border: 1px solid #2e2e42;
        border-radius: 14px;
        padding: 1.4rem 1.6rem 1rem;
        margin-bottom: 1.2rem;
    }
    .card-label {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: #64ffda;
        margin-bottom: 0.6rem;
    }

    /* Result box */
    .result-box {
        background: linear-gradient(145deg, #0f2744 0%, #1a3a5c 100%);
        border: 1px solid #2563eb55;
        border-radius: 16px;
        padding: 2.2rem 2rem 1.8rem;
        text-align: center;
        margin: 1rem 0 0.6rem;
    }
    .cal-number {
        font-size: 5.5rem;
        font-weight: 800;
        color: #60a5fa;
        line-height: 1;
        letter-spacing: -2px;
    }
    .cal-unit {
        font-size: 1.25rem;
        font-weight: 500;
        color: #93c5fd;
        margin-top: 0.35rem;
    }
    .divider-line {
        border: none;
        border-top: 1px solid #1e40af44;
        margin: 1.2rem 0 1rem;
    }
    .context-row {
        display: flex;
        justify-content: center;
        gap: 2.5rem;
        flex-wrap: wrap;
    }
    .context-item { text-align: center; }
    .context-value {
        font-size: 1.5rem;
        font-weight: 700;
        color: #60a5fa;
        display: block;
    }
    .context-desc {
        font-size: 0.8rem;
        color: #94a3b8;
        margin-top: 0.15rem;
    }

    /* Intensity badge */
    .intensity-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        border-radius: 999px;
        padding: 0.45rem 1.1rem;
        font-size: 0.95rem;
        font-weight: 700;
        letter-spacing: 0.03em;
        margin-top: 0.9rem;
    }
    .intensity-light    { background: #14532d33; color: #4ade80; border: 1px solid #4ade8055; }
    .intensity-moderate { background: #78350f33; color: #fbbf24; border: 1px solid #fbbf2455; }
    .intensity-intense  { background: #7f1d1d33; color: #f87171; border: 1px solid #f8717155; }

    /* Footer */
    .footer {
        text-align: center;
        font-size: 0.75rem;
        color: #4b5563;
        padding-top: 1rem;
        margin-top: 1.5rem;
        border-top: 1px solid #1f2937;
    }

    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, scaler

model, scaler = load_artifacts()


# ── Session state — defaults & interaction flag ───────────────────────────────
DEFAULTS = dict(
    gender="Male",
    age=35,
    height=174,
    weight=75,
    duration=15,
    heart_rate=96,
    body_temp=40.0,
    has_interacted=False,
)

for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val

def set_interacted():
    st.session_state.has_interacted = True

def reset_all():
    for key, val in DEFAULTS.items():
        st.session_state[key] = val


# ── Header ────────────────────────────────────────────────────────────────────
st.title("Calories Burnt Predictor")
if st.session_state.has_interacted:
    st.markdown("Move any slider to update the prediction instantly.")
else:
    st.markdown("Adjust your stats below — the prediction updates live as you move the sliders.")
st.divider()


# ── Inputs ────────────────────────────────────────────────────────────────────

# Personal stats
st.markdown('<div class="card"><p class="card-label">Personal Stats</p>', unsafe_allow_html=True)

col1, col2 = st.columns([1, 2])
with col1:
    gender = st.radio(
        "Gender", ["Male", "Female"],
        key="gender",
    )
with col2:
    age    = st.slider("Age",    min_value=20,  max_value=79,  step=1,   format="%d yrs", key="age")
    height = st.slider("Height", min_value=123, max_value=222, step=1,   format="%d cm",  key="height")
    weight = st.slider("Weight", min_value=36,  max_value=132, step=1,   format="%d kg",  key="weight")

st.markdown('</div>', unsafe_allow_html=True)

# Workout stats
st.markdown('<div class="card"><p class="card-label">Workout Stats</p>', unsafe_allow_html=True)

col3, col4, col5 = st.columns(3)
with col3:
    duration   = st.slider("Duration",   min_value=1,    max_value=30,  step=1,   format="%d min",  key="duration",   help="Training data covers sessions up to 30 minutes")
with col4:
    heart_rate = st.slider("Heart Rate", min_value=67,   max_value=128, step=1,   format="%d bpm",  key="heart_rate")
with col5:
    body_temp  = st.slider("Body Temp",  min_value=37.1, max_value=41.5, step=0.1, format="%.1f °C", key="body_temp")

st.markdown('</div>', unsafe_allow_html=True)


# ── Predict button (initial state) / Reset button (live state) ───────────────
if not st.session_state.has_interacted:
    st.button("Predict Calories Burnt", on_click=set_interacted,
              type="primary", use_container_width=True)
else:
    st.button("↺  Reset to defaults", on_click=reset_all, use_container_width=False)


# ── Live result ───────────────────────────────────────────────────────────────
if st.session_state.has_interacted:
    gender_enc = 0 if st.session_state.gender == "Male" else 1
    X = pd.DataFrame(
        [[gender_enc, st.session_state.age, st.session_state.height,
          st.session_state.weight, st.session_state.duration,
          st.session_state.heart_rate, st.session_state.body_temp]],
        columns=["Gender", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"],
    )
    calories = float(model.predict(scaler.transform(X))[0])
    calories = max(1.0, calories)

    # Context
    km_running   = round(calories / 70, 1)
    daily_pct    = round((calories / 2000) * 100, 1)
    slices_bread = round(calories / 79, 1)

    # Intensity
    if calories < 50:
        intensity_class, intensity_icon, intensity_label = "intensity-light",    "🟢", "Light workout"
    elif calories <= 150:
        intensity_class, intensity_icon, intensity_label = "intensity-moderate", "🟡", "Moderate workout"
    else:
        intensity_class, intensity_icon, intensity_label = "intensity-intense",  "🔴", "Intense workout"

    st.markdown(f"""
    <div class="result-box">
        <div class="cal-number">{calories:.0f}</div>
        <div class="cal-unit">kilocalories burned</div>
        <div>
            <span class="intensity-badge {intensity_class}">
                {intensity_icon}&nbsp;{intensity_label}
            </span>
        </div>
        <hr class="divider-line">
        <div class="context-row">
            <div class="context-item">
                <span class="context-value">{km_running} km</span>
                <div class="context-desc">equivalent running distance</div>
            </div>
            <div class="context-item">
                <span class="context-value">{daily_pct}%</span>
                <div class="context-desc">of average daily intake (2,000 kcal)</div>
            </div>
            <div class="context-item">
                <span class="context-value">{slices_bread}</span>
                <div class="context-desc">slices of bread (~79 kcal each)</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Duration curve chart ──────────────────────────────────────────────────
    durations = np.arange(1, 31)
    curve_rows = pd.DataFrame({
        "Gender":     gender_enc,
        "Age":        st.session_state.age,
        "Height":     st.session_state.height,
        "Weight":     st.session_state.weight,
        "Duration":   durations,
        "Heart_Rate": st.session_state.heart_rate,
        "Body_Temp":  st.session_state.body_temp,
    })
    curve_cals = model.predict(scaler.transform(curve_rows)).clip(min=1)

    fig = go.Figure()

    # Filled area under the line for depth
    fig.add_trace(go.Scatter(
        x=durations, y=curve_cals,
        fill="tozeroy", fillcolor="rgba(13,148,136,0.08)",
        line=dict(color="#0d9488", width=2.5),
        mode="lines",
        hovertemplate="<b>%{x} min</b><br>%{y:.0f} kcal<extra></extra>",
    ))

    # Vertical marker at user's current duration
    fig.add_vline(
        x=st.session_state.duration,
        line=dict(color="#0d9488", width=1.5, dash="dash"),
        annotation_text=f"  {st.session_state.duration} min — {calories:.0f} kcal",
        annotation_position="top right",
        annotation_font=dict(color="#0d9488", size=12),
    )

    fig.update_layout(
        title=dict(
            text="How calories scale with workout duration for your profile",
            font=dict(size=13, color="#374151"),
            x=0, xanchor="left",
        ),
        xaxis=dict(
            title="Duration (min)", tickmode="linear", dtick=5,
            showgrid=True, gridcolor="#f0f0f0", gridwidth=1,
            zeroline=False, tickfont=dict(color="#6b7280"),
            title_font=dict(color="#6b7280"),
        ),
        yaxis=dict(
            title="Calories (kcal)",
            showgrid=True, gridcolor="#f0f0f0", gridwidth=1,
            zeroline=False, tickfont=dict(color="#6b7280"),
            title_font=dict(color="#6b7280"),
        ),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=0, r=0, t=40, b=0),
        height=300,
        showlegend=False,
    )

    st.plotly_chart(fig, use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<p class="footer">This prediction is based on a machine learning model trained on 15,000 workout sessions · XGBoost · R² = 0.999 · MAE ≈ 1.4 kcal</p>',
    unsafe_allow_html=True,
)
