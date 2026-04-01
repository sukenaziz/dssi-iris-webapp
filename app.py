"""
Iris Species Classifier — Streamlit Application
Tabs: Overview | Explore Data | Model Performance | Live Predict
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier

from src.inference import get_prediction, SPECIES_MAP, SPECIES_EMOJI
from src.model_registry import retrieve, get_metadata
from src.config import appconfig

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌸 Iris Classifier",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #f8f9ff; }

    /* Metric cards */
    [data-testid="metric-container"] {
        background: white;
        border: 1px solid #e0e4f0;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: white;
        border-radius: 12px;
        padding: 6px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 20px;
        font-weight: 500;
    }

    /* Prediction result card */
    .pred-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 24px;
        color: white;
        text-align: center;
        margin: 16px 0;
    }
    .pred-card h1 { color: white; margin: 0; font-size: 3rem; }
    .pred-card h3 { color: rgba(255,255,255,0.9); margin: 4px 0; }

    /* Section headers */
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #4a4a8a;
        border-left: 4px solid #667eea;
        padding-left: 10px;
        margin: 16px 0 8px 0;
    }
</style>
""", unsafe_allow_html=True)

# ── Data helpers ──────────────────────────────────────────────────────────────
FEATURES  = appconfig['Model']['features'].split(',')
LABEL     = appconfig['Model']['label']
PALETTE   = {"setosa": "#FF6B9D", "versicolor": "#4ECDC4", "virginica": "#45B7D1"}
PALETTE_ID = {0: "#FF6B9D", 1: "#4ECDC4", 2: "#45B7D1"}

@st.cache_data
def load_data():
    df = pd.read_csv("data/iris.csv")
    df['species_name'] = df['species'].map(SPECIES_MAP)
    return df

@st.cache_resource
def load_model_artifacts():
    clf, features = retrieve(appconfig['Model']['name'])
    meta = get_metadata(appconfig['Model']['name'])
    return clf, features, meta

df = load_data()
clf, features, meta = load_model_artifacts()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌸 Iris Classifier")
    st.markdown("*Decision Tree · NUS-ISS DSSI*")
    st.divider()

    st.markdown("### 🔬 Live Prediction Inputs")
    st.caption("Adjust the sliders and switch to the **Live Predict** tab.")

    sl = st.slider("Sepal Length (cm)", 4.0, 8.0, 5.8, 0.1)
    sw = st.slider("Sepal Width (cm)",  2.0, 4.5, 3.0, 0.1)
    pl = st.slider("Petal Length (cm)", 1.0, 7.0, 3.8, 0.1)
    pw = st.slider("Petal Width (cm)",  0.1, 2.5, 1.2, 0.1)

    st.divider()
    if meta:
        st.markdown("### 📊 Deployed Model")
        st.markdown(f"**Algorithm:** Decision Tree")
        st.markdown(f"**Version:** v{meta.get('version', 1)}")
        st.markdown(f"**Accuracy:** {meta['metrics']['accuracy']*100:.1f}%")
        st.markdown(f"**F1 Score:** {meta['metrics'].get('f1', 0)*100:.1f}%")

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "🏠 Overview",
    "📊 Explore Data",
    "🎯 Model Performance",
    "🔮 Live Predict"
])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Overview
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("# 🌸 Iris Species Classifier")
    st.markdown(
        "A machine learning application that classifies Iris flowers into three species "
        "using a **Decision Tree Classifier** trained on 150 flower samples. "
        "Use the sidebar sliders to adjust measurements and explore the **Live Predict** tab."
    )

    # KPI cards
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🎯 Accuracy",   f"{meta['metrics']['accuracy']*100:.1f}%")
    with col2:
        st.metric("📐 F1 Score",   f"{meta['metrics'].get('f1', 0)*100:.1f}%")
    with col3:
        st.metric("🌿 Training samples", "120")
    with col4:
        st.metric("🔢 Features",   "4")

    st.divider()

    col_a, col_b = st.columns([1.1, 1])

    with col_a:
        st.markdown('<div class="section-header">About the Dataset</div>', unsafe_allow_html=True)
        st.markdown("""
The **Iris dataset** is a classic multiclass classification benchmark introduced by
Ronald Fisher in 1936. It contains **150 samples** from three Iris species:

| Species | Colour code | Count |
|---------|-------------|-------|
| 🌸 *Iris setosa* | Pink | 50 |
| 🌿 *Iris versicolor* | Teal | 50 |
| 🌺 *Iris virginica* | Blue | 50 |

**Features measured (in cm):**
- Sepal length & width
- Petal length & width
        """)

    with col_b:
        st.markdown('<div class="section-header">Species Distribution</div>', unsafe_allow_html=True)
        dist = df['species_name'].value_counts().reset_index()
        dist.columns = ['species', 'count']
        fig_pie = px.pie(
            dist, values='count', names='species',
            color='species', color_discrete_map=PALETTE,
            hole=0.45
        )
        fig_pie.update_layout(
            margin=dict(t=10, b=10, l=10, r=10),
            legend=dict(orientation="h", yanchor="bottom", y=-0.2),
            height=280
        )
        fig_pie.update_traces(textposition='outside', textinfo='percent+label')
        st.plotly_chart(fig_pie, use_container_width=True)

    # Raw data preview
    st.markdown('<div class="section-header">Dataset Preview</div>', unsafe_allow_html=True)
    st.dataframe(
        df.style.background_gradient(
            subset=FEATURES, cmap='Blues', axis=0
        ).format({f: "{:.1f}" for f in FEATURES}),
        use_container_width=True, height=260
    )

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Explore Data
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 📊 Exploratory Data Analysis")

    # ── Summary stats
    st.markdown('<div class="section-header">Summary Statistics by Species</div>', unsafe_allow_html=True)
    selected_species = st.multiselect(
        "Filter species:", ["setosa", "versicolor", "virginica"],
        default=["setosa", "versicolor", "virginica"]
    )
    filtered_df = df[df['species_name'].isin(selected_species)]

    stats = filtered_df.groupby('species_name')[FEATURES].mean().round(2).reset_index()
    st.dataframe(
        stats.style.background_gradient(subset=FEATURES, cmap='RdYlGn'),
        use_container_width=True
    )

    st.divider()

    # ── Box plots
    st.markdown('<div class="section-header">Feature Distributions (Box Plot)</div>', unsafe_allow_html=True)
    feat_choice = st.selectbox("Select feature:", FEATURES, index=2)

    fig_box = px.box(
        filtered_df, x='species_name', y=feat_choice,
        color='species_name', color_discrete_map=PALETTE,
        points="all", notched=True,
        labels={'species_name': 'Species', feat_choice: feat_choice.replace('_', ' ').title() + ' (cm)'}
    )
    fig_box.update_layout(
        showlegend=False, height=380,
        plot_bgcolor='white', paper_bgcolor='white',
        xaxis=dict(showgrid=False),
        yaxis=dict(gridcolor='#f0f0f0')
    )
    st.plotly_chart(fig_box, use_container_width=True)

    st.divider()

    # ── Scatter plot
    st.markdown('<div class="section-header">Interactive Scatter Plot</div>', unsafe_allow_html=True)
    col_x, col_y = st.columns(2)
    with col_x:
        x_axis = st.selectbox("X-axis:", FEATURES, index=0)
    with col_y:
        y_axis = st.selectbox("Y-axis:", FEATURES, index=2)

    fig_scatter = px.scatter(
        filtered_df, x=x_axis, y=y_axis,
        color='species_name', color_discrete_map=PALETTE,
        symbol='species_name', size_max=10,
        labels={
            x_axis: x_axis.replace('_', ' ').title() + ' (cm)',
            y_axis: y_axis.replace('_', ' ').title() + ' (cm)',
            'species_name': 'Species'
        },
        marginal_x="histogram", marginal_y="histogram"
    )
    fig_scatter.update_traces(marker=dict(size=9, opacity=0.8))
    fig_scatter.update_layout(
        height=480, plot_bgcolor='white', paper_bgcolor='white',
        legend=dict(orientation="h", yanchor="bottom", y=-0.25)
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.divider()

    # ── Correlation heatmap
    st.markdown('<div class="section-header">Feature Correlation Heatmap</div>', unsafe_allow_html=True)
    corr = filtered_df[FEATURES].corr().round(2)
    fig_heat = px.imshow(
        corr, text_auto=True, color_continuous_scale='RdBu_r',
        zmin=-1, zmax=1, aspect='auto'
    )
    fig_heat.update_layout(height=380, margin=dict(t=20, b=20))
    st.plotly_chart(fig_heat, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Performance
# ═══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 🎯 Model Performance")

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy",  f"{meta['metrics']['accuracy']*100:.2f}%")
    col2.metric("Precision", f"{meta['metrics'].get('precision', 0)*100:.2f}%")
    col3.metric("Recall",    f"{meta['metrics'].get('recall', 0)*100:.2f}%")
    col4.metric("F1 Score",  f"{meta['metrics'].get('f1', 0)*100:.2f}%")

    st.divider()

    col_cm, col_fi = st.columns(2)

    # ── Confusion matrix
    with col_cm:
        st.markdown('<div class="section-header">Confusion Matrix (Test Set)</div>', unsafe_allow_html=True)
        X = df[FEATURES]
        y = df[LABEL]
        _, X_test, _, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        species_labels = ["setosa", "versicolor", "virginica"]

        fig_cm = px.imshow(
            cm, text_auto=True,
            x=species_labels, y=species_labels,
            color_continuous_scale='Blues',
            labels=dict(x="Predicted", y="Actual", color="Count")
        )
        fig_cm.update_layout(height=360, margin=dict(t=20))
        st.plotly_chart(fig_cm, use_container_width=True)

    # ── Feature importance
    with col_fi:
        st.markdown('<div class="section-header">Feature Importance</div>', unsafe_allow_html=True)
        importances = clf.feature_importances_
        fi_df = pd.DataFrame({
            'Feature': [f.replace('_', ' ').title() for f in FEATURES],
            'Importance': importances
        }).sort_values('Importance', ascending=True)

        fig_fi = px.bar(
            fi_df, x='Importance', y='Feature',
            orientation='h',
            color='Importance', color_continuous_scale='Purples',
            text=fi_df['Importance'].apply(lambda x: f"{x:.3f}")
        )
        fig_fi.update_layout(
            height=360, showlegend=False,
            plot_bgcolor='white', paper_bgcolor='white',
            coloraxis_showscale=False,
            xaxis=dict(gridcolor='#f0f0f0'),
            yaxis=dict(showgrid=False)
        )
        fig_fi.update_traces(textposition='outside')
        st.plotly_chart(fig_fi, use_container_width=True)

    st.divider()

    # ── Decision boundary (2D projection)
    st.markdown('<div class="section-header">Decision Boundary — Petal Length vs Petal Width</div>', unsafe_allow_html=True)
    st.caption("Visualises how the Decision Tree separates species in 2D feature space.")

    # Build a 2D mesh
    feat_a, feat_b = 'petal_length', 'petal_width'
    fa_idx, fb_idx = FEATURES.index(feat_a), FEATURES.index(feat_b)
    x_min, x_max = df[feat_a].min() - 0.3, df[feat_a].max() + 0.3
    y_min, y_max = df[feat_b].min() - 0.2, df[feat_b].max() + 0.2
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))

    # Fill other features with their means for prediction
    grid_data = np.full((xx.ravel().shape[0], len(FEATURES)), df[FEATURES].mean().values)
    grid_data[:, fa_idx] = xx.ravel()
    grid_data[:, fb_idx] = yy.ravel()
    Z = clf.predict(grid_data).reshape(xx.shape)

    fig_bd = go.Figure()
    bg_colors = ["#FFCCDD", "#CCEFEC", "#CCE8F4"]
    for cls_id in [0, 1, 2]:
        mask = (Z == cls_id)
        fig_bd.add_trace(go.Scatter(
            x=xx[mask].ravel(), y=yy[mask].ravel(),
            mode='markers',
            marker=dict(color=bg_colors[cls_id], size=3, opacity=0.35),
            name=f"{SPECIES_EMOJI[cls_id]} {SPECIES_MAP[cls_id]} region",
            showlegend=True
        ))

    # Actual data points
    for cls_id in [0, 1, 2]:
        sub = df[df['species'] == cls_id]
        fig_bd.add_trace(go.Scatter(
            x=sub[feat_a], y=sub[feat_b],
            mode='markers',
            marker=dict(color=PALETTE_ID[cls_id], size=9,
                        line=dict(color='white', width=1.5)),
            name=f"{SPECIES_EMOJI[cls_id]} {SPECIES_MAP[cls_id]}",
        ))

    fig_bd.update_layout(
        xaxis_title="Petal Length (cm)", yaxis_title="Petal Width (cm)",
        height=420, plot_bgcolor='white', paper_bgcolor='white',
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        xaxis=dict(gridcolor='#f0f0f0'), yaxis=dict(gridcolor='#f0f0f0')
    )
    st.plotly_chart(fig_bd, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Live Predict
# ═══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## 🔮 Live Prediction")
    st.markdown("Adjust the **sidebar sliders** and click **Classify** to see the prediction.")

    col_in, col_out = st.columns([1, 1.4])

    with col_in:
        st.markdown('<div class="section-header">Current Input Values</div>', unsafe_allow_html=True)
        input_df = pd.DataFrame({
            "Feature": ["Sepal Length", "Sepal Width", "Petal Length", "Petal Width"],
            "Value (cm)": [sl, sw, pl, pw],
            "Min": [df[f].min() for f in FEATURES],
            "Max": [df[f].max() for f in FEATURES],
        })
        # Add normalised bar
        input_df["Range %"] = [
            int((v - mn) / (mx - mn) * 100)
            for v, mn, mx in zip(input_df["Value (cm)"], input_df["Min"], input_df["Max"])
        ]
        st.dataframe(
            input_df[["Feature", "Value (cm)", "Range %"]].style.bar(
                subset=["Range %"], color="#667eea", vmin=0, vmax=100
            ).format({"Value (cm)": "{:.1f}", "Range %": "{}%"}),
            use_container_width=True, hide_index=True
        )

        predict_btn = st.button("🔮 Classify", type="primary", use_container_width=True)

    with col_out:
        if predict_btn or 'last_prediction' in st.session_state:
            if predict_btn:
                result = get_prediction(
                    sepal_length=sl, sepal_width=sw,
                    petal_length=pl, petal_width=pw
                )
                st.session_state['last_prediction'] = result
            else:
                result = st.session_state['last_prediction']

            sid  = result['species_id']
            name = result['species_name']
            emoji = result['emoji']
            proba = result['probabilities']

            # Result card
            card_color = {"setosa": "#e91e8c", "versicolor": "#00b4a0", "virginica": "#0096cc"}[name]
            st.markdown(f"""
            <div class="pred-card" style="background: linear-gradient(135deg, {card_color}cc, {card_color}88);">
                <h1>{emoji}</h1>
                <h3>Predicted Species</h3>
                <h2 style="color:white; margin:4px 0; font-size:2rem;">Iris {name.capitalize()}</h2>
            </div>
            """, unsafe_allow_html=True)

            # Probability bars
            st.markdown('<div class="section-header">Prediction Confidence</div>', unsafe_allow_html=True)
            for sp, prob in proba.items():
                col_lbl, col_bar = st.columns([1, 3])
                with col_lbl:
                    st.markdown(f"**{SPECIES_EMOJI[list(SPECIES_MAP.values()).index(sp)]} {sp}**")
                with col_bar:
                    color = PALETTE[sp]
                    bar_w = int(prob * 100)
                    st.markdown(f"""
                    <div style="background:#eee; border-radius:8px; height:22px; margin-top:4px;">
                        <div style="background:{color}; width:{bar_w}%; border-radius:8px;
                                    height:22px; display:flex; align-items:center;
                                    padding-left:8px; color:white; font-size:0.85rem; font-weight:600;">
                            {prob*100:.1f}%
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    st.markdown("")
        else:
            st.info("👈 Adjust the sliders in the sidebar and click **Classify**.")

    st.divider()

    # ── Where does my flower sit? scatter
    st.markdown('<div class="section-header">Where Does Your Flower Sit?</div>', unsafe_allow_html=True)
    col_ax1, col_ax2 = st.columns(2)
    with col_ax1:
        live_x = st.selectbox("X-axis (live chart):", FEATURES, index=2, key='lx')
    with col_ax2:
        live_y = st.selectbox("Y-axis (live chart):", FEATURES, index=3, key='ly')

    live_vals = {
        'sepal_length': sl, 'sepal_width': sw,
        'petal_length': pl, 'petal_width': pw
    }
    fig_live = px.scatter(
        df, x=live_x, y=live_y,
        color='species_name', color_discrete_map=PALETTE,
        symbol='species_name', opacity=0.55,
        labels={
            live_x: live_x.replace('_', ' ').title() + ' (cm)',
            live_y: live_y.replace('_', ' ').title() + ' (cm)',
            'species_name': 'Species'
        }
    )
    # Plot user point
    fig_live.add_trace(go.Scatter(
        x=[live_vals[live_x]], y=[live_vals[live_y]],
        mode='markers+text',
        marker=dict(size=18, color='gold', symbol='star',
                    line=dict(color='black', width=2)),
        text=["⭐ Your flower"],
        textposition="top center",
        name="Your Input"
    ))
    fig_live.update_layout(
        height=400, plot_bgcolor='white', paper_bgcolor='white',
        legend=dict(orientation="h", yanchor="bottom", y=-0.25),
        xaxis=dict(gridcolor='#f0f0f0'), yaxis=dict(gridcolor='#f0f0f0')
    )
    st.plotly_chart(fig_live, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center; color:#888; font-size:0.85rem;'>"
    "🌸 Iris Species Classifier · Built with Streamlit · NUS-ISS DSSI Workshop II"
    "</div>",
    unsafe_allow_html=True
)
