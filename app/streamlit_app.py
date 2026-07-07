"""Streamlit simulation dashboard for the Spectral Instability Model.

Launch: streamlit run app/streamlit_app.py

Pages:
  1. Dashboard — metrics, rankings, world map
  2. World Map — choropleth + 3D globe
  3. Network Communities — spectral clustering and Fiedler analysis
  4. Shock Simulator — factor/edge/cascade simulation
  5. Spectral Diagnostics — scree plots, eigenvalue spectra, heatmaps
  6. Spectral Embedding — 3D country network visualization
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "data" / "output"

# ── Page Config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Spectral Instability Model",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .stMetric { background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
                border-radius: 12px; padding: 16px; border: 1px solid rgba(255,255,255,0.1); }
    h1 { background: linear-gradient(90deg, #e94560, #0f3460);
         -webkit-background-clip: text; -webkit-text-fill-color: transparent;
         font-weight: 700; }
    .highlight-red { color: #e94560; font-weight: 600; }
    .highlight-green { color: #2ecc71; font-weight: 600; }
    .explainer-box {
        background: rgba(255,255,255,0.03);
        border-left: 3px solid #0f3460;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 8px 0 16px 0;
        font-size: 0.88rem;
        color: rgba(255,255,255,0.72);
        line-height: 1.6;
    }
    .term-pill {
        display: inline-block;
        background: rgba(14,52,96,0.6);
        border: 1px solid rgba(14,52,96,0.9);
        border-radius: 4px;
        padding: 1px 7px;
        font-size: 0.82rem;
        font-weight: 600;
        color: #7eb8f7;
        margin: 0 2px;
    }
</style>
""", unsafe_allow_html=True)


# ── Country Name Mapping ──────────────────────────────────────────────────────

COUNTRY_NAMES: dict[str, str] = {
    "USA": "United States", "CHN": "China", "DEU": "Germany",
    "JPN": "Japan", "IND": "India", "GBR": "United Kingdom",
    "FRA": "France", "ITA": "Italy", "CAN": "Canada",
    "RUS": "Russian Federation", "BRA": "Brazil", "KOR": "Korea, Republic of",
    "MEX": "Mexico", "AUS": "Australia", "ESP": "Spain",
    "IDN": "Indonesia", "TUR": "Türkiye", "SAU": "Saudi Arabia",
    "NLD": "Netherlands", "CHE": "Switzerland", "POL": "Poland",
    "TWN": "Taiwan", "BEL": "Belgium", "ARG": "Argentina",
    "IRL": "Ireland", "SWE": "Sweden", "SGP": "Singapore",
    "ARE": "United Arab Emirates", "ISR": "Israel", "AUT": "Austria",
    "THA": "Thailand", "NOR": "Norway", "PHL": "Philippines",
    "VNM": "Viet Nam", "BGD": "Bangladesh", "DNK": "Denmark",
    "MYS": "Malaysia", "COL": "Colombia", "IRN": "Iran",
    "HKG": "Hong Kong", "ZAF": "South Africa", "EGY": "Egypt",
    "ROU": "Romania", "PAK": "Pakistan", "CZE": "Czechia",
    "CHL": "Chile", "PRT": "Portugal", "FIN": "Finland",
    "PER": "Peru", "KAZ": "Kazakhstan", "IRQ": "Iraq",
    "DZA": "Algeria", "NZL": "New Zealand", "GRC": "Greece",
    "NGA": "Nigeria", "HUN": "Hungary", "QAT": "Qatar",
    "UKR": "Ukraine", "KWT": "Kuwait", "MAR": "Morocco",
    "ETH": "Ethiopia", "SVK": "Slovakia", "PRI": "Puerto Rico",
    "DOM": "Dominican Republic", "ECU": "Ecuador", "UZB": "Uzbekistan",
    "VEN": "Venezuela", "AGO": "Angola", "KEN": "Kenya",
    "BGR": "Bulgaria", "GTM": "Guatemala", "OMN": "Oman",
    "LKA": "Sri Lanka", "CRI": "Costa Rica", "LUX": "Luxembourg",
    "HRV": "Croatia", "SRB": "Serbia", "CIV": "Côte d'Ivoire",
    "PAN": "Panama", "LTU": "Lithuania", "GHA": "Ghana",
    "URY": "Uruguay", "TZA": "Tanzania", "BLR": "Belarus",
    "MMR": "Myanmar", "COD": "DR Congo", "AZE": "Azerbaijan",
    "SVN": "Slovenia", "TKM": "Turkmenistan", "JOR": "Jordan",
    "UGA": "Uganda", "BOL": "Bolivia", "CMR": "Cameroon",
    "TUN": "Tunisia", "MAC": "Macao", "LBY": "Libya",
    "ZWE": "Zimbabwe", "BHR": "Bahrain", "KHM": "Cambodia",
    "PRY": "Paraguay", "LVA": "Latvia", "EST": "Estonia",
    "NPL": "Nepal", "CYP": "Cyprus", "HND": "Honduras",
    "SLV": "El Salvador", "GEO": "Georgia", "ISL": "Iceland",
    "SEN": "Senegal", "PNG": "Papua New Guinea", "BIH": "Bosnia & Herzegovina",
    "LBN": "Lebanon", "SDN": "Sudan", "ALB": "Albania",
    "MLI": "Mali", "ARM": "Armenia", "TTO": "Trinidad & Tobago",
    "ZMB": "Zambia", "HTI": "Haiti", "MLT": "Malta",
    "GUY": "Guyana", "GIN": "Guinea", "MNG": "Mongolia",
    "BFA": "Burkina Faso", "MOZ": "Mozambique",
}


def country_label(iso3: str) -> str:
    """Return 'Full Name (ISO3)' for a given ISO3 code."""
    return f"{COUNTRY_NAMES.get(iso3, iso3)} ({iso3})"


def enrich_with_names(df: pd.DataFrame, iso_col: str = "iso3") -> pd.DataFrame:
    """Add a 'country_name' column and a 'label' column to any DataFrame that has an iso3 column."""
    df = df.copy()
    df["country_name"] = df[iso_col].map(COUNTRY_NAMES).fillna(df[iso_col])
    df["label"] = df[iso_col] + " · " + df["country_name"]
    return df


def explainer(text: str):
    """Render a styled plain-English explanation box."""
    st.markdown(f'<div class="explainer-box">{text}</div>', unsafe_allow_html=True)


# ── Data Loading ──────────────────────────────────────────────────────────────

@st.cache_data
def load_model_outputs():
    """Load all pre-computed model outputs."""
    data = {}

    loaders = {
        "scores":       ("composite_scores.parquet", pd.read_parquet),
        "steady_state": ("steady_state.parquet",     pd.read_parquet),
        "centrality_out": ("centrality_out.parquet", pd.read_parquet),
        "centrality_in":  ("centrality_in.parquet",  pd.read_parquet),
        "W":            ("coupling_matrix.npy",      lambda p: np.load(p)),
        "pca_eigenvalues": ("eigenvalues.npy",       lambda p: np.load(p)),
        "W_eigenvalues":   ("W_eigenvalues.npy",     lambda p: np.load(p, allow_pickle=True)),
        "metadata":     ("model_metadata.parquet",   lambda p: pd.read_parquet(p).iloc[0].to_dict()),
        "countries":    ("countries_order.csv",      lambda p: pd.read_csv(p)["iso3"].tolist()),
        "communities":  ("communities.parquet",      pd.read_parquet),
        "fiedler":      ("fiedler_vector.parquet",   pd.read_parquet),
        "pagerank":     ("pagerank.parquet",          pd.read_parquet),
        "systemic_risk":("systemic_risk.parquet",    pd.read_parquet),
        "laplacian_eigenvalues": ("laplacian_eigenvalues.npy", lambda p: np.load(p)),
        "scores_ci":    ("composite_scores_ci.parquet", lambda p: pd.read_parquet(p) if Path(p).exists() else None),
        "pagerank_ci":  ("pagerank_ci.parquet",       lambda p: pd.read_parquet(p) if Path(p).exists() else None),
        "alpha_ci":     ("alpha_ci.npy",              lambda p: np.load(p) if Path(p).exists() else None),
        "pca_eigenvalues_ci": ("pca_eigenvalues_ci.npy", lambda p: np.load(p) if Path(p).exists() else None),
    }

    for key, (filename, loader) in loaders.items():
        p = OUTPUT_DIR / filename
        if p.exists():
            data[key] = loader(p)

    return data


# ── Load Data ─────────────────────────────────────────────────────────────────

data = load_model_outputs()

if not data:
    st.error("⚠️ **No model outputs found.** Run `make all` to generate results first.")
    st.code("cd spectral-instability-model\nmake install\nmake all", language="bash")
    st.stop()

meta = data.get("metadata", {})
n_ind = meta.get("n_indicators", 20)

# ── Sidebar Navigation ────────────────────────────────────────────────────────

st.sidebar.title("🌍 Spectral Instability")
st.sidebar.markdown("---")

pages = [
    "📊 Dashboard",
    "🗺️ World Map",
    "🔗 Network Communities",
    "💥 Shock Simulator",
    "🔬 Spectral Diagnostics",
    "🌐 Spectral Embedding",
]
page = st.sidebar.radio("Navigation", pages)

st.sidebar.markdown("---")
st.sidebar.caption("Spectral Model of Global Regime Instability")
st.sidebar.caption("Top 125 Economies by Nominal GDP")
st.sidebar.markdown(f"""
<small style='color:rgba(255,255,255,0.4)'>
Scores reflect structural fragility inferred from {n_ind} indicators across governance,
economics, security, and civil society — network-propagated via the Friedkin-Johnsen model.
</small>
""", unsafe_allow_html=True)



# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════

if page == "📊 Dashboard":
    st.title("Global Regime Instability — Dashboard")

    explainer(
        "This dashboard summarizes a <b>spectral network model</b> of political regime instability. "
        f"Each of the 125 largest economies is scored on {n_ind} indicators (e.g. governance quality, "
        "inflation, conflict events, debt levels). Those scores are compressed via <b>PCA</b> "
        "(Principal Component Analysis) into a single <em>composite instability score</em> per country. "
        "Countries are then connected to one another via a <b>coupling matrix</b> (based on "
        "geographic proximity, GDP-weighted trade, and political bloc membership) and a "
        "network propagation model is run so that instability in one country can spill over "
        "to its neighbours. The metrics below describe the model structure."
    )

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Countries", meta.get("n_countries", "—"),
                help="Number of countries in the model (top 125 by nominal GDP).")
    col2.metric("Indicators", meta.get("n_indicators", "—"),
                help="Number of input data indicators used across governance, economics, conflict, and civil society.")
    col3.metric("PCA Components", meta.get("n_components", "—"),
                help="Number of principal components retained (Kaiser criterion + ≥80% variance explained).")
    alpha_val = meta.get('alpha', 0)
    alpha_str = f"{alpha_val:.3f}"
    if "alpha_ci" in data and data["alpha_ci"] is not None:
        ci = data["alpha_ci"]
        alpha_str += f" [{ci[0]:.2f}, {ci[1]:.2f}]"
    col4.metric("α (coupling)", alpha_str, "calibrated (1-yr)",
                help="Coupling strength α: how strongly a country's instability is influenced by its network neighbours vs. its own structural baseline. α=0 means fully isolated; α→1 means fully determined by neighbours.")

    col5, col6, col7, col8 = st.columns(4)
    col5.metric("Spectral Radius ρ(W)", f"{meta.get('spectral_radius_W', 0):.4f}",
                help="The largest eigenvalue magnitude of the coupling matrix W. For a row-stochastic W this equals 1. Stability requires α·ρ(W) < 1.")
    col6.metric("Spectral Gap", f"{meta.get('spectral_gap', 0):.4f}",
                help="The gap between the two largest eigenvalue magnitudes of W. A large gap indicates one dominant propagation mode — shocks spread uniformly. A small gap means multiple propagation pathways exist.")
    col7.metric("Communities", meta.get("n_communities", "—"),
                help="Number of instability blocs detected by spectral clustering — groups of countries whose coupling structure makes them likely to experience correlated instability.")
    col8.metric("Modularity", f"{meta.get('modularity', 0):.4f}",
                help="Newman modularity Q ∈ [-1, 1]. Positive values indicate the communities are meaningfully denser internally than you'd expect by chance. Q > 0.3 is generally considered 'good' community structure.")

    st.markdown("---")

    tab1, tab2, tab3 = st.tabs([
        "🏴 Most Unstable", "📡 Systemic Transmitters",
        "🎯 Most Exposed"
    ])

    with tab1:
        explainer(
            "<b>Steady-state instability score</b>: after running the network propagation model to "
            "equilibrium, each country's score reflects both its own structural fragility and the "
            "instability it absorbs from its network neighbours. Countries at the top are the most "
            "fragile <em>once network spillovers are accounted for</em>. "
            "The <em>network amplification</em> column shows how much the network added or subtracted "
            "versus the country's purely structural score."
        )
        if "steady_state" in data:
            df = enrich_with_names(data["steady_state"].head(20))
            fig = px.bar(
                df, x="label", y="steady_state_score",
                color="steady_state_score",
                color_continuous_scale="Reds",
                title="Top 20 Countries by Steady-State Instability Score",
                labels={"steady_state_score": "Instability Score", "label": "Country"},
                hover_name="country_name",
                hover_data={"country_name": False, "iso3": True,
                            "structural_score": ":.3f", "network_amplification": ":.3f"},
            )
            fig.update_layout(template="plotly_dark", height=450,
                              xaxis_tickangle=-35, xaxis_title="")
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "**Chart Interpretation:** This chart ranks the top 20 countries by their steady-state instability score "
                "(computed using network propagation). The x-axis lists the countries, and the y-axis shows their "
                "instability score (unitless z-score, where higher is more fragile). Hovering over a bar reveals "
                "the country's full name, ISO3 code, its baseline 'Structural Score' (PCA score before network spillovers), "
                "and its 'Network Amplification' (how much spillovers from neighbors changed its score)."
            )

            display_df = df[["rank", "iso3", "country_name", "structural_score",
                              "steady_state_score", "network_amplification"]].copy()
            display_df.columns = ["Rank", "ISO3", "Country", "Structural Score",
                                   "Steady-State Score", "Network Amplification"]
            st.dataframe(display_df, width="stretch", hide_index=True)

    with tab2:
        explainer(
            "<b>Systemic transmitters</b> are countries whose instability, if it spikes, would "
            "propagate most strongly to other countries through the coupling network. This is measured "
            "by <b>eigenvector centrality</b> on the raw (non-normalised) coupling matrix — a country "
            "scores highly if it is strongly connected to other highly-connected countries. "
            "Think of it as 'which countries are hubs in the instability transmission network?' "
            "The <b>PageRank</b> column shows an alternative centrality measure (like Google's original "
            "link-ranking algorithm) that accounts for the directionality of influence."
        )
        if "centrality_out" in data:
            df = enrich_with_names(data["centrality_out"].head(20))
            fig = px.bar(
                df, x="label", y="centrality_out",
                color="centrality_out",
                color_continuous_scale="Oranges",
                title="Top 20 Systemic Transmitters (Outbound Eigenvector Centrality)",
                labels={"centrality_out": "Eigenvector Centrality", "label": "Country"},
                hover_name="country_name",
                hover_data={"country_name": False, "iso3": True, "pagerank": ":.5f"},
            )
            fig.update_layout(template="plotly_dark", height=450,
                              xaxis_tickangle=-35, xaxis_title="")
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "**Chart Interpretation:** This chart shows the top 20 countries by Outbound Eigenvector Centrality. "
                "The y-axis represents the centrality score (0 to 1 scale). Outbound centrality measures how strongly a country "
                "transmits its own instability outward to the global system. A country has high outbound centrality if it is strongly "
                "connected (via trade, geographic, or political linkages) to other highly-central countries. Hovering over a bar reveals "
                "the country's full name, ISO3, and its PageRank centrality score."
            )

            display_df = df[["iso3", "country_name", "centrality_out", "pagerank"]].copy()
            display_df.columns = ["ISO3", "Country", "Eigenvector Centrality", "PageRank"]
            st.dataframe(display_df, width="stretch", hide_index=True)

    with tab3:
        explainer(
            "<b>Most exposed countries</b> are those whose steady-state instability level is "
            "most sensitive to instability shocks arriving from other countries in the network. "
            "This is the <em>inbound</em> eigenvector centrality — measured on the transpose of "
            "the coupling matrix, capturing who is on the receiving end of instability flows. "
            "A country can be both a transmitter and a receiver, or specialise in one role."
        )
        if "centrality_in" in data:
            df = enrich_with_names(data["centrality_in"].head(20))
            fig = px.bar(
                df, x="label", y="centrality_in",
                color="centrality_in",
                color_continuous_scale="Purples",
                title="Top 20 Most Exposed Countries (Inbound Eigenvector Centrality)",
                labels={"centrality_in": "Inbound Centrality", "label": "Country"},
                hover_name="country_name",
                hover_data={"country_name": False, "iso3": True},
            )
            fig.update_layout(template="plotly_dark", height=450,
                              xaxis_tickangle=-35, xaxis_title="")
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "**Chart Interpretation:** This chart ranks the top 20 countries by Inbound Eigenvector Centrality. "
                "The y-axis represents the inbound centrality score. Inbound centrality measures a country's exposure to instability "
                "spilling in from the rest of the network. High inbound centrality indicates that the country is highly vulnerable to "
                "external shocks due to its dense dependencies on other central or fragile countries. Hovering over a bar reveals the "
                "country's full name and ISO3 code."
            )

            display_df = df[["iso3", "country_name", "centrality_in"]].copy()
            display_df.columns = ["ISO3", "Country", "Inbound Centrality"]
            st.dataframe(display_df, width="stretch", hide_index=True)




# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: WORLD MAP
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🗺️ World Map":
    st.title("Global Instability — World Map")

    explainer(
        "Each country is shaded according to the selected metric. Hover over a country to see "
        "its full name and exact value. Use the <b>Color by</b> dropdown to switch between "
        "different model outputs, and the <b>Projection</b> selector to change the map style "
        "(try <em>orthographic</em> for a rotatable globe). "
        "Red/dark shading always indicates higher instability or exposure."
    )

    col_a, col_b = st.columns([2, 1])
    with col_a:
        layer = st.selectbox("Color by", [
            "Steady-State Score", "Structural Score", "Network Amplification",
            "PageRank Centrality", "Inbound Centrality", "Outbound Centrality",
            "Fiedler Vector", "Community"
        ])
    with col_b:
        projection = st.selectbox("Projection", [
            "natural earth", "orthographic", "robinson",
            "mercator", "equirectangular"
        ])

    layer_descriptions = {
        "Steady-State Score":
            "Network-propagated instability score at equilibrium. Combines a country's own fragility "
            "with the instability it absorbs from neighbours via the coupling matrix.",
        "Structural Score":
            f"A country's raw instability score from PCA on {n_ind} indicators alone, before any network "
            "propagation. This is the 'own fragility' baseline.",
        "Network Amplification":
            "Steady-state minus structural score. Positive (red) = the network amplifies this country's "
            "instability; negative (blue) = the network has a stabilising effect (neighbours are stable).",
        "PageRank Centrality":
            "PageRank measures a country's systemic importance in the transmission network. "
            "Like the original web PageRank, a country scores highly if it receives many links "
            "from other high-scoring countries. Major economic hubs (USA, DEU, CHN) dominate.",
        "Outbound Centrality":
            "Eigenvector centrality on the raw coupling matrix — how much instability a country "
            "could push outward to the rest of the network.",
        "Inbound Centrality":
            "How exposed a country is to instability flowing in from the rest of the network.",

        "Fiedler Vector":
            "The Fiedler vector is the eigenvector corresponding to the second-smallest Laplacian "
            "eigenvalue. Countries with similar values cluster together structurally. "
            "Positive (red) vs. negative (blue) indicates the natural binary partition of the network.",
        "Community":
            "Discrete community membership from spectral clustering — countries in the same community "
            "are more tightly coupled and likely to experience correlated instability dynamics.",
    }

    if layer in layer_descriptions:
        explainer(f"<b>{layer}</b>: {layer_descriptions[layer]}")

    # Build map data
    map_df, color_col, color_scale = None, None, "RdYlGn_r"

    if layer == "Steady-State Score" and "steady_state" in data:
        map_df = enrich_with_names(data["steady_state"])
        color_col = "steady_state_score"
    elif layer == "Structural Score" and "steady_state" in data:
        map_df = enrich_with_names(data["steady_state"])
        color_col = "structural_score"
    elif layer == "Network Amplification" and "steady_state" in data:
        map_df = enrich_with_names(data["steady_state"])
        color_col = "network_amplification"
        color_scale = "RdBu_r"
    elif layer == "PageRank Centrality" and "pagerank" in data:
        map_df = enrich_with_names(data["pagerank"])
        color_col = "pagerank"
        color_scale = "Viridis"
    elif layer == "Outbound Centrality" and "centrality_out" in data:
        map_df = enrich_with_names(data["centrality_out"])
        color_col = "centrality_out"
        color_scale = "Oranges"
    elif layer == "Inbound Centrality" and "centrality_in" in data:
        map_df = enrich_with_names(data["centrality_in"])
        color_col = "centrality_in"
        color_scale = "Purples"

    elif layer == "Fiedler Vector" and "fiedler" in data:
        map_df = enrich_with_names(data["fiedler"])
        color_col = "fiedler_vector"
        color_scale = "RdBu"
    elif layer == "Community" and "communities" in data:
        map_df = enrich_with_names(data["communities"])
        map_df["cluster"] = map_df["cluster"].astype(str)
        color_col = "cluster"
        color_scale = None

    if map_df is not None and color_col is not None:
        # Avoid duplicating country_name in hover text since it's the header, and explicitly show iso3
        hover_data = {"iso3": True, "country_name": False}
        for c in [color_col, "rank", "structural_score", "steady_state_score"]:
            if c in map_df.columns and c != color_col:
                hover_data[c] = True

        if color_col == "cluster":
            fig = px.choropleth(
                map_df, locations="iso3", locationmode="ISO-3",
                color=color_col,
                color_discrete_sequence=px.colors.qualitative.Set1,
                projection=projection,
                title=f"World Map: {layer}",
                hover_name="country_name",
                hover_data=hover_data,
            )
        else:
            fig = px.choropleth(
                map_df, locations="iso3", locationmode="ISO-3",
                color=color_col,
                color_continuous_scale=color_scale,
                projection=projection,
                title=f"World Map: {layer}",
                hover_name="country_name",
                hover_data=hover_data,
            )
        fig.update_layout(
            template="plotly_dark", height=700,
            margin=dict(l=0, r=0, t=40, b=0),
            geo=dict(
                showocean=True, oceancolor="rgb(20,30,50)",
                showland=True, landcolor="rgb(40,50,70)",
                showlakes=True, lakecolor="rgb(20,30,50)",
                showcountries=True, countrycolor="rgb(80,90,110)",
                showframe=False,
            ),
        )
        st.plotly_chart(fig, width="stretch")
        st.caption(
            f"**Map Interpretation:** Shading represents the selected metric (**{layer}**). "
            f"Hover over any country to see its **full country name**, ISO3 code, and associated values. "
            f"Red/darker colors indicate higher instability, greater exposure, or higher centrality depending on the layer selected."
        )
    else:
        st.warning(f"Data for '{layer}' not available. Run the model first.")


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: NETWORK COMMUNITIES
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🔗 Network Communities":
    st.title("Network Communities & Fiedler Partition")

    explainer(
        "The coupling matrix W describes how each country transmits instability to others. "
        "This page uses <b>spectral graph theory</b> to uncover the hidden community structure "
        "of that network — which countries are tightly coupled and would likely experience "
        "correlated instability events — and to measure how resilient the network is to shocks."
    )

    tab1, tab2, tab3 = st.tabs(["🏘️ Instability Blocs", "〰️ Fiedler Partition", "📈 Network Metrics"])

    with tab1:
        explainer(
            "<b>Spectral clustering</b> finds groups of countries (blocs) that are more tightly "
            "coupled to each other than to the rest of the world. The algorithm works by computing "
            "the eigenvectors of the <em>normalised Laplacian</em> of the coupling matrix, "
            "projecting countries into that eigenvector space, and then running k-means. "
            "The number of blocs k is chosen automatically via the <em>eigengap heuristic</em> — "
            "we look for the largest 'jump' in the eigenvalue sequence, which signals the natural "
            "number of clusters. "
            f"<b>Modularity Q = {meta.get('modularity', 0):.3f}</b>: values above 0.3 indicate "
            "meaningful community structure (not random)."
        )
        if "communities" in data and "countries" in data:
            comm_df = enrich_with_names(data["communities"])

            fig = px.choropleth(
                comm_df, locations="iso3", locationmode="ISO-3",
                color="cluster",
                color_discrete_sequence=px.colors.qualitative.Set1,
                projection="natural earth",
                title=f"Instability Blocs — {meta.get('n_communities', '?')} Communities "
                      f"(Modularity Q = {meta.get('modularity', 0):.3f})",
                hover_name="country_name",
                hover_data={"iso3": True, "cluster": True, "country_name": False},
            )
            fig.update_layout(
                template="plotly_dark", height=600,
                margin=dict(l=0, r=0, t=40, b=0),
                geo=dict(showocean=True, oceancolor="rgb(20,30,50)",
                         showland=True, landcolor="rgb(40,50,70)",
                         showcountries=True, countrycolor="rgb(80,90,110)"),
            )
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "**Map Interpretation:** Countries are colored by their detected community (instability bloc). "
                "These communities are clusters of countries that are tightly linked to each other via trade, geography, and "
                "political alliances. Modularity (Q) measures the strength of the community structure: positive values close to 0.3 "
                "or higher indicate strong, non-random clustering. Hovering over a country shows its full name and community ID."
            )

            for cid in sorted(comm_df["cluster"].unique()):
                members_df = comm_df[comm_df["cluster"] == cid]
                member_names = [f"{row['country_name']} ({row['iso3']})"
                                for _, row in members_df.iterrows()]
                with st.expander(f"**Community {cid}** — {len(member_names)} countries"):
                    st.write(", ".join(member_names))

    with tab2:
        explainer(
            "The <b>Fiedler vector</b> (named after mathematician Miroslav Fiedler) is the "
            "eigenvector corresponding to the <em>second-smallest eigenvalue</em> of the graph "
            "Laplacian L = D − W. It is the most informative single eigenvector for understanding "
            "network structure: its sign naturally divides the network into two halves that are "
            "most weakly connected to each other. Countries with similar Fiedler values have "
            "similar structural roles in the network, even if they are geographically distant. "
            "<br><br>"
            f"<b>Algebraic connectivity λ₂ = {meta.get('algebraic_connectivity', 0):.4f}</b>: "
            "This is the Fiedler eigenvalue itself. Higher = more robustly connected; if λ₂ → 0 "
            "the network is near a fragmentation point. "
            f"<b>Diffusion timescale τ = {meta.get('diffusion_timescale', 0):.1f} steps</b>: "
            "how many propagation steps are needed for a shock to traverse the full network."
        )
        if "fiedler" in data:
            fiedler_df = enrich_with_names(data["fiedler"])
            fiedler_sorted = fiedler_df.sort_values("fiedler_vector")

            fig = px.bar(
                fiedler_sorted, x="label", y="fiedler_vector",
                color="fiedler_vector",
                color_continuous_scale="RdBu",
                title="Fiedler Vector — Countries Ordered by Network Partition Value",
                labels={"fiedler_vector": "Fiedler Component (v₂)",
                        "label": "Country"},
                hover_name="country_name",
                hover_data={"iso3": True, "fiedler_vector": ":.4f", "country_name": False},
            )
            fig.add_hline(y=0, line_dash="dash", line_color="rgba(255,255,255,0.4)",
                          annotation_text="Partition boundary (sign = 0)")
            fig.update_layout(template="plotly_dark", height=500,
                              xaxis_tickangle=-90, xaxis_title="",
                              xaxis_showticklabels=False)
            st.plotly_chart(fig, width="stretch")

            st.caption(
                "**Chart Interpretation:** This bar chart orders all 125 countries by their value in the Fiedler vector (v₂). "
                "The Fiedler vector is the eigenvector associated with the second-smallest eigenvalue of the graph Laplacian. "
                "Its sign naturally splits the network into two most weakly connected halves (negative/blue vs. positive/red), "
                "revealing the primary geopolitical fault line. Hovering over any bar reveals the country's full name, ISO3 code, and Fiedler value."
            )

            fig2 = px.choropleth(
                fiedler_df, locations="iso3", locationmode="ISO-3",
                color="fiedler_vector",
                color_continuous_scale="RdBu",
                projection="natural earth",
                title="Fiedler Vector — Geospatial View (Red = positive bloc, Blue = negative bloc)",
                hover_name="country_name",
                hover_data={"iso3": True, "fiedler_vector": ":.4f", "country_name": False},
            )
            fig2.update_layout(
                template="plotly_dark", height=600,
                margin=dict(l=0, r=0, t=40, b=0),
                geo=dict(showocean=True, oceancolor="rgb(20,30,50)",
                         showland=True, landcolor="rgb(40,50,70)"),
            )
            st.plotly_chart(fig2, width="stretch")
            st.caption(
                "**Map Interpretation:** Geospatial distribution of Fiedler vector values. Red states are on one side of the "
                "primary network partition, and blue states are on the other. Hovering over a country displays its full name, "
                "ISO3 code, and Fiedler value."
            )

    with tab3:
        explainer(
            "<b>PageRank</b> (left chart): adapted from the original Google search algorithm. "
            "A country receives a high PageRank if it is connected to many other high-PageRank "
            "countries in the coupling network. In this model, PageRank measures which countries "
            "are the most central <em>hubs</em> for instability transmission — major economies "
            "with dense trade and political linkages dominate. "
            "<br><br>"
            "<b>Betweenness centrality</b> (right chart): measures how often a country sits on "
            "the <em>shortest path</em> between two other countries in the network. High betweenness "
            "= a 'bridge' or 'bottleneck' country. If a bridge country destabilises, it can "
            "sever the network's connectivity and isolate entire blocs."
        )
        if "pagerank" in data:
            pr_df = enrich_with_names(data["pagerank"])
            col1, col2 = st.columns(2)

            with col1:
                top_pr = pr_df.head(20)
                if "pagerank_ci" in data and data["pagerank_ci"] is not None:
                    pr_ci = data["pagerank_ci"]
                    top_pr = top_pr.merge(pr_ci, on="iso3", how="left")
                    top_pr["error_plus"] = top_pr["pagerank_ci_upper"] - top_pr["pagerank_base"]
                    top_pr["error_minus"] = top_pr["pagerank_base"] - top_pr["pagerank_ci_lower"]
                    fig = px.bar(
                        top_pr, x="label", y="pagerank",
                        color="pagerank", color_continuous_scale="Viridis",
                        title="Top 20 by PageRank (Network Hubs)",
                        labels={"pagerank": "PageRank Score", "label": "Country"},
                        hover_name="country_name",
                        hover_data={"iso3": True, "pagerank": ":.5f"},
                        error_y="error_plus", error_y_minus="error_minus"
                    )
                else:
                    fig = px.bar(
                        top_pr, x="label", y="pagerank",
                        color="pagerank", color_continuous_scale="Viridis",
                        title="Top 20 by PageRank (Network Hubs)",
                        labels={"pagerank": "PageRank Score", "label": "Country"},
                        hover_name="country_name",
                        hover_data={"iso3": True, "pagerank": ":.5f"},
                    )
                fig.update_layout(template="plotly_dark", height=420,
                                  xaxis_tickangle=-45, xaxis_title="")
                st.plotly_chart(fig, width="stretch")

            with col2:
                top_bw = pr_df.sort_values("betweenness", ascending=False).head(20)
                top_bw = enrich_with_names(top_bw)
                fig = px.bar(
                    top_bw, x="label", y="betweenness",
                    color="betweenness", color_continuous_scale="Inferno",
                    title="Top 20 by Betweenness Centrality (Network Bridges)",
                    labels={"betweenness": "Betweenness (normalised)", "label": "Country"},
                    hover_name="country_name",
                    hover_data={"iso3": True, "betweenness": ":.4f"},
                )
                fig.update_layout(template="plotly_dark", height=420,
                                  xaxis_tickangle=-45, xaxis_title="")
                st.plotly_chart(fig, width="stretch")

            # Full table
            display_df = pr_df[["iso3", "country_name", "pagerank", "betweenness"]].copy()
            display_df.columns = ["ISO3", "Country", "PageRank", "Betweenness"]
            st.dataframe(display_df, width="stretch", hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SHOCK SIMULATOR
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "💥 Shock Simulator":
    st.title("Shock & Cascade Simulator")

    explainer(
        "This page lets you simulate <b>what happens to the global instability network</b> "
        "when a country is hit by a shock. The model propagates the shock through the coupling "
        "matrix using the Friedkin-Johnsen equation: "
        "<code>x(t+1) = α·W·x(t) + (1−α)·s</code>, "
        "where x(t) is the instability vector at time t, W is the coupling matrix, s is each "
        "country's structural baseline, and α controls how strongly neighbours influence each other. "
        "The system converges to a new equilibrium (steady state) after the shock is applied."
    )

    if "W" not in data or "steady_state" not in data:
        st.warning("Model outputs required. Run `make model` first.")
        st.stop()

    countries = data.get("countries", [])
    country_options = [country_label(c) for c in countries]
    W = data["W"]
    ss_df = data["steady_state"]

    sim_type = st.radio(
        "Simulation Type",
        ["Factor Shock", "Cascade Analysis", "Edge Shock"],
        horizontal=True,
    )
    st.markdown("---")

    if sim_type == "Factor Shock":
        explainer(
            "<b>Factor Shock</b>: a sudden increase in a country's underlying fragility "
            "(e.g. a coup, currency collapse, or major conflict eruption). The shock raises "
            "that country's structural instability score by the specified number of standard "
            "deviations (σ). The model then iterates forward in time, showing how the instability "
            "propagates to the country's network neighbours. The trajectory plot shows instability "
            "over time steps for the shocked country and its top-5 most affected partners. "
            "The impact map shows the Δ instability across all countries at the new equilibrium."
        )
        col1, col2 = st.columns(2)
        with col1:
            selected_label = st.selectbox("Select Country to Shock", country_options)
            selected_iso = countries[country_options.index(selected_label)]
        with col2:
            magnitude = st.slider("Shock Magnitude (σ standard deviations)", 0.0, 4.0, 2.0, 0.1,
                                  help="1σ = a significant but not extreme shock; 2σ = a severe crisis; 4σ = a catastrophic event (e.g. state collapse).")

        n_steps = st.slider("Propagation Steps", 10, 100, 50,
                            help="Number of time periods to simulate. The model typically converges within 20-30 steps.")

        if st.button("🚀 Run Factor Shock", type="primary"):
            from model.dynamics import factor_shock

            country_idx = countries.index(selected_iso)
            composite = ss_df["structural_score"].values

            with st.spinner("Running simulation..."):
                result = factor_shock(W, composite, meta.get("alpha", 0.4),
                                      country_idx, magnitude, n_steps)

            traj = result["trajectory"]
            top_partners = result["top10_partners"]
            plot_indices = [country_idx] + list(top_partners[:5])

            fig = go.Figure()
            for idx in plot_indices:
                iso = countries[idx] if idx < len(countries) else f"C{idx}"
                name_full = COUNTRY_NAMES.get(iso, iso)
                is_shocked = (idx == country_idx)
                fig.add_trace(go.Scatter(
                    x=list(range(len(traj))),
                    y=traj[:, idx],
                    name=f"{'⚡ ' if is_shocked else ''}{name_full} ({iso})",
                    mode="lines",
                    line={"width": 3 if is_shocked else 1.5,
                          "dash": "solid" if is_shocked else "dot"},
                ))
            fig.add_vline(x=0, line_dash="dash", line_color="rgba(255,80,80,0.5)",
                          annotation_text="Shock applied")
            fig.update_layout(
                template="plotly_dark",
                title=f"Instability Trajectory: {magnitude}σ Shock to {COUNTRY_NAMES.get(selected_iso, selected_iso)} ({selected_iso})",
                xaxis_title="Time Step (each step = one propagation period)",
                yaxis_title="Instability Score",
                height=480,
                legend_title="Country",
            )
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "⚡ = shocked country (thick line). Dotted lines = top-5 most affected partners. "
                f"Convergence step: {result['convergence_step']} of {n_steps}."
            )

            # Impact map
            delta = result["delta_ss"]
            delta_df = enrich_with_names(pd.DataFrame({"iso3": countries, "delta_instability": delta}))
            fig_map = px.choropleth(
                delta_df, locations="iso3", locationmode="ISO-3",
                color="delta_instability",
                color_continuous_scale="RdBu_r",
                projection="natural earth",
                title=f"Impact Map: Change in Equilibrium Instability from {magnitude}σ Shock to {COUNTRY_NAMES.get(selected_iso, selected_iso)}",
                hover_name="country_name",
                hover_data={"iso3": True, "delta_instability": ":.4f"},
            )
            fig_map.update_layout(
                template="plotly_dark", height=520,
                margin=dict(l=0, r=0, t=40, b=0),
                geo=dict(showocean=True, oceancolor="rgb(20,30,50)",
                         showland=True, landcolor="rgb(40,50,70)"),
                coloraxis_colorbar_title="Δ Instability",
            )
            st.plotly_chart(fig_map, width="stretch")
            st.caption("Red = instability increased (harm); Blue = instability decreased. "
                       "Countries not in the model are shown in neutral grey.")

            # Table of top impacts
            top_delta = delta_df.nlargest(10, "delta_instability")[["iso3", "country_name", "delta_instability"]]
            top_delta.columns = ["ISO3", "Country", "Δ Instability"]
            st.markdown("**Top 10 most impacted countries:**")
            st.dataframe(top_delta, width="stretch", hide_index=True)

    elif sim_type == "Cascade Analysis":
        explainer(
            "<b>Cascade Analysis</b>: tests whether a shock to one country triggers a "
            "<em>cascade</em> — a chain reaction where the initial shock pushes neighbours "
            "above a crisis threshold, and those neighbours in turn push their neighbours over. "
            "The <em>crisis threshold</em> is set as a percentile of the current instability "
            "distribution (e.g. 90th percentile = only the top 10% most fragile states are "
            "already in 'crisis'). The model reports cascade breadth (% of countries affected), "
            "depth (how many cascade 'generations' occur), and the amplification ratio."
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            shock_label = st.selectbox("Shock Country", country_options)
            shock_iso = countries[country_options.index(shock_label)]
        with col2:
            shock_mag = st.slider("Shock (σ)", 0.5, 5.0, 2.0, 0.5)
        with col3:
            threshold_pct = st.slider("Crisis Threshold (percentile)", 70, 99, 90,
                                      help="Countries above this instability percentile are considered 'in crisis'. Lower = more countries vulnerable.")

        if st.button("💥 Run Cascade", type="primary"):
            from model.cascade import cascade_analysis

            country_idx = countries.index(shock_iso)
            composite = ss_df["structural_score"].values
            threshold = np.percentile(composite, threshold_pct)

            with st.spinner("Running cascade analysis..."):
                result = cascade_analysis(W, composite, meta.get("alpha", 0.4),
                                          [country_idx], shock_mag, threshold)

            col_a, col_b, col_c = st.columns(3)
            col_a.metric("Cascade Breadth", f"{result['cascade_breadth']:.1%}",
                         help="Fraction of all 125 countries whose instability increased above the crisis threshold.")
            col_b.metric("Cascade Depth", result["cascade_depth"],
                         help="Number of 'generations' of propagation before the cascade stopped.")
            col_c.metric("Amplification Ratio", f"{result['amplification_ratio']:.3f}",
                         help="Total network-wide instability increase divided by the original shock size. >1 = network amplifies the shock.")

            delta = result["delta"]
            impact_df = enrich_with_names(pd.DataFrame({
                "iso3": countries,
                "delta": delta,
                "status": ["⚡ Shocked" if i == country_idx
                           else "🔴 Cascaded" if i in result["newly_crossed"]
                           else "🟢 Stable" for i in range(len(countries))],
            }))

            fig = px.choropleth(
                impact_df, locations="iso3", locationmode="ISO-3",
                color="delta", color_continuous_scale="RdYlGn_r",
                projection="natural earth",
                title=f"Cascade Map: {shock_mag}σ Shock to {COUNTRY_NAMES.get(shock_iso, shock_iso)}",
                hover_name="country_name",
                hover_data={"iso3": True, "status": True, "delta": ":.4f"},
            )
            fig.update_layout(
                template="plotly_dark", height=600,
                margin=dict(l=0, r=0, t=40, b=0),
                geo=dict(showocean=True, oceancolor="rgb(20,30,50)",
                         showland=True, landcolor="rgb(40,50,70)"),
            )
            st.plotly_chart(fig, width="stretch")

            if result["newly_crossed"]:
                st.markdown("**Countries that crossed the crisis threshold:**")
                crossed_df = pd.DataFrame([{
                    "ISO3": countries[i],
                    "Country": COUNTRY_NAMES.get(countries[i], countries[i]),
                    "Δ Instability": f"{delta[i]:+.4f}",
                } for i in result["newly_crossed"]])
                st.dataframe(crossed_df, width="stretch", hide_index=True)
            else:
                st.success("No countries crossed the crisis threshold — the shock was contained.")

    elif sim_type == "Edge Shock":
        explainer(
            "<b>Edge Shock</b>: models a disruption to the <em>link</em> between two specific countries, "
            "rather than to a country's internal state. For example, severing trade (multiplier=0), "
            "doubling financial exposure (multiplier=2), or exploring the effect of an alliance breakup. "
            "The model recomputes the coupling matrix with the modified edge and shows how the new "
            "equilibrium differs from the baseline across all 125 countries."
        )
        col1, col2, col3 = st.columns(3)
        with col1:
            label_a = st.selectbox("Country A", country_options)
            iso_a = countries[country_options.index(label_a)]
        with col2:
            label_b = st.selectbox("Country B", country_options,
                                   index=min(1, len(country_options) - 1))
            iso_b = countries[country_options.index(label_b)]
        with col3:
            multiplier = st.slider(
                "Link Multiplier",
                0.0, 3.0, 0.0, 0.1,
                help="0 = completely sever the link; 1 = no change; 2 = double the coupling strength; 3 = triple."
            )

        if st.button("🔗 Apply Edge Shock", type="primary"):
            from model.dynamics import edge_shock

            i = countries.index(iso_a)
            j = countries.index(iso_b)
            composite = ss_df["structural_score"].values

            with st.spinner("Computing new equilibrium..."):
                result = edge_shock(W, composite, meta.get("alpha", 0.4), i, j, multiplier)

            delta = result["delta_ss"]
            delta_df = enrich_with_names(pd.DataFrame({"iso3": countries, "delta": delta}))

            col_x, col_y = st.columns(2)
            col_x.metric(
                "Spectral Radius Change",
                f"{result['new_spectrum']['spectral_radius']:.4f}",
                delta=f"{result['new_spectrum']['spectral_radius'] - result['orig_spectrum']['spectral_radius']:+.4f}",
                help="Change in the dominant eigenvalue of W — measures overall network stability.",
            )
            col_y.metric(
                "Max Country Impact",
                f"Δ = {np.abs(delta).max():.4f}",
                delta=f"({COUNTRY_NAMES.get(countries[np.argmax(np.abs(delta))], '?')})",
            )

            fig_map = px.choropleth(
                delta_df, locations="iso3", locationmode="ISO-3",
                color="delta",
                color_continuous_scale="RdBu_r",
                projection="natural earth",
                title=f"Impact of Modifying {COUNTRY_NAMES.get(iso_a, iso_a)} ↔ {COUNTRY_NAMES.get(iso_b, iso_b)} link (×{multiplier})",
                hover_name="country_name",
                hover_data={"iso3": True, "delta": ":.4f"},
            )
            fig_map.update_layout(
                template="plotly_dark", height=500,
                margin=dict(l=0, r=0, t=40, b=0),
                geo=dict(showocean=True, oceancolor="rgb(20,30,50)",
                         showland=True, landcolor="rgb(40,50,70)"),
            )
            st.plotly_chart(fig_map, width="stretch")

            top_affected = delta_df.reindex(delta_df["delta"].abs().sort_values(ascending=False).index).head(15)
            display = top_affected[["iso3", "country_name", "delta"]].copy()
            display.columns = ["ISO3", "Country", "Δ Equilibrium Instability"]
            st.markdown("**Top 15 most affected countries:**")
            st.dataframe(display, width="stretch", hide_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SPECTRAL DIAGNOSTICS
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🔬 Spectral Diagnostics":
    st.title("Spectral Diagnostics")

    explainer(
        "This page exposes the mathematical 'internals' of the model — the eigenvalue spectra "
        "that determine how instability compresses, propagates, and clusters. Understanding "
        "eigenvalues helps assess whether the model is well-specified and whether the network "
        "has meaningful structure."
    )

    tab1, tab2, tab3, tab4 = st.tabs([
        "📉 PCA Scree Plot", "🌀 Coupling Eigenvalues",
        "📐 Laplacian Spectrum", "🔥 Coupling Heatmap"
    ])

    with tab1:
        explainer(
            "<b>PCA Scree Plot</b>: Principal Component Analysis decomposes the 19-indicator matrix "
            "into orthogonal 'directions of variance' (principal components). Each bar shows the "
            "<b>eigenvalue</b> of a component — how much variance it explains. "
            "The <b>Kaiser threshold (eigenvalue > 1)</b> is the classic rule for retaining components: "
            "only keep components that explain more variance than a single raw indicator would. "
            "The model retains the components above this line. A sharply declining scree curve "
            "is healthy — it means the first few components capture most of the information."
        )
        if "pca_eigenvalues" in data:
            evals = data["pca_eigenvalues"]
            n_show = min(20, len(evals))
            cumvar = np.cumsum(evals[:n_show]) / evals.sum() * 100

            fig = go.Figure()
            error_y = None
            if "pca_eigenvalues_ci" in data and data["pca_eigenvalues_ci"] is not None:
                ci = data["pca_eigenvalues_ci"]
                n_ci = min(len(ci), n_show)
                error_y = dict(
                    type='data', symmetric=False,
                    array=ci[:n_ci, 1] - evals[:n_ci],
                    arrayminus=evals[:n_ci] - ci[:n_ci, 0]
                )
                
            fig.add_trace(go.Bar(
                x=list(range(1, n_show + 1)), y=evals[:n_show],
                name="Eigenvalue", marker_color="#e94560",
                hovertemplate="Component %{x}<br>Eigenvalue: %{y:.3f}<extra></extra>",
                error_y=error_y
            ))
            fig.add_trace(go.Scatter(
                x=list(range(1, n_show + 1)), y=cumvar,
                name="Cumulative Variance (%)", yaxis="y2",
                line=dict(color="#2ecc71", width=2), mode="lines+markers",
                hovertemplate="Component %{x}<br>Cumulative Variance: %{y:.1f}%<extra></extra>",
            ))
            fig.add_hline(y=1.0, line_dash="dash", line_color="rgba(255,255,255,0.6)",
                          annotation_text="Kaiser threshold (eigenvalue = 1)",
                          annotation_position="bottom right")
            fig.update_layout(
                template="plotly_dark",
                title=f"PCA Scree Plot — {meta.get('n_components', '?')} components retained, "
                      f"{meta.get('cumulative_variance', 0)*100:.1f}% variance explained",
                xaxis_title="Principal Component Number",
                yaxis_title="Eigenvalue",
                yaxis2=dict(title="Cumulative Variance (%)", overlaying="y",
                            side="right", range=[0, 105], showgrid=False),
                height=470,
                legend=dict(x=0.6, y=0.9),
            )
            st.plotly_chart(fig, width="stretch")

    with tab2:
        explainer(
            "<b>Coupling matrix eigenvalues</b>: the eigenvalues of W determine the stability "
            "and propagation modes of the dynamic model. "
            "<br><br>"
            "• <b>Magnitude plot (top)</b>: for a row-stochastic W, the largest eigenvalue is "
            "always 1. The remaining eigenvalues (|λ| < 1) decay and determine how fast shocks "
            "dissipate. A large <em>spectral gap</em> (fast drop from λ₁ to λ₂) means the "
            "network has one dominant propagation mode and shocks spread uniformly. "
            "<br><br>"
            "• <b>Complex plane plot (bottom)</b>: the unit circle is the stability boundary — "
            "all eigenvalues of a row-stochastic W lie inside or on it. Complex eigenvalues "
            "(off the real axis) indicate oscillatory propagation patterns. Clustering near the "
            "origin = fast decay; clustering near the unit circle = slow decay = long memory."
        )
        if "W_eigenvalues" in data:
            w_evals = data["W_eigenvalues"]
            mags = np.sort(np.abs(w_evals))[::-1]
            n_show = min(40, len(mags))

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(1, n_show + 1)), y=mags[:n_show],
                mode="lines+markers", name="|λ|",
                line={"color": "#e94560", "width": 2},
                marker=dict(size=6),
                hovertemplate="λ_%{x}<br>Magnitude |λ|: %{y:.4f}<extra></extra>",
            ))
            fig.add_hline(y=meta.get("alpha", 0.4),
                          line_dash="dot", line_color="rgba(255,200,0,0.7)",
                          annotation_text=f"α = {meta.get('alpha', 0.4):.2f} (stability margin)",
                          annotation_position="bottom right")
            fig.update_layout(
                template="plotly_dark",
                title=f"Coupling Matrix Eigenvalue Magnitudes (spectral gap = {meta.get('spectral_gap', 0):.4f})",
                xaxis_title="Eigenvalue Index (sorted descending)",
                yaxis_title="|λ| — Eigenvalue Magnitude",
                height=400,
            )
            st.plotly_chart(fig, width="stretch")

            # Complex plane
            fig2 = go.Figure()
            theta = np.linspace(0, 2 * np.pi, 200)
            fig2.add_trace(go.Scatter(
                x=np.cos(theta), y=np.sin(theta),
                mode="lines", line=dict(color="rgba(255,255,255,0.3)", dash="dot"),
                name="Unit circle (stability boundary)", showlegend=True,
            ))
            fig2.add_trace(go.Scatter(
                x=w_evals.real, y=w_evals.imag,
                mode="markers",
                marker=dict(size=7, color=mags[:len(w_evals)],
                            colorscale="Plasma", showscale=True,
                            colorbar=dict(title="|λ|")),
                name="Eigenvalues",
                hovertemplate="Re(λ): %{x:.4f}<br>Im(λ): %{y:.4f}<br>|λ|: %{marker.color:.4f}<extra></extra>",
            ))
            fig2.update_layout(
                template="plotly_dark",
                title="Eigenvalue Distribution in Complex Plane",
                xaxis_title="Re(λ) — Real part (magnitude of non-oscillatory decay)",
                yaxis_title="Im(λ) — Imaginary part (frequency of oscillation)",
                height=520,
                xaxis=dict(scaleanchor="y", range=[-1.15, 1.15]),
                yaxis=dict(range=[-1.15, 1.15]),
            )
            st.plotly_chart(fig2, width="stretch")
            st.caption(
                "**Complex Plane Interpretation:** This scatter plot maps the eigenvalues of the coupling matrix in the complex plane. "
                "The dashed circle is the unit circle (radius = 1), representing the stability boundary. All eigenvalues must lie on or inside it. "
                "Eigenvalues off the horizontal axis (complex numbers) have imaginary parts, which indicate that shocks will oscillate or ripple "
                "through the network rather than decaying monotonically."
            )

    with tab3:
        explainer(
            "<b>Laplacian eigenvalue spectrum</b>: the Laplacian L = D − W_sym (where W_sym is "
            "the symmetrised coupling matrix and D is the degree matrix) encodes the network's "
            "diffusion properties. Its eigenvalues are always real and non-negative. "
            "<br><br>"
            "• <b>λ₁ = 0</b> always (every connected graph has one zero eigenvalue). "
            "<br>"
            f"• <b>λ₂ = {meta.get('algebraic_connectivity', 0):.4f} (Fiedler value)</b>: "
            "the second-smallest eigenvalue, measuring algebraic connectivity. Larger = more "
            "robust network. λ₂ → 0 means the network is near a disconnection (fragmentation). "
            "<br>"
            "• <b>Eigengap chart (bottom)</b>: jumps in the eigenvalue sequence reveal the "
            "natural number of clusters. The largest gap after the first eigenvalue tells you "
            "how many communities the network wants to break into."
        )
        if "laplacian_eigenvalues" in data:
            lap_evals = np.sort(data["laplacian_eigenvalues"])
            n_show = min(40, len(lap_evals))

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=list(range(1, n_show + 1)), y=lap_evals[:n_show],
                mode="lines+markers", name="Laplacian λ",
                line={"color": "#2ecc71", "width": 2},
                marker=dict(size=6),
                hovertemplate="λ_%{x}: %{y:.4f}<extra></extra>",
            ))
            fig.add_annotation(
                x=2, y=lap_evals[1],
                text=f"λ₂ = {lap_evals[1]:.4f} (Fiedler value)",
                showarrow=True, arrowhead=2, arrowcolor="#fff",
                font=dict(color="#2ecc71"),
            )
            fig.update_layout(
                template="plotly_dark",
                title="Laplacian Eigenvalue Spectrum",
                xaxis_title="Index (sorted ascending)",
                yaxis_title="Eigenvalue λ",
                height=420,
            )
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "**Laplacian Spectrum Interpretation:** The eigenvalues of the Graph Laplacian (L = D - W) are always real and non-negative. "
                "λ₁ is always 0. λ₂ is the Fiedler value (or algebraic connectivity). Higher values indicate a more robust, "
                "tightly connected network that is harder to fragment. Hover to see exact values."
            )

            # Eigengap
            gaps = np.diff(lap_evals[:n_show])
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(
                x=list(range(2, n_show + 1)), y=gaps,
                marker_color=["#e94560" if g == gaps.max() else "#3498db" for g in gaps],
                hovertemplate="Gap between λ_%{x} and λ_{x-1}: %{y:.4f}<extra></extra>",
            ))
            fig2.update_layout(
                template="plotly_dark",
                title="Eigengap Δλ — Peaks indicate natural cluster boundaries",
                xaxis_title="Index (between eigenvalue k−1 and k)",
                yaxis_title="Gap Δλ",
                height=350,
            )
            st.plotly_chart(fig2, width="stretch")
            st.caption(
                "**Eigengap Interpretation:** The eigengap is the difference between consecutive eigenvalues of the graph Laplacian. "
                "The largest gap (colored in red) indicates the natural number of clusters (communities) in the network. "
                f"Auto-detected: k = {meta.get('n_communities', '?')} communities."
            )

    with tab4:
        explainer(
            "<b>Coupling matrix heatmap</b>: shows the raw coupling strength W[i, j] — "
            "how strongly country i's instability affects country j's. Each row sums to 1 "
            "(row-stochastic). Brighter cells = stronger coupling. Hover to see exact values. "
            "Visible block structure along the diagonal is evidence of the community structure "
            "detected on the Network Communities page."
        )
        if "W" in data:
            W_disp = data["W"]
            countries_list = data.get("countries", [f"C{i}" for i in range(W_disp.shape[0])])
            n_show = st.slider("Countries to show (top N)", 10, min(60, W_disp.shape[0]), 30)
            labels = [f"{COUNTRY_NAMES.get(c, c)} ({c})" for c in countries_list[:n_show]]

            fig = px.imshow(
                W_disp[:n_show, :n_show],
                x=labels, y=labels,
                color_continuous_scale="Viridis",
                title=f"Coupling Matrix W — top {n_show} countries (row = sender, col = receiver)",
                labels={"color": "Coupling Weight W[i,j]"},
            )
            fig.update_traces(
                hovertemplate="<b>Sender (Row):</b> %{y}<br><b>Receiver (Col):</b> %{x}<br><b>Coupling Weight W[i,j]:</b> %{z:.4f}<extra></extra>"
            )
            fig.update_layout(template="plotly_dark", height=700,
                              xaxis_tickangle=-45)
            st.plotly_chart(fig, width="stretch")
            st.caption(
                "**Heatmap Interpretation:** The cell at row i (sender) and column j (receiver) represents the coupling weight W[i,j], "
                "which is the fraction of country i's instability that spills over to country j. Bright cells represent strong connections. "
                "Dense regional blocks on the diagonal represent communities of highly coupled countries."
            )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE: SPECTRAL EMBEDDING
# ═══════════════════════════════════════════════════════════════════════════════

elif page == "🌐 Spectral Embedding":
    st.title("Spectral Embedding — Network Topology in 3D")

    explainer(
        "<b>What is spectral embedding?</b> Instead of placing countries on a geographic map, "
        "we position them based on their <em>structural role</em> in the coupling network. "
        "We use the second, third, and fourth eigenvectors of the Laplacian (v₂, v₃, v₄) as "
        "3D coordinates — this is called the <b>spectral embedding</b>. "
        "<br><br>"
        "Countries that <b>cluster together</b> in this space share similar coupling patterns — "
        "they transmit and receive instability in similar ways — even if they are geographically "
        "distant. For example, two small open economies with similar trade partner profiles will "
        "cluster together even if they are on opposite sides of the world. "
        "<br><br>"
        "<b>Axes</b>: v₂ = Fiedler vector (primary partition), v₃ and v₄ = secondary and tertiary "
        "structural dimensions. Colour can be set to any model metric."
    )

    if "W" not in data or "countries" not in data:
        st.warning("Coupling matrix required. Run model first.")
        st.stop()

    W = data["W"]
    countries = data["countries"]
    n = len(countries)

    from model.laplacian import compute_laplacian
    from scipy.linalg import eigh

    lap = compute_laplacian(W)
    eigenvalues_lap, eigenvectors_lap = eigh(lap["L_sym"])
    coords = eigenvectors_lap[:, 1:4]  # v₂, v₃, v₄

    color_source = st.selectbox("Color by", [
        "Community", "Steady-State Score", "PageRank Centrality",
        "Structural Score", "Fiedler Vector"
    ])

    embed_df = pd.DataFrame({
        "iso3": countries,
        "x": coords[:, 0],
        "y": coords[:, 1],
        "z": coords[:, 2] if coords.shape[1] > 2 else 0,
    })
    embed_df = enrich_with_names(embed_df)

    if color_source == "Community" and "communities" in data:
        embed_df = embed_df.merge(data["communities"].assign(
            cluster=data["communities"]["cluster"].astype(str)
        ), on="iso3", how="left")
        color_col = "cluster"
        color_kw = dict(color_discrete_sequence=px.colors.qualitative.Set1)
    elif color_source == "Steady-State Score" and "steady_state" in data:
        embed_df = embed_df.merge(data["steady_state"][["iso3", "steady_state_score"]], on="iso3", how="left")
        color_col = "steady_state_score"
        color_kw = dict(color_continuous_scale="RdYlGn_r")
    elif color_source == "PageRank Centrality" and "pagerank" in data:
        embed_df = embed_df.merge(data["pagerank"][["iso3", "pagerank"]], on="iso3", how="left")
        color_col = "pagerank"
        color_kw = dict(color_continuous_scale="Viridis")
    elif color_source == "Structural Score" and "steady_state" in data:
        embed_df = embed_df.merge(data["steady_state"][["iso3", "structural_score"]], on="iso3", how="left")
        color_col = "structural_score"
        color_kw = dict(color_continuous_scale="RdYlGn_r")
    elif color_source == "Fiedler Vector" and "fiedler" in data:
        embed_df = embed_df.merge(data["fiedler"], on="iso3", how="left")
        color_col = "fiedler_vector"
        color_kw = dict(color_continuous_scale="RdBu")
    else:
        embed_df["value"] = 1.0
        color_col = "value"
        color_kw = dict(color_continuous_scale="Blues")

    fig = px.scatter_3d(
        embed_df, x="x", y="y", z="z",
        color=color_col,
        text="iso3",
        hover_name="country_name",
        hover_data={"iso3": True, "x": ":.4f", "y": ":.4f", "z": ":.4f", "country_name": False},
        title="3D Spectral Embedding — Countries Positioned by Network Role (not geography)",
        **color_kw,
    )
    fig.update_traces(
        marker=dict(size=8, opacity=0.85, line=dict(width=1, color="rgba(255,255,255,0.3)")),
        textfont_size=8,
    )
    fig.update_layout(
        template="plotly_dark",
        height=800,
        scene=dict(
            xaxis_title="v₂ — Fiedler (primary partition)",
            yaxis_title="v₃ — Secondary structural dimension",
            zaxis_title="v₄ — Tertiary structural dimension",
            xaxis=dict(backgroundcolor="rgb(20,30,50)"),
            yaxis=dict(backgroundcolor="rgb(20,30,50)"),
            zaxis=dict(backgroundcolor="rgb(20,30,50)"),
        ),
    )
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "**3D Scatter Plot Interpretation:** This 3D scatter plot positions countries based on their structural role in the coupling network. "
        "The coordinates are the 2nd, 3rd, and 4th eigenvectors of the graph Laplacian. Geography is ignored: proximity in this 3D space "
        "indicates that countries have similar connection topologies (e.g. they trade with the same partners or belong to the same regional blocs). "
        "Color represents the selected metric. Hovering over a point displays the country's full name, ISO3 code, and coordinates."
    )

    st.caption(
        "🖱️ Drag to rotate · Scroll to zoom · Hover over any point for the full country name. "
        "Countries that cluster together share similar structural coupling roles in the instability network."
    )

    # Edge overlay
    show_edges = st.checkbox("Show strongest coupling edges", value=False)
    if show_edges:
        n_edges = st.slider("Number of edges to show", 10, 200, 50)
        W_flat = W.copy()
        np.fill_diagonal(W_flat, 0)
        top_indices = np.argsort(W_flat.ravel())[::-1][:n_edges]
        edge_i, edge_j = np.unravel_index(top_indices, W.shape)

        edge_x, edge_y, edge_z = [], [], []
        for ei, ej in zip(edge_i, edge_j):
            edge_x += [coords[ei, 0], coords[ej, 0], None]
            edge_y += [coords[ei, 1], coords[ej, 1], None]
            edge_z += [coords[ei, 2] if coords.shape[1] > 2 else 0,
                       coords[ej, 2] if coords.shape[1] > 2 else 0, None]

        fig.add_trace(go.Scatter3d(
            x=edge_x, y=edge_y, z=edge_z,
            mode="lines",
            line=dict(color="rgba(255,255,255,0.12)", width=1),
            showlegend=False,
            hoverinfo="skip",
        ))
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Showing the {n_edges} strongest coupling links in the network.")
