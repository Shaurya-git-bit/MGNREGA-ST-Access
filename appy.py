"""
MGNREGA ST Access - Village Risk Watchlist
Run:  pip install streamlit plotly pandas
      streamlit run app.py
Needs village_risk_2021.csv in the same folder.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

st._config.set_option("theme.base", "light")
st.set_page_config(page_title="MGNREGA Early Warning", page_icon="🌾", layout="wide")

st.markdown(
    """
    <style>
      .block-container {padding-top: 1.6rem; padding-bottom: 2rem; max-width: 1250px;}
      div[data-testid="stMetric"] {
          background: #F6F8FA; border: 1px solid #E3E8EE; border-radius: 12px;
          padding: 14px 16px;
      }
      div[data-testid="stMetricLabel"] p {font-size: 0.82rem; color: #57606A;}
      .risk-pill {display:inline-block; padding:3px 12px; border-radius:999px;
                  font-weight:600; font-size:0.85rem; color:white;}
      .reason {background:#F6F8FA; border-left:4px solid #D1495B; border-radius:6px;
               padding:8px 12px; margin:6px 0; font-size:0.93rem;}
      .reason.down {border-left-color:#2E8B57;}
      .note {background:#FFF8E5; border:1px solid #F2E1A8; border-radius:10px;
             padding:10px 14px; font-size:0.9rem;}
      h1 {font-size: 1.9rem !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

RISK_COLORS = {"High": "#D1495B", "Medium": "#EDAE49", "Low": "#2E8B57"}


@st.cache_data
def load(path: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    d["risk_pct"] = (d["risk"] * 100).round(1)
    d["access_pct"] = (d["st_access_rate_lag1"] * 100).round(0)
    d["roll3_pct"] = (d["st_access_rate_roll3"] * 100).round(0)
    d["Actual 2021"] = d["is_low_access_current"].map({1: "Low access", 0: "OK"})
    d["Village"] = d["Village Name"].str.title()
    d["Block"] = d["Block"].astype(str)
    d["District"] = d["District"].astype(str)
    return d


try:
    df = load("village_risk_2021.csv")
except FileNotFoundError:
    st.error("village_risk_2021.csv not found. Run the export cell in the notebook first, "
             "then put the CSV next to app.py.")
    st.stop()

# ---------------------------------------------------------------- header
st.title("🌾 MGNREGA ST Access — Village Risk Watchlist")
st.caption("Which villages are most likely to give Scheduled Tribe households too little work "
           "this year? Ranked from each village's own history. Demo on held-out year 2021 "
           "(model trained on earlier years only).")

# ---------------------------------------------------------------- sidebar
with st.sidebar:
    st.header("Filters")
    districts = ["All districts"] + sorted(df["District"].unique())
    district = st.selectbox("District", districts)

    pool = df if district == "All districts" else df[df["District"] == district]
    blocks = ["All blocks"] + sorted(pool["Block"].unique())
    block = st.selectbox("Block", blocks)

    review_pct = st.slider("Review budget: top % of villages by risk", 5, 100, 20, step=5,
                           help="How many villages can you visit? The riskiest come first.")
    min_cards = st.number_input("Hide villages with fewer ST job cards than", 0, 500, 10,
                                help="Tiny villages give extreme scores from very few households.")
    search = st.text_input("Search village name")

    st.divider()
    st.caption("Risk bands: **High** ≥ 50%, **Medium** 25–50%, **Low** < 25%.")

view = df.copy()
if district != "All districts":
    view = view[view["District"] == district]
if block != "All blocks":
    view = view[view["Block"] == block]
view = view[view["st_jobcards"] >= min_cards].sort_values("risk", ascending=False)

n_review = max(1, int(len(view) * review_pct / 100)) if len(view) else 0
flagged = view.head(n_review).copy()
if search:
    flagged = flagged[flagged["Village"].str.contains(search, case=False, na=False)]
flagged["Band"] = pd.cut(flagged["risk"], [-0.01, 0.25, 0.50, 1.01],
                         labels=["Low", "Medium", "High"])

# ---------------------------------------------------------------- KPIs
k1, k2, k3, k4 = st.columns(4)
k1.metric("Villages in scope", f"{len(view):,}")
k2.metric("Flagged for review", f"{len(flagged):,}")
k3.metric("Average risk (flagged)", f"{flagged['risk'].mean():.0%}" if len(flagged) else "–")
if len(flagged):
    k4.metric("Actually low-access in 2021",
              f"{(flagged['is_low_access_current'] == 1).mean():.0%}",
              help="Hit rate among flagged villages. About 1 in 5 is typical: "
                   "this is a watchlist, not a verdict.")
else:
    k4.metric("Actually low-access in 2021", "–")

st.markdown(
    "<div class='note'>💡 <b>How to use this:</b> start at the top of the list, verify that work "
    "was recorded, then check whether it was denied. About 1 in 5 flagged villages turns out "
    "low-access, so treat flags as a prompt to look, not proof of exclusion.</div>",
    unsafe_allow_html=True)
st.write("")

if not len(flagged):
    st.warning("No villages match these filters. Widen the review budget or lower the "
               "job-card minimum.")
    st.stop()

# ---------------------------------------------------------------- tabs
tab_list, tab_charts, tab_village, tab_about = st.tabs(
    ["📋 Watchlist", "📊 Overview", "🔍 Village detail", "ℹ️ About the model"])

with tab_list:
    show = flagged[["Village", "Block", "District", "risk_pct", "Band", "st_jobcards",
                    "access_pct", "roll3_pct", "Actual 2021"]].rename(columns={
        "risk_pct": "Risk %", "st_jobcards": "ST job cards",
        "access_pct": "Last year's access %", "roll3_pct": "3-yr avg access %"})
    st.dataframe(
        show, use_container_width=True, hide_index=True, height=520,
        column_config={
            "Risk %": st.column_config.ProgressColumn("Risk", min_value=0, max_value=100,
                                                      format="%.0f%%"),
            "Last year's access %": st.column_config.NumberColumn(format="%.0f%%"),
            "3-yr avg access %": st.column_config.NumberColumn(format="%.0f%%"),
            "ST job cards": st.column_config.NumberColumn(format="%d"),
        })
    st.download_button("⬇️ Download this watchlist (CSV)",
                       show.to_csv(index=False).encode("utf-8"),
                       file_name="village_watchlist.csv", mime="text/csv")

with tab_charts:
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Risk by block")
        by_block = (flagged.groupby("Block")
                    .agg(villages=("Village", "count"), avg_risk=("risk", "mean"))
                    .reset_index().sort_values("avg_risk", ascending=True).tail(15))
        fig = px.bar(by_block, x="avg_risk", y="Block", orientation="h",
                     hover_data={"villages": True, "avg_risk": ":.0%"},
                     color="avg_risk", color_continuous_scale=["#EDAE49", "#D1495B"])
        fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0),
                          coloraxis_showscale=False, xaxis_tickformat=".0%",
                          xaxis_title="Average risk", yaxis_title="")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Last year's access vs risk")
        fig = px.scatter(flagged, x="access_pct", y="risk_pct", color="Actual 2021",
                         hover_name="Village", hover_data=["Block", "st_jobcards"],
                         color_discrete_map={"Low access": "#D1495B", "OK": "#2E8B57"},
                         opacity=0.75)
        fig.update_layout(height=420, margin=dict(l=0, r=0, t=10, b=0),
                          xaxis_title="Last year's ST access rate (%)",
                          yaxis_title="Predicted risk (%)", legend_title="")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("How well does the ranking work?")
    st.caption("If you could only visit the top X% of villages (ranked by risk), what share of all "
               "low-access villages would you reach? The dashed line is random visits.")
    ranked = view.sort_values("risk", ascending=False).reset_index(drop=True)
    total_pos = ranked["is_low_access_current"].sum()
    if total_pos > 0:
        ranked["cum_recall"] = ranked["is_low_access_current"].cumsum() / total_pos
        ranked["visited"] = (ranked.index + 1) / len(ranked)
        gain = go.Figure()
        gain.add_trace(go.Scatter(x=ranked["visited"], y=ranked["cum_recall"], mode="lines",
                                  name="Model ranking", line=dict(color="#D1495B", width=3)))
        gain.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random visits",
                                  line=dict(color="#8B949E", dash="dash")))
        gain.add_vline(x=review_pct / 100, line_dash="dot", line_color="#57606A",
                       annotation_text=f"your budget: {review_pct}%")
        gain.update_layout(height=380, margin=dict(l=0, r=0, t=10, b=0),
                           xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                           xaxis_title="Share of villages visited (highest risk first)",
                           yaxis_title="Share of low-access villages reached",
                           legend=dict(orientation="h", y=1.1))
        st.plotly_chart(gain, use_container_width=True)
        at_budget = (ranked.loc[ranked["visited"] <= review_pct / 100,
                                "is_low_access_current"].sum() / total_pos)
        st.success(f"Visiting the top {review_pct}% of villages in this scope reaches "
                   f"**{at_budget:.0%}** of the villages that actually ended up low-access.")
    else:
        st.info("No low-access villages in this scope, so the ranking chart is not shown.")

with tab_village:
    labels = flagged["Village"] + "  ·  " + flagged["Block"]
    pick = st.selectbox("Choose a flagged village", labels.tolist())
    row = flagged[labels == pick].iloc[0]
    band = str(row["Band"])

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Risk score", f"{row['risk']:.0%}")
    d2.metric("ST job cards", f"{int(row['st_jobcards'])}")
    d3.metric("Last year's access", f"{row['access_pct']:.0f}%")
    d4.metric("3-year average", f"{row['roll3_pct']:.0f}%")
    st.markdown(f"<span class='risk-pill' style='background:{RISK_COLORS[band]}'>{band} risk</span>"
                f" &nbsp; {row['District']} district · {row['Block']} block",
                unsafe_allow_html=True)
    st.write("")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Why was this village flagged?**")
        for r in str(row["reasons"]).split(" | "):
            cls = "down" if "lowers" in r else ""
            st.markdown(f"<div class='reason {cls}'>{r}</div>", unsafe_allow_html=True)
        st.caption("Red = pushes risk up, green = pushes risk down (from the model's SHAP "
                   "explanation).")
    with right:
        gauge = go.Figure(go.Indicator(
            mode="gauge+number", value=row["risk_pct"], number={"suffix": "%"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": RISK_COLORS[band]},
                   "steps": [{"range": [0, 25], "color": "#E6F4EA"},
                             {"range": [25, 50], "color": "#FFF3D6"},
                             {"range": [50, 100], "color": "#FBE3E6"}]}))
        gauge.update_layout(height=230, margin=dict(l=10, r=10, t=10, b=0))
        st.plotly_chart(gauge, use_container_width=True)

    if row["Actual 2021"] == "Low access":
        st.error("What actually happened in 2021: **low access** (below 30%).")
    else:
        st.success("What actually happened in 2021: **access was OK** (a false alarm).")

with tab_about:
    st.markdown(
        """
**What it predicts.** For each village, the chance that Scheduled Tribe households will get work at
a rate **below 30%** of job-card households this fiscal year. It uses only the village's *previous*
years (last 1 to 3 years' access rate, the 3-year average, workdays per household, 100-day
completion) plus district and block.

**How it was tested.** Trained on earlier years and scored on years it never saw. Across four
unseen years (2018 to 2021), reviewing the **top 20%** of villages reached roughly **72 to 76%**
of the villages that ended up low-access, versus about 20% for random visits.

**What to keep in mind**
- **It is a watchlist.** About 1 in 5 flagged villages is truly low-access. Use flags to decide
  where to look first.
- **Zero employment can mean two things.** Roughly two-thirds of low-access cases show *exactly
  zero* recorded employment, which may be real denial or a reporting gap. Check that work was
  recorded first.
- **It shows where, not why.** The explanations describe what drove the score, not the cause.
- **Data.** NDAP Dataset 8071, Meghalaya, fiscal years 2014 to 2021. The 2022 extract was
  excluded because it records almost no employment (a reporting artifact).
        """)