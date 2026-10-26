"""Streamlit app: Smartphone Value-for-Money Recommender (Business Analytics DA2)."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import phone_model as pm

st.set_page_config(page_title="Smartphone Recommender", page_icon="📱", layout="wide")


@st.cache_data
def load():
    return pd.read_csv(Path(__file__).parent / "phones_scored.csv")


df = load()

st.title("📱 Smartphone Value-for-Money Recommender")
st.caption("Business Analytics · Lakshay Ujlan (24BCE5385) · Based on a ~2023 snapshot of 980 phones (Kaggle). "
           "Prices in ₹ — check current prices before buying.")

tab_rec, tab_market, tab_about = st.tabs(["Recommend", "Market insights", "Method"])

with st.sidebar:
    st.header("Your needs")
    budget = st.slider("Maximum budget (₹)", 5000, 150000, 25000, step=1000)
    st.subheader("What matters to you? (0 = ignore, 5 = critical)")
    w_perf = st.slider("Performance (RAM, processor, storage)", 0, 5, 3)
    w_cam = st.slider("Camera", 0, 5, 2)
    w_batt = st.slider("Battery & charging", 0, 5, 3)
    w_disp = st.slider("Display (refresh rate, resolution, size)", 0, 5, 1)
    w_value = st.slider("Value for money", 0, 5, 4)
    st.subheader("Filters")
    need_5g = st.checkbox("Must support 5G", value=False)
    min_rating = st.slider("Minimum user rating (out of 10)", 6.0, 9.0, 7.5, step=0.1)
    brand_options = sorted(df["brand_name"].unique())
    brands = st.multiselect("Brands (leave empty for all)", brand_options)

with tab_rec:
    res = pm.recommend(df, budget, w_perf, w_cam, w_batt, w_disp, w_value,
                       need_5g=need_5g, brands=brands or None, min_rating=min_rating, top_n=5)
    if res.empty:
        st.warning("No phones match these filters. Try a higher budget, a lower minimum rating, or fewer brand filters.")
    else:
        st.subheader(f"Top {len(res)} phones for ₹{budget:,}")
        for rank, (_, r) in enumerate(res.iterrows(), start=1):
            with st.container(border=True):
                c1, c2, c3, c4 = st.columns([3, 1.2, 1.2, 1.2])
                c1.markdown(f"**{rank}. {r['model']}**  \n{r['why']}")
                c2.metric("Price", f"₹{r['price']:,.0f}")
                c3.metric("Fair price (from specs)", f"₹{r['fair_price']:,.0f}")
                c4.metric("Match score", f"{r['match_score']:.0f}/100")
                st.caption(
                    f"{int(r['ram_capacity'])} GB RAM · {int(r['internal_memory'])} GB storage · "
                    f"{r['battery_capacity']:.0f} mAh · {int(r['refresh_rate'])} Hz · "
                    f"{r['primary_camera_rear']:.0f} MP rear camera · {'5G' if r['5G_or_not'] == 1 else 'no 5G'} · "
                    f"rating {r['avg_rating']:.1f}/10")

        pool = df[df["price"] <= budget]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(pool["fair_price"], pool["price"], s=10, alpha=.35, color="#9AA5B1", label="Other phones in budget")
        ax.scatter(res["fair_price"], res["price"], s=70, color="#E4572E", label="Recommended", zorder=3)
        lim = max(pool["fair_price"].max(), budget)
        ax.plot([0, lim], [0, lim], "k--", lw=1, label="Price = fair price")
        ax.set_xlim(0, lim); ax.set_ylim(0, budget * 1.05)
        ax.set_xlabel("Fair price predicted from specs (₹)"); ax.set_ylabel("Actual price (₹)")
        ax.set_title("Phones below the dashed line are cheaper than their specs justify"); ax.legend(fontsize=8)
        st.pyplot(fig)

with tab_market:
    st.subheader("Market segments (K-means clustering)")
    prof = df.groupby("cluster")[["price", "ram_capacity", "internal_memory", "battery_capacity",
                                  "refresh_rate", "primary_camera_rear", "avg_rating"]].median().round(1)
    prof.insert(0, "phones", df["cluster"].value_counts())
    st.dataframe(prof.loc[["Budget", "Mid-range", "Premium"]], width="stretch")
    st.subheader("Which brands give the best value?")
    bv = (df.groupby("brand_group")["value_score"].median().sort_values(ascending=False)
          .rename("median value score").to_frame())
    st.bar_chart(bv)
    st.caption("Value score = (fair price − actual price) / fair price. Positive = cheaper than its specs justify. "
               "The model controls for the iOS platform premium (about +148% at equal specs), so Apple phones are judged against other iPhones' pricing, not against Android phones.")

with tab_about:
    st.markdown("""
**How it works**
1. **Data:** 980 smartphones (Kaggle *Smartphone Dataset*), cleaned and missing values imputed.
2. **Fair price:** a multiple linear regression of log(price) on specifications (adjusted R² ≈ 0.81, cross-validated R² ≈ 0.80) estimates what each phone *should* cost.
3. **Value score:** (fair price − actual price) / fair price.
4. **Segments:** K-means clustering groups phones into Budget / Mid-range / Premium.
5. **Recommendation:** within your budget and filters, each phone is scored 0–1 on performance, camera, battery and display (percentile ranks). Your weights blend these with the value-for-money rank to give the match score.

**Limitations:** data is a ~2023 snapshot from third-party listings; a few records are anomalous; ratings are partly imputed (brand median) where missing.
""")
