import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(page_title="Shipping Efficiency Dashboard", layout="wide")

st.title("🚚 Factory → Customer Shipping Route Efficiency Dashboard")

# -------------------------------
# 🎨 PREMIUM CSS (UNCHANGED)
# -------------------------------
st.markdown("""
<style>
body {background: linear-gradient(135deg, #0b1220, #111827);}
section[data-testid="stSidebar"] {
    background: rgba(15, 23, 42, 0.95);
}
div[role="radiogroup"] > label {
    padding: 10px;
    border-radius: 10px;
    transition: 0.3s;
}
div[role="radiogroup"] > label:hover {
    background: linear-gradient(135deg, #1f77b4, #ff4d6d);
    transform: translateX(5px);
    color: white;
}
.kpi-card {
    background: rgba(255,255,255,0.05);
    backdrop-filter: blur(12px);
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    color: white;
    transition: 0.3s;
}
.kpi-card:hover {
    transform: scale(1.05);
    box-shadow: 0px 8px 30px rgba(255,77,109,0.4);
}
.section-box {
    background: rgba(255,255,255,0.03);
    padding: 20px;
    border-radius: 16px;
    margin-bottom: 20px;
}
h1,h2,h3 {color:white;}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# KPI CARD
# -------------------------------
def kpi_card(title, value):
    st.markdown(f"""
    <div class="kpi-card">
        <h4>{title}</h4>
        <h2>{value}</h2>
    </div>
    """, unsafe_allow_html=True)

# -------------------------------
# LOAD REAL DATA
# -------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Nassau Candy Distributor.csv")

    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True)

    df["Lead_Time"] = (df["Ship Date"] - df["Order Date"]).dt.days

    df = df.drop_duplicates()
    df["Sales"] = df["Sales"].fillna(df["Sales"].mean())
    df["Lead_Time"] = df["Lead_Time"].fillna(df["Lead_Time"].median())

    df = df[df["Lead_Time"] > 0]

    df = df.rename(columns={"Ship Mode": "Ship_Mode"})

    avg_lead = df["Lead_Time"].mean()
    df["Status"] = df["Lead_Time"].apply(
        lambda x: "Delayed" if x > avg_lead else "On-Time"
    )

    return df
with st.spinner("Loading Dashboard..."):
    df = load_data()

# -------------------------------
# GEO (FIXED SIMPLE MAP)
# -------------------------------
city_coords = {
    "New York": (40.7,-74),
    "Los Angeles": (34,-118),
    "Chicago": (41.8,-87),
    "Houston": (29.7,-95),
    "Philadelphia": (39.9,-75),
    "Phoenix": (33.4,-112),
    "San Antonio": (29.4,-98),
    "San Diego": (32.7,-117),
    "Dallas": (32.8,-96),
    "San Jose": (37.3,-121)
}

df["lat"] = df["City"].map(lambda x: city_coords.get(x, (None,None))[0])
df["lon"] = df["City"].map(lambda x: city_coords.get(x, (None,None))[1])
df_geo = df.dropna(subset=["lat","lon"]) 

state_map = {
    "California": "CA",
    "Texas": "TX",
    "New York": "NY",
    "Florida": "FL",
    "Illinois": "IL",
    "Pennsylvania": "PA",
    "Ohio": "OH",
    "Georgia": "GA",
    "North Carolina": "NC",
    "Michigan": "MI"
}

df["state_code"] = df["State/Province"].map(state_map)

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.title("📊 Nassau Dashboard")

page = st.sidebar.radio("Navigation",[
    "Overview","Route Efficiency","Geographical Analysis",
    "Ship Mode Comparison","Route Drill Down","ML Prediction"
])

# -------------------------------
# FILTERS
# -------------------------------
st.sidebar.markdown("## Filters")
region = st.sidebar.multiselect(
    "Region",
    df["Region"].unique(),
    default=df["Region"].unique()
)

if region:
    cities = df[df["Region"].isin(region)]["City"].unique()
else:
    cities = df["City"].unique()

city = st.sidebar.multiselect("City", cities)
lead = st.sidebar.slider(
    "Lead Time Threshold",
    int(df["Lead_Time"].min()),
    int(df["Lead_Time"].max()),
    int(df["Lead_Time"].max())
)

df_f = df.copy()

if region:
    df_f = df_f[df_f["Region"].isin(region)]
if city:
    df_f = df_f[df_f["City"].isin(city)]

df_f = df_f[df_f["Lead_Time"] <= lead]

if df_f.empty:
    st.warning("No data available")
    st.stop()
st.write("Filtered Data Shape:", df_f.shape)

csv = df_f.to_csv(index=False).encode('utf-8')
st.sidebar.download_button("📥 Download Data", csv, "report.csv")

# -------------------------------
# ROUTE KPI
# -------------------------------
route_kpi = df_f.groupby("City").agg(
    avg_lead_time=("Lead_Time","mean"),
    route_volume=("Sales","count"),
    delayed_shipments=("Status", lambda x: (x=="Delayed").sum())
).reset_index()

route_kpi["efficiency_score"] = 1 - (
    (route_kpi["avg_lead_time"] - route_kpi["avg_lead_time"].min()) /
    (route_kpi["avg_lead_time"].max() - route_kpi["avg_lead_time"].min() + 1e-5)
)

# ===============================
# OVERVIEW
# ===============================
if page=="Overview":

    col1,col2,col3,col4 = st.columns(4)
    with col1: kpi_card("Total Shipments", len(df_f))
    with col2: kpi_card("Revenue", int(df_f["Sales"].sum()))
    with col3: kpi_card("Avg Lead Time", round(df_f["Lead_Time"].mean(),2))
    with col4: kpi_card("Delay Count", len(df_f[df_f["Status"]=="Delayed"]))

    st.markdown("---")

    col1,col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.histogram(df_f,x="Lead_Time",title="📊 Lead Time Distribution",color_discrete_sequence=["#AF4DFF"]),use_container_width=True)
    with col2:
        st.plotly_chart(px.pie(df_f,names="Ship_Mode",title="🚚 Shipment Mode Distribution",color_discrete_sequence=["#00C49F","#FF4D6D","#FFD166"]),use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    # -------------------------------
    # TOP / BOTTOM ROUTES
    # -------------------------------
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🏆 Top 10 Efficient Routes")
        st.dataframe(
            route_kpi.sort_values("efficiency_score", ascending=False)
            .head(10)
        )

    with col2:
        st.subheader("⚠️ Bottom 10 Inefficient Routes")
        st.dataframe(
            route_kpi.sort_values("efficiency_score", ascending=True)
            .head(10)
        )
    # -------------------------------
# 🧠 SMART INSIGHTS
# -------------------------------
    st.markdown("### 🧠 Key Insights")

    col1, col2, col3 = st.columns(3)

    fastest = df_f.groupby("City")["Lead_Time"].mean().idxmin()
    slowest = df_f.groupby("City")["Lead_Time"].mean().idxmax()
    highest_delay = df_f[df_f["Status"]=="Delayed"]["City"].value_counts().idxmax()

    with col1:
        st.success(f"🚀 Fastest Route: {fastest}")

    with col2:
        st.error(f"⚠️ Slowest Route: {slowest}")

    with col3:
        st.warning(f"📍 Highest Delay City: {highest_delay}")  
    # ⚠️ DELAY ALERT
# -------------------------------
    delay_rate = (df_f["Status"]=="Delayed").mean()*100

    if delay_rate > 30:
        st.error(f"⚠️ High Delay Rate: {round(delay_rate,2)}%")
    elif delay_rate > 15:
        st.warning(f"⚠️ Moderate Delay Rate: {round(delay_rate,2)}%")
    else:
        st.success(f"✅ Good Performance: {round(delay_rate,2)}%")  
        

# ===============================
# ROUTE EFFICIENCY
# ===============================
elif page=="Route Efficiency":

    route = df_f.groupby("City")["Lead_Time"].mean().reset_index()

    st.plotly_chart(px.bar(route,x="City",y="Lead_Time", title="⏱ Avg Lead Time by Route",color="Lead_Time",
    color_continuous_scale="RdYlGn_r"),use_container_width=True)
    st.subheader("📊 Volume vs Lead Time")

    scatter_df = df_f.groupby("City").agg(
    avg_lead_time=("Lead_Time","mean"),
    volume=("Sales","count")
).reset_index()

    fig = px.scatter(
    scatter_df,
    x="volume",
    y="avg_lead_time",
    size="volume",
    color="City",
    title="Shipment Volume vs Lead Time"
)

    st.plotly_chart(fig, use_container_width=True)

# ===============================
# GEO
# ===============================
elif page=="Geographical Analysis":

    st.header("🌍 Geographical Intelligence Dashboard")

    # -------------------------------
    # COMMON GEO DATA (IMPORTANT FIX)
    # -------------------------------
    geo_df = df_geo[df_geo.index.isin(df_f.index)].groupby(
        ["City","lat","lon"]
    ).agg(
        avg_lead_time=("Lead_Time","mean"),
        volume=("Sales","count")
    ).reset_index()

    # -------------------------------
    # TABS
    # -------------------------------
    tab1, tab2, tab3 = st.tabs([
        "US Shipping Map",
        "Bottleneck Analysis",
        "State Rankings"
    ])

    # -------------------------------
    # TAB 1: MAP
    # -------------------------------
    with tab1:
        st.subheader("🗺 US Shipping Efficiency Map")

        fig = px.scatter_mapbox(
            geo_df,
            lat="lat",
            lon="lon",
            size="volume",
            color="avg_lead_time",
            color_continuous_scale="RdYlGn_r",
            zoom=3,
            title="Average Lead Time by Location"
        )

        fig.update_layout(mapbox_style="carto-darkmatter")
        st.plotly_chart(fig,use_container_width=True)

    # -------------------------------
    # TAB 2: BOTTLENECK
    # -------------------------------
    with tab2:
        st.subheader("⚠️ Bottleneck Analysis")

        bottleneck = geo_df[
            geo_df["avg_lead_time"] > geo_df["avg_lead_time"].mean()
        ]

        fig = px.scatter_mapbox(
            bottleneck,
            lat="lat",
            lon="lon",
            size="volume",
            color="avg_lead_time",
            color_continuous_scale="Reds",
            zoom=3,
            title="High Delay Areas"
        )

        fig.update_layout(mapbox_style="carto-darkmatter")
        st.plotly_chart(fig,use_container_width=True)

    # -------------------------------
    # TAB 3: STATE RANKING
    # -------------------------------
    with tab3:

        st.subheader("📊 State Ranking")

        # TABLE
        state_df = df_f.groupby("State/Province").agg(
            avg_lead_time=("Lead_Time","mean")
        ).reset_index()

        st.dataframe(state_df)

        # -------------------------------
        # CHOROPLETH MAP
        # -------------------------------
        st.subheader("🌍 State-wise Shipping Map")

        state_map = {
            'Alabama':'AL','Alaska':'AK','Arizona':'AZ','Arkansas':'AR','California':'CA',
            'Colorado':'CO','Connecticut':'CT','Delaware':'DE','Florida':'FL','Georgia':'GA',
            'Hawaii':'HI','Idaho':'ID','Illinois':'IL','Indiana':'IN','Iowa':'IA',
            'Kansas':'KS','Kentucky':'KY','Louisiana':'LA','Maine':'ME','Maryland':'MD',
            'Massachusetts':'MA','Michigan':'MI','Minnesota':'MN','Mississippi':'MS','Missouri':'MO',
            'Montana':'MT','Nebraska':'NE','Nevada':'NV','New Hampshire':'NH','New Jersey':'NJ',
            'New Mexico':'NM','New York':'NY','North Carolina':'NC','North Dakota':'ND','Ohio':'OH',
            'Oklahoma':'OK','Oregon':'OR','Pennsylvania':'PA','Rhode Island':'RI','South Carolina':'SC',
            'South Dakota':'SD','Tennessee':'TN','Texas':'TX','Utah':'UT','Vermont':'VT',
            'Virginia':'VA','Washington':'WA','West Virginia':'WV','Wisconsin':'WI','Wyoming':'WY'
        }

        state_df["state_code"] = state_df["State/Province"].str.strip().map(state_map)
        state_df = state_df.dropna(subset=["state_code"])

        fig = px.choropleth(
            state_df,
            locations="state_code",
            locationmode="USA-states",
            color="avg_lead_time",
            color_continuous_scale="RdYlGn_r",
            scope="usa"
        )

        fig.update_layout(
            geo=dict(bgcolor="rgba(0,0,0,0)"),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white")
        )

        st.plotly_chart(fig, use_container_width=True)

# ===============================
# SHIP MODE
# ===============================
elif page=="Ship Mode Comparison":

    st.plotly_chart(
        px.bar(df_f.groupby("Ship_Mode")["Lead_Time"].mean().reset_index(),
               x="Ship_Mode",y="Lead_Time",title="📦 Avg Lead Time by Ship Mode"),
        use_container_width=True
    )
    st.subheader("📈 Ship Mode Lead Time Trend Over Time")

# create time column
    df_f["Month"] = df_f["Order Date"].dt.to_period("M").astype(str)

    trend_df = df_f.groupby(["Month","Ship_Mode"]).agg(
    avg_lead_time=("Lead_Time","mean")
).reset_index()

    fig = px.line(
    trend_df,
    x="Month",
    y="avg_lead_time",
    color="Ship_Mode",
    markers=True,
    title="Lead Time Trend by Ship Mode"
)

    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("⚠️ Delay Rate by Ship Mode")

    delay_df = df_f.groupby("Ship_Mode").agg(
    delay_rate=("Status", lambda x: (x=="Delayed").mean()*100)
).reset_index()

    fig = px.bar(
    delay_df,
    x="Ship_Mode",
    y="delay_rate",
    title="Delay Percentage by Ship Mode",
    text_auto=True
)

    st.plotly_chart(fig, use_container_width=True)

# ===============================
# DRILL DOWN
# ===============================
elif page=="Route Drill Down":

    r = st.selectbox("Route", df_f["City"].unique())
    d = df_f[df_f["City"]==r]

    st.plotly_chart(px.histogram(d,x="Lead_Time"),use_container_width=True)
    st.subheader("📊 Avg Lead Time by State")

    state_df = df_f.groupby("State/Province").agg(
    avg_lead_time=("Lead_Time","mean")
).reset_index()

    fig = px.bar(
    state_df,
    x="State/Province",
    y="avg_lead_time",
    title="Average Lead Time by State"
)

    st.plotly_chart(fig, use_container_width=True)

# ===============================
# ML
# ===============================
elif page=="ML Prediction":

    df_ml = df.copy()
    df_ml["Status"] = df_ml["Status"].map({"On-Time":0,"Delayed":1})

    X = pd.get_dummies(df_ml[["Ship_Mode","Region","Sales","Lead_Time"]])
    y = df_ml["Status"]

    model = RandomForestClassifier().fit(X,y)

    sm = st.selectbox("Ship Mode",df["Ship_Mode"].unique())
    rg = st.selectbox("Region",df["Region"].unique())
    s = st.slider("Sales",100,10000,500)
    lt = st.slider("Lead Time",1,20,5)

    inp = pd.DataFrame({
        "Ship_Mode":[sm],
        "Region":[rg],
        "Sales":[s],
        "Lead_Time":[lt]
    })

    inp = pd.get_dummies(inp).reindex(columns=X.columns,fill_value=0)

    pred = model.predict(inp)[0]

    if pred==1:
        st.error("Delay Risk ⚠️")
    else:
        st.success("On-Time ✅")