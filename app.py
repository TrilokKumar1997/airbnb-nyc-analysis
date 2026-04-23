import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
import numpy as np

st.set_page_config(
    page_title="NYC Airbnb Price Analyzer",
    layout="wide",
    page_icon="🏙️"
)

@st.cache_data
def load_data():
    df = pd.read_csv("data/airbnb_clean.csv")
    borough_stats = pd.read_csv("data/borough_stats.csv")
    roomtype_stats = pd.read_csv("data/roomtype_stats.csv")
    neighbourhood_stats = pd.read_csv("data/neighbourhood_stats.csv")
    return df, borough_stats, roomtype_stats, neighbourhood_stats

@st.cache_resource
def load_model():
    model = joblib.load("output/price_model.pkl")
    le_borough = joblib.load("output/le_borough.pkl")
    le_neighbourhood = joblib.load("output/le_neighbourhood.pkl")
    le_room = joblib.load("output/le_room.pkl")
    return model, le_borough, le_neighbourhood, le_room

df, borough_stats, roomtype_stats, neighbourhood_stats = load_data()
model, le_borough, le_neighbourhood, le_room = load_model()

st.title("NYC Airbnb Price Analyzer")
st.caption("Data source: Kaggle NYC Airbnb Open Data 2019 · 48,616 listings")
st.divider()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total listings",    f"{len(df):,}")
col2.metric("Average price",     f"${df['price'].mean():.0f}/night")
col3.metric("Median price",      f"${df['price'].median():.0f}/night")
col4.metric("Neighbourhoods",    f"{df['neighbourhood'].nunique()}")
st.divider()

st.subheader("Price predictor")
st.caption("Enter listing details to predict the nightly price")

pc1, pc2, pc3 = st.columns(3)

with pc1:
    borough = st.selectbox("Borough",
        sorted(df["neighbourhood_group"].unique()))

with pc2:
    neighbourhoods = sorted(
        df[df["neighbourhood_group"] == borough]["neighbourhood"].unique()
    )
    neighbourhood = st.selectbox("Neighbourhood", neighbourhoods)

with pc3:
    room_type = st.selectbox("Room type",
        sorted(df["room_type"].unique()))

pc4, pc5, pc6 = st.columns(3)

with pc4:
    minimum_nights = st.slider("Minimum nights", 1, 30, 2)

with pc5:
    availability = st.slider("Availability (days/year)", 0, 365, 180)

with pc6:
    reviews = st.slider("Number of reviews", 0, 500, 50)

if st.button("Predict price"):
    try:
        borough_enc = le_borough.transform([borough])[0]
        neighbourhood_enc = le_neighbourhood.transform([neighbourhood])[0]
        room_enc = le_room.transform([room_type])[0]

        features = np.array([[
            borough_enc,
            neighbourhood_enc,
            room_enc,
            minimum_nights,
            reviews,
            reviews / 12,
            3,
            availability
        ]])

        predicted_price = model.predict(features)[0]

        st.success(f"Predicted nightly price: ${predicted_price:.0f}")

        similar = df[
            (df["neighbourhood_group"] == borough) &
            (df["room_type"] == room_type)
        ]["price"]

        st.info(f"Similar listings in {borough}: avg ${similar.mean():.0f}, "
                f"median ${similar.median():.0f}/night")

    except Exception as e:
        st.error(f"Error: {e}")

st.divider()

st.subheader("Market analysis")

tab1, tab2, tab3, tab4 = st.tabs([
    "By borough", "By room type",
    "Top neighbourhoods", "Listings map"
])

with tab1:
    fig = px.box(df, x="neighbourhood_group", y="price",
        color="neighbourhood_group",
        title="Price distribution by borough",
        labels={"neighbourhood_group": "Borough", "price": "Price ($)"})
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(borough_stats, use_container_width=True)

with tab2:
    fig = px.bar(roomtype_stats, x="room_type", y="avg_price",
        color="room_type",
        title="Average price by room type",
        labels={"room_type": "Room type", "avg_price": "Avg price ($)"})
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(roomtype_stats, use_container_width=True)

with tab3:
    fig = px.bar(neighbourhood_stats, x="avg_price", y="neighbourhood",
        orientation="h", color="neighbourhood_group",
        title="Top 10 most expensive neighbourhoods",
        labels={"avg_price": "Avg price ($)", "neighbourhood": ""})
    fig.update_layout(yaxis=dict(autorange="reversed"))
    st.plotly_chart(fig, use_container_width=True)

with tab4:
    fig = px.scatter_mapbox(
        df.sample(5000),
        lat="latitude", lon="longitude",
        color="neighbourhood_group",
        size="price", hover_name="name",
        hover_data={"price": True, "room_type": True},
        zoom=10, height=500
    )
    fig.update_layout(mapbox_style="open-street-map")
    st.plotly_chart(fig, use_container_width=True)

st.divider()
st.caption("Built by Trilok Kumar · Data Science MS, University of New Haven")
