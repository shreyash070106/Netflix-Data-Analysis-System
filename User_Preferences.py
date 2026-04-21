import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="User Preference Filtering",
    layout="wide"
)

st.title("Find Content Based on Your Preferences")

# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("NetFlix.csv")

df.fillna("Unknown", inplace=True)

# -----------------------------
# Sidebar Filters
# -----------------------------

st.sidebar.header("Filter Preferences")

type_filter = st.sidebar.multiselect(
    "Content Type",
    options=df["type"].unique(),
    default=df["type"].unique()
)

genre_filter = st.sidebar.multiselect(
    "Genre",
    options=sorted(df["genres"].unique()),
    default=df["genres"].unique()
)

rating_filter = st.sidebar.multiselect(
    "Rating",
    options=df["rating"].unique(),
    default=df["rating"].unique()
)

year_range = st.sidebar.slider(
    "Release Year Range",
    int(df["release_year"].min()),
    int(df["release_year"].max()),
    (int(df["release_year"].min()), int(df["release_year"].max()))
)

# -----------------------------
# Apply Filters
# -----------------------------

filtered_df = df[
    (df["type"].isin(type_filter)) &
    (df["genres"].isin(genre_filter)) &
    (df["rating"].isin(rating_filter)) &
    (df["release_year"].between(year_range[0], year_range[1]))
]

# -----------------------------
# Display Results
# -----------------------------

st.subheader("Matching Titles")

st.write("Results Found:", len(filtered_df))

st.dataframe(
    filtered_df[[
        "title",
        "type",
        "genres",
        "release_year",
        "rating",
        "duration"
    ]],
    use_container_width=True
)