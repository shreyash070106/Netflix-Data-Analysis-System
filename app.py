import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor

# -----------------------------
# Page Config
# -----------------------------

st.set_page_config(
    page_title="Netflix Analytics Dashboard",
    layout="wide"
)

# -----------------------------
# Netflix Styling
# -----------------------------

st.markdown("""
<style>

/* Main background */

.stApp {
    background-color: #0f0f0f;
    color: white;
    font-family: Arial, Helvetica, sans-serif;
}

/* Title styling */

h1 {
    color: #E50914;
    text-align: center;
    font-weight: 800;
}

/* Section titles */

h2, h3 {
    color: white;
    margin-top: 20px;
}

/* Metric cards */

div[data-testid="metric-container"] {
    background-color: #181818;
    border: 1px solid #E50914;
    border-radius: 10px;
    padding: 15px;
}

/* Chart container */

.chart-card {
    background-color: #181818;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #2a2a2a;
}

/* Trending movie cards */

.movie-card {
    background-color: #181818;
    border: 1px solid #E50914;
    border-radius: 12px;
    padding: 12px;
    text-align: center;
    height: 100%;
}
            
.movie-rank {
    color: #E50914;
    font-size: 22px;
    font-weight: bold;
}
    

</style>
""", unsafe_allow_html=True)
# -----------------------------
# Title
# -----------------------------

st.title("NETFLIX ANALYTICS DASHBOARD")

st.markdown(
    "<div style='text-align:center;color:#bbbbbb;'>Content insights and popularity prediction</div>",
    unsafe_allow_html=True
)

# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("NetFlix.csv")



df.fillna("Unknown", inplace=True)

df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
df['year_added'] = df['date_added'].dt.year

# -----------------------------
# Poster Fetch Function
# -----------------------------
@st.cache_data
def fetch_poster(movie_title):

    api_key = "869041fc5ee78743f0ae07adc0c69a0d"

    url = "https://api.themoviedb.org/3/search/multi"

    params = {
        "api_key": api_key,
        "query": movie_title
    }

    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
        "Accept": "application/json",
        "Connection": "keep-alive"
    }

    session = requests.Session()

    for _ in range(3):   # retry up to 3 times

        try:

            response = session.get(
                url,
                params=params,
                headers=headers,
                timeout=8
            )

            if response.status_code == 200:

                data = response.json()

                if "results" in data and len(data["results"]) > 0:

                    poster_path = data["results"][0].get("poster_path")

                    if poster_path:
                        return "https://image.tmdb.org/t/p/w500" + poster_path

                return None

        except requests.exceptions.RequestException:
            continue

    return None

# -----------------------------
# Popularity Score
# -----------------------------

# -----------------------------
# Realistic Popularity Calculation
# -----------------------------

# 1️ Recency Score (newer movies are more popular)
current_year = df['release_year'].max()

df['recency_score'] = 1 - (
    (current_year - df['release_year']) /
    (current_year - df['release_year'].min())
)

# 2️ Genre Popularity
genre_counts = df['genres'].value_counts()

df['genre_score'] = df['genres'].map(genre_counts)

df['genre_score'] = df['genre_score'] / df['genre_score'].max()

# 3️ Content Type Score
df['type_score'] = df['type'].apply(lambda x: 1 if x == "Movie" else 0.8)

# 4️ Duration Score
df['duration_num'] = df['duration'].astype(str).str.extract('(\d+)').astype(float)

df['duration_score'] = df['duration_num'] / df['duration_num'].max()

# 5️ Final Popularity Score (0-10 scale)
df['popularity'] = (
    0.4 * df['recency_score'] +
    0.3 * df['genre_score'] +
    0.2 * df['type_score'] +
    0.1 * df['duration_score']
) * 10
tv_df = df[df['type'] == "TV Show"].copy()
movie_df = df[df['type'] == "Movie"].copy()
# -----------------------------
# Encode Data
# -----------------------------

le = LabelEncoder()

df['type'] = le.fit_transform(df['type'])
df['rating'] = le.fit_transform(df['rating'])
df['country'] = le.fit_transform(df['country'])
df['genres'] = le.fit_transform(df['genres'])

# -----------------------------
# Train Model
# -----------------------------

X = df[['type','release_year','rating','country','genres']]
y = df['popularity']

model = RandomForestRegressor()
model.fit(X,y)

# -----------------------------
# Dashboard Metrics
# -----------------------------

# -----------------------------
# Dashboard Overview
# -----------------------------

st.subheader("Catalog Overview")

total_titles = len(df)

total_movies = len(df[df['type'] == 0])   # encoded value for Movie
total_tv = len(df[df['type'] == 1])       # encoded value for TV Show

avg_release_year = int(df['release_year'].mean())

most_common_rating = df['rating'].mode()[0]

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Titles", total_titles)

col2.metric("Movies", total_movies)

col3.metric("TV Shows", total_tv)

col4.metric("Average Release Year", avg_release_year)

st.divider()
# -----------------------------
# Charts Section
# -----------------------------

# -----------------------------
# Netflix Chart Theme
# -----------------------------

plt.style.use("dark_background")

NETFLIX_RED = "#E50914"
NETFLIX_DARK_RED = "#B20710"
NETFLIX_BG = "#0f0f0f"
NETFLIX_PANEL = "#1c1c1c"
GRID_COLOR = "#333333"

# -----------------------------
# Trending Movies (TOP SECTION)
# -----------------------------



# -----------------------------
# Top 10 Trending Movies
# -----------------------------

st.subheader("Top 10 Trending Movies")

top_movies = movie_df.sort_values("popularity", ascending=False).head(10)

for row_start in range(0, 10, 5):

    cols = st.columns(5)

    for col, (_, row) in zip(cols, top_movies.iloc[row_start:row_start+5].iterrows()):

        poster = fetch_poster(row["title"])

        with col:

            st.markdown('<div class="movie-card">', unsafe_allow_html=True)

            if poster:
                st.image(poster, use_container_width=True)

            st.markdown(
                f"""
                <div class="movie-rank">#{row_start + list(top_movies.iloc[row_start:row_start+5].index).index(row.name) + 1}</div>
                <div><b>{row['title']}</b></div>
                <div>Year: {row['release_year']}</div>
                <div>Popularity: {round(row['popularity'],2)}</div>
                """,
                unsafe_allow_html=True
            )

            st.markdown('</div>', unsafe_allow_html=True)

 # -----------------------------
# Top 10 Trending TV Shows
# -----------------------------

st.subheader("Top 10 Trending TV Shows")

top_tv = tv_df.sort_values("popularity", ascending=False).head(10)

for row_start in range(0, 10, 5):

    cols = st.columns(5)

    for col, (_, row) in zip(cols, top_tv.iloc[row_start:row_start+5].iterrows()):

        poster = fetch_poster(row["title"])

        with col:

            st.markdown('<div class="movie-card">', unsafe_allow_html=True)

            if poster:
                st.image(poster, use_container_width=True)

            st.markdown(
                f"""
                <div class="movie-rank">#{row_start + list(top_tv.iloc[row_start:row_start+5].index).index(row.name) + 1}</div>
                <div><b>{row['title']}</b></div>
                <div>Year: {row['release_year']}</div>
                <div>Popularity: {round(row['popularity'],2)}</div>
                """,
                unsafe_allow_html=True
            )

            st.markdown('</div>', unsafe_allow_html=True)
# -----------------------------
# Charts Section
# -----------------------------

st.subheader("Content Comparisons")

# -------- FIRST ROW --------

col1, col2 = st.columns(2)

with col1:

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)

    fig1, ax1 = plt.subplots()

    sns.countplot(
        x='type',
        data=df,
        ax=ax1,
        palette=["#E50914","#B20710"]
    )

    ax1.set_facecolor("#181818")
    fig1.patch.set_facecolor("#0f0f0f")

    ax1.set_title("Movies vs TV Shows Distribution", color="white")

    st.pyplot(fig1)

    st.markdown('</div>', unsafe_allow_html=True)

with col2:

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)

    fig2, ax2 = plt.subplots()

    df['release_year'].value_counts().sort_index().plot(
        ax=ax2,
        color="#E50914",
        linewidth=2
    )

    ax2.set_facecolor("#181818")
    fig2.patch.set_facecolor("#0f0f0f")

    ax2.set_title("Content Growth Over Years", color="white")

    st.pyplot(fig2)

    st.markdown('</div>', unsafe_allow_html=True)

# -------- SECOND ROW --------

col3, col4 = st.columns(2)

# Rating Distribution
with col3:

    fig3, ax3 = plt.subplots()

    fig3.patch.set_facecolor(NETFLIX_BG)
    ax3.set_facecolor(NETFLIX_PANEL)

    sns.countplot(
        y='rating',
        data=df,
        ax=ax3,
        color=NETFLIX_RED
    )

    ax3.set_title("Rating Distribution", color="white")
    ax3.tick_params(colors="white")
    ax3.grid(color=GRID_COLOR)

    st.pyplot(fig3)

# Content Release Trend
with col4:

    fig4, ax4 = plt.subplots()

    fig4.patch.set_facecolor(NETFLIX_BG)
    ax4.set_facecolor(NETFLIX_PANEL)

    df['release_year'].value_counts().sort_index().plot(
        ax=ax4,
        color=NETFLIX_RED,
        linewidth=2
    )

    ax4.set_title("Content Release Trend", color="white")
    ax4.tick_params(colors="white")
    ax4.grid(color=GRID_COLOR)

    st.pyplot(fig4)

st.divider()
