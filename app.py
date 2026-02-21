"""
🎬 Movie Recommendation System — Streamlit App
================================================
Content-based movie recommender using TMDB 5000 dataset.
Uses TF-IDF + Cosine Similarity for recommendations.

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import ast
from pathlib import Path

# ─────────────────────────────────────────────────
# 🎨 PAGE CONFIGURATION
# ─────────────────────────────────────────────────
st.set_page_config(
    page_title="🎬 CineMatch — Movie Recommender",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────
# 🎨 CUSTOM CSS — Premium Dark Theme
# ─────────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Import Google Font ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    /* ── Global Styles ── */
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* ── Header Styling ── */
    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid rgba(108, 92, 231, 0.3);
        box-shadow: 0 20px 60px rgba(108, 92, 231, 0.15);
    }
    
    .main-header h1 {
        color: #ffffff;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        letter-spacing: -0.5px;
        background: linear-gradient(135deg, #a29bfe, #6c5ce7, #fd79a8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .main-header p {
        color: #b2bec3;
        font-size: 1.15rem;
        font-weight: 300;
        margin: 0;
    }
    
    /* ── Movie Card ── */
    .movie-card {
        background: linear-gradient(145deg, #1e1e2e, #2a2a3e);
        border-radius: 16px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(108, 92, 231, 0.2);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
    }
    
    .movie-card:hover {
        transform: translateY(-4px);
        border-color: rgba(108, 92, 231, 0.5);
        box-shadow: 0 12px 40px rgba(108, 92, 231, 0.2);
    }
    
    .movie-title {
        color: #ffffff;
        font-size: 1.3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        letter-spacing: -0.3px;
    }
    
    .movie-meta {
        color: #a0a0b8;
        font-size: 0.9rem;
        margin-bottom: 0.4rem;
        line-height: 1.5;
    }
    
    .movie-overview {
        color: #8a8aa0;
        font-size: 0.88rem;
        line-height: 1.6;
        margin-top: 0.75rem;
        border-top: 1px solid rgba(255,255,255,0.06);
        padding-top: 0.75rem;
    }
    
    /* ── Badges ── */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 0.3rem;
        margin-bottom: 0.3rem;
    }
    
    .badge-genre {
        background: rgba(108, 92, 231, 0.2);
        color: #a29bfe;
        border: 1px solid rgba(108, 92, 231, 0.3);
    }
    
    .badge-sim {
        background: rgba(0, 184, 148, 0.2);
        color: #55efc4;
        border: 1px solid rgba(0, 184, 148, 0.3);
    }
    
    .badge-rating {
        background: rgba(253, 203, 110, 0.2);
        color: #fdcb6e;
        border: 1px solid rgba(253, 203, 110, 0.3);
    }
    
    /* ── Sidebar ── */
    .sidebar-info {
        background: linear-gradient(145deg, #1e1e2e, #2a2a3e);
        border-radius: 12px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        border: 1px solid rgba(108, 92, 231, 0.15);
    }
    
    .sidebar-info h4 {
        color: #a29bfe;
        margin-bottom: 0.5rem;
    }
    
    .sidebar-info p {
        color: #b2bec3;
        font-size: 0.85rem;
        margin: 0.3rem 0;
    }
    
    /* ── Stats Cards ── */
    .stat-card {
        background: linear-gradient(145deg, #1e1e2e, #2a2a3e);
        border-radius: 16px;
        padding: 1.5rem;
        text-align: center;
        border: 1px solid rgba(108, 92, 231, 0.2);
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.2);
    }
    
    .stat-number {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #a29bfe, #6c5ce7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .stat-label {
        color: #a0a0b8;
        font-size: 0.9rem;
        font-weight: 500;
        margin-top: 0.3rem;
    }
    
    /* ── Input Movie Card ── */
    .input-movie-card {
        background: linear-gradient(145deg, #1a1a3e, #2d2b55);
        border-radius: 16px;
        padding: 1.8rem;
        margin-bottom: 2rem;
        border: 1px solid rgba(108, 92, 231, 0.4);
        box-shadow: 0 8px 30px rgba(108, 92, 231, 0.15);
    }
    
    .input-movie-card h3 {
        color: #a29bfe;
        font-size: 1.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* ── Rank Badge ── */
    .rank-badge {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 32px;
        height: 32px;
        border-radius: 50%;
        background: linear-gradient(135deg, #6c5ce7, #a29bfe);
        color: white;
        font-weight: 700;
        font-size: 0.85rem;
        margin-right: 0.75rem;
        flex-shrink: 0;
    }
    
    /* ── Divider ── */
    .section-divider {
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(108, 92, 231, 0.4), transparent);
        margin: 2rem 0;
        border: none;
    }
    
    /* ── Hide default Streamlit elements ── */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Metric styling */
    div[data-testid="metric-container"] {
        background: linear-gradient(145deg, #1e1e2e, #2a2a3e);
        border-radius: 12px;
        padding: 1rem;
        border: 1px solid rgba(108, 92, 231, 0.15);
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────
# 📦 DATA LOADING
# ─────────────────────────────────────────────────
@st.cache_data
def load_data():
    """Load preprocessed data. Falls back to raw CSV processing if models aren't exported yet."""
    models_dir = Path("models")
    
    if models_dir.exists() and (models_dir / "movies_processed.pkl").exists():
        # Load from exported pickle files
        df = pd.read_pickle(models_dir / "movies_processed.pkl")
        with open(models_dir / "cosine_sim.pkl", "rb") as f:
            cosine_sim = pickle.load(f)
        with open(models_dir / "indices.pkl", "rb") as f:
            indices = pickle.load(f)
        return df, cosine_sim, indices, True
    else:
        # Process raw data if models haven't been exported
        return process_raw_data()


def safe_parse(val):
    """Safely parse JSON-like string columns."""
    if isinstance(val, list):
        return val
    try:
        return ast.literal_eval(str(val))
    except (ValueError, SyntaxError):
        return []


def extract_names(json_col, max_items=None):
    """Extract 'name' field from a list of dicts."""
    parsed = safe_parse(json_col)
    names = [item['name'] for item in parsed if 'name' in item]
    if max_items:
        names = names[:max_items]
    return names


def get_director(crew_data):
    """Extract the director's name from the crew list."""
    parsed = safe_parse(crew_data)
    for member in parsed:
        if member.get('job') == 'Director':
            return member.get('name', '')
    return ''


@st.cache_data
def process_raw_data():
    """Process raw TMDB CSV files and build the recommendation engine."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import linear_kernel
    
    movies = pd.read_csv("Data/tmdb_5000_movies.csv")
    credits = pd.read_csv("Data/tmdb_5000_credits.csv")
    
    credits.rename(columns={'movie_id': 'id', 'title': 'title_credits'}, inplace=True)
    df = movies.merge(credits, on='id', how='left')
    
    # Parse metadata
    df['genres_list'] = df['genres'].apply(lambda x: extract_names(x))
    df['genres_clean'] = df['genres_list'].apply(lambda x: ' '.join([g.replace(' ', '') for g in x]))
    df['keywords_list'] = df['keywords'].apply(lambda x: extract_names(x))
    df['keywords_clean'] = df['keywords_list'].apply(lambda x: ' '.join([k.replace(' ', '') for k in x]))
    df['cast_list'] = df['cast'].apply(lambda x: extract_names(x, max_items=5))
    df['cast_clean'] = df['cast_list'].apply(lambda x: ' '.join([c.replace(' ', '') for c in x]))
    df['director'] = df['crew'].apply(get_director)
    df['director_clean'] = df['director'].apply(lambda x: x.replace(' ', '') if isinstance(x, str) else '')
    df['overview'] = df['overview'].fillna('')
    
    df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')
    df['release_year'] = df['release_date'].dt.year
    
    # Weighted rating
    C = df['vote_average'].mean()
    m = df['vote_count'].quantile(0.60)
    df['weighted_score'] = df.apply(
        lambda row: (row['vote_count'] / (row['vote_count'] + m)) * row['vote_average'] + 
                     (m / (row['vote_count'] + m)) * C, axis=1
    )
    
    # Build soup
    def build_soup(row):
        parts = [
            row['genres_clean'],
            row['keywords_clean'],
            row['cast_clean'],
            row['director_clean'] * 3,
            row['overview'].lower()
        ]
        return ' '.join(parts)
    
    df['soup'] = df.apply(build_soup, axis=1)
    
    # TF-IDF + Cosine Similarity
    tfidf = TfidfVectorizer(max_features=20000, stop_words='english',
                            ngram_range=(1, 2), min_df=2, max_df=0.85)
    tfidf_matrix = tfidf.fit_transform(df['soup'])
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    
    df = df.reset_index(drop=True)
    indices = pd.Series(df.index, index=df['title']).drop_duplicates()
    
    # Keep needed columns
    keep_cols = ['id', 'title', 'overview', 'genres_list', 'keywords_list', 'cast_list',
                 'director', 'vote_average', 'vote_count', 'popularity', 'runtime',
                 'release_year', 'weighted_score', 'budget', 'revenue', 'soup']
    df = df[[c for c in keep_cols if c in df.columns]]
    
    return df, cosine_sim, indices, False


def get_recommendations(title, cosine_sim, df, indices, top_n=10):
    """Get content-based movie recommendations."""
    if title not in indices:
        matches = df[df['title'].str.contains(title, case=False, na=False)]
        if len(matches) == 0:
            return None, f"Movie '{title}' not found."
        title = matches.iloc[0]['title']
    
    idx = indices[title]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]
    
    movie_indices = [i[0] for i in sim_scores]
    scores = [round(i[1], 4) for i in sim_scores]
    
    result = df.iloc[movie_indices].copy()
    result['similarity'] = scores
    result = result.reset_index(drop=True)
    
    return result, title


# ─────────────────────────────────────────────────
# 🚀 MAIN APP
# ─────────────────────────────────────────────────
def main():
    # Load data
    try:
        df, cosine_sim, indices, from_cache = load_data()
    except Exception as e:
        st.error(f"⚠️ Error loading data: {e}")
        st.info("Please run the Jupyter notebook first to generate the model files, or ensure the raw data files are in the `Data/` folder.")
        return
    
    # ── Header ──
    st.markdown("""
    <div class="main-header">
        <h1>🎬 CineMatch</h1>
        <p>AI-Powered Movie Recommendation Engine · Content-Based Filtering · TMDB 5000</p>
    </div>
    """, unsafe_allow_html=True)
    
    # ── Sidebar ──
    with st.sidebar:
        st.markdown("## 🎯 Configuration")
        
        # Number of recommendations
        top_n = st.slider("Number of recommendations", min_value=3, max_value=20, value=10, step=1)
        
        st.markdown("---")
        
        # Stats
        st.markdown("""
        <div class="sidebar-info">
            <h4>📊 Dataset Stats</h4>
            <p>🎥 <strong>{:,}</strong> Movies</p>
            <p>🎭 <strong>{}</strong> Unique Genres</p>
            <p>📅 Years: <strong>{:.0f}</strong> — <strong>{:.0f}</strong></p>
        </div>
        """.format(
            len(df),
            len(set(g for gl in df['genres_list'] for g in gl)) if 'genres_list' in df.columns else 'N/A',
            df['release_year'].min() if 'release_year' in df.columns and df['release_year'].notna().any() else 0,
            df['release_year'].max() if 'release_year' in df.columns and df['release_year'].notna().any() else 0
        ), unsafe_allow_html=True)
        
        st.markdown("""
        <div class="sidebar-info">
            <h4>🧠 How It Works</h4>
            <p>1️⃣ Select a movie you enjoy</p>
            <p>2️⃣ Our engine analyzes genres, cast, director, keywords & plot</p>
            <p>3️⃣ TF-IDF vectorization creates a text fingerprint</p>
            <p>4️⃣ Cosine similarity finds the closest matches</p>
            <p>5️⃣ Results ranked by similarity score</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Genre filter
        st.markdown("---")
        st.markdown("### 🎭 Filter by Genre")
        all_genres = sorted(set(g for gl in df['genres_list'] for g in gl))
        selected_genres = st.multiselect("Include genres", all_genres, default=[])
        
        # Source info
        data_src = "📦 Pre-computed models" if from_cache else "🔄 Processed from raw CSVs"
        st.markdown(f"---\n*Data source: {data_src}*")
    
    # ── Main Content ──
    # Movie search
    all_titles = sorted(df['title'].unique().tolist())
    
    col_search1, col_search2 = st.columns([3, 1])
    with col_search1:
        selected_movie = st.selectbox(
            "🔍 Search for a movie",
            options=all_titles,
            index=all_titles.index("The Dark Knight") if "The Dark Knight" in all_titles else 0,
            help="Start typing to search for any movie in the TMDB 5000 dataset"
        )
    with col_search2:
        st.markdown("<br>", unsafe_allow_html=True)
        search_clicked = st.button("🚀 Get Recommendations", type="primary", use_container_width=True)
    
    if selected_movie and search_clicked:
        # Show selected movie info
        movie_info = df[df['title'] == selected_movie].iloc[0]
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Input movie card
        genres_html = ' '.join([f'<span class="badge badge-genre">{g}</span>' for g in movie_info.get('genres_list', [])])
        cast_names = ', '.join(movie_info.get('cast_list', [])[:5]) if isinstance(movie_info.get('cast_list'), list) else ''
        
        st.markdown(f"""
        <div class="input-movie-card">
            <h3>🎬 {selected_movie}</h3>
            <div style="margin: 0.75rem 0;">{genres_html}</div>
            <div class="movie-meta">🎬 Director: <strong>{movie_info.get('director', 'N/A')}</strong> · 
            📅 Year: <strong>{int(movie_info.get('release_year', 0)) if pd.notna(movie_info.get('release_year')) else 'N/A'}</strong> · 
            ⭐ Rating: <strong>{movie_info.get('vote_average', 0):.1f}</strong>/10 
            ({int(movie_info.get('vote_count', 0)):,} votes)</div>
            <div class="movie-meta">🎭 Cast: {cast_names}</div>
            <div class="movie-overview">{movie_info.get('overview', '')[:300]}{'...' if len(str(movie_info.get('overview', ''))) > 300 else ''}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Get recommendations
        with st.spinner("🧠 Computing similarity scores..."):
            results, matched_title = get_recommendations(selected_movie, cosine_sim, df, indices, top_n=top_n)
        
        if results is None:
            st.error(f"❌ {matched_title}")
            return
        
        # Apply genre filter if selected
        if selected_genres:
            results = results[results['genres_list'].apply(
                lambda gl: any(g in gl for g in selected_genres) if isinstance(gl, list) else False
            )]
        
        # Stats row
        st.markdown(f"### 🎯 Top {len(results)} Recommendations for *{matched_title}*")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("🎯 Top Match", results.iloc[0]['title'][:20] + "..." if len(results.iloc[0]['title']) > 20 else results.iloc[0]['title'])
        with col2:
            st.metric("📊 Avg Similarity", f"{results['similarity'].mean():.2%}")
        with col3:
            st.metric("⭐ Avg Rating", f"{results['vote_average'].mean():.1f}/10")
        with col4:
            st.metric("🏆 Best Similarity", f"{results['similarity'].max():.2%}")
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        # Recommendation cards
        for rank, (_, row) in enumerate(results.iterrows(), 1):
            genres_list = row.get('genres_list', [])
            if isinstance(genres_list, list):
                genres_badges = ' '.join([f'<span class="badge badge-genre">{g}</span>' for g in genres_list])
            else:
                genres_badges = ''
            
            cast_text = ', '.join(row.get('cast_list', [])[:4]) if isinstance(row.get('cast_list'), list) else ''
            year = int(row['release_year']) if pd.notna(row.get('release_year')) else 'N/A'
            runtime = f"{int(row['runtime'])} min" if pd.notna(row.get('runtime')) else 'N/A'
            overview = str(row.get('overview', ''))[:250]
            if len(str(row.get('overview', ''))) > 250:
                overview += '...'
            
            sim_pct = row['similarity'] * 100
            
            # Color gradient based on similarity
            if sim_pct > 30:
                sim_color = '#55efc4'
            elif sim_pct > 15:
                sim_color = '#fdcb6e'
            else:
                sim_color = '#fab1a0'
            
            st.markdown(f"""
            <div class="movie-card">
                <div style="display: flex; align-items: flex-start;">
                    <span class="rank-badge">{rank}</span>
                    <div style="flex: 1;">
                        <div class="movie-title">{row['title']}</div>
                        <div style="margin: 0.4rem 0;">
                            {genres_badges}
                            <span class="badge badge-sim" style="color: {sim_color};">📊 {sim_pct:.1f}% match</span>
                            <span class="badge badge-rating">⭐ {row.get('vote_average', 0):.1f}</span>
                        </div>
                        <div class="movie-meta">
                            🎬 Director: <strong>{row.get('director', 'N/A')}</strong> · 
                            📅 {year} · ⏱️ {runtime} · 
                            👥 {int(row.get('vote_count', 0)):,} votes
                        </div>
                        <div class="movie-meta">🎭 Cast: {cast_text}</div>
                        <div class="movie-overview">{overview}</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        # ── Similarity Distribution Chart ──
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown("### 📊 Similarity Score Distribution")
        
        import plotly.express as px
        
        chart_df = results[['title', 'similarity']].copy()
        chart_df['similarity_pct'] = chart_df['similarity'] * 100
        chart_df = chart_df.sort_values('similarity_pct', ascending=True)
        
        fig = px.bar(
            chart_df, x='similarity_pct', y='title',
            orientation='h',
            labels={'similarity_pct': 'Similarity (%)', 'title': ''},
            color='similarity_pct',
            color_continuous_scale='viridis'
        )
        fig.update_layout(
            template='plotly_dark',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=max(400, len(chart_df) * 40),
            coloraxis_showscale=False,
            margin=dict(l=200, r=20, t=10, b=40),
            xaxis=dict(gridcolor='rgba(255,255,255,0.05)'),
            yaxis=dict(gridcolor='rgba(255,255,255,0.05)')
        )
        st.plotly_chart(fig, use_container_width=True)
    
    elif not search_clicked:
        # Show welcome state
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        st.markdown("### 🌟 Popular Movies to Try")
        
        popular_movies = [
            ("🦇", "The Dark Knight", "Action, Crime, Drama"),
            ("🧸", "Toy Story", "Animation, Comedy, Family"),
            ("🚀", "Interstellar", "Adventure, Drama, Sci-Fi"),
            ("🚢", "Titanic", "Drama, Romance"),
            ("🦸", "The Avengers", "Sci-Fi, Action, Adventure"),
            ("🎩", "The Godfather", "Drama, Crime"),
            ("🌍", "Avatar", "Action, Adventure, Fantasy"),
            ("🧊", "Inception", "Action, Thriller, Sci-Fi"),
        ]
        
        cols = st.columns(4)
        for i, (emoji, title, genres) in enumerate(popular_movies):
            with cols[i % 4]:
                st.markdown(f"""
                <div class="movie-card" style="text-align: center; padding: 1.5rem 1rem;">
                    <div style="font-size: 2.5rem; margin-bottom: 0.5rem;">{emoji}</div>
                    <div class="movie-title" style="font-size: 1.05rem;">{title}</div>
                    <div class="movie-meta">{genres}</div>
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("""
        <div style="text-align: center; padding: 2rem; color: #636e72; font-size: 1rem;">
            👆 Select a movie from the dropdown above and click <strong>Get Recommendations</strong>
        </div>
        """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
