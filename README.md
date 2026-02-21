# Syntecxhub_Movie-Recommendation-System
Project 1 - Week 4
# 🎬 CineMatch — Movie Recommendation System

A **content-based movie recommendation system** built using the TMDB 5000 dataset. This project includes comprehensive EDA, metadata cleaning, a TF-IDF + cosine similarity recommender engine, qualitative evaluation, and a premium Streamlit web interface.

---

## 📂 Project Structure

```
Project 1/
├── Data/
│   ├── tmdb_5000_movies.csv     # TMDB movies metadata  
│   └── tmdb_5000_credits.csv    # Cast & crew data
├── models/                      # Generated after running notebook
│   ├── movies_processed.pkl     # Cleaned & processed dataframe
│   ├── cosine_sim.pkl           # Precomputed similarity matrix
│   └── indices.pkl              # Title-to-index mapping
├── movie_recommender.ipynb      # Main notebook (EDA + Model)
├── app.py                       # Streamlit UI
├── requirements.txt             # Python dependencies
└── README.md                    # This file
```

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Notebook
Open `movie_recommender.ipynb` in Jupyter and run all cells. This will:
- Perform EDA with interactive visualizations
- Clean and parse metadata (genres, cast, crew, keywords)
- Build TF-IDF vectors and cosine similarity matrix
- Evaluate with sample queries
- Export model artifacts to `models/`

### 3. Launch Streamlit App
```bash
streamlit run app.py
```

> **Note:** The Streamlit app can also process raw CSV data directly if the notebook hasn't been run yet.

---

## 🧠 Methodology

### Data Pipeline
1. **Merge** movies + credits datasets on movie ID
2. **Parse** JSON columns: genres, keywords, cast, crew
3. **Extract** director from crew, top-5 cast members
4. **Build "soup"** — combined text of genres + keywords + cast + director + overview

### Recommendation Engine
- **TF-IDF Vectorization** (20K features, unigrams + bigrams)
- **Cosine Similarity** between all movie pairs
- **Weighted Rating** (IMDB formula) for ranking

### Features
| Feature | Description |
|---------|-------------|
| Genres | Parsed from JSON, space-joined |
| Keywords | Plot keywords from TMDB |
| Cast | Top 5 actors, name-concatenated |
| Director | Extracted from crew, 3× weighted |
| Overview | Full plot description (lowercased) |

---

## 📊 EDA Highlights

- **4,803 movies** spanning decades of cinema
- **20 genres** — Drama & Comedy are most common
- Strong **budget-revenue correlation**
- Interactive visualizations: distributions, correlations, word clouds, genre analysis

---

## 🧪 Evaluation

Qualitative testing across 6 diverse movies:

| Query Movie | Genre | Top Recommendation | Genre Overlap |
|------------|-------|-------------------|---------------|
| The Dark Knight | Action/Crime | Batman Begins | High |
| Toy Story | Animation/Family | Toy Story 2 | High |
| The Godfather | Crime/Drama | The Godfather: Part II | High |
| Interstellar | Sci-Fi/Drama | Gravity | High |
| Titanic | Romance/Drama | Pearl Harbor | High |
| The Avengers | Action/Sci-Fi | Iron Man | High |

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| Data | Pandas, NumPy |
| ML/NLP | Scikit-learn (TF-IDF, Cosine Similarity) |
| Visualization | Matplotlib, Seaborn, Plotly, WordCloud |
| UI | Streamlit |
| Dataset | TMDB 5000 Movies & Credits |

---

*SYNTECXHUB Intern — Week 4, Project 1*


