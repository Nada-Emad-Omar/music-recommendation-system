# music-recommendation-system
Spotify-like Music Recommender 🎵

A hybrid music recommendation system.  
It leverages semantic embeddings and audio features to provide accurate and personalized recommendations for songs, artists, genres, or release years.

---

## 🔹 Features

- Search by **song, artist, genre, or year**  
- Hybrid recommendations combining:  
  - **Text embeddings** with SentenceTransformer  
  - **Audio features** like danceability, energy, valence, popularity  
- Spotify-like interface with embedded track info and thumbnails  
- Preprocessing and visualizations for dataset exploration  
- Save and reuse **embeddings** and **FAISS index** for faster queries  

---

## 🔹 Dataset

**Source:** Spotify Dataset 1921-2020 by Vatsal Mavani  
LINK: https://www.kaggle.com/datasets/vatsalmavani/spotify-dataset/data

**Files used:**
- `data.csv` → Track-level details  
- `data_by_artist.csv` → Aggregated artist info  
- `data_by_genres.csv` → Aggregated genre info  
- `data_by_year.csv` → Tracks by release year  
- `data_w_genres.csv` → Tracks with multiple genres  

**Preprocessed outputs:**
- `tracks_modern.csv` → cleaned track data  
- `tracks_embeddings.npy` → saved embeddings  
- `faiss_index.bin` → FAISS index for fast retrieval  

---

## 🔹 Installation

### Create virtual environment (optional)
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

### Install dependencies
pip install -r requirements.txt


**Dependencies include:**
`flask`, `flask_cors`, `numpy`, `pandas`, `faiss-cpu`, `scikit-learn`, `sentence-transformers`, `ipywidgets==7.7.2`, `matplotlib`, `seaborn`, `wordcloud`

---
## 🔹 Workflow Overview

### 1️⃣ Google Colab Notebook
- Used for **data exploration, preprocessing, and generating embeddings**  
- Steps include:  
  1. Load CSVs and explore datasets with `pandas`  
  2. Preprocess tracks, artists, genres, and years  
  3. Normalize audio features and handle missing values  
  4. Create text columns for embedding (`name + artists`, `genres_list`, etc.)  
  5. Generate embeddings with `SentenceTransformer`  
  6. Save `tracks_modern.csv`, `tracks_embeddings.npy`, and `faiss_index.bin`  

> **Note:** Colab is mainly for computation-heavy preprocessing and visualization.

### 2️⃣ Local VS Code Project (`Spotify_Recommendation_System`)
- Production-ready **Flask app + recommender logic**  
- Steps include:  
  1. Install dependencies with `requirements.txt`  
  2. Load preprocessed files (`tracks_modern.csv`, `tracks_embeddings.npy`, `faiss_index.bin`)  
  3. Use `recommender.py` for hybrid recommendations  
  4. Run `app.py` to launch **Spotify-like web interface**  
  5. Search by track, artist, genre, or year and display top recommendations  

---


## 🔹 Project Structure

```
# VS Code / Flask folder
Spotify_Recommendation_System/
├── app.py                  # Flask web app
├── recommender.py          # Core recommendation logic
├── tracks_modern.csv       # Preprocessed track data
├── tracks_embeddings.npy   # Saved embeddings
├── faiss_index.bin         # FAISS index
├── templates/
│   └── index.html          # Web UI template
├── static/
│   └── style.css           # Web UI styles
└── requirements.txt        # Python dependencies

# Colab Notebook 
colab_notebook.ipynb        # Exploratory analysis, preprocessing, embedding generation

```

---

## 🔹 Example Recommendations

![example](https://github.com/user-attachments/assets/39264e17-46c4-4d6e-8adf-e1e21e9177e4)

---

## 🔹 License

MIT License

