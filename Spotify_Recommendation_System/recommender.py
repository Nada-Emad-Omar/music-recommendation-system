import numpy as np
import pandas as pd
import faiss
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

# --- Load CSV & embeddings ---
CSV_FILE = "tracks_modern.csv"
EMB_NPY = "tracks_embeddings.npy"

df_songs = pd.read_csv(CSV_FILE)
df_songs.reset_index(drop=True, inplace=True)

embeddings = np.load(EMB_NPY).astype('float32')
embeddings_norm = normalize(embeddings, axis=1)

# --- FAISS index ---
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings_norm)

# --- SentenceTransformer for queries ---
model = SentenceTransformer('all-MiniLM-L6-v2')

# --- Recommendation function ---
def get_recommendations(query, top_k=10, sort_by='relevance'):
    query = str(query).strip().lower()
    if query == "":
        return pd.DataFrame(columns=df_songs.columns)

    # Text match first
    matches_text = df_songs[df_songs['text'].str.lower().str.contains(query, na=False)]
    matches_artist = df_songs[df_songs['artists_joined'].str.lower().str.contains(query, na=False)]
    matches = pd.concat([matches_text, matches_artist]).drop_duplicates()

    if not matches.empty:
        seed_idx = int(matches.sort_values('popularity', ascending=False).index[0])
    else:
        # embedding search fallback
        q_emb = model.encode([query]).astype('float32')
        q_emb = normalize(q_emb, axis=1)
        _, I = index.search(q_emb, 1)
        seed_idx = int(I[0][0])

    # Search similar embeddings
    num_candidates = min(top_k*5, len(df_songs))
    D_emb, I_emb = index.search(embeddings_norm[seed_idx:seed_idx+1], num_candidates+1)
    indices = I_emb[0][1:top_k+1]

    results = df_songs.iloc[indices].copy()
    results['score'] = 1 - D_emb[0][1:top_k+1]  # similarity score

    # Optional sorting
    if sort_by == 'popularity':
        results = results.sort_values('popularity', ascending=False)
    elif sort_by == 'year':
        results = results.sort_values('year', ascending=False)

    # Reset index before returning
    return results.reset_index(drop=True)
