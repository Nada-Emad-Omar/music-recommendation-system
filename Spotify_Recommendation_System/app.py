from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from recommender import get_recommendations, df_songs
import os

app = Flask(__name__)
CORS(app)

# --- Add embed URLs to dataframe ---
def add_embed_url(df):
    df['embed_url'] = df['id'].apply(lambda x: f"https://open.spotify.com/embed/track/{x}")
    return df

df_songs = add_embed_url(df_songs)

# --- Flask routes ---
@app.route("/")
def home():
    popular = df_songs.sort_values('popularity', ascending=False).head(12)
    seeds = popular[['name','artists_joined']].to_dict(orient='records')
    return render_template("index.html", seeds=seeds)

@app.route("/api/recommend", methods=["POST"])
def api_recommend():
    data = request.json or {}
    query = data.get('query', '')
    top_k = int(data.get('top_k', 10))
    sort_by = data.get('sort_by', 'relevance')
    
    recs = get_recommendations(query=query, top_k=top_k, sort_by=sort_by)
    
    results = []
    for _, row in recs.iterrows():
        results.append({
            "name": row['name'],
            "artists_joined": row['artists_joined'],
            "year": row['year'],
            "popularity": row['popularity'],
            "embed_url": row['embed_url']  # use embed URL
        })
    
    return jsonify({"results": results})

if __name__ == "__main__":
    app.run(debug=True)
