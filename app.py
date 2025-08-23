from flask import Flask, render_template, request
import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import math

app = Flask(__name__)

# Load data & model
news_df = pd.read_csv('news_with_cluster.csv')
news_df['Waktu'] = pd.to_datetime(news_df['Waktu'], errors='coerce')
news_df = news_df.dropna(subset=['Waktu'])

with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)
with open('tfidf_matrix.pkl', 'rb') as f:
    tfidf_matrix = pickle.load(f)

@app.route('/')
def home():
    page = int(request.args.get('page', 1))
    per_page = 9
    search_query = request.args.get('q')
    
    # --- LOGIKA DISATUKAN DI SINI ---
    # 1. Tentukan dulu sumber datanya (hasil pencarian atau data acak)
    if search_query:
        # Jika ada query pencarian, filter DataFrame
        mask = news_df['Judul'].str.contains(search_query, case=False, na=False) | \
               news_df['Content'].str.contains(search_query, case=False, na=False)
        news_to_display = news_df[mask].reset_index(drop=True)
    else:
        # Jika tidak, gunakan sampel acak
        news_to_display = news_df.sample(n=60).reset_index(drop=True)

    # 2. Terapkan pagination ke sumber data yang sudah ditentukan
    total_items = len(news_to_display)
    total_pages = math.ceil(total_items / per_page)
    start = (page - 1) * per_page
    end = start + per_page
    paged_news = news_to_display.iloc[start:end]

    return render_template('index.html',
                           samples=paged_news.to_dict(orient='records'),
                           total_pages=total_pages,
                           current_page=page,
                           search_query=search_query,
                           total_items=total_items) # Kirim total item untuk info

@app.route('/detail/<judul>')
def detail(judul):
    try:
        article = news_df[news_df['Judul'] == judul].iloc[0]
    except IndexError:
        return "Artikel tidak ditemukan", 404

    idx = article.name
    cosine_sim = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
    similar_idx = cosine_sim.argsort()[-6:-1][::-1]
    recommendations = news_df.iloc[similar_idx].to_dict(orient='records')
    
    return render_template('detail.html', 
                           article=article.to_dict(), 
                           recommendations=recommendations)

if __name__ == '__main__':
    app.run(debug=True)