import os
import re
import pickle
import random
import pandas as pd
from flask import Flask, render_template, request, abort
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# ==========================
# Load data
# ==========================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "news_with_cluster.csv")
VEC_PATH = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")
MAT_PATH = os.path.join(BASE_DIR, "tfidf_matrix.pkl")

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"File CSV tidak ditemukan: {CSV_PATH}")

df = pd.read_csv(CSV_PATH)

# pastikan kolom pokok ada
for col in ["Judul", "Content"]:
    if col not in df.columns:
        raise ValueError(f"Kolom '{col}' wajib ada di {CSV_PATH}")

# pastikan rowid stabil
if "rowid" not in df.columns:
    df = df.reset_index().rename(columns={"index": "rowid"})

# ==========================
# Preprocess & combined text
# ==========================
def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"\d+", " ", text)         # hapus angka
    text = re.sub(r"[^\w\s]", " ", text)     # hapus simbol
    text = re.sub(r"\s+", " ", text).strip()
    return text

if "combined" not in df.columns:
    # bobotkan judul sedikit dengan mengulang sekali
    df["cleaned"] = df["Content"].astype(str).apply(clean_text)
    df["combined"] = df["Judul"].astype(str) + " " + df["cleaned"] + " " + df["Judul"].astype(str)
else:
    # jika sudah ada, tetap sediakan cleaned untuk jaga2
    if "cleaned" not in df.columns:
        df["cleaned"] = df["Content"].astype(str).apply(clean_text)

# ==========================
# TF-IDF: load kalau ada, kalau mismatch -> fit ulang
# ==========================
def fit_tfidf(corpus):
    vec = TfidfVectorizer(max_features=10000, ngram_range=(1, 3))
    mat = vec.fit_transform(corpus)
    return vec, mat

vectorizer = None
tfidf_matrix = None

if os.path.exists(VEC_PATH) and os.path.exists(MAT_PATH):
    try:
        vectorizer = pickle.load(open(VEC_PATH, "rb"))
        tfidf_matrix = pickle.load(open(MAT_PATH, "rb"))
        # cek ukuran
        if tfidf_matrix.shape[0] != len(df):
            # beda jumlah dokumen -> refit
            vectorizer, tfidf_matrix = fit_tfidf(df["combined"])
    except Exception:
        vectorizer, tfidf_matrix = fit_tfidf(df["combined"])
else:
    vectorizer, tfidf_matrix = fit_tfidf(df["combined"])
    # optional: simpan ulang agar konsisten ke depan
    try:
        pickle.dump(vectorizer, open(VEC_PATH, "wb"))
        pickle.dump(tfidf_matrix, open(MAT_PATH, "wb"))
    except Exception:
        pass

# ==========================
# Search: tampilkan semua yang match kata/phrase, urut skor
# ==========================
def search_results(q: str) -> pd.DataFrame:
    q = (q or "").strip()
    if not q:
        return pd.DataFrame(columns=["rowid", "Judul", "Content", "score"])

    # vektor query berbasis cleaned
    q_vec = vectorizer.transform([clean_text(q)])
    scores = cosine_similarity(q_vec, tfidf_matrix).flatten()

    # word-boundary filter biar "bunga" ≠ "Bung"
    # dukung multi-kata: cocokkan frasa penuh ATAU salah-satu token
    q_lower = q.lower().strip()
    tokens = [t for t in re.findall(r"\b[\w-]+\b", q_lower) if t]
    patterns = []
    # frasa penuh
    if len(q_lower.split()) > 1:
        patterns.append(rf"\b{re.escape(q_lower)}\b")
    # token per kata
    for t in tokens:
        patterns.append(rf"\b{re.escape(t)}\b")

    if patterns:
        big_pattern = "(" + "|".join(patterns) + ")"
        mask = (
            df["Judul"].astype(str).str.lower().str.contains(big_pattern, regex=True, na=False) |
            df["Content"].astype(str).str.lower().str.contains(big_pattern, regex=True, na=False)
        )
    else:
        mask = True  # fallback (harusnya ga kejadian)

    res = df.loc[mask, ["rowid", "Judul", "Content"]].copy()
    res["score"] = scores[mask]
    res = res.sort_values("score", ascending=False)
    return res

# ==========================
# Recommendation: pure cosine top-N (urut relevansi)
# ==========================
def get_recommendations_by_rowid(rowid: int, top_n: int = 5):
    if rowid not in df["rowid"].values:
        return []

    idx = df.index[df["rowid"] == rowid][0]
    sims = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()

    order = sims.argsort()[::-1]
    recs = []
    for i in order:
        if i == idx:
            continue
        recs.append({
            "rowid": int(df.iloc[i]["rowid"]),
            "Judul": str(df.iloc[i]["Judul"]),
            "snippet": str(df.iloc[i]["Content"])[:200]
        })
        if len(recs) >= top_n:
            break
    return recs

# ==========================
# Routes
# ==========================
@app.route("/")
def home():
    q = request.args.get("q", "").strip()
    page = int(request.args.get("page", 1) or 1)
    seed = request.args.get("seed", "")

    per_page = 9

    if q:
        results_df = search_results(q)
    else:
        # beranda: selalu acak
        results_df = df.sample(frac=1).copy()

    total_items = len(results_df)
    total_pages = (total_items // per_page) + (1 if total_items % per_page else 0) if total_items else 1
    page = max(1, min(page, total_pages))
    start = (page - 1) * per_page
    end = start + per_page

    page_df = results_df.iloc[start:end].copy()

    news = []
    for _, row in page_df.iterrows():
        news.append({
            "rowid": int(row["rowid"]),
            "Judul": str(row["Judul"]),
            "snippet": str(row.get("Content", ""))[:200]
        })

    return render_template(
        "index.html",
        news=news,
        q=q,
        seed=seed,
        page=page,
        total_pages=total_pages,
        total_items=total_items
    )

@app.route("/detail/<int:rowid>")
def detail(rowid):
    article_row = df[df["rowid"] == rowid]
    if article_row.empty:
        abort(404)
    article = article_row.iloc[0].to_dict()

    recs = get_recommendations_by_rowid(rowid=rowid, top_n=5)

    return render_template(
        "detail.html",
        article=article,
        recommendations=recs,
        q=request.args.get("q", ""),
        seed=request.args.get("seed", ""),
        page=int(request.args.get("page", 1) or 1)
    )

if __name__ == "__main__":
    app.run(debug=True)
