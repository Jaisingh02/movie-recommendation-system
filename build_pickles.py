import pandas as pd
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

print("📌 Loading dataset...")
df = pd.read_csv("movies_metadata.csv", low_memory=False)

# ensure required cols
if "overview" not in df.columns or "title" not in df.columns:
    raise Exception("movies_metadata.csv must contain 'title' and 'overview' columns")

df["overview"] = df["overview"].fillna("")
df = df.dropna(subset=["title"]).reset_index(drop=True)

print("📌 Creating TF-IDF...")
tfidf = TfidfVectorizer(stop_words="english")
tfidf_matrix = tfidf.fit_transform(df["overview"])

indices = pd.Series(df.index, index=df["title"]).drop_duplicates()

print("✅ Saving pickle files...")
with open("df.pkl", "wb") as f:
    pickle.dump(df, f, protocol=pickle.HIGHEST_PROTOCOL)

with open("tfidf.pkl", "wb") as f:
    pickle.dump(tfidf, f, protocol=pickle.HIGHEST_PROTOCOL)

with open("tfidf_matrix.pkl", "wb") as f:
    pickle.dump(tfidf_matrix, f, protocol=pickle.HIGHEST_PROTOCOL)

with open("indices.pkl", "wb") as f:
    pickle.dump(indices, f, protocol=pickle.HIGHEST_PROTOCOL)

print("✅ Done! Pickle files created successfully.")
