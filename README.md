## Content-Based Movie Recommendation System

A machine learning recommendation engine built in Python, utilizing TF-IDF vectorization and Cosine Similarity to recommend movies based on plot descriptions, genres, and metadata. It features a dual-interface architecture: a high-performance **FastAPI REST API** backend and an interactive **Streamlit dashboard** frontend.

---

## 🚀 Key Architectural Components

### 1. Vectorization & Similarity Computations
* **TF-IDF Vectorizer**: Translates text-based plots, overview summaries, and metadata descriptions into numerical feature vectors, isolating term frequencies weighted across the dataset.
* **Cosine Similarity**: Measures the cosine angle between multidimensional TF-IDF vectors, yielding a semantic matching score between 0% and 100% to evaluate movie recommendations.

### 2. High-Performance Caching (Pickling)
* To bypass costly matrix computations on every API call, `build_pickles.py` pre-processes the raw `movies_metadata.csv` (34MB) and caches the matrices into serialized Python pickles:
  * `df.pkl`: The processed pandas DataFrame containing movie records.
  * `indices.pkl`: Mapping indices of movie titles for O(1) reverse lookup.
  * `tfidf.pkl`: Fitted Scikit-learn TF-IDF Vectorizer object.
  * `tfidf_matrix.pkl`: Pre-computed vector matrices representing plot dimensions.

### 3. Dual Interfaces
* **FastAPI Backend (`main.py`)**: Restful API endpoints built with Uvicorn and Gunicorn to serve recommendation queries programmatically with minimal latency.
* **Streamlit Frontend (`app.py`)**: A visual, interactive web application providing a search bar and sandbox controls to explore matches.

---

## 📂 Project Directory Structure

* `movies_metadata.csv`: Raw metadata dataset containing movie details.
* `build_pickles.py`: Precomputes vector models and exports pickle caches.
* `app.py`: Streamlit frontend dashboard logic.
* `main.py`: FastAPI server routes.
* `requirements.txt`: Python package dependency listings.
* `df.pkl` / `tfidf_matrix.pkl` / `tfidf.pkl` / `indices.pkl`: Serialized data pickles.

---

## 🛠️ Local Installation & Setup

1. **Clone & Navigate to directory**:
   ```bash
   cd "Movie recommendatio system"
   ```

2. **Set up a Virtual Environment**:
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Precompute Pickles** (if missing or dataset changes):
   ```bash
   python build_pickles.py
   ```

5. **Run Streamlit Dashboard**:
   ```bash
   streamlit run app.py
   ```

6. **Run FastAPI Backend Server**:
   ```bash
   uvicorn main:app --reload
   ```
