# import os 
# import pickle
# from typing import Optional, List, Dict, Any, Tuple

# import numpy as np
# import pandas as pd
# import httpx
# from fastapi import FastAPI,HTTPException , Query    #Import "fastapi"  
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from dotenv import load_dotenv


# # .env , CORS CONFIG 
# load_dotenv()
# TMDB_API_KEY = os.getenv("TMDB_API_KEY")

# TMDB_BASE="https://api.themoviedb.org/3"    # this the api for selecting the images adn movies
# TMDB_IMG_500="https://image.tmdb.org/t/p/w500"

# if not TMDB_API_KEY:  #NOTE:this will handle if tmdb api key is not there
#     # don't crash import-time in production if you prefer; but for oyu better fail early
#     raise RuntimeError("TMDB_API_KEY missing. Put it in .env as TMDB_API_KEY=xxxx")

# app = FastAPI(title="Movie Recommender API", version="1.0")

# #NOTE: CORS = Cross-Origin Resource Sharing : It is a security rule in browsers that decides whether your frontend (Streamlit/React/etc.) can access your FastAPI backend.
# # these are the CORS of the FasAPI 
# app.add_middleware(       # they allows to access from any frontend to out end point(FastAPI)
#     CORSMiddleware,
#     allow_origins=["*"],   # for local streamlit
#     allow_credentials=["*"],
#     allow_headers=["*"],
# )


# # PATH AND GLOBAL VAR CONFIG
# # Declaring GLOBAL VARIABLES 

# BASE_DIR = os.path.dirname(os.path.abspath(__file__))    # from here we will picking all the pickle file 

# DF_PATH=os.path.join(BASE_DIR,r"C:\Users\DELL\Desktop\Movie recommendatio system\df.pkl")
# INDICES_PATH=os.path.join(BASE_DIR, r"C:\Users\DELL\Desktop\Movie recommendatio system\indices.pkl")
# TFIDF_MATRIX_PATH=os.path.join(BASE_DIR, r"C:\Users\DELL\Desktop\Movie recommendatio system\tfidf_matrix.pkl")
# TFIDF_PATH=os.path.join(BASE_DIR,r"C:\Users\DELL\Desktop\Movie recommendatio system\tfidf.pkl")

# # we have declared here variable with type as None 
# df: Optional[pd.DataFrame]=None
# indices_obj: Any = None
# tfidf_matrix: Any = None
# tfidf_obj: Any = None

# # this is a global variable
# TITLE_TO_IDX: Optional[Dict[str, int]]=None

# # MODELS 
# # now we will be using pydentic : kinda help data to be in structured form 
# class TMDBMovieCard(BaseModel):
#     tmdb_id:int
#     title: str
#     poster_url: Optional[str]=None
#     release_date: Optional[str]=None
#     vote_average: Optional[str]=None

# # Movies detail
# class TMDBMovieDetails(BaseModel):
#     tmdb_id:int
#     title: str
#     overview: Optional[str]=None
#     release_date: Optional[str]=None
#     poster_url: Optional[str]=None
#     backdrop_url: Optional[str]=None
#     genres: List[dict]=[]

# # this will match the score of the recommendation 
# class TDIDFRecItem(BaseModel):
#     title: str
#     score: float
#     tmdb: Optional[TMDBMovieCard]=None

# # this for the search method and this is kinda main class which includes inheritance as well 
# class SearchBundleResponse(BaseModel):
#     query: str
#     movie_details: TMDBMovieDetails
#     tfidf_recommendations: List[TDIDFRecItem]
#     genre_recommendations: List[TMDBMovieCard]


# # UTILITY FUNCTION 
# def _norm_title(t:str)->str:
#     return str(t).strip().lower()

# def make_img_url(path:Optional[str])-> Optional[str]:
#     if not path:
#         return None
#     return f"{TMDB_IMG_500}{path}"

# # this is our basic utility function : whose work is to bring just MOVIES to us 
# # params : which movie , name and according it will get path 
# async def tmdb_get(path:str, params: Dict[str,Any])-> Dict[str,Any]:
#     q=dict(params)
#     q["api_key"]=TMDB_API_KEY    #calling the api and whatever the output will be coming wil be going in the r 

#     try:
#         async with httpx.AsyncClient(timeout=20) as client:
#             r=await client.get(f"{TMDB_BASE}{path}",params=q)
#     except httpx.RequestError as e:
#         raise HTTPException(
#             status_code=502,
#             detail=f"TMDB request error: {type(e).__name__} | {repr(e)}"
#         )
    
#     # then we will check the status score of the r 
#     if r.status_code !=200:
#         raise HTTPException(
#             status_code=502, detail=f"TMDB error {r.status_code}: {r.test}"
#         )
    
#     return r.json()

# # out is giving us details of the movie in the form Dict/list form 
# #NOTE: THIS WILL GIVE US THE CARD OF THE MOVIE WHICH WILL BE COMING IN THE HOMEPAGE
# async def tmdb_cards_from_results(
#         results: List[dict], limit: int =20
# ) -> List[TMDBMovieCard]:
#     out: List[TMDBMovieCard]=[]
#     for m in (results or [])[:limit]:
#         out.append(
#             TMDBMovieCard(
#                 tmdb_id=int(m["id"]),
#                 title=m.get("title") or m.get("name") or "",
#                 poster_url=make_img_url(m.get("poster_path")),
#                 release_date=m.get("release_date"),
#                 vote_average=m.get("vote_average"),
#             )
#         )
#     return out

# #NOTE: This will give us the DETAILS of the MOVIE 
# async def tmdb_movie_details(movie_id:int) -> TMDBMovieDetails:
#         data=await tmdb_get(f"/movie/{movie_id}",{"language":"en-US"})   #tmdb_get it will give the details of the movie 
#         return TMDBMovieDetails (
#                 tmdb_id=int(data["id"]),
#                 title=data.get("title") or "",
#                 overview=data.get("overview"),
#                 release_date=data.get("release_date"),
#                 poster_url=make_img_url(data.get("poster_path")),
#                 backdrop_url=make_img_url(data.get("backdrop_path")),
#                 genres=data.get("genres",[]) or [],
#         )

# #NOTE: SEARCH MOVIE FUNCTION
# async def tmdb_search_movies(query: str, page:int =1)-> Dict[str, Any]:
#     return await tmdb_get(
#         "/search/movie",
#         {
#             "query": query,
#             "include_adult": "false",
#             "language":"en-US",
#             "page":page,
#         },
#     )

# async def tmdb_search_first(query: str)-> Optional[dict]:
#     data=await tmdb_search_movies(query=query, page=1)
#     results=data.get("results",[])
#     return results[0] if results else None

# # ALPHA FUNCTION : this will convert title_to_idx into mapping form 

# def build_title_to_idx_map(indices:Any)-> Dict[str, int]:
# # This will extract data from indicies.pkl into dict type and pandas into index=title , value=index form 
#     title_to_idx: Dict[str.int]={}

#     if isinstance(indices,dict):
#         for k,v in indices.items():
#             title_to_idx[_norm_title(k)]=int(v)
#         return title_to_idx
    
#     # pandas series or similar mapping 
#     try: 
#         for k.v in indices.items():
#             title_to_idx[_norm_title(k)]=int(v)
#         return title_to_idx
#     except Exception:
#         # last resort: if it's list-like etc
#         raise RuntimeError(
#             "indices.pkl must be dict or pandsa Series-like (with .items() )"
#         )
    
# # convert local indexes according to title 
# def get_local_idx_by_title(title:str)-> int:
#     global TITLE_TO_IDX
#     if TITLE_TO_IDX is None:
#         raise HTTPException(status_code=500, detail="TF-IDF index map not initialized")
    
#     key=_norm_title(title)
#     if key in TITLE_TO_IDX:
#         return int(TITLE_TO_IDX[key])
#     raise HTTPException(
#         status_code=404, detail=f"Title not found in local dataset: '{title}' "
#     )

# # RECOMMENDS TITLES 
# # It will return list of title and score using the cosine similarity and ALSO HANDLING THE MISSING VALUES 
# def tfidf_recommend_titles(
#         query_title: str, top_n:int=10
# ) -> List[Tuple[str,float]]:
#     global df , tfidf_matrix
#     if df is None or tfidf_matrix is None:
#         raise HTTPException(status_code=500, detail="TF-IDF resource not loaded")

#     idx=get_local_idx_by_title(query_title)

#     # query vector 
#     qv=tfidf_matrix[idx]
#     scores=(tfidf_matrix @ qv.T).toarray().ravel()

#     # sort descending
#     order=np.argsort(-scores)

#     out:List[Tuple[str,float]]=[]
#     for i in order:
#         if int(i)==int(idx):
#             continue
#         try:
#             title_i=str(df.iloc[int(i)]["title"])
#         except Exception:
#             continue
#         out.append((title_i,float(scores[int(i)])))
#         if len(out)>=top_n:
#             break
#     return out

# # this will give images according to the title searched and ALSO HANDLE THE crash if not found 
# async def attach_tmdb_card_by_title(title: str)-> Optional[TMDBMovieCard]:
#     try:
#         m=await tmdb_search_first(title)
#         if not m:
#             return None
#         return TMDBMovieCard(
#             tmdb_id=int(m["id"]),
#             title=m.get("title") or title,
#             poster_url=make_img_url(m.get("poster_path")),
#             release_date=m.get("release_date"),
#             vote_average=m.get("vote_average"),
#         )
#     # handling the crash 
#     except Exception:
#         return None

# # ---------------------------------------------------------------------------------------------------------------------------

# # STARTUP : LOADS PICKLES 
# # now we will load our pickle file and create API
# @app.on_event("startup")   # as soon as our api starts running will tell what to do 
# def load_pickles():        # load all the pickle files 
#     global df , indices_obj, tfidf_matrix, tfidf_obj , TITLE_TO_IDX 

#     # load df 
#     with open(DF_PATH, "rb") as f:
#         df=pickle.load(f)

#     # load indices
#     with open(INDICES_PATH,"rb") as f:
#         indices_obj=pickle.load(f)

#     # load TF-IDF matrix (usually scipy sparse)
#     with open (TFIDF_MATRIX_PATH,"rb") as f:
#         tfidf_obj=pickle.load(f)

#     # build normalized map
#     TITLE_TO_IDX=build_title_to_idx_map(indices_obj)

#     # sanity
#     if df is None or "title" not in df.columns:
#         raise RuntimeError("df.pkl must contain a DataFrame with a 'title column")
    
# # ROUTES 
# @app.get("/health")
# def health():
#     return {"status":"ok"}

# # HOME ROUTE 
# @app.get("/home", response_model=List[TMDBMovieCard])
# async def home(
#     category: str = Query("Popular"),
#     limit:int =Query(24,ge=1,le=50),
# ):
    
#     try: 
#         if category=="trending":
#             data=await tmdb_get("/trending/movie/day",{"language":"en-US"})
#             return await tmdb_cards_from_results(data.get("results",[]),limit=limit)
        
#         if category not in {"popular","top-rated","upcoming","now-playing"}:
#             raise HTTPException(status_code=400, detail="Invalid category")
        
#         data=await tmdb_get(f"/movie/{category}", {"language":"en-US", "page":1})
#         return await tmdb_cards_from_results(data.get("results",[]),limit=limit)
    
#     except HTTPException:
#         raise
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Home route failed: {e}")
    
# # SEARCH ROUTE : according to the keyword 
# @app.get("/tmdb/search")
# async def tmdb_search(
#     query: str = Query(..., min_length=1),
#     page: int = Query(1,ge=1,le=10),
# ):
    
#     return await tmdb_search_movies(query=query, page=page)
 
# # MOVIE DETAILS 
# @app.get("/movies/id/{tmdb_id}", response_model=TMDBMovieDetails)
# async def movie_details_route(tmdb_id: int):
#     return await tmdb_movie_details(tmdb_id)

# # GENRE RECOMMENDATION

# @app.get("recommend/genre", response_model=List[TMDBMovieCard])
# async def recommend_genre(
#     tmdb_id: int= Query(...),
#     limit: int=Query(18, ge=1,le=50),
# ):
#     details=await tmdb_movie_details(tmdb_id)
#     if not details.genres:
#         return []
    
#     genre_id=details.genres[0]["id"]
#     discover=await tmdb_get(
#         "/discovery/movie",
#         {
#             "with_genres":genre_id,
#             "language":"en-US",
#             "sort-by":"popularity.desc",
#             "page":1,
#         },
#     )
#     cards= await tmdb_cards_from_results(discover.get("results",[]) , limit=limit)
#     return [c for c in cards if c.tmdb_id !=tmdb_id]


# # Provide Recommendation titles 
# @app.get("recommend/tfidf")
# async def recommend_tfidf(
#     title: str=Query(..., min_length=1),
#     top_n: int=Query(10,ge=1,le=50),
# ):
#     recs = tfidf_recommend_titles(title,top_n=top_n)

#     return [{"title":t, "score":s} for t,s in recs] 

# # NOTE: BUNDLE: DETAILS+TF-IDF recommendation + GENRE recommendation  NOTE: it is the sum of above 3 functions created 
# # GIVES ALL THE DETAILS OF THE MOVIE when we OPEN to view the about MOVIE
# # if you want multiple matches then use /tmdb/search
# @app.get("/movie/search",response_model=SearchBundleResponse)
# async def search_bundle(
#     query: str=Query(..., min_length=1),
#     tfidf_top_n: int = Query(12,ge=1,le=30),
#     genre_limit: int = Query(12,ge=1,le=30),
# ):
#     best=await tmdb_search_first(query)
#     if not best:
#         raise HTTPException(
#             status_code=404, detail=f"No TMDB movie found for query: {query}"
#         )
#     tmdb_id=int(best["id"])
#     details=await tmdb_movie_details(tmdb_id)

#     # 1. tf-idf recommendations (never crash endpoint)
#     tfidf_items: List[TFIDFRecItem]=[]
    
#     recs: List[Tuple[str,float]]=[]
#     try:
#         # try local dataset by MDB title
#         recs=tfidf_recommend_titles(details.title, top_n=tfidf_top_n)
#     except Exception:
#         # fall back to user query
#         try:
#             recs=tfidf_recommend_titles(query, top_n=tfidf_top_n)
#         except Exception:
#             recs=[]

#     for title, score in recs:
#         card= await attach_tmdb_card_by_title(title)
#         tfidf_items.append(TFIDFRecItem(title=title, score=score, tmdb=card))
    
#     # 2. Genre recommendations (TMDB discover bt first genre)
#     genre_recs: List[TMDBMovieCard]=[]
#     if details.genres:
#         genre_id=details.genres[0]["id"]
#         discover= await tmdb_get(
#             "/discover/movie",
#             {
#                 "with_genres":"genre_id",
#                 "language":"en-US",
#                 "sort_by":"popularity.desc",
#                 "page":1,
#             },
#         )
#         cards=await tmdb_cards_from_results(
#             discover.get("results",[]),limit=genre_limit
#         )
#         genre_recs=[c for c in cards if c.tmdb_id != details.tmdb_id]
    
#     return SearchBundleResponse(
#         query=query,
#         movie_details=details,
#         tfidf_recommendations=tdidf_items,
#         genre_recommendations=genre_recs,
#     )