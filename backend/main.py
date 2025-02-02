from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import random
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.stem import PorterStemmer

with open('../word_lists/codenames.txt') as f:
    original_words = np.array(f.read().splitlines())
with open('../word_lists/duet.txt') as f:
    duet_words = np.array(f.read().splitlines())
with open('../word_lists/deep_undercover.txt') as f:
    undercover_words = np.array(f.read().splitlines())

WORD_POOL = duet_words
GRID_SIZE = 5

# Embedding Models Placeholder
ml_models = {}

# Utility Functions
def generate_grids(word_pool=WORD_POOL):
    word_grid = np.random.choice(word_pool, GRID_SIZE**2, replace=False).reshape([GRID_SIZE,GRID_SIZE])
    key_cards = list(zip(
        ['A','B','B','B','B','B','F','F','F','F','F','F','F','F','F','B','A','B','B','B','B','B','B','B','A'], 
        ['F','F','F','F','F','F','F','F','F','B','B','B','B','B','A','A','A','B','B','B','B','B','B','B','B']
    ))
    random.shuffle(key_cards)
    key_grid_player, key_grid_model = zip(*key_cards)
    key_grid_player = np.reshape(key_grid_player, [GRID_SIZE,GRID_SIZE])
    key_grid_model = np.reshape(key_grid_model, [GRID_SIZE,GRID_SIZE])

    word_grid = [list(row) for row in word_grid]
    key_grid_player = [list(row) for row in key_grid_player]
    key_grid_model = [list(row) for row in key_grid_model]

    return word_grid, key_grid_player, key_grid_model

def get_word_groups(word_grid, key_grid):
    word_groups = {}
    for i in range(len(key_grid)):
        for j in range(len(key_grid[i])):
            key = key_grid[i][j]
            word = word_grid[i][j]
            
            if key in word_groups:
                word_groups[key].append(word)
            else:
                word_groups[key] = [word]
    
    return word_groups

def filter_dict_by_stemmed_keys(data_dict, word_list):
    stemmer = PorterStemmer()

    # Stem all words in the list
    stemmed_words = {stemmer.stem(word) for word in word_list}

    def should_remove(key):
        """Check if a given key should be removed based on the conditions."""
        stemmed_key = stemmer.stem(key)
        for word in stemmed_words:
            if (
                stemmed_key == word or 
                word in stemmed_key or 
                stemmed_key in word or 
                word in key or 
                key in word
            ):
                return True
        return False

    # Filter dictionary based on the should_remove condition
    filtered_dict = {key: value for key, value in data_dict.items() if not should_remove(key)}

    return filtered_dict

def generate_embeddings(word_grid, model):
    words = np.array(word_grid).flatten()
    embeddings = model.encode(words)
    return {w:e.tolist() for w,e in zip(words, embeddings)}

def normalize(v):
    """Return the unit vector of v."""
    norm = np.linalg.norm(v)
    return v if norm == 0 else v / norm

def nearest_neighbor(v_star, candidate_embs):
    """
    Given an ideal clue vector v_star and a dictionary mapping candidate clue words
    to their precomputed embeddings, return the candidate with the highest cosine similarity.
    """
    best_word = None
    best_sim = -np.inf
    for word, emb in candidate_embs.items():
        sim = np.dot(v_star, emb)
        if sim > best_sim:
            best_sim = sim
            best_word = word
    return best_word

def find_best_cluster(
    friend_words, 
    non_friend_words,
    friend_embs,  
    non_friend_embs, 
    candidate_embs,
    target_N=3, 
    penalty_coeff=0.2,
    verbose=True
):
    """
    For each candidate cluster size N (from 1 to len(friend_words)), this function selects
    a tightly grouped cluster of friendly words (using a greedy strategy) and computes its centroid.
    It then evaluates a margin — defined as:
    
         margin = min_{f in cluster} cosine_similarity(centroid, f)
                  - max_{w in non-friends} cosine_similarity(centroid, w)
    
    To encourage clusters in a desired range (e.g. 2 to 4), we subtract a quadratic penalty:
    
         penalty = penalty_coeff * (N - target_N)^2
    
    The overall score is then:
    
         score = margin - penalty
    
    The best cluster (and corresponding centroid) is chosen based on the highest overall score.
    
    Parameters:
      - friend_words: list of friendly words.
      - non_friend_words: list of non-friendly words (e.g., bystanders + assassins).
      - candidate_words: list of candidate clue words.
      - friend_embs: dict mapping each friendly word to its embedding.
      - non_friend_embs: dict mapping each non-friendly word to its embedding.
      - candidate_embs: dict mapping each candidate clue word to its embedding.
      - target_N: desired number of friendlies (default 3).
      - penalty_coeff: coefficient for the quadratic penalty (default 0.2).
    
    Returns:
      - best_cluster: list of friendly words in the chosen cluster.
      - best_N: the size of the cluster.
      - best_v_star: the centroid vector of the best cluster.
      - best_clue: the candidate clue word (from candidate_words) whose embedding is closest to best_v_star.
    """
    best_score = -np.inf
    best_cluster = None
    best_N = None
    best_v_star = None

    # Loop over all possible cluster sizes.
    for N in range(1, len(friend_words) + 1):
        if verbose:
            print(f"N = {N}")
            print()
        # Try every friendly word as a potential seed for the cluster.
        for seed in friend_words:
            if verbose:
                print(f"Cluster seed = \"{seed}\"")
            cluster = [seed]
            remaining = set(friend_words) - {seed}
            # Greedily grow the cluster until it has N words.
            while len(cluster) < N and remaining:
                best_candidate = None
                best_avg_sim = -np.inf
                for candidate in remaining:
                    # Compute average cosine similarity between candidate and the current cluster.
                    sim_sum = sum(np.dot(friend_embs[candidate], friend_embs[w])
                                  for w in cluster)
                    avg_sim = sim_sum / len(cluster)
                    if avg_sim > best_avg_sim:
                        best_avg_sim = avg_sim
                        best_candidate = candidate
                if best_candidate is not None:
                    cluster.append(best_candidate)
                    remaining.remove(best_candidate)
                else:
                    break
            # Only consider complete clusters of size N.
            if len(cluster) != N:
                continue
            # Compute the centroid of the cluster.
            centroid = normalize(sum(friend_embs[w] for w in cluster))
            # Calculate the minimum similarity from the cluster words to the centroid.
            friend_sims = [np.dot(centroid, friend_embs[w]) for w in cluster]
            min_friend_sim = min(friend_sims)
            # Calculate the maximum similarity from any non-friendly word to the centroid.
            non_friend_sims = [np.dot(centroid, non_friend_embs[w]) for w in non_friend_words]
            max_non_friend_sim = max(non_friend_sims) if non_friend_sims else -np.inf
            # Define a margin that measures the separation.
            margin = min_friend_sim - max_non_friend_sim
            # Optionally, you could add a bonus for larger clusters (e.g., margin + alpha * N).
            score = margin - (penalty_coeff * (N - target_N)**2)
            if verbose:
                print(f"Cluster words = {cluster}")
                print(f"Cluster margin = {margin}")
                print(f"Cluster score = {score}")
                print(f"Cluster clue = {nearest_neighbor(centroid, candidate_embs)}")
                print()
            if score > best_score:
                best_score = score
                best_cluster = list(cluster)
                best_N = N
                best_v_star = centroid
        if verbose:
            print()

    # Once the best centroid is chosen, "round" it to a candidate clue word.
    best_clue = nearest_neighbor(best_v_star, candidate_embs)
    if verbose:
        print(f"BEST N: {best_N}")
        print(f"BEST CLUSTER: {best_cluster}")
        print(f"BEST SCORE: {best_score}")
        print(f"BEST CLUE: {best_clue}")
    return best_cluster, best_N, best_v_star, best_clue

def generate_clue(remaining_friendlies, remaining_bystanders, remaining_assassins, model):
    friend_words = remaining_friendlies
    non_friend_words = remaining_bystanders + remaining_assassins
    all_words = friend_words + non_friend_words

    friend_embeddings = {word: model.encode(word) for word in friend_words}
    non_friend_embeddings = {word: model.encode(word) for word in non_friend_words}

    with open('../clue_candidates.txt') as f:
        candidate_words = np.array(f.read().splitlines())

    candidate_embeddings = {word:embedding for word, embedding in zip(candidate_words, np.load("../clue_candidate_embeddings.npy"))}

    candidate_embeddings = filter_dict_by_stemmed_keys(candidate_embeddings, all_words)

    best_cluster, best_N, best_v_star, best_clue = find_best_cluster(
        friend_words, 
        non_friend_words, 
        friend_embeddings, 
        non_friend_embeddings, 
        candidate_embeddings, 
        verbose=False
    )
    
    return best_clue, best_N

def generate_guesses(clue_word, clue_count, remaining_words, model, sim_thresh=0.0):
    # generate word embeddings for remaining_words and clue_word
    remaining_word_embeddings = model.encode(remaining_words)
    clue_embedding = model.encode([clue_word])

    # calculate similarity scores for clue_word
    similarities = cosine_similarity(clue_embedding, remaining_word_embeddings).flatten()

    guesses = sorted(list(zip(remaining_words, similarities)), key=lambda x: x[1], reverse=True)[:clue_count]
    guesses = [str(g) for g in guesses if g[1] > sim_thresh]

    return guesses
    

# FastAPI Application
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML models
    ml_models["default"] = SentenceTransformer('all-MiniLM-L6-v2')
    # TODO: Add more models
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

# Routes
@app.get("/api/start-game")
async def start_game():
    """Initialize a new game."""
    word_grid, key_grid_player, key_grid_model = generate_grids()
    word_groups_player = get_word_groups(word_grid, key_grid_player)
    word_groups_model = get_word_groups(word_grid, key_grid_model)
    model_name = "default"
    word_embeddings = generate_embeddings(word_grid, ml_models[model_name])
    return {
        "word_grid": word_grid,
        "key_grid_player": key_grid_player,
        "key_grid_model": key_grid_model,
        "word_groups_player": word_groups_player,
        "word_groups_model": word_groups_model,
        "num_timer_tokens": 9,
        "num_acceptable_mistakes": 9,
        "model_name": model_name,
        "word_embeddings": word_embeddings
    }

class ClueRequest(BaseModel):
    remaining_friendlies: List[str]
    remaining_bystanders: List[str]
    remaining_assassins: List[str]
    model_name: str

@app.post("/api/generate-clue")
async def generate_clue_endpoint(data: ClueRequest):
    """Generate a clue."""
    remaining_friendlies = data.remaining_friendlies
    remaining_bystanders = data.remaining_bystanders
    remaining_assassins = data.remaining_assassins
    model_name = data.model_name
    clue_word, clue_count = generate_clue(remaining_friendlies, remaining_bystanders, remaining_assassins, ml_models[model_name])
    return {"clue_word": clue_word, "clue_count": clue_count}

class GuessRequest(BaseModel):
    clue_word: str
    clue_count: int
    remaining_words: List[str]
    model_name: str

@app.post("/api/generate-guess")
async def generate_guess_endpoint(data: GuessRequest):
    """Generate guesses."""
    clue_word = data.clue_word
    clue_count = data.clue_count
    remaining_words = data.remaining_words
    model_name = data.model_name
    guesses = generate_guesses(clue_word, clue_count, remaining_words, ml_models[model_name])
    return {"guesses": guesses}
