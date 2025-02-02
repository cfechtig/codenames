from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import random
from typing import List, Dict, Tuple
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

with open('word_lists/codenames.txt') as f:
    original_words = np.array(f.read().splitlines())
with open('word_lists/duet.txt') as f:
    duet_words = np.array(f.read().splitlines())
with open('word_lists/deep_undercover.txt') as f:
    undercover_words = np.array(f.read().splitlines())

WORD_POOL = duet_words
GRID_SIZE = 5

# Embedding Models Placeholder
ml_models = {}

# Utility Functions
def generate_grids(word_pool=WORD_POOL):
    word_grid = list(np.random.choice(word_pool, GRID_SIZE**2, replace=False).reshape([GRID_SIZE,GRID_SIZE]))
    key_cards = list(zip(
        ['A','B','B','B','B','B','F','F','F','F','F','F','F','F','F','B','A','B','B','B','B','B','B','B','A'], 
        ['F','F','F','F','F','F','F','F','F','B','B','B','B','B','A','A','A','B','B','B','B','B','B','B','B']
    ))
    random.shuffle(key_cards)
    key_grid_player, key_grid_model = zip(*key_cards)
    key_grid_player = list(np.reshape(key_grid_player, [GRID_SIZE,GRID_SIZE]))
    key_grid_model = list(np.reshape(key_grid_model, [GRID_SIZE,GRID_SIZE]))
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

def generate_embeddings(word_grid, model):
    words = np.array(word_grid).flatten()
    embeddings = model.encode(words)
    return {w:e for w,e in zip(words, embeddings)}

def generate_clue(remaining_friendlies, remaining_bystanders, assassins, model):
    clue_word = None
    clue_count = None
    return clue_word, clue_count

def generate_guesses(clue_word, clue_count, remaining_words, model, sim_thresh=0.0):
    # generate word embeddings for remaining_words and clue_word
    remaining_word_embeddings = model.encode(remaining_words)
    clue_embedding = model.encode([clue_word])

    # calculate similarity scores for clue_word
    similarities = cosine_similarity(clue_embedding, remaining_word_embeddings).flatten()

    guesses = sorted(list(zip(remaining_words, similarities)), key=lambda x: x[1], reverse=True)[:clue_count]
    guesses = [g for g in guesses if g[1] > sim_thresh]

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
    word_groups_model = get_word_groups(word_grid, key_grid_player)
    word_embeddings = generate_embeddings(word_grid)
    return {
        "word_grid": word_grid,
        "key_grid_player": key_grid_player,
        "key_grid_model": key_grid_model,
        "word_groups_player": word_groups_player,
        "word_groups_model": word_groups_model,
        "num_timer_tokens": 9,
        "num_acceptable_mistakes": 9,
        "word_embeddings": word_embeddings
    }

@app.post("/api/generate-clue")
async def generate_clue_endpoint(data):
    """Generate a clue."""
    remaining_friendlies = data.get("remaining_friendlies")
    remaining_bystanders = data.get("remaining_bystanders")
    assassins = data.get("assassins")
    model = ml_models["default"] # or data.get("model")
    clue_word, clue_count = generate_clue(remaining_friendlies, remaining_bystanders, assassins, model)
    return {"clue_word": clue_word, "clue_count": clue_count}

@app.post("/api/generate-guess")
async def generate_guess_endpoint(data):
    """Generate guesses."""
    clue_word = data.get("clue_word")
    clue_count = data.get("clue_count")
    remaining_words = data.get("remaining_words")
    model = ml_models["default"] # or data.get("model")
    guesses = generate_guesses(clue_word, clue_count, remaining_words, model)
    return {"guesses": guesses}
