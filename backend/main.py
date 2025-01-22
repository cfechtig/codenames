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
def generate_grids():
    word_grid = list(np.random.choice(original_words, GRID_SIZE**2, replace=False).reshape([GRID_SIZE,GRID_SIZE]))
    key_cards = list(zip(
        ['A','B','B','B','B','B','F','F','F','F','F','F','F','F','F','B','A','B','B','B','B','B','B','B','A'], 
        ['F','F','F','F','F','F','F','F','F','B','B','B','B','B','A','A','A','B','B','B','B','B','B','B','B']
    ))  
    random.shuffle(key_cards)
    key_grid_player, key_grid_model = zip(*key_cards)
    key_grid_player = list(np.reshape(key_grid_player, [GRID_SIZE,GRID_SIZE]))
    key_grid_model = list(np.reshape(key_grid_model, [GRID_SIZE,GRID_SIZE]))
    return word_grid, key_grid_player, key_grid_model

def generate_clue(word_grid, key_grid_model, state_grid, model):
    remaining_friendlies = []
    remaining_bystanders = []
    assassins = []

    clue_word = None
    clue_count = None
    return clue_word, clue_count

def update_guess_queue(clue_word, clue_count, guess_queue, word_grid, key_grid_model, state_grid, model, sim_thresh=None):
    # determine remaining guessable words based on word_grid, key_grid_model, and state_grid
    remaining_words = []

    # generate word embeddings for remaining_words and clue_word
    word_embeddings = model.encode(remaining_words) # MAY BE REDUNDANT (update to have word embeddings generated at game start and saved client side)
    clue_embedding = model.encode([clue_word])

    # calculate similarity scores for clue_word
    similarities = cosine_similarity(clue_embedding, word_embeddings).flatten()

    guesses = sorted(list(zip(remaining_words, similarities, clue_word*len(remaining_words))), key=lambda x: x[1])

    # update guess_queue based on remaining_words
    guess_queue = [x for x in guess_queue if x[0] in remaining_words]

    return guesses

# FastAPI Application
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML models
    ml_models["default"] = SentenceTransformer('all-MiniLM-L6-v2')
    yield
    # Clean up the ML models and release the resources
    ml_models.clear()

app = FastAPI(lifespan=lifespan)

# Routes
@app.get("/api/start-game")
async def start_game():
    """Initialize a new game."""
    word_grid, key_grid_player, key_grid_model = generate_grids()
    return {
        "word_grid": word_grid,
        "key_grid_player": key_grid_player,
        "key_grid_model": key_grid_model,
        "num_timer_tokens": 9,
        "num_acceptable_mistakes": 9
    }

@app.post("/api/generate-clue")
async def generate_clue_endpoint(data):
    """Generate a clue."""
    grid = data.get("grid")
    model = ml_models["default"]
    clue_word, clue_count = generate_clue(grid, model)
    return {"clue_word": clue_word, "clue_count": clue_count}

@app.post("/api/generate-guess")
async def generate_guess_endpoint(data):
    """Generate guesses."""
    grid = data.get("grid")
    clue = data.get("clue")
    model = ml_models["default"]
    guesses = generate_guess(clue, grid, model)
    return {"guesses": guesses}
