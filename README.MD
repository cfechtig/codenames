# Information needed for game

### Defined globally:
- word_pool: list of all words in 'deck'
- grid_size: integer defining size of grid side
- embedding_models: dictionary of all embedding models


### Defined per game instance:

##### Static (defined at game start, never updated):
- word_grid: grid_size * grid_size array defining the words and word positions in play
- key_grid_player: grid_size * grid_size array defining the positions of friendlies (0), bystanders (1), and assassins (2) for player's side of key card
- key_grid_model: grid_size * grid_size array defining the positions of friendlies, bystanders, and assassins for model's side of key card
- num_timer_tokens: total number of turns before game loss
- num_acceptable_mistakes: total number of mistakes before game loss

##### Dynamic (defined at game start, updated per turn):
- state_grid: grid_size * grid_size array defining the current play grid state, including positions of blank cards, player-facing bystanders, model-facing bystanders, and discovered friendlies
- current_giver: current player who is giving the clue
- current_guesser: current player who is guessing based on the clue
- turn_count: current number of turns played
- mistake_count: current number of mistakes made
- user_guesses: set of all previous user guesses
- model_guesses: set of all previous model guesses


### Defined per turn:
- clue: word and integer
- guess(es): word(s)


# Backend functions
- load global stuff at start of application
- function to generate and return static game instance stuff for a new game (no input)
- function to generate an embedding model clue using game state variables (passed via dict)
- function to generate an embedding model guess using game state variables (passed via dict)
