from enum import Enum, auto

class PlayerType(Enum):
    USER = 'user'
    OFFLINE = 'offline'
    RANDOM = 'random_agent'
    MINIMAX = 'minimax'

# Note: SquareType moved to game_logic.board.py
# Note: HeuristicType moved to ai.minimax.evaluator.py