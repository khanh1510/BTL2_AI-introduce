from enum import Enum, auto

class PlayerType(Enum):
    USER = 'user'
    OFFLINE = 'offline'
    RANDOM = 'random_agent'
    MINIMAX = 'minimax'
    LSTM = 'lstm_model'