import random
from ..gameLogic.board import SquareType
from ..constants import BOARD_SIZE

def getRandomMove(game):
    validMoves = []
    board_state = game.board.state
    for row in range(BOARD_SIZE):
        for col in range(BOARD_SIZE):
            if board_state[row, col] == SquareType.VALID:
                validMoves.append((row, col))

    if not validMoves:
        return None

    return random.choice(validMoves)