import numpy as np
from enum import Enum
from ..constants import BOARD_SIZE


class SquareType(Enum):
    EMPTY = ' '
    BLACK = '●'
    WHITE = '○'
    VALID = '☆'


class Board:
    def __init__(self):
        self.state = np.full((BOARD_SIZE, BOARD_SIZE), SquareType.EMPTY)

        mid = BOARD_SIZE // 2
        self.state[mid - 1, mid - 1] = SquareType.WHITE
        self.state[mid - 1, mid] = SquareType.BLACK
        self.state[mid, mid - 1] = SquareType.BLACK
        self.state[mid, mid] = SquareType.WHITE

        # Set up initial valid moves
        self.state[mid - 2, mid - 1] = SquareType.VALID
        self.state[mid - 1, mid - 2] = SquareType.VALID
        self.state[mid, mid + 1] = SquareType.VALID
        self.state[mid + 1, mid] = SquareType.VALID

    def display(self):
        boardRepr = np.array([
            [" " + square.value + " " for square in row]
            for row in self.state
        ])

        colLabel = '   '.join(chr(ord('A') + i) for i in range(BOARD_SIZE))
        print(f'     {colLabel}  ')
        print('  +' + '-' * (BOARD_SIZE * 4 + 1) + '+')
        for i, row in enumerate(boardRepr, start=1):
            row_str = '|'.join(row)
            print(f'{i:<2}| {row_str} |')
            print('  +' + '-' * (BOARD_SIZE * 4 + 1) + '+')
