BOARD_SIZE = 8

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1),
              (-1, -1), (-1, 1), (1, -1), (1, 1)]

DEFAULT_AI_DEPTH = 3

CORNERS = [(0, 0), (0, BOARD_SIZE - 1), (BOARD_SIZE - 1, 0), (BOARD_SIZE - 1, BOARD_SIZE - 1)]

BLACK_WIN = 1
DRAW_SCORE = 0
WHITE_WIN = -1