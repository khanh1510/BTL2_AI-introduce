import numpy as np
from enum import Enum, auto
from ...gameLogic.board import SquareType
from ...constants import CORNERS, BOARD_SIZE, BLACK_WIN, WHITE_WIN, DRAW_SCORE


class HeuristicType(Enum):
    DISC_DIFF = auto()
    MOBILITY = auto()
    CORNERS = auto()


class StateEvaluator:

    def __init__(self, weights: dict = None):

        self.heuristic_methods = {
            HeuristicType.DISC_DIFF: self.disc_diff_heuristic,
            HeuristicType.MOBILITY: self.mobility_heuristic,
            HeuristicType.CORNERS: self.corner_heuristic,
        }

        default_weights = {
            HeuristicType.DISC_DIFF: 0.5,
            HeuristicType.MOBILITY: 0.5,
            HeuristicType.CORNERS: 0
        }

        self.weights = weights if weights else default_weights

        if not np.isclose(sum(self.weights.values()), 1):

            raise ValueError("Heuristic weights must sum to 1.")

    def evaluate(self, game):
        if game.is_finished:

            if game.game_result is None:
                game.determine_winner()

            if game.game_result == "Black Wins":
                return BLACK_WIN
            elif game.game_result == "White Wins":
                return WHITE_WIN

            elif game.game_result == "Draw":
                return DRAW_SCORE
            else:

                if game.black_score > game.white_score:
                    return BLACK_WIN
                if game.white_score > game.black_score:
                    return WHITE_WIN
                return DRAW_SCORE

        score = 0.0
        active_heuristics = {ht for ht, w in self.weights.items() if w != 0}

        if HeuristicType.DISC_DIFF in active_heuristics:
            score += self.weights[HeuristicType.DISC_DIFF] * \
                self.disc_diff_heuristic(game)
        if HeuristicType.MOBILITY in active_heuristics:
            score += self.weights[HeuristicType.MOBILITY] * \
                self.mobility_heuristic(game)
        if HeuristicType.CORNERS in active_heuristics:
            score += self.weights[HeuristicType.CORNERS] * \
                self.corner_heuristic(game)

        return score

    def count_valid_moves(self, game, disc_color):
        return len(game.get_valid_moves_by_color(disc_color))

    def mobility_heuristic(self, game):
        black_moves = self.count_valid_moves(game, SquareType.BLACK)
        white_moves = self.count_valid_moves(game, SquareType.WHITE)

        if black_moves + white_moves == 0:
            return 0.0

        return (black_moves - white_moves) / (black_moves + white_moves)

    def count_discs(self, game, disc_color):

        if disc_color == SquareType.BLACK:
            return game.black_score
        elif disc_color == SquareType.WHITE:
            return game.white_score
        else:
            raise ValueError("Invalid color specified for counting discs.")

    def disc_diff_heuristic(self, game):
        black_discs = self.count_discs(game, SquareType.BLACK)
        white_discs = self.count_discs(game, SquareType.WHITE)

        if black_discs + white_discs == 0:
            return 0.0

        return (black_discs - white_discs) / (black_discs + white_discs)

    def count_corners(self, game, disc_color):

        count = 0
        for row, col in CORNERS:
            if 0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE:
                if game.board.state[row, col] == disc_color:
                    count += 1
        return count

    def corner_heuristic(self, game):
        black_corners = self.count_corners(game, SquareType.BLACK)
        white_corners = self.count_corners(game, SquareType.WHITE)

        if black_corners + white_corners == 0:
            return 0.0

        return (black_corners - white_corners) / (black_corners + white_corners)
