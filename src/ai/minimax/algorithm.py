import copy
import numpy as np
import time
from ...gameLogic.board import SquareType

def simulate_move_for_ai(game, move):
    new_game = copy.deepcopy(game)

    new_game.next_move = move
    new_game.make_move()

    new_game.change_turn()
    new_game.update_valid_moves()
    new_game.check_finished()
    if not new_game.is_finished:
        new_game.update_scores()

    return new_game

def minimax_alpha_beta(game, depth: int, alpha: float, beta: float, maximizing_player: bool, evaluator):
    if depth == 0 or game.is_finished:
        return evaluator.evaluate(game)

    current_player_color = game.active.disc_color
    validMoves = game.get_valid_moves_by_color(current_player_color)

    if not validMoves:
        simulated_pass_game = simulate_move_for_ai(game, None)
        if simulated_pass_game.is_finished:
            return evaluator.evaluate(simulated_pass_game)
        return minimax_alpha_beta(simulated_pass_game, depth, alpha, beta, not maximizing_player, evaluator)

    if maximizing_player:
        max_eval = float('-inf')
        for move_candidate in validMoves:
            simulated_game = simulate_move_for_ai(game, move_candidate)
            eval_score = minimax_alpha_beta(simulated_game, depth - 1, alpha, beta, False, evaluator)
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            if beta <= alpha:
                break
        return max_eval
    else:
        min_eval = float('inf')
        for move_candidate in validMoves:
            simulated_game = simulate_move_for_ai(game, move_candidate)
            eval_score = minimax_alpha_beta(simulated_game, depth - 1, alpha, beta, True, evaluator)
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            if beta <= alpha:
                break
        return min_eval

def minimax_evaluate_moves(game, depth: int, evaluator):
    moves_with_values = []
    active_player = game.active
    validMoves = game.get_valid_moves_by_color(active_player.disc_color)

    if not validMoves:
        return moves_with_values

    is_maximizing = (active_player.disc_color == SquareType.BLACK)

    for move_candidate in validMoves:
        simulated_game = simulate_move_for_ai(game, move_candidate)

        if simulated_game.is_finished:
            minimax_value = evaluator.evaluate(simulated_game)
        else:
            minimax_value = minimax_alpha_beta(
                simulated_game,
                depth - 1,
                float('-inf'),
                float('inf'),
                not is_maximizing,
                evaluator
            )
        moves_with_values.append((move_candidate, minimax_value))

    return moves_with_values

def get_minimax_move(game, depth: int = 3, evaluator = None):
    if evaluator is None:
        raise ValueError("Evaluator must be provided to get_minimax_move")

    evaluated_moves = minimax_evaluate_moves(game, depth, evaluator)

    if not evaluated_moves:
        return None

    active_player_color = game.active.disc_color

    if active_player_color == SquareType.BLACK:
        best_move = max(evaluated_moves, key=lambda item: item[1])[0]
    else:
        best_move = min(evaluated_moves, key=lambda item: item[1])[0]

    return best_move