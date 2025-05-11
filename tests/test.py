
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(os.path.join(project_root, 'src'))

import unittest
import numpy as np
from src.gameLogic.game import Game
from src.gameLogic.board import Board, SquareType
from src.gameLogic.player import Player, PlayerType
from src.ai.minimax.evaluator import StateEvaluator, HeuristicType
from src.ai.minimax.algorithm import minimax_alpha_beta, minimax_evaluate_moves, get_minimax_move, simulate_move_for_ai

class TestGame(unittest.TestCase):
    

    def setUp(self):
        self.black_player = Player(PlayerType.RANDOM, SquareType.BLACK)
        self.white_player = Player(PlayerType.RANDOM, SquareType.WHITE)
        self.game = Game(self.black_player, self.white_player)

    def test_player_colors(self):

        self.assertEqual(self.game.player_black.disc_color, SquareType.BLACK)
        self.assertEqual(self.game.player_white.disc_color, SquareType.WHITE)

    def test_change_turn(self):
        self.game.change_turn()

        self.assertEqual(self.game.active, self.game.player_white)
        self.assertEqual(self.game.inactive, self.game.player_black)

    def test_reset_valid_moves(self):
        self.game.reset_valid_moves()

        EXPECTED_BOARD_STATE = np.full((8, 8), SquareType.EMPTY)
        EXPECTED_BOARD_STATE[3, 3] = SquareType.WHITE
        EXPECTED_BOARD_STATE[3, 4] = SquareType.BLACK
        EXPECTED_BOARD_STATE[4, 3] = SquareType.BLACK
        EXPECTED_BOARD_STATE[4, 4] = SquareType.WHITE

        assert np.array_equal(self.game.board.state, EXPECTED_BOARD_STATE) 

    def test_valid_moves_initial_board(self):

        self.game.reset_valid_moves()

        valid_moves = self.game.get_valid_moves()

        EXPECTED_VALID_MOVES = [(2, 3), (3, 2), (4, 5), (5, 4)]
        self.assertEqual(set(valid_moves), set(EXPECTED_VALID_MOVES))

    def test_valid_moves_near_full_board(self):
        

        NEARLY_FULL_BOARD = [
            [SquareType.BLACK] * 8,
            [SquareType.BLACK] * 8,
            [SquareType.BLACK] * 8,
            [SquareType.BLACK] * 8,
            [SquareType.WHITE] * 8,
            [SquareType.WHITE] * 8,
            [SquareType.WHITE] * 8,
            [SquareType.WHITE] * 7 + [SquareType.EMPTY],
        ]
        self.game.board.state = np.array(NEARLY_FULL_BOARD)
        valid_moves = self.game.get_valid_moves()
        EXPECTED_VALID_MOVES = [(7, 7)]
        self.assertEqual(set(valid_moves), set(EXPECTED_VALID_MOVES))

    
    def test_update_valid_moves(self):
        

        NEARLY_FULL_BOARD = [
            [SquareType.BLACK] * 8,
            [SquareType.BLACK] * 8,
            [SquareType.BLACK] * 8,
            [SquareType.BLACK] * 8,
            [SquareType.WHITE] * 8,
            [SquareType.WHITE] * 8,
            [SquareType.WHITE] * 8,
            [SquareType.WHITE] * 7 + [SquareType.EMPTY],
        ]
        self.game.board.state = np.array(NEARLY_FULL_BOARD)
        self.game.update_valid_moves()
        self.assertEqual(self.game.board.state[7,7], SquareType.VALID)

    
    def test_disc_to_flip(self):        
        self.assertEqual(self.game.discs_to_flip(0 , 0), [])
        self.assertEqual(self.game.discs_to_flip(2 , 3), [(3, 3)])

    def test_is_board_full_empty_board(self):
        self.assertFalse(self.game.is_board_full())

    def test_is_board_full_full_board(self):
        self.game.board.state = np.full((8, 8), SquareType.BLACK)
        self.assertTrue(self.game.is_board_full())

    def test_is_finished(self):
        self.game.next_move = (2, 3)
        self.game.check_finished()
        assert not self.game.is_finished, "Game should not be finished."

        self.game.next_move = None
        self.game.prev_move = None
        self.game.check_finished()
        assert self.game.is_finished, \
            "Game should finnish when there are no moves for either player."

        self.game.board.state = np.full((8, 8), SquareType.BLACK)
        self.game.check_finished()
        assert self.game.is_finished, \
            "Game should be finished when the board is full."

    def test_no_valid_moves(self):
        NO_VALID_MOVES_BOARD = [
            [SquareType.BLACK] * 8,
            [SquareType.BLACK] * 8,
            [SquareType.BLACK] * 8,
            [SquareType.BLACK] * 8,
            [SquareType.BLACK] * 8,
            [SquareType.BLACK] * 8,
            [SquareType.BLACK] * 8,
            [SquareType.EMPTY] * 8 
        ]
        self.game.board.state = np.array(NO_VALID_MOVES_BOARD)
        self.game.get_player_move()

        self.assertIsNone(self.game.next_move, "There should be no valid moves.")

    def test_get_valid_moves_by_color(self):
        EXPECTED_BLACK_VALID_MOVES = [(2, 3), (3, 2), (4, 5), (5, 4)]
        EXPECTED_WHITE_VALID_MOVES = [(2, 4), (3, 5), (4, 2), (5, 3)]

        black_valid_moves = self.game.get_valid_moves_by_color(SquareType.BLACK)
        white_valid_moves = self.game.get_valid_moves_by_color(SquareType.WHITE)

        self.assertListEqual(black_valid_moves, EXPECTED_BLACK_VALID_MOVES)
        self.assertListEqual(white_valid_moves, EXPECTED_WHITE_VALID_MOVES)

    
    def test_white_wins(self):

        self.game.board.state.fill(SquareType.WHITE)
        self.game.update_scores()

        self.game.check_finished()
        self.game.determine_winner()

        self.assertEqual(self.game.game_result, "White Wins", \
                         "White should be the winner but isn't.")
        
        
    def test_draw(self):

        for row in range(4):
            for col in range(8):
                self.game.board.state[row, col] = SquareType.WHITE
        for row in range(4, 8):
            for col in range(8):
                self.game.board.state[row, col] = SquareType.BLACK
        self.game.update_scores()

        self.game.check_finished()
        self.game.determine_winner()

        self.assertEqual(self.game.game_result, "Draw", \
                         "The game should be a draw but isn't.")

class TestPlayer(unittest.TestCase):
    

    def test_state_evaluator_integration_with_player(self):

        custom_weights = {
            HeuristicType.DISC_DIFF: 0.7,
            HeuristicType.MOBILITY: 0.3
        }
        state_evaluator = StateEvaluator(weights=custom_weights)
        
        black_player = Player(PlayerType.MINIMAX, SquareType.BLACK, state_evaluator)
        white_player = Player(PlayerType.RANDOM, SquareType.WHITE)
        game = Game(black_player, white_player)

        valid_moves_count = black_player.state_eval.count_valid_moves(game, black_player.disc_color)
        disc_count = black_player.state_eval.count_discs(game, black_player.disc_color)
        self.assertEqual(valid_moves_count, 4)
        self.assertEqual(disc_count, 2)

class TestStateEvaluator(unittest.TestCase):
    

    def setUp(self):

        self.evaluator = StateEvaluator()

        black_player = Player(PlayerType.MINIMAX, SquareType.BLACK, self.evaluator)
        white_player = Player(PlayerType.RANDOM, SquareType.WHITE)

        self.game = Game(black_player, white_player)

    def test_count_valid_moves(self):
        EXPECTED_BLACK_VALID_MOVES_COUNT = 4
        EXPECTED_WHITE_VALID_MOVES_COUNT = 4

        black_valid_moves_count = self.evaluator.count_valid_moves(self.game, SquareType.BLACK)
        white_valid_moves_count = self.evaluator.count_valid_moves(self.game, SquareType.WHITE)

        self.assertEqual(black_valid_moves_count, EXPECTED_BLACK_VALID_MOVES_COUNT)
        self.assertEqual(white_valid_moves_count, EXPECTED_WHITE_VALID_MOVES_COUNT)

class TestHeuristics(unittest.TestCase):
    
        
    def setUp(self):
        

        custom_weights = {
            HeuristicType.DISC_DIFF: 0.7,
            HeuristicType.MOBILITY: 0.3
        }

        self.evaluator = StateEvaluator(weights=custom_weights)

        black_player = Player(PlayerType.MINIMAX, SquareType.BLACK, self.evaluator)
        white_player = Player(PlayerType.RANDOM, SquareType.WHITE)
        self.game = Game(black_player, white_player)

        self.game.board.state = np.full((8, 8), SquareType.EMPTY)

        self.game.board.state[2, 3] = SquareType.BLACK  

        self.game.board.state[3, 3] = SquareType.BLACK
        self.game.board.state[3, 4] = SquareType.BLACK
        self.game.board.state[4, 3] = SquareType.BLACK
        self.game.board.state[4, 4] = SquareType.WHITE

        self.game.update_scores()

    def test_disc_diff_heuristic(self):
        
        EXPECTED_DISC_DIFF = 0.6
        disc_diff = self.evaluator.disc_diff_heuristic(self.game)
        self.assertEqual(disc_diff, EXPECTED_DISC_DIFF, 
                         "Disc difference heuristic evaluated incorrectly.")

    def test_mobility_heuristic(self):
        
        EXPECTED_MOBILITY = 0.0
        mobility = self.evaluator.mobility_heuristic(self.game)
        self.assertEqual(mobility, EXPECTED_MOBILITY, 
                         "Mobility heuristic evaluated incorrectly.")

    def test_count_corners(self):
        

        self.game.board.state[0, 0] = SquareType.BLACK
        self.game.board.state[0, 7] = SquareType.WHITE
        self.game.board.state[7, 0] = SquareType.EMPTY
        self.game.board.state[7, 7] = SquareType.BLACK

        black_corners = self.evaluator.count_corners(self.game, SquareType.BLACK)
        white_corners = self.evaluator.count_corners(self.game, SquareType.WHITE)

        self.assertEqual(black_corners, 2, "Incorrect corner count for Black.")
        self.assertEqual(white_corners, 1, "Incorrect corner count for White.")

    
    def test_corner_heuristic(self):
        

        self.game.board.state[0, 0] = SquareType.BLACK
        self.game.board.state[0, 7] = SquareType.WHITE
        self.game.board.state[7, 0] = SquareType.EMPTY
        self.game.board.state[7, 7] = SquareType.BLACK

        corner_heuristic = self.evaluator.corner_heuristic(self.game)


        EXPECTED_VALUE = 1/3

        self.assertAlmostEqual(corner_heuristic, EXPECTED_VALUE, \
                               msg="Corner heuristic calculated incorrectly.")

    def test_evaluate(self):
        
        EXPECTED_EVALUATION_SCORE = 0.42
        evaluation_score = self.evaluator.evaluate(self.game)
        self.assertEqual(evaluation_score, EXPECTED_EVALUATION_SCORE, 
                         "Evaluate function didn't return expected score.")

    def test_evaluate_black_win(self):
        

        self.game.board.state.fill(SquareType.BLACK)
        self.game.board.state[0, 0] = SquareType.WHITE 

        self.game.update_scores() 

        self.game.check_finished()

        evaluator = StateEvaluator()
        evaluation_score = evaluator.evaluate(self.game)

        self.assertEqual(evaluation_score, 1, 
                         "Evaluate function should return +1 for a Black win.")

class TestSimulateMove(unittest.TestCase):

    def setUp(self):
        

        black_player = Player(PlayerType.USER, SquareType.BLACK, StateEvaluator())
        white_player = Player(PlayerType.RANDOM, SquareType.WHITE)

        self.game = Game(black_player, white_player)
        self.initial_board = np.copy(self.game.board.state)

    def test_simulate_move_d3(self):
        

        move = (2, 3)
        simulated_game = simulate_move_for_ai(self.game, move)

        self.assertFalse(np.array_equal(self.initial_board, simulated_game.board.state),
                         "The board state should have changed after simulating the move.")

        self.assertEqual(simulated_game.board.state[move], SquareType.BLACK,
                         "D3 should be occupied by black.")

        self.assertEqual(simulated_game.active.disc_color, SquareType.WHITE,
                         "The active player should now be white.")

        white_valid_moves = simulated_game.get_valid_moves_by_color(SquareType.WHITE)
        self.assertTrue(len(white_valid_moves) == 3,
                        "White should have three valid moves after black plays D3.")

        self.assertEqual(simulated_game.black_score, 4,
                         "Black should have a score of 4 after playing D3.")
        self.assertEqual(simulated_game.white_score, 1,
                         "White should have a score of 1 after black plays D3.")

class TestMinimax(unittest.TestCase):
    

    def setUp(self):
        

        state_evaluator = StateEvaluator()

        minimax_player_black = Player(PlayerType.MINIMAX, SquareType.BLACK, state_evaluator)
        minimax_player_white = Player(PlayerType.MINIMAX, SquareType.WHITE, state_evaluator)
        self.game = Game(minimax_player_black, minimax_player_white)

    def test_minimax_base_case(self):
        

        initial_moves = [(2, 3), (3, 2), (4, 5), (5, 4)]

        EXPECTED_MINIMAX_VALUE = 0.3
        
        for move in initial_moves:

            simulated_game = simulate_move_for_ai(self.game, move)
            player = simulated_game.player_black
            minimax_value = minimax_alpha_beta(simulated_game, 0,float('-inf'), float('inf'), True, player.state_eval)

            self.assertEqual(minimax_value, EXPECTED_MINIMAX_VALUE,
                f"Minimax value for {move}  expected to be {EXPECTED_MINIMAX_VALUE}, but got {minimax_value}.")

    def test_minimax_recursion(self):
        

        simulated_game = simulate_move_for_ai(self.game, (2,3))


        simulated_game.player_white.depth = 2
        player = simulated_game.player_white

        white_moves = minimax_evaluate_moves(simulated_game, player.depth, player.state_eval)

        EXPECTED_MINIMAX_VALUES = {
            (2, 2): 0.4642857142857143, 

            (2, 4): 0.38095238095238093, 

            (4, 2): 0.38095238095238093  

        }

        for move, value in white_moves:
            self.assertAlmostEqual(value, EXPECTED_MINIMAX_VALUES[move],
                msg=f"Minimax value for move {move} expected to be {EXPECTED_MINIMAX_VALUES[move]}, but got {value}.")

    def test_get_minimax_move(self):
        

        simulated_game = simulate_move_for_ai(self.game, (2,3))

        simulated_game.player_white.depth = 2
        best_move_white = get_minimax_move(simulated_game, simulated_game.player_white.depth, simulated_game.player_white.state_eval)

        EXPECTED_BEST_MOVE = (2, 4)

        self.assertEqual(best_move_white, EXPECTED_BEST_MOVE,
                        f"The best minimax move for White expected to be {EXPECTED_BEST_MOVE}, but got {best_move_white}.")
        
if __name__ == '__main__':
    unittest.main(verbosity=2)