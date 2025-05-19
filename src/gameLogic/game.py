
import numpy as np
import copy
from .board import Board, SquareType
from .player import Player, PlayerType
from ..ai.randomAgent import getRandomMove
from ..constants import DIRECTIONS, BOARD_SIZE
from ..ai.minimax.algorithm import get_minimax_move
import logging

class Game:
    def __init__(self, player_black, player_white):
        self.board = Board()
        self.is_finished = False
        self.player_black = player_black
        self.player_white = player_white
        self.active = self.player_black
        self.inactive = self.player_white
        self.next_move = None
        self.prev_move = None
        self.black_score = 2
        self.white_score = 2
        self.game_result = None
        self.move_history_str = ""


    def __deepcopy__(self, memo):
        cls = self.__class__
        new_game = cls.__new__(cls)
        memo[id(self)] = new_game
        new_game.board = copy.deepcopy(self.board, memo)
        new_game.is_finished = self.is_finished
        new_game.player_black = Player(
            player_type=self.player_black.player_type, disc_color=self.player_black.disc_color,
            state_eval=self.player_black.state_eval, depth=self.player_black.depth,
            ml_model_instance=self.player_black.ml_model
        )
        new_game.player_white = Player(
            player_type=self.player_white.player_type, disc_color=self.player_white.disc_color,
            state_eval=self.player_white.state_eval, depth=self.player_white.depth,
            ml_model_instance=self.player_white.ml_model
        )
        if self.active is self.player_black: new_game.active = new_game.player_black
        else: new_game.active = new_game.player_white
        if self.inactive is self.player_black: new_game.inactive = new_game.player_black
        else: new_game.inactive = new_game.player_white

        new_game.next_move = copy.deepcopy(self.next_move, memo)
        new_game.prev_move = copy.deepcopy(self.prev_move, memo)
        new_game.black_score = self.black_score
        new_game.white_score = self.white_score
        new_game.game_result = self.game_result
        new_game.move_history_str = self.move_history_str

        return new_game

    def _coordinate_to_notation(self, row, col):
        if row is None or col is None: return ""
        return chr(ord('a') + col) + str(row + 1)

    def _notation_to_coordinate(self, notation_str):
        if not notation_str or len(notation_str) != 2: return None
        try:
            col_char, row_char = notation_str[0].lower(), notation_str[1]
            if not ('a' <= col_char <= 'h' and '1' <= row_char <= '8'): return None
            return int(row_char) - 1, ord(col_char) - ord('a')
        except: return None
            
    def change_turn(self):
        self.active, self.inactive = self.inactive, self.active

    def update_scores(self):
        self.black_score = np.count_nonzero(self.board.state == SquareType.BLACK)
        self.white_score = np.count_nonzero(self.board.state == SquareType.WHITE)    

    def _is_move_valid_for_player(self, row, col, player_color_obj, opponent_color_obj):
        if not (0 <= row < BOARD_SIZE and 0 <= col < BOARD_SIZE): return False
        if self.board.state[row, col] != SquareType.EMPTY and self.board.state[row, col] != SquareType.VALID: return False
        for dr, dc in DIRECTIONS:
            r, c = row + dr, col + dc
            found_opponent = False
            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                if self.board.state[r, c] == opponent_color_obj:
                    found_opponent = True
                elif self.board.state[r, c] == player_color_obj:
                    if found_opponent: return True
                    break
                else: break
                r, c = r + dr, c + dc
        return False

    def get_valid_moves_by_color(self, color_enum):
        valid_moves = []
        player_color_obj = color_enum
        opponent_color_obj = SquareType.WHITE if color_enum == SquareType.BLACK else SquareType.BLACK
        for r_idx in range(BOARD_SIZE):
            for c_idx in range(BOARD_SIZE):
                if self._is_move_valid_for_player(r_idx, c_idx, player_color_obj, opponent_color_obj):
                    valid_moves.append((r_idx, c_idx))
        return valid_moves
    
    def reset_valid_moves(self):
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board.state[r,c] == SquareType.VALID: self.board.state[r,c] = SquareType.EMPTY

    def update_valid_moves(self):
        self.reset_valid_moves()
        current_player_valid_moves = self.get_valid_moves_by_color(self.active.disc_color)
        for r, c in current_player_valid_moves:
            self.board.state[r,c] = SquareType.VALID
            
    def is_any_valid_move_on_board(self):
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                if self.board.state[r,c] == SquareType.VALID: return True
        return False
        
    def get_player_move(self):
        self.prev_move = self.next_move
        player = self.active
        self.next_move = None
        self.update_valid_moves()

        if not self.is_any_valid_move_on_board():
            self.next_move = None
            return

        if player.player_type == PlayerType.OFFLINE:
            self.next_move = player.get_offline_move(self)
        elif player.player_type == PlayerType.USER: pass
        elif player.player_type == PlayerType.RANDOM:
            self.next_move = getRandomMove(self)
        elif player.player_type == PlayerType.MINIMAX:
            if player.depth is None or player.state_eval is None: raise ValueError("Minimax player misconfigured.")
            self.next_move = get_minimax_move(self, player.depth, player.state_eval)
        elif player.player_type == PlayerType.LSTM:
            if player.ml_model and hasattr(player.ml_model, 'infer_lstm'):
                notation = player.ml_model.infer_lstm(self.move_history_str)
                self.next_move = self._notation_to_coordinate(notation)
                if self.next_move is None or self.board.state[self.next_move[0], self.next_move[1]] != SquareType.VALID:
                    self.next_move = getRandomMove(self)
            else: self.next_move = getRandomMove(self)
        elif player.player_type == PlayerType.GPT2:
            if player.ml_model and hasattr(player.ml_model, 'infer_gpt2'):
                notation = player.ml_model.infer_gpt2(self.move_history_str)
                self.next_move = self._notation_to_coordinate(notation)
                if self.next_move is None or self.board.state[self.next_move[0], self.next_move[1]] != SquareType.VALID:
                    self.next_move = getRandomMove(self)
            else: self.next_move = getRandomMove(self)
        elif player.player_type == PlayerType.RF_MODEL:
            if player.ml_model and hasattr(player.ml_model, 'infer_rf'):
                self.next_move = player.ml_model.infer_rf(self)
                if self.next_move is None or self.board.state[self.next_move[0], self.next_move[1]] != SquareType.VALID:
                    logging.warning(f"RF Model fallback to random.")
                    self.next_move = getRandomMove(self)
            else: self.next_move = getRandomMove(self)
        elif player.player_type == PlayerType.XGB_MODEL:
            if player.ml_model and hasattr(player.ml_model, 'infer_xgb'):
                self.next_move = player.ml_model.infer_xgb(self)
                if self.next_move is None or self.board.state[self.next_move[0], self.next_move[1]] != SquareType.VALID:
                    logging.warning(f"XGB Model fallback to random.")
                    self.next_move = getRandomMove(self)
            else: self.next_move = getRandomMove(self)

    def _discs_to_flip_for_move(self, row, col, active_color, inactive_color):
        discs_to_flip_list = []
        for dr, dc in DIRECTIONS:
            r, c = row + dr, col + dc
            seq_discs = []
            found_opponent = False
            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                if self.board.state[r, c] == inactive_color:
                    seq_discs.append((r, c))
                    found_opponent = True
                elif self.board.state[r, c] == active_color:
                    if found_opponent: discs_to_flip_list.extend(seq_discs)
                    break
                else: break
                r, c = r + dr, c + dc
        return discs_to_flip_list

    def make_move(self):
        if self.next_move is None: return
        row, col = self.next_move[0], self.next_move[1]
        self.board.state[row, col] = self.active.disc_color
        
        discs_to_change = self._discs_to_flip_for_move(row, col, self.active.disc_color, self.inactive.disc_color)
        for r_f, c_f in discs_to_change:
            self.board.state[r_f, c_f] = self.active.disc_color
            
        move_notation = self._coordinate_to_notation(row, col)
        if move_notation: self.move_history_str += move_notation

    def is_board_full(self):
        return not np.any((self.board.state == SquareType.EMPTY) | (self.board.state == SquareType.VALID))

    def check_finished(self):
        current_player_has_moves = self.is_any_valid_move_on_board()

        temp_game_check_opponent = copy.deepcopy(self)
        temp_game_check_opponent.change_turn()
        temp_game_check_opponent.update_valid_moves()
        opponent_has_moves = temp_game_check_opponent.is_any_valid_move_on_board()

        if not current_player_has_moves and not opponent_has_moves:
            self.is_finished = True
        elif self.is_board_full():
            self.is_finished = True
        elif self.black_score == 0 or self.white_score == 0:
            self.is_finished = True
        
        if self.is_finished and not self.game_result:
            self.determine_winner()
    
    def determine_winner(self):
        if self.black_score > self.white_score: self.game_result = "Black Wins"
        elif self.white_score > self.black_score: self.game_result = "White Wins"
        else: self.game_result = "Draw"