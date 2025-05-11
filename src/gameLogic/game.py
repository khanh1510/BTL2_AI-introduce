import numpy as np
import copy
from .board import Board, SquareType
from .player import Player, PlayerType
from ..ai.randomAgent import getRandomMove
from ..constants import DIRECTIONS, BOARD_SIZE
from ..ai.minimax.algorithm import get_minimax_move

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

    def __deepcopy__(self, memo):
        cls = self.__class__
        new_game = cls.__new__(cls)
        memo[id(self)] = new_game

        new_game.board = copy.deepcopy(self.board, memo)
        new_game.is_finished = self.is_finished
        
        new_game.player_black = Player(
            player_type=self.player_black.player_type,
            disc_color=self.player_black.disc_color,
            state_eval=self.player_black.state_eval, 
            depth=self.player_black.depth
        )
        memo[id(self.player_black)] = new_game.player_black

        new_game.player_white = Player(
            player_type=self.player_white.player_type,
            disc_color=self.player_white.disc_color,
            state_eval=self.player_white.state_eval, 
            depth=self.player_white.depth
        )
        memo[id(self.player_white)] = new_game.player_white
        
        if self.active is self.player_black:
            new_game.active = new_game.player_black
            new_game.inactive = new_game.player_white
        elif self.active is self.player_white: # else if to be explicit
            new_game.active = new_game.player_white
            new_game.inactive = new_game.player_black
        else: # Should ideally not be reached if active is always one of the two players
            new_game.active = None 
            new_game.inactive = None


        new_game.next_move = copy.deepcopy(self.next_move, memo)
        new_game.prev_move = copy.deepcopy(self.prev_move, memo)
        new_game.black_score = self.black_score
        new_game.white_score = self.white_score
        new_game.game_result = self.game_result

        return new_game

    def change_turn(self):
        self.active, self.inactive = self.inactive, self.active

    def update_scores(self):
        self.black_score = np.count_nonzero(self.board.state == SquareType.BLACK)
        self.white_score = np.count_nonzero(self.board.state == SquareType.WHITE)    

    def is_valid_move(self, row, col):

        self.reset_valid_moves()

        if self.board.state[row, col] != SquareType.EMPTY:
            return False

        for direction in DIRECTIONS:
            d_row, d_col = direction
            r, c = row + d_row, col + d_col

            found_opposing_disc = False

            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                if self.board.state[r, c] == SquareType.EMPTY:
                    break

                if self.board.state[r, c] == self.active.disc_color:
                    if found_opposing_disc:

                        return True
                    break

                if self.board.state[r, c] == self.inactive.disc_color:
                    found_opposing_disc = True

                r += d_row
                c += d_col

        return False

    def get_valid_moves(self):

        return [
            (row, col)
            for row in range(BOARD_SIZE)
            for col in range(BOARD_SIZE)
            if self.is_valid_move(row, col)
        ]

    def get_valid_moves_by_color(self, color):

        original_active = self.active
        original_inactive = self.inactive

        if color == SquareType.BLACK:
            self.active = self.player_black
            self.inactive = self.player_white
        elif color == SquareType.WHITE:
            self.active = self.player_white
            self.inactive = self.player_black
        else:
            raise ValueError("Invalid color specified.")

        valid_moves = [
            (row, col)
            for row in range(BOARD_SIZE)
            for col in range(BOARD_SIZE)
            if self.is_valid_move(row, col)
        ]

        self.active = original_active
        self.inactive = original_inactive

        return valid_moves

    def reset_valid_moves(self):

        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                if self.board.state[row, col] == SquareType.VALID:
                    self.board.state[row, col] = SquareType.EMPTY

    def update_valid_moves(self):

        self.reset_valid_moves()

        valid_moves = self.get_valid_moves()

        if not valid_moves:
            return
        
        for move in valid_moves:

            row, col = move
            self.board.state[row, col] = SquareType.VALID

    def is_valid_moves(self):

        return any(
            cell == SquareType.VALID 
            for row in self.board.state 
            for cell in row
        )    

    def get_player_move(self):
        self.prev_move = self.next_move
        player = self.active
        self.next_move = None
        
        self.update_valid_moves() 

        if player.player_type == PlayerType.OFFLINE:
            move_coords = player.get_offline_move(self)
            if move_coords:
                self.next_move = move_coords
        elif player.player_type == PlayerType.USER:
            pass
        elif player.player_type == PlayerType.RANDOM:
            self.next_move = getRandomMove(self)
        elif player.player_type == PlayerType.MINIMAX:
            if player.depth is None or player.state_eval is None:
                raise ValueError("Minimax player must have depth and state_eval configured.")
            self.next_move = get_minimax_move(self, player.depth, player.state_eval)

    def discs_to_flip(self, row, col):

        discs_to_flip = []
        for direction in DIRECTIONS:
            d_row, d_col = direction
            r, c = row + d_row, col + d_col

            seq_discs = []

            flip_flag = False

            while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                if self.board.state[r, c] == self.inactive.disc_color:

                    seq_discs.append((r, c))
                    flip_flag = True

                elif self.board.state[r, c] == self.active.disc_color:
                    if flip_flag:

                        discs_to_flip.extend(seq_discs)
                    break
                else:
                    break

                r += d_row
                c += d_col

        return discs_to_flip

    def flip(self):

        row, col = self.next_move[0], self.next_move[1]
        discs = self.discs_to_flip(row, col)

        for disc in discs:

            self.board.state[disc[0], disc[1]] = self.active.disc_color

    def make_move(self):

        if self.next_move is None:
            return

        row, col = self.next_move[0], self.next_move[1]
        self.board.state[row, col] = self.active.disc_color

        self.flip()

    def is_board_full(self):

        for row in range(BOARD_SIZE):
            for col in range(BOARD_SIZE):
                cell_state = self.board.state[row, col]
                if cell_state in [SquareType.EMPTY, SquareType.VALID]:
                    return False
        
        return True

    def check_finished(self):

        if self.next_move is None and self.prev_move is None:
            self.is_finished = True
            self.determine_winner()

        elif self.is_board_full():
            self.is_finished = True
            self.determine_winner()

        elif self.black_score == 0 or self.white_score == 0:
            self.is_finished = True
            self.determine_winner()

    
    def determine_winner(self):

        if self.black_score > self.white_score:
            self.game_result = "Black Wins"
        elif self.white_score > self.black_score:
            self.game_result = "White Wins"
        else:
            self.game_result = "Draw"