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
        self.move_history_str = ""

    def __deepcopy__(self, memo):
        cls = self.__class__
        new_game = cls.__new__(cls)
        memo[id(self)] = new_game

        new_game.board = copy.deepcopy(self.board, memo)
        new_game.is_finished = self.is_finished

        # Deepcopy players, bao gồm cả ml_model nếu có
        new_game.player_black = Player(
            player_type=self.player_black.player_type,
            disc_color=self.player_black.disc_color,
            state_eval=self.player_black.state_eval,
            depth=self.player_black.depth,
            ml_model_instance=self.player_black.ml_model # Truyền instance model
        )
        memo[id(self.player_black)] = new_game.player_black

        new_game.player_white = Player(
            player_type=self.player_white.player_type,
            disc_color=self.player_white.disc_color,
            state_eval=self.player_white.state_eval,
            depth=self.player_white.depth,
            ml_model_instance=self.player_white.ml_model # Truyền instance model
        )
        memo[id(self.player_white)] = new_game.player_white

        if self.active is self.player_black:
            new_game.active = new_game.player_black
            new_game.inactive = new_game.player_white
        elif self.active is self.player_white:
            new_game.active = new_game.player_white
            new_game.inactive = new_game.player_black
        else:
            new_game.active = None
            new_game.inactive = None

        new_game.next_move = copy.deepcopy(self.next_move, memo)
        new_game.prev_move = copy.deepcopy(self.prev_move, memo)
        new_game.black_score = self.black_score
        new_game.white_score = self.white_score
        new_game.game_result = self.game_result
        new_game.move_history_str = self.move_history_str # <-- SAO CHÉP LỊCH SỬ NƯỚC ĐI

        return new_game
    
    # --- CÁC HÀM HELPER CHO VIỆC CHUYỂN ĐỔI TỌA ĐỘ ---
    def _coordinate_to_notation(self, row, col):
        if row is None or col is None:
            return ""
        return chr(ord('a') + col) + str(row + 1)

    def _notation_to_coordinate(self, notation_str):
        if not notation_str or len(notation_str) != 2:
            # print(f"Cảnh báo: Chuỗi ký hiệu không hợp lệ để chuyển đổi: '{notation_str}'")
            return None
        try:
            col_char = notation_str[0].lower()
            row_char = notation_str[1]
            if not ('a' <= col_char <= 'h' and '1' <= row_char <= '8'):
                # print(f"Cảnh báo: Ký hiệu ngoài phạm vi bàn cờ: '{notation_str}'")
                return None
            col = ord(col_char) - ord('a')
            row = int(row_char) - 1
            # Không cần kiểm tra 0 <= row < BOARD_SIZE nữa vì đã kiểm tra ở trên
            return row, col
        except (ValueError, TypeError, IndexError) as e:
            # print(f"Cảnh báo: Lỗi khi chuyển đổi chuỗi ký hiệu: '{notation_str}', Lỗi: {e}")
            return None
    # --- KẾT THÚC HÀM HELPER ---

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

        self.update_valid_moves() # Luôn cập nhật các ô VALID trên bảng cho người chơi hiện tại

        if not self.is_valid_moves(): # Nếu không có nước đi hợp lệ nào
            self.next_move = None
            return

        if player.player_type == PlayerType.OFFLINE:
            move_coords = player.get_offline_move(self) # get_offline_move sẽ cần kiểm tra với self.board.state[r,c] == SquareType.VALID
            if move_coords:
                self.next_move = move_coords
        elif player.player_type == PlayerType.USER:
            pass # Nước đi của USER sẽ được đặt từ Flask view vào self.next_move
        elif player.player_type == PlayerType.RANDOM:
            self.next_move = getRandomMove(self) # getRandomMove nên chọn từ các ô SquareType.VALID
        elif player.player_type == PlayerType.MINIMAX:
            if player.depth is None or player.state_eval is None:
                raise ValueError("Minimax player must have depth and state_eval configured.")
            self.next_move = get_minimax_move(self, player.depth, player.state_eval)
        elif player.player_type == PlayerType.LSTM: # <-- XỬ LÝ CHO LSTM
            if player.ml_model and hasattr(player.ml_model, 'infer_lstm'):
                print(f"LSTM input history: '{self.move_history_str}'")
                notation_move = player.ml_model.infer_lstm(self.move_history_str)
                print(f"LSTM raw output notation: '{notation_move}'")
                self.next_move = self._notation_to_coordinate(notation_move)

                # Kiểm tra nếu nước đi từ LSTM không hợp lệ (ví dụ: model trả về ô không phải VALID)
                # hoặc model không trả về gì
                if self.next_move is None or self.board.state[self.next_move[0], self.next_move[1]] != SquareType.VALID:
                    print(f"Cảnh báo: LSTM model trả về nước đi không hợp lệ ('{notation_move}' -> {self.next_move}) hoặc ô không phải VALID.")
                    print("Các nước đi hợp lệ hiện tại là:")
                    for r_idx in range(BOARD_SIZE):
                        for c_idx in range(BOARD_SIZE):
                            if self.board.state[r_idx, c_idx] == SquareType.VALID:
                                print(f"  - {self._coordinate_to_notation(r_idx, c_idx)}")
                    print("LSTM sẽ bỏ lượt hoặc chọn ngẫu nhiên.")
                    # Chiến lược dự phòng: chọn ngẫu nhiên nếu model lỗi
                    # Hoặc có thể để self.next_move = None (bỏ lượt)
                    self.next_move = getRandomMove(self)
                    if self.next_move:
                         print(f"LSTM fallback to random move: {self._coordinate_to_notation(self.next_move[0], self.next_move[1])}")
                    else:
                         print("LSTM fallback, no random move available either.")

            else:
                print("Lỗi: LSTM Player không có instance ml_model hoặc thiếu hàm infer_lstm.")
                self.next_move = getRandomMove(self) # Dự phòng
                if self.next_move:
                    print(f"LSTM player missing model, fallback to random: {self._coordinate_to_notation(self.next_move[0], self.next_move[1])}")

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

        # Cập nhật lịch sử nước đi SAU KHI nước đi đã được thực hiện
        # VÀ TRƯỚC KHI thay đổi lượt chơi
        move_notation = self._coordinate_to_notation(row, col)
        if move_notation: # Chỉ thêm vào lịch sử nếu nước đi hợp lệ
            self.move_history_str += move_notation
            # logging.debug(f"Move history updated in make_move: '{self.move_history_str}'")
        else:
            # logging.warning(f"Không thể chuyển đổi nước đi ({row},{col}) thành ký hiệu, không cập nhật lịch sử.")
            pass

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