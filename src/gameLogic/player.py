from ..common.enums import PlayerType
from .board import SquareType
from ..ai.ML_DL_models.model import Model


class Player:

    def __init__(self,
                 player_type: PlayerType,
                 disc_color: SquareType,
                 state_eval=None,
                 depth: int = None,
                 ml_model_instance: Model = None):

        if not isinstance(player_type, PlayerType):
            raise TypeError(
                "player_type must be an instance of PlayerType Enum")
        if not isinstance(disc_color, SquareType):
            raise TypeError(
                "disc_color must be an instance of SquareType Enum")
        if disc_color == SquareType.EMPTY or disc_color == SquareType.VALID:
            raise ValueError("Player disc_color cannot be EMPTY or VALID")
        self.player_type = player_type
        self.disc_color = disc_color
        self.state_eval = state_eval
        self.depth = depth
        self.ml_model = None

        if player_type == PlayerType.LSTM: # Hoặc PlayerType.GPT2 nếu bạn thêm
            if ml_model_instance:
                self.ml_model = ml_model_instance
            else:
                # Tùy chọn: Tự động tạo instance model nếu không được cung cấp
                # Điều này tiện cho console, nhưng cho web, model nên được quản lý tập trung
                print(f"Cảnh báo: Player {player_type.value} đang tự tạo instance ML model. "
                      "Cân nhắc quản lý tập trung cho hiệu suất tốt hơn trong web app.")
                self.ml_model = Model()
                if self.ml_model.lstm is None: # Kiểm tra model có được load thành công không
                    raise RuntimeError("Không thể tải model LSTM cho Player.")

    def get_offline_move(self, game):
        while True:
            try:
                move = input(
                    f"Enter move for {self.disc_color.name}: ").strip().lower()
                if len(move) != 2:
                    print(
                        "Invalid format. Use column letter and row number (e.g., 'e4').")
                    continue
                col = ord(move[0]) - ord('a')
                row = int(move[1]) - 1
                if not (0 <= row < game.board.state.shape[0] and 0 <= col < game.board.state.shape[1]):
                    print("Move out of bounds.")
                    continue
                if game.is_valid_move(row, col):
                    return row, col
                else:
                    valid_moves_list = game.get_valid_moves()
                    if (row, col) in valid_moves_list:
                        print("Move is listed as valid, proceeding (DEBUG).")
                        return row, col
                    else:
                        print(
                            "Invalid move (not empty or doesn't flip discs). Try again.")
            except (ValueError, IndexError):
                print("Invalid input format. Enter the move in the format 'e4'.")
            except Exception as e:
                print(f"An error occurred: {e}")
