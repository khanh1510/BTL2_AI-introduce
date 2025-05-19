
from ..common.enums import PlayerType
from .board import SquareType




class Player:
    def __init__(self,
                 player_type: PlayerType,
                 disc_color: SquareType,
                 state_eval=None,
                 depth: int = None,
                 ml_model_instance = None):

        if not isinstance(player_type, PlayerType):
            raise TypeError("player_type must be an instance of PlayerType Enum")
        if not isinstance(disc_color, SquareType):
            raise TypeError("disc_color must be an instance of SquareType Enum")
        if disc_color == SquareType.EMPTY or disc_color == SquareType.VALID:
            raise ValueError("Player disc_color cannot be EMPTY or VALID")
        
        self.player_type = player_type
        self.disc_color = disc_color
        self.state_eval = state_eval
        self.depth = depth
        self.ml_model = None


        if ml_model_instance:
            self.ml_model = ml_model_instance


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

                if game.board.state[row, col] == SquareType.VALID:
                    return row, col
                else:
                    print("Invalid move (not a valid empty square). Try again.")
            except (ValueError, IndexError):
                print("Invalid input format. Enter the move in the format 'e4'.")
            except Exception as e:
                print(f"An error occurred: {e}")