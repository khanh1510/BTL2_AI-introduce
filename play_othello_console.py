# play_othello_console.py

import time
from src.gameLogic.game import Game
from src.gameLogic.player import Player, PlayerType
from src.gameLogic.board import SquareType
from src.ai.ML_DL_models.model import Model # <-- THÊM IMPORT NÀY

# Tạo một instance model dùng chung để tránh tải lại nhiều lần
# Model sẽ được tải khi script bắt đầu
shared_ml_model = None
try:
    print("Đang tải ML model...")
    shared_ml_model = Model()
    if shared_ml_model.lstm is None: # Kiểm tra nếu LSTM không tải được
        print("Không thể tải model LSTM. Chức năng LSTM AI sẽ không khả dụng.")
        # Không raise lỗi ở đây để script vẫn có thể chạy với các AI khác
except Exception as e:
    print(f"Lỗi nghiêm trọng khi khởi tạo ML Model: {e}")
    print("Chức năng ML AI sẽ không khả dụng.")


# Ask offline user to choose their color
print("Bạn muốn chơi với quân màu gì (black/white), hoặc chọn chế độ AI vs AI (ví dụ: 'lstm_vs_random', 'random_vs_minimax1')?")
mode_or_color = input("Nhập lựa chọn: ").lower()

player_black = None
player_white = None

def create_player_console(player_str, color_enum, model_instance=None):
    if player_str == "offline":
        return Player(PlayerType.OFFLINE, color_enum)
    elif player_str == "random":
        return Player(PlayerType.RANDOM, color_enum)
    elif player_str == "lstm":
        if model_instance and model_instance.lstm:
            return Player(PlayerType.LSTM, color_enum, ml_model_instance=model_instance)
        else:
            print("Model LSTM không khả dụng, sử dụng Random thay thế.")
            return Player(PlayerType.RANDOM, color_enum)
    # Thêm các AI khác nếu cần, ví dụ Minimax
    # elif player_str.startswith("minimax"):
    #     try:
    #         depth = int(player_str.replace("minimax",""))
    #         from src.ai.minimax.evaluator import StateEvaluator, HeuristicType
    #         state_eval = StateEvaluator(weights={ # Ví dụ weights, bạn có thể tùy chỉnh
    #             HeuristicType.DISC_DIFF: 5/60,
    #             HeuristicType.MOBILITY: 15/60,
    #             HeuristicType.CORNERS: 40/60
    #         })
    #         return Player(PlayerType.MINIMAX, color_enum, state_eval, depth)
    #     except ValueError:
    #         print(f"Độ sâu Minimax không hợp lệ: {player_str}. Sử dụng Random thay thế.")
    #         return Player(PlayerType.RANDOM, color_enum)
    else:
        print(f"Loại người chơi không xác định: {player_str}. Sử dụng Random thay thế.")
        return Player(PlayerType.RANDOM, color_enum)

if "_vs_" in mode_or_color:
    parts = mode_or_color.split("_vs_")
    p1_type, p2_type = parts[0], parts[1]
    print(f"Chế độ AI vs AI: {p1_type.upper()} (BLACK) vs {p2_type.upper()} (WHITE)")
    player_black = create_player_console(p1_type, SquareType.BLACK, shared_ml_model)
    player_white = create_player_console(p2_type, SquareType.WHITE, shared_ml_model)
else: # Chế độ người chơi vs AI
    user_color_choice = mode_or_color
    while user_color_choice not in ['black', 'white']:
        print("Bạn muốn chơi với quân màu gì (black/white)?")
        user_color_choice = input("Nhập: ").lower()

    ai_opponent_type = "lstm" # Mặc định đối thủ là LSTM nếu có thể
    if not (shared_ml_model and shared_ml_model.lstm):
        ai_opponent_type = "random" # Dự phòng nếu LSTM không tải được
        print("Model LSTM không khả dụng, bạn sẽ chơi với Random AI.")
    # Hoặc bạn có thể hỏi người dùng muốn chơi với AI nào:
    # ai_opponent_type = input(f"Chọn AI đối thủ (ví dụ: random, lstm, minimax1): ").lower()


    if user_color_choice == 'black':
        player_black = Player(PlayerType.OFFLINE, SquareType.BLACK)
        player_white = create_player_console(ai_opponent_type, SquareType.WHITE, shared_ml_model)
        print(f"Bạn (BLACK) vs {ai_opponent_type.upper()} (WHITE)")
    else:
        player_black = create_player_console(ai_opponent_type, SquareType.BLACK, shared_ml_model)
        player_white = Player(PlayerType.OFFLINE, SquareType.WHITE)
        print(f"{ai_opponent_type.upper()} (BLACK) vs Bạn (WHITE)")


# Initialise game
game = Game(player_black, player_white)

while not game.is_finished:
    print("\n\n")
    game.board.display()
    # time.sleep(1.5) # Bỏ comment nếu muốn game chậm lại

    current_player_color_name = game.active.disc_color.name
    current_player_type_name = game.active.player_type.value

    if game.prev_move is not None:
        prev_player_color_name = game.inactive.disc_color.name
        prev_move_notation = game._coordinate_to_notation(game.prev_move[0], game.prev_move[1])
        print(f"{prev_player_color_name} đã đi {prev_move_notation}.")
        # time.sleep(1)

    print(f"Lượt của {current_player_color_name} ({current_player_type_name}).")
    # time.sleep(1)

    game.get_player_move() # Lấy nước đi

    if game.next_move is None : # Xử lý trường hợp không có nước đi hợp lệ (bỏ lượt)
        if game.is_valid_moves(): # Kiểm tra lại, nếu get_player_move() set next_move là None nhưng vẫn có nước đi hợp lệ (ví dụ model lỗi)
            print(f"Lỗi: {current_player_color_name} có nước đi hợp lệ nhưng next_move là None. Bỏ qua lượt.")
        else:
            print(f"{current_player_color_name} không có nước đi hợp lệ. Bỏ lượt.")
        game.prev_move = game.next_move # Cập nhật prev_move dù là None
        game.change_turn()
        game.update_valid_moves() # Cập nhật cho người chơi tiếp theo
        game.check_finished() # Kiểm tra kết thúc game sau khi bỏ lượt
        if game.is_finished: # Nếu game kết thúc ngay sau khi bỏ lượt
            break
        else: # Nếu chưa kết thúc, chuyển sang người chơi tiếp theo
            # Kiểm tra xem người chơi tiếp theo có nước đi không, nếu không thì game kết thúc
            if not game.is_valid_moves():
                print(f"{game.active.disc_color.name} cũng không có nước đi hợp lệ.")
                game.prev_move = None # Đặt lại prev_move cho người chơi thứ hai không đi được
                game.check_finished()
                if game.is_finished:
                    break
        continue # Bỏ qua phần còn lại của vòng lặp


    game.make_move() # Thực hiện nước đi
    game.change_turn()
    game.update_valid_moves() # Cập nhật valid_moves cho người chơi tiếp theo
    game.update_scores()

    game.check_finished()
    # time.sleep(1.5)

# Display final game board
print("\n\n--- TRẠNG THÁI CUỐI CÙNG ---")
game.board.display()
game.determine_winner() # Đảm bảo game_result được set
print(f"Trò chơi kết thúc. {game.game_result}.")
print(f"Tỉ số: BLACK {game.black_score} - WHITE {game.white_score}")