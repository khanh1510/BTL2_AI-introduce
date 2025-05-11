# play_othello_console.py
import time
import logging # Thêm logging
from src.gameLogic.game import Game
from src.gameLogic.player import Player, PlayerType
from src.gameLogic.board import SquareType
from src.ai.ML_DL_models.model import Model

# Cấu hình logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


shared_ml_model = None
try:
    logging.info("Đang tải ML model chung...")
    shared_ml_model = Model() # Model.__init__ sẽ log trạng thái tải của LSTM và GPT-2
    if not (shared_ml_model.lstm or (shared_ml_model.gpt2 and shared_ml_model.tokenizer)):
        logging.warning("Không có model ML nào (LSTM hoặc GPT-2) được tải thành công.")
except Exception as e:
    logging.error(f"Lỗi nghiêm trọng khi khởi tạo ML Model: {e}")
    logging.warning("Chức năng ML AI sẽ không khả dụng.")


print("Bạn muốn chơi với quân màu gì (black/white), hoặc chọn chế độ AI vs AI?")
print("Ví dụ AI vs AI: 'lstm_vs_random', 'gpt2_vs_lstm', 'random_vs_gpt2'")
mode_or_color = input("Nhập lựa chọn: ").lower()

player_black = None
player_white = None

def create_player_console(player_str, color_enum, model_instance=None):
    player_str = player_str.lower() # Chuẩn hóa input
    if player_str == "offline":
        return Player(PlayerType.OFFLINE, color_enum)
    elif player_str == "random":
        return Player(PlayerType.RANDOM, color_enum)
    elif player_str == "lstm":
        if model_instance and model_instance.lstm:
            logging.info(f"Tạo người chơi LSTM cho màu {color_enum.name}")
            return Player(PlayerType.LSTM, color_enum, ml_model_instance=model_instance)
        else:
            logging.warning("Model LSTM không khả dụng, sử dụng Random thay thế cho LSTM.")
            return Player(PlayerType.RANDOM, color_enum)
    elif player_str == "gpt2":
        if model_instance and model_instance.gpt2 and model_instance.tokenizer:
            logging.info(f"Tạo người chơi GPT-2 cho màu {color_enum.name}")
            return Player(PlayerType.GPT2, color_enum, ml_model_instance=model_instance)
        else:
            logging.warning("Model GPT-2 không khả dụng, sử dụng Random thay thế cho GPT-2.")
            return Player(PlayerType.RANDOM, color_enum)
    # ... (Minimax nếu bạn muốn thêm lại) ...
    else:
        logging.warning(f"Loại người chơi không xác định: '{player_str}'. Sử dụng Random thay thế.")
        return Player(PlayerType.RANDOM, color_enum)

if "_vs_" in mode_or_color:
    parts = mode_or_color.split("_vs_")
    if len(parts) == 2:
        p1_type, p2_type = parts[0], parts[1]
        print(f"Chế độ AI vs AI: {p1_type.upper()} (BLACK) vs {p2_type.upper()} (WHITE)")
        player_black = create_player_console(p1_type, SquareType.BLACK, shared_ml_model)
        player_white = create_player_console(p2_type, SquareType.WHITE, shared_ml_model)
    else:
        print("Định dạng AI vs AI không hợp lệ. Sử dụng Random vs Random.")
        player_black = Player(PlayerType.RANDOM, SquareType.BLACK)
        player_white = Player(PlayerType.RANDOM, SquareType.WHITE)

else: # Chế độ người chơi vs AI
    user_color_choice = mode_or_color
    while user_color_choice not in ['black', 'white']:
        print("Bạn muốn chơi với quân màu gì (black/white)?")
        user_color_choice = input("Nhập: ").lower()

    # Hỏi người dùng muốn đối đầu với AI nào
    available_ais = ["random"]
    if shared_ml_model and shared_ml_model.lstm:
        available_ais.append("lstm")
    if shared_ml_model and shared_ml_model.gpt2 and shared_ml_model.tokenizer:
        available_ais.append("gpt2")
    
    ai_opponent_type_input = "random" # Mặc định
    if len(available_ais) > 1: # Nếu có nhiều hơn chỉ random
        ai_opponent_type_input = input(f"Chọn AI đối thủ ({', '.join(available_ais)}): ").lower()
        if ai_opponent_type_input not in available_ais:
            logging.warning(f"Lựa chọn AI '{ai_opponent_type_input}' không hợp lệ. Dùng Random.")
            ai_opponent_type_input = "random"
    else:
        logging.info("Chỉ có Random AI khả dụng làm đối thủ.")


    if user_color_choice == 'black':
        player_black = Player(PlayerType.OFFLINE, SquareType.BLACK)
        player_white = create_player_console(ai_opponent_type_input, SquareType.WHITE, shared_ml_model)
        print(f"Bạn (BLACK) vs {ai_opponent_type_input.upper()} (WHITE)")
    else:
        player_black = create_player_console(ai_opponent_type_input, SquareType.BLACK, shared_ml_model)
        player_white = Player(PlayerType.OFFLINE, SquareType.WHITE)
        print(f"{ai_opponent_type_input.upper()} (BLACK) vs Bạn (WHITE)")

if player_black is None or player_white is None:
    logging.error("Không thể khởi tạo người chơi. Thoát.")
    exit()

game = Game(player_black, player_white)
game.move_history_str = "" # Đảm bảo lịch sử rỗng khi bắt đầu game mới

logging.info(f"Bắt đầu game: {game.player_black.player_type.value} (BLACK) vs {game.player_white.player_type.value} (WHITE)")

while not game.is_finished:
    print("\n\n")
    game.board.display()
    # time.sleep(1.5)

    current_player_color_name = game.active.disc_color.name
    current_player_type_name = game.active.player_type.value

    if game.prev_move is not None:
        prev_player_color_name = game.inactive.disc_color.name
        prev_move_notation = game._coordinate_to_notation(game.prev_move[0], game.prev_move[1])
        if prev_move_notation: # Chỉ in nếu có nước đi trước đó
             print(f"{prev_player_color_name} đã đi {prev_move_notation}.")
        # time.sleep(1)

    print(f"Lượt của {current_player_color_name} ({current_player_type_name}). Lịch sử hiện tại: '{game.move_history_str}'") # In lịch sử
    # time.sleep(1)

    game.get_player_move()

    if game.next_move is None:
        # ... (logic xử lý bỏ lượt giữ nguyên)
        current_player_passed = False
        if not game.is_valid_moves(): # Thực sự không có nước đi
            print(f"{current_player_color_name} không có nước đi hợp lệ. Bỏ lượt.")
            current_player_passed = True
        else: # Có nước đi nhưng model/logic lỗi
            print(f"Lỗi: {current_player_color_name} có nước đi hợp lệ nhưng next_move là None. Coi như bỏ lượt.")
            current_player_passed = True # Vẫn coi như bỏ lượt để game tiếp tục

        game.prev_move = game.next_move # next_move là None
        game.change_turn()
        game.update_valid_moves() # Cho người chơi tiếp theo
        
        if current_player_passed: # Chỉ kiểm tra kết thúc nếu người chơi hiện tại thực sự bỏ lượt
            # Kiểm tra xem người chơi tiếp theo có nước đi không
            if not game.is_valid_moves():
                print(f"{game.active.disc_color.name} (sau khi người trước bỏ lượt) cũng không có nước đi hợp lệ.")
                game.is_finished = True # Cả hai người chơi liên tiếp không có nước đi
            else: # Người chơi tiếp theo có nước đi
                pass # Game tiếp tục với người chơi tiếp theo
        
        if game.is_finished: # Kiểm tra lại is_finished sau logic trên
             game.determine_winner() # Xác định người thắng nếu game kết thúc
             break
        continue


    game.make_move()
    game.change_turn()
    game.update_valid_moves()
    game.update_scores()
    game.check_finished() # Sẽ gọi determine_winner nếu is_finished = True
    # time.sleep(1.5)

# Kết thúc game
print("\n\n--- TRẠNG THÁI CUỐI CÙNG ---")
game.board.display()
if not game.game_result: # Đảm bảo game_result được set
    game.determine_winner()
print(f"Trò chơi kết thúc. {game.game_result}.")
print(f"Tỉ số: BLACK {game.black_score} - WHITE {game.white_score}")