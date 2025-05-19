
import time
import logging
from src.gameLogic.game import Game
from src.gameLogic.player import Player, PlayerType
from src.gameLogic.board import SquareType
from src.ai.ML_DL_models.model import Model

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

shared_ml_model = None
try:
    logging.info("Đang tải ML model chung...")
    shared_ml_model = Model()
    if not (shared_ml_model.lstm or \
            (shared_ml_model.gpt2 and shared_ml_model.tokenizer) or \
            shared_ml_model.rf_model or \
            shared_ml_model.xgb_model):
        logging.warning("Không có model ML nào (LSTM, GPT-2, RF, XGB) được tải thành công.")
except Exception as e:
    logging.error(f"Lỗi nghiêm trọng khi khởi tạo ML Model: {e}")
    logging.warning("Chức năng ML AI sẽ không khả dụng.")

print("Bạn muốn chơi với quân màu gì (black/white), hoặc chọn chế độ AI vs AI?")
print("Ví dụ AI vs AI: 'lstm_vs_random', 'gpt2_vs_lstm', 'rf_vs_xgb', 'xgb_vs_minimax'")
mode_or_color = input("Nhập lựa chọn: ").lower()

player_black = None
player_white = None

def create_player_console(player_str, color_enum, model_instance=None):
    player_str = player_str.lower()
    if player_str == "offline":
        return Player(PlayerType.OFFLINE, color_enum)
    elif player_str == "random":
        return Player(PlayerType.RANDOM, color_enum)
    elif player_str == "lstm":
        if model_instance and model_instance.lstm:
            return Player(PlayerType.LSTM, color_enum, ml_model_instance=model_instance)
        else:
            logging.warning("Model LSTM không khả dụng, sử dụng Random thay thế.")
            return Player(PlayerType.RANDOM, color_enum)
    elif player_str == "gpt2":
        if model_instance and model_instance.gpt2 and model_instance.tokenizer:
            return Player(PlayerType.GPT2, color_enum, ml_model_instance=model_instance)
        else:
            logging.warning("Model GPT-2 không khả dụng, sử dụng Random thay thế.")
            return Player(PlayerType.RANDOM, color_enum)
    elif player_str == "rf" or player_str == "rf_model":
        if model_instance and model_instance.rf_model:
            logging.info(f"Tạo người chơi Random Forest cho màu {color_enum.name}")
            return Player(PlayerType.RF_MODEL, color_enum, ml_model_instance=model_instance)
        else:
            logging.warning("Model Random Forest không khả dụng, sử dụng Random thay thế.")
            return Player(PlayerType.RANDOM, color_enum)
    elif player_str == "xgb" or player_str == "xgb_model":
        if model_instance and model_instance.xgb_model:
            logging.info(f"Tạo người chơi XGBoost cho màu {color_enum.name}")
            return Player(PlayerType.XGB_MODEL, color_enum, ml_model_instance=model_instance)
        else:
            logging.warning("Model XGBoost không khả dụng, sử dụng Random thay thế.")
            return Player(PlayerType.RANDOM, color_enum)
    elif player_str.startswith("minimax_"):
        try:
            from src.ai.minimax.evaluator import StateEvaluator, HeuristicType
            depth = int(player_str.split('_')[1])
            state_eval = StateEvaluator(weights={
                HeuristicType.DISC_DIFF: 5/60, HeuristicType.MOBILITY: 15/60, HeuristicType.CORNERS: 40/60
            })
            logging.info(f"Tạo người chơi Minimax (depth {depth}) cho màu {color_enum.name}")
            return Player(PlayerType.MINIMAX, color_enum, state_eval, depth)
        except (ImportError, IndexError, ValueError) as e:
            logging.warning(f"Lỗi tạo Minimax player '{player_str}': {e}. Dùng Random.")
            return Player(PlayerType.RANDOM, color_enum)
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
        player_black = Player(PlayerType.RANDOM, SquareType.BLACK)
        player_white = Player(PlayerType.RANDOM, SquareType.WHITE)
else:
    user_color_choice = mode_or_color
    if user_color_choice not in ['black', 'white']: user_color_choice = 'black'

    available_ais = ["random"]
    if shared_ml_model and shared_ml_model.lstm: available_ais.append("lstm")
    if shared_ml_model and shared_ml_model.gpt2: available_ais.append("gpt2")
    if shared_ml_model and shared_ml_model.rf_model: available_ais.append("rf")
    if shared_ml_model and shared_ml_model.xgb_model: available_ais.append("xgb")
    available_ais.extend(["minimax_1", "minimax_2", "minimax_3"])

    ai_opponent_type_input = "random"
    if len(available_ais) > 1:
        ai_opponent_type_input = input(f"Chọn AI đối thủ ({', '.join(available_ais)}): ").lower()
        if ai_opponent_type_input not in available_ais:
            ai_opponent_type_input = "random"

    if user_color_choice == 'black':
        player_black = Player(PlayerType.OFFLINE, SquareType.BLACK)
        player_white = create_player_console(ai_opponent_type_input, SquareType.WHITE, shared_ml_model)
    else:
        player_black = create_player_console(ai_opponent_type_input, SquareType.BLACK, shared_ml_model)
        player_white = Player(PlayerType.OFFLINE, SquareType.WHITE)
    print(f"Bạn ({user_color_choice.upper()}) vs {ai_opponent_type_input.upper()} ({'WHITE' if user_color_choice == 'black' else 'BLACK'})")


if player_black is None or player_white is None:
    logging.error("Không thể khởi tạo người chơi. Thoát.")
    exit()

game = Game(player_black, player_white)
game.move_history_str = ""

logging.info(f"Bắt đầu game: {game.player_black.player_type.value} (BLACK) vs {game.player_white.player_type.value} (WHITE)")

while not game.is_finished:
    print("\n\n")
    game.board.display()

    current_player_color_name = game.active.disc_color.name
    current_player_type_name = game.active.player_type.value

    if game.prev_move is not None:
        prev_player_color_name = game.inactive.disc_color.name
        prev_move_notation = game._coordinate_to_notation(game.prev_move[0], game.prev_move[1])
        if prev_move_notation:
             print(f"{prev_player_color_name} đã đi {prev_move_notation}.")

    print(f"Lượt của {current_player_color_name} ({current_player_type_name}). Lịch sử: '{game.move_history_str}'")

    start_time = time.time()
    game.get_player_move()
    end_time = time.time()
    move_time = end_time - start_time
    if game.active.player_type != PlayerType.OFFLINE and game.active.player_type != PlayerType.USER :
        print(f"({current_player_type_name} suy nghĩ trong {move_time:.4f} giây)")

    if game.next_move is None:
        current_player_passed = False

        if not game.is_any_valid_move_on_board():
            print(f"{current_player_color_name} không có nước đi hợp lệ. Bỏ lượt.")
            current_player_passed = True


        if current_player_passed:
            game.update_valid_moves()
            if not game.is_any_valid_move_on_board():
                print(f"{game.active.disc_color.name} (sau khi người trước bỏ lượt) cũng không có nước đi hợp lệ.")
                game.is_finished = True 
            else: 
                pass

        if game.is_finished:
             game.determine_winner()
             break
        continue

    game.make_move()
    game.change_turn()
    game.update_valid_moves()
    game.update_scores()
    game.check_finished()

print("\n\n--- TRẠNG THÁI CUỐI CÙNG ---")
game.board.display()
if not game.game_result and game.is_finished:
    game.determine_winner()
print(f"Trò chơi kết thúc. {game.game_result if game.game_result else 'Chưa xác định'}.")
print(f"Tỉ số: BLACK {game.black_score} - WHITE {game.white_score}")
print(f"Lịch sử nước đi cuối cùng: {game.move_history_str}")