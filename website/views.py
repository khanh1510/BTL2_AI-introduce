from flask import (
    Blueprint, render_template, request, 
    jsonify, session, redirect, url_for
)
import pickle
import logging 
import numpy as np
logging.basicConfig(level=logging.DEBUG)

from src.gameLogic.game import Game
from src.gameLogic.board import SquareType, Board 
from src.gameLogic.player import Player, PlayerType 
from src.ai.minimax.evaluator import StateEvaluator, HeuristicType
from src.ai.ML_DL_models.model import Model

views = Blueprint("views", __name__)


_ml_model_global_instance = None

def get_global_ml_model():
    global _ml_model_global_instance
    if _ml_model_global_instance is None:
        try:
            logging.info("Đang tải ML model cho ứng dụng web...")
            _ml_model_global_instance = Model()
            if _ml_model_global_instance.lstm is None:
                 logging.warning("CẢNH BÁO WEB: Không thể tải model LSTM.")
            else:
                 logging.info("Model LSTM đã được tải thành công cho web.")
        except Exception as e:
            logging.error(f"LỖI WEB: Không thể khởi tạo ML Model: {e}")
    return _ml_model_global_instance

def _attach_models_to_game_players(game_instance):
    if not game_instance:
        return
    
    ml_model = get_global_ml_model()
    if ml_model:
        if game_instance.player_black.player_type in [PlayerType.LSTM]: # Thêm PlayerType.GPT2 nếu có
            game_instance.player_black.ml_model = ml_model
            # logging.debug(f"Gắn model cho BLACK player ({game_instance.player_black.player_type.value})")
        if game_instance.player_white.player_type in [PlayerType.LSTM]: # Thêm PlayerType.GPT2 nếu có
            game_instance.player_white.ml_model = ml_model


def create_player_from_type_str(type_str, color_enum, state_evaluator):
    model_instance = get_global_ml_model() # Lấy instance model

    if type_str == "USER":
        return Player(PlayerType.USER, color_enum)
    elif type_str == "RANDOM":
        return Player(PlayerType.RANDOM, color_enum)
    elif type_str == "MINIMAX_1":
        return Player(PlayerType.MINIMAX, color_enum, state_evaluator, 1)
    elif type_str == "MINIMAX_2":
        return Player(PlayerType.MINIMAX, color_enum, state_evaluator, 2)
    elif type_str == "MINIMAX_3":
        return Player(PlayerType.MINIMAX, color_enum, state_evaluator, 3)
    elif type_str == "MINIMAX_4":
        return Player(PlayerType.MINIMAX, color_enum, state_evaluator, 4)
    elif type_str == "LSTM":
        # Khi tạo player ban đầu, ta có thể truyền model instance.
        # Lúc này player_black/white là các object mới, không phải từ session.
        if model_instance and model_instance.lstm:
            return Player(PlayerType.LSTM, color_enum, ml_model_instance=model_instance)
        else:
            logging.warning("Model LSTM không khả dụng cho web, sử dụng Random AI thay thế.")
            return Player(PlayerType.RANDOM, color_enum)
    # ...
    logging.warning(f"Loại người chơi không xác định '{type_str}', sử dụng Random AI.")
    return Player(PlayerType.RANDOM, color_enum)

@views.route("/")
def home():
    return render_template("home.html")


@views.route('/play_game', methods=['GET', 'POST'])
def play_game():
    if request.method == 'POST':
        player_black_type_str = request.form.get('player_black_type')
        player_white_type_str = request.form.get('player_white_type')

        session['player_black_type'] = player_black_type_str
        session['player_white_type'] = player_white_type_str
        session.pop('user_color', None)

        state_eval = StateEvaluator(weights={
            HeuristicType.DISC_DIFF: 5/60,
            HeuristicType.MOBILITY: 15/60,
            HeuristicType.CORNERS: 40/60
        })
    
        player_black = create_player_from_type_str(player_black_type_str, SquareType.BLACK, state_eval)
        player_white = create_player_from_type_str(player_white_type_str, SquareType.WHITE, state_eval)
        
        game = Game(player_black, player_white)
        game.move_history_str = ""

        temp_black_model = game.player_black.ml_model
        temp_white_model = game.player_white.ml_model
        game.player_black.ml_model = None
        game.player_white.ml_model = None
        
        serialized_game = pickle.dumps(game)
        session['game_instance'] = serialized_game
        session['game_started'] = True
        
        game.player_black.ml_model = temp_black_model
        game.player_white.ml_model = temp_white_model

        return redirect(url_for('views.play_game'))
    
    else:
        serialized_game = session.get('game_instance')
        game = None
        
        if serialized_game:
            try:
                game = pickle.loads(serialized_game)
                _attach_models_to_game_players(game) # Gắn lại model sau khi unpickle
            except Exception as e:
                logging.error(f"Lỗi khi unpickle game từ session: {e}")
                session.pop('game_instance', None) # Xóa session hỏng
                session.pop('game_started', None)
                # game sẽ vẫn là None
        
        if game is None: # Nếu không có game trong session hoặc unpickle lỗi
            logging.info("Không có game trong session hoặc unpickle lỗi, tạo game mặc định.")
            # Tạo game mặc định (ví dụ: User vs Random)
            # Lấy player types từ session nếu có, hoặc dùng mặc định
            p_black_type_session = session.get('player_black_type', 'USER')
            p_white_type_session = session.get('player_white_type', 'RANDOM') # Hoặc MINIMAX_3 như trong HTML

            state_eval_default = StateEvaluator(weights={
                HeuristicType.DISC_DIFF: 5/60,
                HeuristicType.MOBILITY: 15/60,
                HeuristicType.CORNERS: 40/60
            })
            user_player_default = create_player_from_type_str(p_black_type_session, SquareType.BLACK, state_eval_default)
            ai_player_default = create_player_from_type_str(p_white_type_session, SquareType.WHITE, state_eval_default)
            game = Game(user_player_default, ai_player_default)
            game.move_history_str = "" # Khởi tạo lịch sử
        
        return render_template('play_game.html', 
                               game=game, 
                               game_started=session.get('game_started', False),
                               black_score=game.black_score if game else 2,
                               white_score=game.white_score if game else 2)


@views.route('/user_move', methods=['POST'])
def user_move():
    data = request.get_json()
    row = data['row']
    col = data['col']
    
    serialized_game = session.get('game_instance')
    
    if serialized_game:
        logging.debug(f"Received move: row={row}, col={col}")
        game = pickle.loads(serialized_game)
        _attach_models_to_game_players(game)

        if game.active.player_type != PlayerType.USER:
             return jsonify({'message': 'Not user turn on server', 'error': True}), 400

        if game.board.state[row, col] != SquareType.VALID:
            logging.warning(f"Nước đi của người dùng ({row},{col}) không phải là ô VALID trên server.")
            # Cập nhật lại game board cho client để hiển thị đúng các ô VALID
            game.update_valid_moves() # Đảm bảo client có trạng thái mới nhất
            session['game_instance'] = pickle.dumps(game)
            return jsonify({
                'message': 'Nước đi không hợp lệ. Hãy chọn một ô màu xám.',
                'error': True,
                'game_over': game.is_finished, # Gửi lại các thông tin cần thiết
                'black_score': game.black_score,
                'white_score': game.white_score,
                'next_player_is_ai': False, # Vì vẫn là lượt user
                'user_has_moves': game.is_valid_moves()
                }), 400

        game.next_move = (row, col)
        game.make_move()
        game.change_turn()
        game.update_valid_moves()
        game.update_scores()
        game.check_finished()

        temp_black_model = game.player_black.ml_model
        temp_white_model = game.player_white.ml_model
        game.player_black.ml_model = None
        game.player_white.ml_model = None

        session['game_instance'] = pickle.dumps(game)

        next_player_is_ai = game.active.player_type != PlayerType.USER if not game.is_finished else False

        response = {
            'message': 'User move received',
            'game_over': game.is_finished,
            'black_score': game.black_score,
            'white_score': game.white_score,
            'next_player_is_ai': next_player_is_ai,
            'user_has_moves': game.is_valid_moves() if next_player_is_ai else (game.is_valid_moves() if not game.is_finished else False)

        }

        return jsonify(response)
    else:
        return jsonify({'message': 'Game instance not found'})
        

@views.route('/agent_move', methods=['POST'])
def agent_move():
    serialized_game = session.get('game_instance')
    
    if serialized_game:
        game = pickle.loads(serialized_game)
        _attach_models_to_game_players(game)

        if game.active.player_type == PlayerType.USER:
            return jsonify({'message': 'Agent cannot move if it is user turn', 'error': True}), 400
            

        game.update_valid_moves()
        valid_moves_for_agent = game.is_valid_moves()

        agent_moved_this_turn = False
        game.get_player_move()

        if valid_moves_for_agent:
            game.get_player_move()
            game.make_move()
            agent_moved_this_turn = True
        
        game.change_turn()
        game.update_valid_moves()
        game.update_scores()
        game.check_finished()

        current_player_has_moves = game.is_valid_moves() if not game.is_finished else False
        current_player_is_ai = game.active.player_type != PlayerType.USER if not game.is_finished else False

        temp_black_model = game.player_black.ml_model
        temp_white_model = game.player_white.ml_model
        game.player_black.ml_model = None
        game.player_white.ml_model = None
        session['game_instance'] = pickle.dumps(game)

        response = {
            'message': 'Agent move processed' if agent_moved_this_turn else 'No valid move for agent, turn passed',
            'game_over': game.is_finished,
            'agent_moved_this_turn': agent_moved_this_turn,
            'current_player_has_moves': current_player_has_moves,
            'current_player_is_ai': current_player_is_ai,
            'black_score': game.black_score,
            'white_score': game.white_score
        }
        return jsonify(response)
    else:
        return jsonify({'message': 'Game instance not found'})
    

@views.route('/get_game_state', methods=['GET'])
def get_game_state():
    serialized_game = session.get('game_instance')
    game = None
    if serialized_game:
        try:
            game = pickle.loads(serialized_game)
            # Không cần _attach_models_to_game_players ở đây vì chỉ lấy state để hiển thị
            # Và game.active.player_type.value vẫn truy cập được mà không cần model
        except Exception as e:
            logging.error(f"Lỗi get_game_state unpickle: {e}")
            # Trả về trạng thái mặc định nếu không unpickle được
            return jsonify({
                'game_state': Board().state.tolist(), # Bảng cờ mặc định
                'black_score': 2,
                'white_score': 2,
                'active_player_color': SquareType.BLACK.name,
                'active_player_type': PlayerType.USER.value, # Mặc định
                'is_finished': False,
                'message': 'Game instance error, showing default.'
            })

        game_state_serializable = [[cell.name for cell in row] for row in game.board.state]
        response = {
            'game_state': game_state_serializable,
            'black_score': game.black_score,
            'white_score': game.white_score,
            'active_player_color': game.active.disc_color.name if not game.is_finished else None,
            'active_player_type': game.active.player_type.value if not game.is_finished else None, # Lấy giá trị của Enum
            'is_finished': game.is_finished
        }
        return jsonify(response)
    else:
        # Trả về trạng thái khởi tạo nếu không có game trong session
        default_board = Board()
        game_state_serializable = [[cell.name for cell in row] for row in default_board.state]
        return jsonify({
            'game_state': game_state_serializable,
            'black_score': 2,
            'white_score': 2,
            'active_player_color': SquareType.BLACK.name, # Giả sử Black bắt đầu
            'active_player_type': PlayerType.USER.value, # Hoặc một giá trị mặc định khác
            'is_finished': False,
            'message': 'Game instance not found, showing default state.'
        })

@views.route('/get_game_outcome')
def get_game_outcome():
    serialized_game = session.get('game_instance')

    if serialized_game:
        game = pickle.loads(serialized_game)
        game.determine_winner()

        outcome_message = f"Game over. {game.game_result}. Score: Black {game.black_score} - White {game.white_score}"
        return jsonify({
            "outcome_message": outcome_message,
            "black_score": game.black_score,
            "white_score": game.white_score
            })
    
    else:
        return jsonify({"outcome_message": "Game instance not found"})
    

@views.route('/reset_game', methods=['POST'])
def reset_game():
    session.pop('user_color', None)
    session.pop('game_instance', None)
    session.pop('game_started', None)
    session.pop('player_black_type', None)
    session.pop('player_white_type', None)
    return jsonify({'message': 'Game reset'})