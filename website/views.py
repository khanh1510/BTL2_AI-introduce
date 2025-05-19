
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
            logging.info("VIEWS: Đang tải ML model cho ứng dụng web...")
            _ml_model_global_instance = Model()
        except Exception as e:
            logging.error(f"VIEWS: LỖI WEB: Không thể khởi tạo ML Model: {e}")
    return _ml_model_global_instance

def _attach_models_to_game_players(game_instance):
    if not game_instance: return
    ml_model = get_global_ml_model()
    if ml_model:
        player_types_needing_model = [PlayerType.LSTM, PlayerType.GPT2, PlayerType.RF_MODEL, PlayerType.XGB_MODEL]
        if game_instance.player_black.player_type in player_types_needing_model:
            game_instance.player_black.ml_model = ml_model
        if game_instance.player_white.player_type in player_types_needing_model:
            game_instance.player_white.ml_model = ml_model

def create_player_from_type_str(type_str, color_enum, state_evaluator):
    model_instance = get_global_ml_model()
    if type_str == "USER":
        return Player(PlayerType.USER, color_enum)
    elif type_str == "RANDOM":
        return Player(PlayerType.RANDOM, color_enum)
    elif type_str.startswith("MINIMAX_"):
        try:
            depth = int(type_str.split('_')[1])
            return Player(PlayerType.MINIMAX, color_enum, state_evaluator, depth)
        except (IndexError, ValueError):
            return Player(PlayerType.RANDOM, color_enum)
    elif type_str == "LSTM":
        if model_instance and model_instance.lstm:
            return Player(PlayerType.LSTM, color_enum, ml_model_instance=model_instance)
        return Player(PlayerType.RANDOM, color_enum)
    elif type_str == "GPT2":
        if model_instance and model_instance.gpt2 and model_instance.tokenizer:
            return Player(PlayerType.GPT2, color_enum, ml_model_instance=model_instance)
        return Player(PlayerType.RANDOM, color_enum)
    elif type_str == "RF_MODEL": 
        if model_instance and model_instance.rf_model:
            return Player(PlayerType.RF_MODEL, color_enum, ml_model_instance=model_instance)
        logging.warning("VIEWS: Model Random Forest không khả dụng, dùng Random.")
        return Player(PlayerType.RANDOM, color_enum)
    elif type_str == "XGB_MODEL": 
        if model_instance and model_instance.xgb_model:
            return Player(PlayerType.XGB_MODEL, color_enum, ml_model_instance=model_instance)
        logging.warning("VIEWS: Model XGBoost không khả dụng, dùng Random.")
        return Player(PlayerType.RANDOM, color_enum)

    logging.warning(f"VIEWS: Loại người chơi không xác định '{type_str}', dùng Random.")
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
        session.pop('game_instance', None)



        state_eval = StateEvaluator(weights={
            HeuristicType.DISC_DIFF: 5/60,
            HeuristicType.MOBILITY: 15/60,
            HeuristicType.CORNERS: 40/60
        })

        player_black = create_player_from_type_str(player_black_type_str, SquareType.BLACK, state_eval)
        player_white = create_player_from_type_str(player_white_type_str, SquareType.WHITE, state_eval)

        game = Game(player_black, player_white)
        game.move_history_str = ""

        temp_black_model_ref = game.player_black.ml_model
        temp_white_model_ref = game.player_white.ml_model
        game.player_black.ml_model = None
        game.player_white.ml_model = None

        serialized_game = pickle.dumps(game)
        session['game_instance'] = serialized_game
        session['game_started'] = True

        game.player_black.ml_model = temp_black_model_ref
        game.player_white.ml_model = temp_white_model_ref

        return redirect(url_for('views.play_game'))
    else:
        serialized_game = session.get('game_instance')
        game = None

        if serialized_game:
            try:
                game = pickle.loads(serialized_game)
                _attach_models_to_game_players(game)
            except Exception as e:
                logging.error(f"Lỗi khi unpickle game từ session: {e}")
                session.pop('game_instance', None)
                session.pop('game_started', None)

        if game is None:
            if not session.get('game_started'):
                logging.info("VIEWS: Không có game trong session hoặc chưa bắt đầu, chờ người dùng chọn player.")

            else:
                logging.error("VIEWS: Game đã bắt đầu nhưng session game_instance bị lỗi. Resetting.")
                session.pop('game_instance', None)
                session.pop('game_started', None)

                return redirect(url_for('views.play_game'))


        return render_template('play_game.html',
                               game=game,
                               game_started=session.get('game_started', False),
                               black_score=game.black_score if game else 2,
                               white_score=game.white_score if game else 2)


@views.route('/user_move', methods=['POST'])
def user_move():
    data = request.get_json()
    row, col = data['row'], data['col']
    serialized_game = session.get('game_instance')
    if not serialized_game:
        return jsonify({'message': 'Game instance not found', 'error': True}), 400
    
    game = pickle.loads(serialized_game)
    _attach_models_to_game_players(game)

    if game.active.player_type != PlayerType.USER:
        return jsonify({'message': 'Not user turn on server', 'error': True}), 400

    game.update_valid_moves() 

    if game.board.state[row, col] != SquareType.VALID:
        logging.warning(f"Nước đi của người dùng ({row},{col}) không phải là ô VALID trên server.")
        return jsonify({
            'message': 'Nước đi không hợp lệ. Hãy chọn một ô màu xám.',
            'error': True,
            'game_over': game.is_finished,
            'black_score': game.black_score,
            'white_score': game.white_score,
            'next_player_is_ai': game.active.player_type != PlayerType.USER if not game.is_finished else False,
            'user_has_moves': game.is_any_valid_move_on_board() if not game.is_finished else False 
            }), 400

    game.next_move = (row, col)
    game.make_move()
    game.change_turn()
    game.update_valid_moves() 
    game.update_scores()
    game.check_finished()

    temp_black_model_ref = game.player_black.ml_model
    temp_white_model_ref = game.player_white.ml_model
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
        'user_has_moves': game.is_any_valid_move_on_board() if not game.is_finished else False
    }
    return jsonify(response)


@views.route('/agent_move', methods=['POST'])
def agent_move():
    serialized_game = session.get('game_instance')
    if not serialized_game:
        return jsonify({'message': 'Game instance not found', 'error': True}), 400

    game = pickle.loads(serialized_game)
    _attach_models_to_game_players(game)

    if game.active.player_type == PlayerType.USER:
        return jsonify({'message': 'Agent cannot move if it is user turn', 'error': True}), 400

    agent_moved_this_turn = False
    game.get_player_move() 

    if game.next_move is not None: 
        game.make_move()
        agent_moved_this_turn = True
    
    game.change_turn()
    game.update_valid_moves()
    game.update_scores()
    game.check_finished()

    current_player_has_moves = game.is_any_valid_move_on_board() if not game.is_finished else False
    current_player_is_ai = game.active.player_type != PlayerType.USER if not game.is_finished else False

    temp_black_model_ref = game.player_black.ml_model
    temp_white_model_ref = game.player_white.ml_model
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



@views.route('/get_game_state', methods=['GET'])
def get_game_state():
    serialized_game = session.get('game_instance')
    game = None
    if serialized_game:
        try:
            game = pickle.loads(serialized_game)

        except Exception as e:
            logging.error(f"Lỗi get_game_state unpickle: {e}")
            default_board = Board()
            return jsonify({
                'game_state': [[cell.name for cell in row] for row in default_board.state],
                'black_score': 2, 'white_score': 2,
                'active_player_color': SquareType.BLACK.name,
                'active_player_type': PlayerType.USER.value,
                'is_finished': False,
                'message': 'Game instance error, showing default.'
            })

        game_state_serializable = [[cell.name for cell in row] for row in game.board.state]
        active_player_type_value = None
        if not game.is_finished and game.active:
            active_player_type_value = game.active.player_type.value

        response = {
            'game_state': game_state_serializable,
            'black_score': game.black_score,
            'white_score': game.white_score,
            'active_player_color': game.active.disc_color.name if not game.is_finished and game.active else None,
            'active_player_type': active_player_type_value,
            'is_finished': game.is_finished
        }
        return jsonify(response)
    else:
        default_board = Board()
        return jsonify({
            'game_state': [[cell.name for cell in row] for row in default_board.state],
            'black_score': 2, 'white_score': 2,
            'active_player_color': SquareType.BLACK.name,
            'active_player_type': PlayerType.USER.value,
            'is_finished': False,
            'message': 'Game instance not found, showing default state.'
        })

@views.route('/get_game_outcome')
def get_game_outcome():
    serialized_game = session.get('game_instance')
    if serialized_game:
        game = pickle.loads(serialized_game)
        if not game.game_result and game.is_finished :
            game.determine_winner()
        outcome_message = f"Game over. {game.game_result if game.game_result else 'Result pending'}. Score: Black {game.black_score} - White {game.white_score}"
        return jsonify({
            "outcome_message": outcome_message,
            "black_score": game.black_score,
            "white_score": game.white_score
            })
    return jsonify({"outcome_message": "Game instance not found"})

@views.route('/reset_game', methods=['POST'])
def reset_game():
    session.pop('game_instance', None)
    session.pop('game_started', None)
    session.pop('player_black_type', None)
    session.pop('player_white_type', None)
    return jsonify({'message': 'Game reset'})