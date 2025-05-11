from flask import (
    Blueprint, render_template, request, 
    jsonify, session, redirect, url_for
)
import pickle
import logging 
import numpy as np
logging.basicConfig(level=logging.DEBUG)

from src.gameLogic.game import Game
from src.gameLogic.board import SquareType 
from src.gameLogic.player import Player, PlayerType 
from src.ai.minimax.evaluator import StateEvaluator, HeuristicType
 
views = Blueprint("views", __name__)


def create_player_from_type_str(type_str, color_enum, state_evaluator):
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
        
        serialized_game = pickle.dumps(game)
        session['game_instance'] = serialized_game
        session['game_started'] = True
        
        return redirect(url_for('views.play_game'))
    
    else:
        serialized_game = session.get('game_instance')
        
        if serialized_game:
            game = pickle.loads(serialized_game)
        else:
            user_player_default = Player(PlayerType.USER, SquareType.BLACK)
            ai_player_default = Player(PlayerType.RANDOM, SquareType.WHITE)
            game = Game(user_player_default, ai_player_default)
        
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

        if game.active.player_type != PlayerType.USER:
             return jsonify({'message': 'Not user turn on server', 'error': True}), 400


        game.next_move = (row, col)
        game.make_move()
        game.change_turn()
        game.update_valid_moves()
        game.update_scores()
        game.check_finished()

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

        if game.active.player_type == PlayerType.USER:
            return jsonify({'message': 'Agent cannot move if it is user turn', 'error': True}), 400
            

        game.update_valid_moves()
        valid_moves_for_agent = game.is_valid_moves()

        agent_moved_this_turn = False
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
    
    if serialized_game:
        game = pickle.loads(serialized_game)
        game_state = [[cell.name for cell in row] for row in game.board.state]
        response = {
            'game_state': game_state,
            'black_score': game.black_score,
            'white_score': game.white_score,
            'active_player_color': game.active.disc_color.name if not game.is_finished else None,
            'active_player_type': game.active.player_type.value if not game.is_finished else None,
            'is_finished': game.is_finished
        }
        
        return jsonify(response)
    
    else:
        return jsonify({'message': 'Game instance not found'})

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