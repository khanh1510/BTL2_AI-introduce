import os
import numpy as np
import pandas as pd 
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import logging
import joblib
import copy
from ...gameLogic.board import SquareType
from ...constants import BOARD_SIZE, DIRECTIONS

class Model:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        lstm_model_path = os.path.join(base_dir, 'built_lstm.keras')
        gpt2_model_path = os.path.join(base_dir, 'finetuned_gpt2')
        rf_model_path = os.path.join(base_dir, 'rf_othello_classifier.pkl')
        xgb_model_path = os.path.join(base_dir, 'xgb_othello_classifier.pkl')

        self.lstm = None
        self.tokenizer = None
        self.gpt2 = None
        self.rf_model = None
        self.xgb_model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        try:
            if os.path.exists(lstm_model_path): self.lstm = keras.models.load_model(lstm_model_path)
        except Exception as e: logging.error(f"Lỗi tải LSTM: {e}")
        try:
            if os.path.isdir(gpt2_model_path):
                self.tokenizer = AutoTokenizer.from_pretrained(gpt2_model_path)
                self.gpt2 = AutoModelForCausalLM.from_pretrained(gpt2_model_path).to(self.device)
        except Exception as e: logging.error(f"Lỗi tải GPT-2: {e}")
        try:
            if os.path.exists(rf_model_path): self.rf_model = joblib.load(rf_model_path)
        except Exception as e: logging.error(f"Lỗi tải RF: {e}")
        try:
            if os.path.exists(xgb_model_path): self.xgb_model = joblib.load(xgb_model_path)
        except Exception as e: logging.error(f"Lỗi tải XGB: {e}")

    def _index_token(self, move):
        if not move or len(move) != 2: return -1
        col_char = move[0].lower(); row_char = move[1]
        if not ('a' <= col_char <= 'h' and '1' <= row_char <= '8'): return -1
        col = ord(col_char) - ord('a'); row = int(row_char) - 1
        return row * 8 + col

    def _deindex_token(self, index_val):
        if index_val < 0 or index_val >= 64: return ""
        row = index_val // 8; col = index_val % 8
        return chr(col + ord('a')) + str(row + 1)

    def infer_lstm(self, game_moves_str):
        if self.lstm is None: return ""
        current_sequence_indices = []
        if game_moves_str:
            for i in range(0, len(game_moves_str), 2):
                move_token = game_moves_str[i:i+2]
                indexed_token = self._index_token(move_token)
                if indexed_token != -1: current_sequence_indices.append(indexed_token)
        padded_sequence = pad_sequences([current_sequence_indices], maxlen=60, value=-1, padding='post', truncating='post')
        prediction_probs = self.lstm.predict(padded_sequence, verbose=0)
        predicted_index = np.argmax(prediction_probs[0])
        return self._deindex_token(predicted_index)

    def infer_gpt2(self, game_moves_str):
        if self.gpt2 is None or self.tokenizer is None: return ""
        prompt_moves = [game_moves_str[i:i+2] for i in range(0, len(game_moves_str), 2)]
        prompt = f"Played moves: {' '.join(prompt_moves)}\nBest Next move:"
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        outputs_ids = self.gpt2.generate(
            inputs['input_ids'], attention_mask=inputs['attention_mask'],
            max_new_tokens=5, pad_token_id=self.tokenizer.eos_token_id,
            eos_token_id=self.tokenizer.eos_token_id, do_sample=False
        )
        generated_tokens = outputs_ids[0][inputs['input_ids'].shape[-1]:]
        gen_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        predicted_move = gen_text.strip().split()[0] if gen_text.strip() else ""
        if len(predicted_move) == 2 and 'a' <= predicted_move[0].lower() <= 'h' and '1' <= predicted_move[1] <= '8':
            return predicted_move.lower()
        return ""

    def _convert_board_to_numeric(self, board_state_enum, current_player_color):
        numeric_board = np.zeros(board_state_enum.shape, dtype=int)
        opponent_color = SquareType.WHITE if current_player_color == SquareType.BLACK else SquareType.BLACK
        for r in range(board_state_enum.shape[0]):
            for c in range(board_state_enum.shape[1]):
                if board_state_enum[r, c] == current_player_color:
                    numeric_board[r, c] = 1
                elif board_state_enum[r, c] == opponent_color:
                    numeric_board[r, c] = -1
                else:
                    numeric_board[r, c] = 0
        return numeric_board.flatten().reshape(1, -1)

    def _get_best_move_from_prediction(self, current_game, model, model_name_for_log="Model"):
        active_player_color = current_game.active.disc_color
        original_valid_moves = current_game.get_valid_moves_by_color(active_player_color)

        if not original_valid_moves:
            return None

        best_move_candidate = None
        priority_map = {1: 0, 0: 1, -1: 2} if active_player_color == SquareType.BLACK else {-1: 0, 0: 1, 1: 2}
        current_best_priority = 3

        feature_names_64 = [f"{r}{c}" for r in range(8) for c in range(8)]

        for move in original_valid_moves:
            temp_board_state_after_move = copy.deepcopy(current_game.board.state)

            temp_board_state_after_move[move[0], move[1]] = active_player_color


            class MiniGameForFlip:
                def __init__(self, board_state, active_color, inactive_color):
                    self.board = lambda: None
                    setattr(self.board, 'state', board_state)
                    self.active = lambda: None
                    setattr(self.active, 'disc_color', active_color)
                    self.inactive = lambda: None
                    setattr(self.inactive, 'disc_color', inactive_color)

                def discs_to_flip(self, r_move, c_move):
                    discs_to_flip_list = []
                    for direction in DIRECTIONS:
                        dr, dc = direction
                        r, c = r_move + dr, c_move + dc
                        seq_discs = []
                        found_opposing = False
                        while 0 <= r < BOARD_SIZE and 0 <= c < BOARD_SIZE:
                            if self.board.state[r,c] == self.inactive.disc_color:
                                seq_discs.append((r,c))
                                found_opposing = True
                            elif self.board.state[r,c] == self.active.disc_color:
                                if found_opposing:
                                    discs_to_flip_list.extend(seq_discs)
                                break
                            else:
                                break
                            r += dr
                            c += dc
                    return discs_to_flip_list

            inactive_color = SquareType.WHITE if active_player_color == SquareType.BLACK else SquareType.BLACK
            mini_game_flipper = MiniGameForFlip(temp_board_state_after_move, active_player_color, inactive_color)
            discs_to_change = mini_game_flipper.discs_to_flip(move[0], move[1])
            
            for r_f, c_f in discs_to_change:
                temp_board_state_after_move[r_f, c_f] = active_player_color
            
            numeric_board_features = self._convert_board_to_numeric(temp_board_state_after_move, active_player_color)
            
            if numeric_board_features.shape[1] != 64:
                logging.error(f"{model_name_for_log}: _convert_board_to_numeric không trả về 64 features.")
                continue

            input_df_for_model = pd.DataFrame(numeric_board_features, columns=feature_names_64)
            
            try:
                prediction_result = model.predict(input_df_for_model)[0]
            except Exception as e:
                logging.error(f"Lỗi khi {model_name_for_log}.predict: {e}. Input shape: {input_df_for_model.shape}")
                continue

            if prediction_result not in priority_map:
                continue 

            predicted_priority = priority_map[prediction_result]

            if predicted_priority < current_best_priority:
                current_best_priority = predicted_priority
                best_move_candidate = move
            elif predicted_priority == current_best_priority and best_move_candidate is None:
                 best_move_candidate = move

        if not best_move_candidate and original_valid_moves:
            best_move_candidate = original_valid_moves[np.random.choice(len(original_valid_moves))]
        
        return best_move_candidate

    def infer_rf(self, current_game):
        if self.rf_model is None: return None
        return self._get_best_move_from_prediction(current_game, self.rf_model, "Random Forest")

    def infer_xgb(self, current_game):
        if self.xgb_model is None: return None
        
        class XGBWrapper:
            def __init__(self, xgb_model_instance):
                self.xgb_model = xgb_model_instance
                self.class_mapping_to_original = {0: -1, 1: 0, 2: 1}
            def predict(self, X):
                xgb_preds = self.xgb_model.predict(X)
                return np.array([self.class_mapping_to_original.get(p, 0) for p in xgb_preds])

        
        xgb_wrapper = XGBWrapper(self.xgb_model)
        return self._get_best_move_from_prediction(current_game, xgb_wrapper, "XGBoost")