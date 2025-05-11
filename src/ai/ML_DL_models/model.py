# src/ai/ML_DL_models/model.py
import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer, AutoModelForCausalLM # Bỏ comment
import torch # Bỏ comment
import logging # Thêm để log lỗi tốt hơn

class Model:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        lstm_model_path = os.path.join(base_dir, 'built_lstm.keras')
        gpt2_model_path = os.path.join(base_dir, 'finetuned_gpt2') # Đường dẫn đến thư mục model GPT-2

        self.lstm = None
        self.tokenizer = None
        self.gpt2 = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logging.info(f"Sử dụng device: {self.device} cho model ML/DL.")

        try:
            if os.path.exists(lstm_model_path):
                self.lstm = keras.models.load_model(lstm_model_path)
                logging.info(f"Model LSTM đã được tải thành công từ: {lstm_model_path}")
            else:
                logging.warning(f"LƯU Ý: Không tìm thấy file model LSTM tại: {lstm_model_path}")
        except Exception as e:
            logging.error(f"Lỗi khi tải model LSTM: {e}")

        try:
            if os.path.isdir(gpt2_model_path): # Kiểm tra nếu là thư mục
                self.tokenizer = AutoTokenizer.from_pretrained(gpt2_model_path)
                self.gpt2 = AutoModelForCausalLM.from_pretrained(gpt2_model_path)
                self.gpt2 = self.gpt2.to(self.device)
                logging.info(f"Model GPT-2 và Tokenizer đã được tải thành công từ: {gpt2_model_path}")
            else:
                logging.warning(f"LƯU Ý: Không tìm thấy thư mục model GPT-2 tại: {gpt2_model_path}")
        except Exception as e:
            logging.error(f"Lỗi khi tải model GPT-2: {e}")

    def _index_token(self, move):
        # ... (giữ nguyên)
        if not move or len(move) != 2: return -1
        col_char = move[0].lower(); row_char = move[1]
        if not ('a' <= col_char <= 'h' and '1' <= row_char <= '8'): return -1
        col = ord(col_char) - ord('a'); row = int(row_char) - 1
        return row * 8 + col

    def _deindex_token(self, index_val):
        # ... (giữ nguyên)
        if index_val < 0 or index_val >= 64: return ""
        row = index_val // 8; col = index_val % 8
        return chr(col + ord('a')) + str(row + 1)

    def infer_lstm(self, game_moves_str):
        # ... (giữ nguyên như phiên bản đã sửa ở trên)
        if self.lstm is None:
            logging.error("Model LSTM chưa được tải, không thể thực hiện infer_lstm.")
            return ""
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

    def infer_gpt2(self, game_moves_str): # Sửa tên tham số để nhất quán
        if self.gpt2 is None or self.tokenizer is None:
            logging.error("Model GPT-2 hoặc Tokenizer chưa được tải, không thể thực hiện infer_gpt2.")
            return ""

        prompt_moves = [game_moves_str[i:i+2] for i in range(0, len(game_moves_str), 2)]
        prompt = f"Played moves: {' '.join(prompt_moves)}\nBest Next move:"
        logging.debug(f"GPT-2 prompt: \"{prompt}\"")

        try:
            inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device) # Sửa lỗi device

            outputs_ids = self.gpt2.generate(
                inputs['input_ids'], # Chỉ truyền input_ids
                attention_mask=inputs['attention_mask'], # Truyền attention_mask
                max_new_tokens=5,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                do_sample=False # Tắt sampling để kết quả nhất quán hơn cho debug
            )
            # Lấy phần token được generate (bỏ qua phần prompt)
            generated_tokens = outputs_ids[0][inputs['input_ids'].shape[-1]:]
            gen_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            logging.debug(f"GPT-2 raw generated text (new tokens only): '{gen_text}'")

            predicted_move = gen_text.strip().split()[0] if gen_text.strip() else ""
            logging.debug(f"GPT-2 predicted move (after split): '{predicted_move}'")

            if len(predicted_move) == 2 and 'a' <= predicted_move[0].lower() <= 'h' and '1' <= predicted_move[1] <= '8':
                return predicted_move.lower() # Chuẩn hóa về chữ thường
            else:
                logging.warning(f"GPT-2 trả về nước đi không hợp lệ: '{predicted_move}' từ text: '{gen_text}'")
                return ""
        except Exception as e:
            logging.error(f"Lỗi trong quá trình GPT-2 inference: {e}")
            return ""