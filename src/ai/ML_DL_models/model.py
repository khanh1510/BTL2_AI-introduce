import os
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch



class Model:
    def __init__(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        lstm_model_path = os.path.join(base_dir, 'built_lstm.keras')
        self.lstm = None
        try:
            if os.path.exists(lstm_model_path):
                self.lstm = keras.models.load_model(lstm_model_path)
                print(f"Model LSTM đã được tải thành công từ: {lstm_model_path}")
            else:
                print(f"LỖI: Không tìm thấy file model LSTM tại: {lstm_model_path}")
                # raise FileNotFoundError(f"Không tìm thấy file model LSTM tại: {lstm_model_path}")
        except Exception as e:
            print(f"Lỗi khi tải model LSTM: {e}")

        # self.tokenizer = AutoTokenizer.from_pretrained('./finetuned_gpt2')
        # self.gpt2 = AutoModelForCausalLM.from_pretrained('./finetuned_gpt2')
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        # self.gpt2 = self.gpt2.to(self.device)


    def _index_token(self, move):
        # ... (code của bạn cho _index_token) ...
        if not move or len(move) != 2:
            return -1
        col_char = move[0].lower()
        row_char = move[1]
        if not ('a' <= col_char <= 'h' and '1' <= row_char <= '8'):
            return -1
        col = ord(col_char) - ord('a')
        row = int(row_char) - 1
        return row * 8 + col


    # def infer_lstm(self, str):
    #     input = [str[i:i+2] for i in range(0, len(str), 2)]
    #     input = [self.index_token(move) for move in input]
    #     input = pad_sequences(input, maxlen=60, value=-1, padding='post')

    #     output = np.argmax(self.lstm.predict(input))

    #     if output < 0 or output >= 64:
    #         return ""
    #     row = output // 8
    #     col = output % 8
    #     return chr(col + ord('a')) + str(row + 1)

    def _deindex_token(self, index_val): # Kiểm tra tên hàm này
        if index_val < 0 or index_val >= 64:
            return ""
        row = index_val // 8
        col = index_val % 8
        return chr(col + ord('a')) + str(row + 1)

    def infer_lstm(self, game_moves_str):
        if self.lstm is None:
            print("Lỗi: Model LSTM chưa được tải.")
            return ""

        current_sequence_indices = []
        if game_moves_str:
            for i in range(0, len(game_moves_str), 2):
                move_token = game_moves_str[i:i+2]
                indexed_token = self._index_token(move_token) # Gọi _index_token
                if indexed_token != -1:
                    current_sequence_indices.append(indexed_token)
        
        padded_sequence = pad_sequences([current_sequence_indices], maxlen=60, value=-1, padding='post', truncating='post')
        prediction_probs = self.lstm.predict(padded_sequence, verbose=0)
        predicted_index = np.argmax(prediction_probs[0])

        return self._deindex_token(predicted_index) # Gọi _deindex_token

    # def infer_gpt2(self, str):
    #     prompt = [str[i:i+2] for i in range(0, len(str), 2)]
    #     prompt = f"Played moves: {' '.join(prompt)}\nBest next move:"
    #     input = self.tokenizer(prompt, return_tensor='pt').to(self.gpt2.devide)

    #     output = self.gpt2.generate(
    #         **input,
    #         max_new_tokens=5,
    #         pad_token_id=self.tokenizer.eos_token_id,
    #         eos_token_id=self.tokenizer.eos_token_id
    #     )
    #     gen_text = self.tokenizer.decode(output[0])

    #     try:
    #         move_part = gen_text.split("Best Next move:")[1].strip()
    #         predicted_move = move_part.split()[0]
    #     except IndexError:
    #         predicted_move = ""

    #     return predicted_move