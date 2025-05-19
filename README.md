# OthelloAI - Game Othello với Trí Tuệ Nhân Tạo

Chào mừng bạn đến với OthelloAI, một dự án game Othello (còn gọi là Reversi) được xây dựng bằng Python với giao diện web sử dụng Flask và các AI đối thủ đa dạng.

## Giới thiệu

Othello là một trò chơi chiến thuật trên bàn cờ dành cho hai người chơi. Mục tiêu của trò chơi là có nhiều quân cờ màu của mình hơn đối thủ khi bàn cờ đã đầy hoặc khi không còn nước đi hợp lệ nào cho cả hai người chơi.

Dự án này cung cấp một nền tảng để chơi Othello với các tùy chọn người chơi đa dạng, bao gồm:
*   Người chơi vs Người chơi (trên cùng một máy)
*   Người chơi vs AI (với các độ khó và loại AI khác nhau)
*   AI vs AI (để quan sát và phân tích)

## Tính năng chính

*   **Logic game Othello hoàn chỉnh:** Triển khai đầy đủ các quy tắc của trò chơi Othello.
*   **Giao diện web tương tác:** Giao diện người dùng trực quan được xây dựng bằng Flask, HTML, CSS và JavaScript.
*   **AI đối thủ thông minh và đa dạng:**
    *   **Random Agent:** AI chơi ngẫu nhiên, phù hợp cho người mới bắt đầu.
    *   **Minimax Agent:** AI sử dụng thuật toán Minimax với cắt tỉa Alpha-Beta và hàm lượng giá dựa trên heuristic (có thể cấu hình độ sâu tìm kiếm từ 1 đến 4).
    *   **LSTM Model:** AI dựa trên mạng Long Short-Term Memory để dự đoán nước đi tiếp theo.
    *   **GPT-2 Model:** AI sử dụng mô hình ngôn ngữ lớn GPT-2 đã được finetune để sinh nước đi.
    *   **Random Forest Model:** AI dựa trên mô hình Random Forest để dự đoán kết quả trận đấu và chọn nước đi.
    *   **XGBoost Model:** AI dựa trên mô hình XGBoost để dự đoán kết quả trận đấu và chọn nước đi.
*   **Lựa chọn người chơi linh hoạt:** Cho phép người dùng chọn loại người chơi cho cả quân Đen và quân Trắng.
*   **Hiển thị nước đi hợp lệ.**
*   **Cập nhật điểm số trực tiếp.**
*   **Khả năng chơi lại (Reset Game).**
*   **Chơi trên console:** Script `play_othello_console.py` để thử nghiệm logic game và AI.
*   **Phân tích AI:** Jupyter Notebooks (`Depth_MoveTime_Analysis.ipynb`, `Heuristic_Analysis.ipynb`, `modelMLDL.ipynb`) và script (`MLmodel.py`, `grand_tournament_runner.py`) để huấn luyện, phân tích thời gian, heuristic và hiệu suất các AI.

## Cấu trúc thư mục

Dự án được tổ chức như sau:
```bash
BTL2_AI-introduce/
├── src/ # Mã nguồn chính của game và AI
│ ├── ai/ # Logic của các tác tử AI (Minimax, Random)
│ ├── common/ # Các enums và hằng số dùng chung
│ ├── gameLogic/ # Logic cốt lõi của game Othello (bàn cờ, người chơi, luật chơi)
│ └── experiments/ # Các Jupyter Notebooks và dữ liệu phân tích AI
├── website/ # Mã nguồn cho giao diện web Flask
│ ├── static/ # Các file tĩnh (CSS, JavaScript)
│ └── templates/ # Các template HTML
├── main.py # Điểm khởi chạy ứng dụng web
├── play_othello_console.py # Script để chơi game trên console
└── requirements.txt # Danh sách các thư viện Python cần thiết
```

## Hướng dẫn cài đặt

### Yêu cầu

*   Python 3.12 trở lên
*   pip (Python package installer)

### Các bước cài đặt

1.  **Clone repository:**
    ```bash
    git clone https://github.com/khanh1510/BTL2_AI-introduce.git
    cd BTL2_AI-introduce
    ```

2.  **Tạo và kích hoạt môi trường ảo (khuyến khích):**
    ```bash
    python -m venv venv
    # Trên Windows
    venv\Scripts\activate
    # Trên macOS/Linux
    source venv/bin/activate
    ```

3.  **Cài đặt các thư viện cần thiết:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Lưu ý: Bạn cần tạo file `requirements.txt` chứa các thư viện như Flask, NumPy. Ví dụ: `Flask==2.x.x`, `numpy==1.x.x`)*

4.  **Chuẩn bị Dữ liệu và Model:**

    4.1 ***Tải Dữ liệu Huấn luyện***

    - Truy cập thư mục Google Drive:  
    👉 [Link Drive Dữ liệu](https://drive.google.com/drive/folders/1aBPETv39HpvDx3p1pVLDe-Ms5c4xGE_u?usp=sharing)

    - Tải xuống hai file:
    - `othello_dataset.csv` (dùng cho mô hình LSTM/GPT-2)
    - `othello_state_dataset.csv` (dùng cho mô hình Random Forest/XGBoost)

    - Đặt cả hai file vào thư mục:  
    `src/experiments/data/`  
    *(Nếu thư mục `data/` chưa tồn tại, bạn cần tạo thủ công.)*

    4.2 ***Chuẩn bị Model GPT-2 đã Finetune***

    - Tải file nén GPT-2 đã huấn luyện:  
    👉 [GPT-2 Checkpoint](https://drive.google.com/file/d/1vbxDIM0UhccnHi3saSyB_XcM5rpj-_er/view?usp=sharing)

    - Giải nén file `gpt2-othello-checkpoint-500.zip` → được thư mục `checkpoint-500`.

    - Di chuyển thư mục này vào:

    - Sau bước này, bạn sẽ có cấu trúc: src/ai/ML_DL_models/finetuned_gpt2/ (chứa các file model của GPT-2).
    
    4.3 ***Huấn luyện Model Random Forest & XGBoost***

    - Mở terminal (hoặc command prompt) tại thư mục gốc dự án (`BTL2_AI-introduce`).

    - Kích hoạt môi trường ảo (nếu có):
    ```bash
    # Trên Windows
    venv\Scripts\activate

    # Trên macOS/Linux
    source venv/bin/activate
    ```
    - Chạy script MLmodel.py: python src/experiments/MLmodel.py
            
    - Script này sẽ đọc othello_state_dataset.csv, huấn luyện model RF và XGBoost, sau đó tự động lưu rf_othello_classifier.pkl và xgb_othello_classifier.pkl vào thư mục src/ai/ML_DL_models/.

## Hướng dẫn sử dụng

### Chạy ứng dụng Web

1.  Đảm bảo bạn đang ở thư mục gốc của dự án (`Othello-main`).
2.  Chạy file `main.py`:
    ```bash
    python main.py
    ```
3.  Mở trình duyệt web và truy cập vào địa chỉ: `http://127.0.0.1:5000/`
4.  Trên trang chủ, nhấp vào "Play OthelloAI".
5.  Chọn loại người chơi cho quân Đen và quân Trắng, sau đó nhấn "Start Game".

### Chơi trên Console (Tùy chọn)

Nếu bạn muốn thử nghiệm logic game hoặc AI mà không cần giao diện web:
```bash
python play_othello_console.py