# OthelloAI - Game Othello với Trí Tuệ Nhân Tạo

Chào mừng bạn đến với OthelloAI, một dự án game Othello (còn gọi là Reversi) được xây dựng bằng Python với giao diện web sử dụng Flask và một AI đối thủ dựa trên thuật toán Minimax.

## Giới thiệu

Othello là một trò chơi chiến thuật trên bàn cờ dành cho hai người chơi. Mục tiêu của trò chơi là có nhiều quân cờ màu của mình hơn đối thủ khi bàn cờ đã đầy hoặc khi không còn nước đi hợp lệ nào cho cả hai người chơi.

Dự án này cung cấp một nền tảng để chơi Othello với các tùy chọn người chơi đa dạng, bao gồm:
*   Người chơi vs Người chơi (trên cùng một máy)
*   Người chơi vs AI (với các độ khó khác nhau)
*   AI vs AI (để quan sát và phân tích)

## Tính năng chính

*   **Logic game Othello hoàn chỉnh:** Triển khai đầy đủ các quy tắc của trò chơi Othello.
*   **Giao diện web tương tác:** Giao diện người dùng trực quan được xây dựng bằng Flask, HTML, CSS và JavaScript.
*   **AI đối thủ thông minh:**
    *   **Random Agent:** AI chơi ngẫu nhiên, phù hợp cho người mới bắt đầu.
    *   **Minimax Agent:** AI sử dụng thuật toán Minimax với cắt tỉa Alpha-Beta và hàm lượng giá dựa trên heuristic.
    *   **Nhiều cấp độ khó:** AI Minimax có thể được cấu hình với các độ sâu tìm kiếm khác nhau (Dễ, Trung bình, Khó, Rất khó), tương ứng với depth 1, 2, 3 và 4.
*   **Lựa chọn người chơi linh hoạt:** Cho phép người dùng chọn loại người chơi (Người, AI các cấp độ) cho cả quân Đen và quân Trắng.
*   **Hiển thị nước đi hợp lệ:** Giúp người chơi dễ dàng nhận biết các nước đi có thể thực hiện.
*   **Cập nhật điểm số trực tiếp.**
*   **Khả năng chơi lại (Reset Game).**
*   **Chơi trên console (Tùy chọn):** Cung cấp script `play_othello_console.py` để thử nghiệm logic game và AI trong môi trường console.
*   **Phân tích AI:** Bao gồm các Jupyter Notebooks để phân tích thời gian thực thi của AI theo độ sâu và hiệu quả của các heuristic.

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