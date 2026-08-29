# KẾ HOẠCH CHI TIẾT - THÀNH VIÊN 1: HOÀNG HÔN (TRƯỞNG NHÓM)
**Đề tài:** Phân tích cảm xúc đánh giá ITviec  
**Môn học:** Máy học (Machine Learning)  
**Phân công:** `Business Understanding, Data Processing & Report Management`  
**Thời gian thực hiện:** 3 Ngày cốt lõi (Giai đoạn 1) & Quản lý điều phối suốt dự án  

---

## 📌 I. DANH SÁCH NHIỆM VỤ CHI TIẾT (CHECKLIST)

### 🟢 Giai đoạn 1: Thiết lập & Tiền xử lý dữ liệu (Data Preprocessing) — ✅ HOÀN THÀNH (28/08/2026)
- [x] **Xác định bài toán & Mục tiêu:**
  - Bài toán: Phân loại văn bản đa lớp (Multi-class Sentiment Classification) với 3 lớp: `Positive` (4-5 sao), `Neutral` (3 sao), `Negative` (1-2 sao).
  - Xác định luồng pipeline Machine Learning chuẩn mực: Dữ liệu thô → Làm sạch → Tách từ → Trích xuất TF-IDF → Mô hình ML → Đánh giá.
- [x] **Thiết lập cấu trúc Repository & Thư viện:**
  - Chuẩn hóa môi trường Python, `requirements.txt`, 10 bộ từ điển tại `data/dictionaries/`.
- [x] **Xây dựng module tiền xử lý văn bản [src/preprocessing.py](file:///d:/Trí%20tuệ%20nhân%20tạo/HK2/Máy%20học/Project/Do_An_May_Hoc_Sentiment_Analysis/src/preprocessing.py):**
  - Chuẩn hóa Unicode NFC (`unicodedata.normalize`).
  - Ánh xạ biểu tượng cảm xúc (Emoji/Emojicon) sang từ ngữ mang cảm xúc.
  - Chuẩn hóa viết tắt (teencode), thuật ngữ IT và sửa lỗi chính tả (`teencode.txt`, `wrong-word.txt`, `english-vnmese.txt`).
  - Tách từ tiếng Việt bằng `pyvi.ViTokenizer` (lazy-load, fallback từ `underthesea`).
  - Lọc bỏ stopwords tiếng Việt ([vietnamese-stopwords.txt](file:///d:/Trí%20tuệ%20nhân%20tạo/HK2/Máy%20học/Project/Do_An_May_Hoc_Sentiment_Analysis/data/dictionaries/vietnamese-stopwords.txt)).
- [x] **Gán nhãn yếu (Weak Labeling) & Xuất dữ liệu sạch:**
  - Script: [scripts/run_step1_preprocessing.py](file:///d:/Trí%20tuệ%20nhân%20tạo/HK2/Máy%20học/Project/Do_An_May_Hoc_Sentiment_Analysis/scripts/run_step1_preprocessing.py).
  - Đã xuất `data/processed/reviews_cleaned.xlsx` và `reviews_cleaned.csv` (8.417 mẫu × 23 cột).
  - Phân bố nhãn: **Positive 6.208 (73.8%)** / **Neutral 1.639 (19.5%)** / **Negative 570 (6.8%)**.
  - ✅ Đã bàn giao cho TV2 & TV3.

### 🟢 Giai đoạn 2: Báo cáo & Quản lý đồ án môn Máy học
- [ ] **Soạn thảo các mục trong Báo cáo Word:**
  - **Mục 1:** Mục tiêu đề tài và tóm tắt bài toán.
  - **Mục 2:** Giới thiệu bộ dữ liệu ITviec & Chiến lược gán nhãn.
  - **Mục 3.2:** Quy trình tiền xử lý văn bản tiếng Việt chi tiết.
- [ ] **Tổng hợp toàn văn Báo cáo (Word / PDF):**
  - Ghép nối các phần báo cáo từ TV2, TV3, TV4 theo cấu trúc chuẩn [De_Cuong_Do_An_Mon_Hoc_May_Hoc.md](file:///d:/Trí%20tuệ%20nhân%20tạo/HK2/Máy%20học/Project/Do_An_May_Hoc_Sentiment_Analysis/reports/De_Cuong_Do_An_Mon_Hoc_May_Hoc.md).
  - Định dạng chuẩn bài báo cáo đồ án đại học (bìa, mục lục, bảng biểu, trích dẫn tài liệu tham khảo).
- [ ] **Thiết kế Slide thuyết trình:**
  - Xây dựng Slide (15-20 trang) bám sát các tiêu chí chấm điểm của giảng viên.
  - Tổ chức buổi thuyết trình thử cho cả nhóm.

---

## 📦 II. ĐẦU VÀO & ĐẦU RA (INPUTS & OUTPUTS)

* **Đầu vào (Inputs):**
  - Dữ liệu thô: [Reviews.xlsx](file:///d:/Trí%20tuệ%20nhân%20tạo/HK2/Máy%20học/Project/Do_An_May_Hoc_Sentiment_Analysis/data/raw/Reviews.xlsx) (8.417 mẫu).
  - Bộ từ điển: `data/dictionaries/`.
* **Đầu ra (Outputs bàn giao):**
  - Module code: [src/preprocessing.py](file:///d:/Trí%20tuệ%20nhân%20tạo/HK2/Máy%20học/Project/Do_An_May_Hoc_Sentiment_Analysis/src/preprocessing.py).
  - Notebook: [notebooks/02_text_preprocessing.ipynb](file:///d:/Trí%20tuệ%20nhân%20tạo/HK2/Máy%20học/Project/Do_An_May_Hoc_Sentiment_Analysis/notebooks/02_text_preprocessing.ipynb).
  - Dữ liệu sạch: `data/processed/reviews_cleaned.xlsx`.
  - Toàn văn Báo cáo Word/PDF và File Slide trình chiếu.
