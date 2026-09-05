# KẾ HOẠCH CHI TIẾT - THÀNH VIÊN 2: VĂN DUY
**Đề tài:** Phân tích cảm xúc đánh giá ITviec  
**Môn học:** Máy học (Machine Learning)  
**Phân công:** `Feature Engineering, Exploratory Data Analysis (EDA) & Data Splitting`  
**Thời gian thực hiện:** 4 Ngày cốt lõi (Giai đoạn 1) & Hỗ trợ kỹ thuật modeling  

---

## 📌 I. DANH SÁCH NHIỆM VỤ CHI TIẾT (CHECKLIST)

### 🟢 Giai đoạn 1: Khám phá phân tích dữ liệu (EDA) — ✅ HOÀN THÀNH (04/09/2026)
- [x] **Khám phá thống kê bộ dữ liệu ([01_data_exploration_eda.ipynb](file:///d:/Trí%20tuệ%20nhân%20tạo/HK2/Máy%20học/Project/Do_An_May_Hoc_Sentiment_Analysis/notebooks/01_data_exploration_eda.ipynb)):**
  - Thống kê số lượng mẫu (8.417 review, 180 công ty).
  - Phân tích phân bố số sao rating (1-5 sao) và mức độ mất cân bằng lớp: Positive (73.76%), Neutral (19.47%), Negative (6.77%).
  - Phân tích phân bố độ dài văn bản đánh giá (ký tự, số từ).
  - Phân tích tương quan giữa 5 khía cạnh thành phần (Lương thưởng, Đào tạo, Quản lý, Môi trường, OT) với Rating tổng.
  - Phân tích phân bố đánh giá theo thời gian và theo từng công ty.
- [x] **Xuất 9 biểu đồ trực quan hóa 300 DPI:**
  - Lưu trữ tại [reports/figures/](file:///d:/Trí%20tuệ%20nhân%20tạo/HK2/Máy%20học/Project/Do_An_May_Hoc_Sentiment_Analysis/reports/figures) để chèn trực tiếp vào Báo cáo và Slide.

### 🟢 Giai đoạn 2: Trích xuất đặc trưng (Feature Engineering) & Chia dữ liệu — ✅ HOÀN THÀNH (04/09/2026)
- [x] **Xây dựng module [src/features.py](file:///d:/Tr%C3%AD%20tu%E1%BB%87%20nh%C3%A2n%20t%E1%BA%A1o/HK2/M%C3%A1y%20h%E1%BB%8Dc/Project/Do_An_May_Hoc_Sentiment_Analysis/src/features.py):**
  - Trích xuất đặc trưng **TF-IDF (Term Frequency - Inverse Document Frequency)** với `ngram_range=(1, 2)`, `sublinear_tf=True`, `max_features=5000`.
  - Ablation study: so sánh **TF-IDF** vs **TF-IDF + Lexicon features** để xác định bộ đặc trưng nào cho Cross-Validation Macro F1 cao hơn.
  - Thử nghiệm tích hợp `neutral_keywords.txt` (có sẵn trong `data/dictionaries/`) vào bộ đặc trưng Lexicon.
- [x] **Phân chia dữ liệu chuẩn chống rò rỉ (Data Splitting):**
  - Áp dụng **Stratified Split 80/20**:
    - **Development Set (80%):** 6.730 mẫu dùng cho Cross-Validation và huấn luyện mô hình.
    - **Final Test Set (20%):** 1.683 mẫu khóa độc lập chống rò rỉ dữ liệu (data leakage).
  - **Kiểm tra phân bố 3 lớp trong cả 2 tập** (Positive/Neutral/Negative) sau khi split, xác nhận Stratified giữ đúng tỷ lệ 73.8/19.5/6.8%.
- [x] **Đóng gói Artifacts bàn giao:**
  - Lưu trữ `models/train_test_features.joblib`, `models/text_tfidf_vectorizer.joblib`, `models/artifact_manifest.json` để bàn giao cho TV3.

### 🟢 Giai đoạn 3: Soạn thảo Báo cáo môn Máy học — ✅ HOÀN THÀNH (bản thảo `reports/eda_feature_engineering.md`)
- [x] **Viết nội dung Báo cáo:**
  - **Mục 2:** Phân tích mô tả bộ dữ liệu, phân bố lớp và tỷ lệ phân chia tập dữ liệu.
  - **Mục 3.1:** Toàn bộ nội dung Phân tích khám phá dữ liệu (EDA) kèm biểu đồ và nhận xét.
  - **Mục 3.2 (Phần Feature Engineering):** Cơ sở lý thuyết của TF-IDF N-gram và cách biến đổi văn bản thành ma trận đặc trưng số.

---

## 📦 II. ĐẦU VÀO & ĐẦU RA (INPUTS & OUTPUTS)

* **Đầu vào (Inputs):**
  - File dữ liệu sạch `data/processed/reviews_cleaned.xlsx` từ TV1.
* **Đầu ra (Outputs bàn giao):**
  - Module [src/features.py](file:///d:/Trí%20tuệ%20nhân%20tạo/HK2/Máy%20học/Project/Do_An_May_Hoc_Sentiment_Analysis/src/features.py).
  - Notebook [01_data_exploration_eda.ipynb](file:///d:/Trí%20tuệ%20nhân%20tạo/HK2/Máy%20học/Project/Do_An_May_Hoc_Sentiment_Analysis/notebooks/01_data_exploration_eda.ipynb).
  - 9 biểu đồ trực quan chất lượng cao trong `reports/figures/`.
  - Artifacts đặc trưng đã đóng gói trong `models/` bàn giao cho TV3.


---

## ✅ III. KẾT QUẢ THỰC TẾ (cập nhật 05/09/2026, chạy lại trên dataset lexicon v2)

| Hạng mục | Kết quả |
| :--- | :--- |
| Dữ liệu đầu vào | 8.417 review / 23 cột, sau khử trùng lặp còn **8.414** dòng modeling |
| Phân bố nhãn | Positive 73,76% · Neutral 19,47% · Negative 6,77% |
| Split | Stratified 80/20, `random_state=2026` → development **6.731** / final test **1.683** (khóa, TV2 không đo metric) |
| Cấu hình TF-IDF chọn theo CV | `ngram_range=(1,2)`, `max_features=5000`, `min_df=2`, `sublinear_tf=True` |
| CV Macro F1 (development, LogReg balanced) | unigram 0,5569 · **unigram+bigram 0,5722** |
| Ablation | Hybrid 0,7492 · Aspect-only 0,7374 · Text+lexicon 0,5775 · **Text-only 0,5722** |
| Biểu đồ | 9 file PNG 300 DPI tại `reports/figures/` |
| Artifacts bàn giao TV3 | `text_tfidf_vectorizer.joblib`, `text_feature_extractor.joblib`, `train_test_features.joblib`, `artifact_manifest.json` |
| Kiểm thử | `pytest tests/` → 11 passed |

**Lưu ý bàn giao cho TV3:** artifact chính là **text-only** (khớp web demo chỉ nhập văn bản). Nhóm feature điểm khía cạnh chỉ là thí nghiệm chẩn đoán, không dùng cho mô hình bàn giao. SMOTE phải áp dụng **bên trong từng fold CV**, không cân bằng trước khi chia.

**Cập nhật 05/09/2026:** hai lỗi lexicon đã được xử lý ở pipeline tiền xử lý (greedy phrase matching, đếm emoji trên văn bản thô). Coverage 12,23% → **98,87%**, emoji 0 → 20 dòng. Toàn bộ artifacts và biểu đồ đã chạy lại trên dataset mới, Macro F1 text-only tăng 0,5597 → **0,5722**. Nhóm Text+lexicon lần đầu vượt Text-only (0,5775 so với 0,5722), tuy chênh lệch vẫn nằm trong dao động giữa các fold nên cấu hình chính giữ nguyên dạng chỉ văn bản.
