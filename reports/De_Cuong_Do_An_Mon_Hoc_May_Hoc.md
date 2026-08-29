# ĐỀ CƯƠNG BÁO CÁO ĐỒ ÁN MÔN HỌC MÁY HỌC
## ĐỀ TÀI: PHÂN TÍCH CẢM XÚC (SENTIMENT ANALYSIS) ĐÁNH GIÁ CÔNG TY ITVIEC

*(Bám sát 100% yêu cầu nội dung môn học Machine Learning - DoAnMonHoc.docx)*

---

## 1. Mục tiêu của đồ án
* **Hiểu và phân tích bộ dữ liệu:** Khám phá đặc tính của tập dữ liệu đánh giá thực tế từ 180 công ty công nghệ trên ITviec.
* **Tiền xử lý dữ liệu:** Xây dựng pipeline làm sạch văn bản tiếng Việt (chuẩn hóa Unicode, giải mã emoji/emojicon, xử lý teencode, tách từ ghép tiếng Việt bằng `underthesea`, lọc bỏ stopwords) và mã hóa sang không gian đặc trưng số **TF-IDF (Unigram + Bigram)**.
* **Áp dụng các thuật toán Machine Learning:** Cài đặt và thực nghiệm **5 thuật toán Machine Learning** giải quyết bài toán phân loại đa lớp:
  1. *Multinomial Naive Bayes (MNB)*
  2. *Logistic Regression (LR)*
  3. *Support Vector Machine (Linear SVM)*
  4. *Random Forest Classifier (RF)*
  5. *Stacking Ensemble Classifier (Kết hợp LR + SVM + RF)*
* **Xử lý mất cân bằng dữ liệu:** Áp dụng kỹ thuật phân bổ trọng số lớp `class_weight='balanced'` để tối ưu hóa việc phân loại các lớp thiểu số.
* **So sánh và đánh giá hiệu năng:** Sử dụng Stratified 5-Fold Cross Validation trên tập Development và kiểm thử độc lập trên tập Final Test bằng các thang đo **Macro F1-Score, Accuracy, Precision, Recall và Confusion Matrix**.
* **Phân tích chuyên sâu & Kết luận:** Phân tích ma trận nhầm lẫn, phân tích lỗi sai (Error Analysis), đánh giá hiện tượng Overfitting/Underfitting và rút ra kết luận thực nghiệm.

---

## 2. Mô tả bộ dữ liệu
* **Tổng số mẫu:** **8.417 đánh giá** từ 180 công ty CNTT tại Việt Nam.
* **Số lượng lớp (3 lớp cảm xúc):**
  * **Positive (Tích cực - 4-5 sao):** 6.208 mẫu (**73.76%**)
  * **Neutral (Trung tính - 3 sao):** 1.639 mẫu (**19.47%**)
  * **Negative (Tiêu cực - 1-2 sao):** 570 mẫu (**6.77%**)
* **Mức độ mất cân bằng:** Mất cân bằng lớp nghiêm trọng (Tỷ lệ Positive : Neutral : Negative $\approx$ 11 : 3 : 1).
* **Cơ chế gán nhãn:** Gán nhãn yếu (*Rating-derived Weak Labels*) từ điểm số sao thực tế của người dùng.
* **Phân chia dữ liệu (Data Splitting):**
  * Phân chia có phân tầng (**Stratified Split**) đảm bảo giữ nguyên tỷ lệ phân bố các lớp.
  * **Tập Development (Huấn luyện & Cross-Validation):** 80% (6.729 mẫu) $\rightarrow$ áp dụng Stratified 5-Fold CV để chọn siêu tham số và mô hình.
  * **Tập Final Test (Kiểm thử độc lập - Khóa chống rò rỉ):** 20% (1.682 mẫu) $\rightarrow$ chỉ đánh giá một lần duy nhất khi đã khóa mô hình tốt nhất.

---

## 3. Khám phá dữ liệu (EDA) & Tiền xử lý
### 3.1. Phân tích khám phá dữ liệu (EDA)
* Thống kê mô tả: Phân bố số sao rating, phân bố độ dài văn bản đánh giá (ký tự / số từ).
* Phân tích tương quan giữa các điểm đánh giá khía cạnh: Lương thưởng & Phúc lợi, Cơ hội đào tạo, Sự quan tâm của quản lý, Văn hóa & Môi trường, Tăng ca (OT).
* Phân tích phân bố số lượng đánh giá theo thời gian và theo từng công ty.

### 3.2. Tiền xử lý & Trích xuất đặc trưng (Feature Engineering)
* **Quy trình tiền xử lý:**
  1. Hợp nhất nội dung đánh giá: `raw_text = Title + " . " + Liked + " . " + Improvement`.
  2. Chuẩn hóa Unicode NFC, loại bỏ link, email, ký tự rác.
  3. Ánh xạ emoji, emojicon sang từ ngữ cảm xúc tương ứng.
  4. Chuẩn hóa teencode và thuật ngữ IT viết tắt (*cty, k, lm, ot, dev, pm,...*).
  5. Tách từ tiếng Việt bằng thư viện `underthesea`.
  6. Loại bỏ từ dừng (Stopwords), bảo lưu các từ phủ định quan trọng (*không, chưa, chẳng*).
* **Trích xuất đặc trưng:**
  * **TF-IDF Vectorizer:** Biến đổi văn bản thành ma trận đặc trưng số, cấu hình `ngram_range=(1, 2)`, `sublinear_tf=True`, `max_features=5000`.

---

## 4. Mô hình Machine Learning & Đánh giá
### 4.1. Thiết kế các mô hình
* **Mô hình 1: Multinomial Naive Bayes:** Mô hình Baseline dựa trên xác suất có điều kiện của từ vựng.
* **Mô hình 2: Logistic Regression:** Sử dụng hàm Sigmoid/Softmax với trọng số lớp `class_weight='balanced'` và điều chuẩn L2.
* **Mô hình 3: Support Vector Machine (Linear SVM):** Tối ưu hóa siêu phẳng phân cách với hàm mất mát Hinge và `class_weight='balanced'`.
* **Mô hình 4: Random Forest Classifier:** Mô hình cây quyết định kết hợp (Bagging Ensemble) với `n_estimators=200`, `class_weight='balanced'`.
* **Mô hình 5: Stacking Ensemble Classifier:** Kết hợp dự đoán xác suất của Logistic Regression, SVM và Random Forest qua Meta-Classifier (Logistic Regression).

### 4.2. Thang đo đánh giá (Evaluation Metrics)
* **Macro F1-Score:** Độ đo trọng tâm vì dữ liệu mất cân bằng nặng.
* **Accuracy, Precision, Recall (per-class & weighted average).**
* **Confusion Matrix (Ma trận nhầm lẫn).**

---

## 5. Kết quả thực nghiệm và Phân tích kết quả
### 5.1. So sánh hiệu suất giữa các mô hình
* Bảng tổng hợp kết quả 5-Fold Cross Validation trên tập Development và điểm số trên Final Test.
* Nhận xét và lý giải: *Tại sao Linear SVM và Logistic Regression lại đạt hiệu năng vượt trội hơn Naive Bayes và Random Forest trên không gian đặc trưng TF-IDF thưa chiều cao (High-dimensional Sparse Space)?*

### 5.2. Phân tích chi tiết kết quả & Phân tích lỗi (Error Analysis)
* **Model tốt nhất:** Chỉ ra mô hình tối ưu nhất và mức độ chênh lệch F1-Score so với baseline.
* **Phân tích Confusion Matrix:** Đánh giá các class hay bị nhầm lẫn nhất (đặc biệt là giữa Neutral và Positive/Negative).
* **Phân tích lỗi sai (Error Analysis):**
  * Dạng 1: Các review mang tính châm biếm, mỉa mai (*"công ty rất tốt cho ai muốn học cách chịu đựng"*).
  * Dạng 2: Review dài chứa cả ý khen và ý chê (*"lương cao nhưng OT sấp mặt, quản lý kém"*).
  * Dạng 3: Bất nhất giữa nội dung văn bản và số sao rating do người dùng chấm.
* **Đánh giá Overfitting/Underfitting:** So sánh chênh lệch giữa điểm Train CV và Test để chứng minh mô hình tổng quát hóa tốt.

---

## 6. Kết luận, Hướng phát triển & Sản phẩm nộp
### 6.1. Kết luận & Đóng góp
* Xây dựng thành công pipeline hoàn chỉnh từ văn bản thô đến mô hình phân loại cảm xúc tối ưu.
* Khai phá các insight quan trọng về mức độ hài lòng của nhân viên đối với các công ty IT tại Việt Nam.

### 6.2. Hướng phát triển
* Mở rộng bài toán phân tích cảm xúc chi tiết theo từng khía cạnh cụ thể (Aspect-Based Sentiment Analysis - ABSA).
* Ứng dụng mô hình ngôn ngữ lớn (LLM) để tự động tạo bản tóm tắt góp ý cho phòng nhân sự (HR).

### 6.3. Sản phẩm bàn giao
* 📄 **File Báo cáo Word/PDF hoàn chỉnh.**
* 💻 **Mã nguồn và Jupyter Notebooks có chú thích rõ ràng.**
* 📊 **Slide thuyết trình báo cáo đồ án.**
* 🌐 **Ứng dụng Demo phân loại cảm xúc tương tác trực tiếp.**
