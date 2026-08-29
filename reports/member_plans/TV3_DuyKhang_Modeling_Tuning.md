# KẾ HOẠCH CHI TIẾT - THÀNH VIÊN 3: DUY KHANG
**Đề tài:** Phân tích cảm xúc đánh giá ITviec  
**Môn học:** Máy học (Machine Learning)  
**Phân công:** `Machine Learning Modeling, Hyperparameter Tuning & Cross-Validation`  
**Thời gian thực hiện:** 5 Ngày cốt lõi (Giai đoạn 2) & Phối hợp hoàn thiện báo cáo  

---

## 📌 I. DANH SÁCH NHIỆM VỤ CHI TIẾT (CHECKLIST)

### 🟢 Giai đoạn 1: Xây dựng & Huấn luyện các mô hình Machine Learning
- [ ] **Làm việc trên [notebooks/03_sentiment_modeling_ml.ipynb](file:///d:/Trí%20tuệ%20nhân%20tạo/HK2/Máy%20học/Project/Do_An_May_Hoc_Sentiment_Analysis/notebooks/03_sentiment_modeling_ml.ipynb) và [src/models.py](file:///d:/Trí%20tuệ%20nhân%20tạo/HK2/Máy%20học/Project/Do_An_May_Hoc_Sentiment_Analysis/src/models.py):**
  - Đọc ma trận đặc trưng `train_test_features.joblib` do TV2 bàn giao.
- [ ] **Cài đặt 5 thuật toán Machine Learning phân loại:**
  1. **Multinomial Naive Bayes (MNB):** Mô hình Baseline đánh giá xác suất từ khóa.
  2. **Logistic Regression (LR):** Mô hình tuyến tính với hàm mất mát Log-loss, áp dụng `class_weight='balanced'`.
  3. **Support Vector Machine (Linear SVM):** Tìm siêu phẳng phân cách cực đại hóa biên độ (Margin), áp dụng `class_weight='balanced'`.
  4. **Random Forest Classifier (RF):** Mô hình Ensemble Bagging nhiều cây quyết định, xử lý phi tuyến.
  5. **Stacking Ensemble Classifier:** Kết hợp các mô hình cơ sở (Base Learners) qua một Meta-Classifier để nâng cao hiệu năng tổng thể.

### 🟢 Giai đoạn 2: Kỹ thuật xử lý mất cân bằng & Tinh chỉnh siêu tham số (Tuning)
- [ ] **Xây dựng sklearn Pipeline chống Data Leakage:**
  - TF-IDF Vectorizer phải nằm bên trong `sklearn.pipeline.Pipeline` cùng với từng model (chỉ `fit` trên tập train, `transform` trên tập val/test).
  - Ví dụ: `Pipeline([('tfidf', TfidfVectorizer(...)), ('clf', LogisticRegression(...))])`.
- [ ] **Xử lý mất cân bằng dữ liệu (Class Imbalance):**
  - Thử nghiệm và so sánh cơ chế phân bổ trọng số lớp nghịch đảo `class_weight='balanced'` vs kỹ thuật tái lấy mẫu (SMOTE / Random Undersampling).
- [ ] **Tối ưu hóa siêu tham số (Hyperparameter Tuning):**
  - Áp dụng **Stratified 5-Fold Cross Validation** với `scoring='f1_macro'` trên tập Development để tìm bộ tham số tốt nhất (GridSearchCV / RandomizedSearchCV).
  - Tinh chỉnh: `C` (Logistic Regression, LinearSVC), `alpha` (Naive Bayes), `n_estimators`, `max_depth` (Random Forest).
  - Định nghĩa Stacking rõ ràng: **Base learners = [MNB, LR, LinearSVC, RF]** → **Meta-learner = Logistic Regression** với `cv=5`.
- [ ] **Khóa mô hình tối ưu:**
  - Huấn luyện lại mô hình có điểm CV Macro F1 cao nhất trên toàn bộ tập Development (80%).
  - Lưu checkpoint mô hình tốt nhất vào `models/best_sentiment_model.joblib`.
  - Bàn giao mô hình đã khóa cho **TV4 (Thành Trung)** để đánh giá đúng 1 lần trên tập Final Test độc lập.

### 🟢 Giai đoạn 3: Soạn thảo Báo cáo môn Máy học
- [ ] **Viết nội dung Báo cáo:**
  - **Mục 3.3:** Cơ sở lý thuyết của 5 mô hình Machine Learning, cơ chế Ensemble Stacking và kỹ thuật xử lý mất cân bằng dữ liệu.
  - **Mục 3.4 & 5.1 (Phần Huấn luyện):** Bảng tổng hợp kết quả 5-Fold Cross Validation, giải thích lý do lựa chọn siêu tham số và lý giải vì sao mô hình tuyến tính (SVM/LR) hoạt động tốt trên không gian đặc trưng TF-IDF.

---

## 📦 II. ĐẦU VÀO & ĐẦU RA (INPUTS & OUTPUTS)

* **Đầu vào (Inputs):**
  - Bộ dữ liệu ma trận TF-IDF `train_test_features.joblib` và `text_tfidf_vectorizer.joblib` từ TV2.
* **Đầu ra (Outputs bàn giao):**
  - Module code [src/models.py](file:///d:/Trí%20tuệ%20nhân%20tạo/HK2/Máy%20học/Project/Do_An_May_Hoc_Sentiment_Analysis/src/models.py).
  - Notebook hoàn chỉnh [notebooks/03_sentiment_modeling_ml.ipynb](file:///d:/Trí%20tuệ%20nhân%20tạo/HK2/Máy%20học/Project/Do_An_May_Hoc_Sentiment_Analysis/notebooks/03_sentiment_modeling_ml.ipynb).
  - File model tốt nhất đã lưu: `models/best_sentiment_model.joblib`.
  - Bảng số liệu Cross-Validation phục vụ viết Báo cáo.
