# TỔNG KẾT TOÀN DIỆN ĐỒ ÁN MÔN HỌC MÁY HỌC (FINAL SUMMARY)

**ĐỀ TÀI:** PHÂN TÍCH CẢM XÚC ĐÁNH GIÁ CÔNG TY ITVIEC (SENTIMENT ANALYSIS ON ITVIEC REVIEWS)  
**Môn học:** Máy học (Machine Learning) — Học kỳ 2  
**Đơn vị:** Trường Đại học Công nghệ Thông tin – ĐHQG-HCM (UIT)  
**Giảng viên hướng dẫn:** Thầy Cáp Phạm Đình Thăng  
**Sinh viên thực hiện:**
- Hoàng Hôn
- Vũ Văn Duy
- Nguyễn Duy Khang
- Phan Thành Trung  

---

## MỤC LỤC TỔNG QUAN

1. [Tổng quan đề tài & Mục tiêu nghiên cứu](#1-tổng-quan-đề-tài--mục-tiêu-nghiên-cứu)
2. [Bộ dữ liệu & Thách thức đặc thù](#2-bộ-dữ-liệu--thách-thức-đặc-thù)
3. [Quy trình tiền xử lý văn bản tiếng Việt 2 tầng](#3-quy-trình-tiền-xử-lý-văn-bản-tiếng-việt-2-tầng)
4. [Trích xuất đặc trưng & Khám phá dữ liệu (EDA)](#4-trích-xuất-đặc-trưng--khám-phá-dữ-liệu-eda)
5. [Thiết kế mô hình & Chống rò rỉ dữ liệu với SMOTE](#5-thiết-kế-mô-hình--chống-rò-rỉ-dữ-liệu-với-smote)
6. [Kết quả thực nghiệm & Đánh giá mô hình (Retrained v2)](#6-kết-quả-thực-nghiệm--đánh-giá-mô-hình-retrained-v2)
7. [Phân tích lỗi chuyên sâu (Error Analysis)](#7-phân-tích-lỗi-chuyên-sâu-error-analysis)
8. [Insight văn hóa & Cảm xúc doanh nghiệp IT](#8-insight-văn-hóa--cảm-xúc-doanh-nghiệp-it)
9. [Ứng dụng Web Demo tương tác (Streamlit)](#9-ứng-dụng-web-demo-tương-tác-streamlit)
10. [Bài học kinh nghiệm, Hạn chế & Hướng phát triển](#10-bài-học-kinh-nghiệm-hạn-chế--hướng-phát-triển)
11. [Hướng dẫn tái lập kết quả (Reproducibility Guide)](#11-hướng-dẫn-tái-lập-kết-quả-reproducibility-guide)

---

## 1. TỔNG QUAN ĐỀ TÀI & MỤC TIÊU NGHIÊN CỨU

Trong kỷ nguyên số, các nền tảng đánh giá tuyển dụng như ITviec đóng vai trò cầu nối thông tin quan trọng giữa ứng viên và doanh nghiệp. Tuy nhiên, việc đọc và tổng hợp hàng nghìn nhận xét dài, chứa văn phong đời thường, teencode và trộn lẫn thuật ngữ tiếng Anh – Việt vượt quá khả năng xử lý thủ công.

### Mục tiêu cốt lõi:
1. **Xây dựng pipeline Machine Learning End-to-End:** Tự động phân loại đánh giá của nhân viên thành 3 sắc thái cảm xúc: **Tích cực (Positive)**, **Trung tính (Neutral)**, và **Tiêu cực (Negative)**.
2. **Tuân thủ kỷ luật học thuật:** Chia tách dữ liệu nghiêm ngặt (*Stratified Train/Test*), khóa tập Test để chống Overfitting và rò rỉ dữ liệu (*Data Leakage*).
3. **Giải quyết vấn đề mất cân bằng lớp:** Xử lý tình trạng mất cân bằng dữ liệu 11:1 thông qua kỹ thuật Resampling (SMOTE) nằm trọn vẹn trong Pipeline cross-validation.
4. **Khai thác Insight doanh nghiệp:** Phân tích các khía cạnh nổi bật (Lương thưởng, Quản lý, Văn phòng, Đào tạo, Phúc lợi) tại các công ty IT hàng đầu Việt Nam.
5. **Triển khai ứng dụng thực tế:** Đóng gói toàn bộ mô hình và dịch vụ suy luận vào ứng dụng Web tương tác Streamlit với giao diện Dark-tech hiện đại.

---

## 2. BỘ DỮ LIỆU & THÁCH THỨC ĐẶC THÙ

### 2.1. Thống kê bộ dữ liệu
- **Dữ liệu thu thập ban đầu:** 8.417 bản ghi review công ty từ ITviec.
- **Sau kiểm tra dữ liệu trùng (Duplicate Audit):** Loại bỏ các mẫu xung đột nhãn, giữ lại **8.414 mẫu văn bản sạch**.
- **Chia tập dữ liệu (Stratified 80/20, Seed 2026):**
  - **Tập Development (Train/Val):** 6.731 mẫu (80%).
  - **Tập Final Test (Khóa độc lập):** 1.683 mẫu (20%).

### 2.2. Phân bố nhãn cảm xúc & Thách thức mất cân bằng
| Nhãn cảm xúc | Rating tương ứng | Số lượng mẫu (Toàn bộ) | Tỷ lệ (%) | Số lượng (Final Test) |
| :--- | :---: | :---: | :---: | :---: |
| **Positive (Tích cực)** | 4 – 5 sao | 5.595 | 66,5% | 1.241 |
| **Neutral (Trung tính)** | 3 sao | 1.664 | 19,8% | 328 |
| **Negative (Tiêu cực)** | 1 – 2 sao | 1.155 | 13,7% | 114 |
| **Tổng cộng** | **1 – 5 sao** | **8.414** | **100%** | **1.683** |

*(Lưu ý: Tỷ lệ giữa Positive và Negative là gần 11 : 1 trên dữ liệu gốc).*

### 2.3. Các thách thức kỹ thuật chính
1. **Bẫy Accuracy (Accuracy Paradox):** Do lớp Positive chiếm áp đảo (>66%), một mô hình tầm thường chỉ cần đoán toàn bộ là "Positive" đã đạt ~70% Accuracy nhưng hoàn toàn vô dụng với lớp Negative. Vì vậy, **Macro F1** bắt buộc phải là thước đo đánh giá chính.
2. **Nhãn yếu (Weak Label Noise):** Nhãn được suy ra từ số sao đánh giá (rating), không phải do chuyên gia gán thủ công từng câu. Một bài đánh giá 4 sao vẫn có thể chứa đoạn phàn nàn gay gắt về lương hoặc OT.
3. **Ngôn ngữ hỗn hợp (Code-switching):** Review IT chứa nhiều từ viết tắt tiếng Anh (OT, onsite, deadline, PM, fresher, senior, review lương, tech stack...).
4. **Teencode & Emojicon:** Sự xuất hiện của `ko`, `dc`, `k`, `wa`, `mn`, `b` cùng các biểu tượng cảm xúc `:))`, `:(`, `^^`.

---

## 3. QUY TRÌNH TIỀN XỬ LÝ VĂN BẢN TIẾNG VIỆT 2 TẦNG

Để giải quyết triệt để tính chất phức tạp của văn bản IT, nhóm thiết kế quy trình tiền xử lý hai tầng rõ ràng:

```
[Review thô: Tiêu đề + Điểm thích + Điểm góp ý]
                     │
                     ▼
       ┌─────────────────────────────┐
       │   TẦNG 1: BASIC CLEANING    │
       │ - Chuẩn hóa Unicode NFC     │
       │ - Loại bỏ URL, email        │
       │ - Chuyển chữ thường         │
       │ - Chuẩn hóa Emoji / Icon    │
       │ - Ánh xạ Teencode & Thuật ngữ│
       └──────────────┬──────────────┘
                      │  clean_basic_text (giữ ngữ cảnh tự nhiên)
                      ▼
       ┌─────────────────────────────┐
       │  TẦNG 2: ADVANCED CLEANING  │
       │ - Tách từ: underthesea      │
       │ - Ghép cụm từ ghép (gạch _) │
       │ - Lọc Stopwords có chọn lọc │
       │   (Giữ từ phủ định, mức độ) │
       └──────────────┬──────────────┘
                      │  clean_advance_text (đầu vào cho TF-IDF & Model)
                      ▼
            [TF-IDF Vectorizer]
```

### Điểm cải tiến cốt lõi của phiên bản Retrained v2:
- **Khắc phục lỗi xóa nhầm từ mang cảm xúc:** Ở phiên bản ban đầu, danh sách từ dừng (stopwords) loại bỏ các từ quan trọng như `không`, `chưa`, `thấp`, `quá`, `kém`. Bản Retrained v2 đã tinh chỉnh từ điển:
  - **Giữ lại nguyên vẹn các từ phủ định:** `không`, `chưa`, `chẳng`, `đừng`.
  - **Giữ lại các từ chỉ mức độ & tiêu cực:** `thấp`, `ít`, `nhiều`, `chậm`, `tệ`, `thiếu`.
  - **Bổ sung ánh xạ teencode chuyên ngành:** `ko` -> `không`, `dc` -> `được`, `k` -> `không`.
- **Ví dụ minh họa:**
  - *Câu gốc:* `"Lương thấp, quản lý thiếu minh bạch và thường xuyên phải OT không lương."`
  - *Sau tiền xử lý:* `"lương thấp quản_lý thiếu minh_bạch thường_xuyên ot không lương"` *(toàn bộ cụm sắc thái then chốt đều được bảo tồn trọn vẹn)*.

---

## 4. TRÍCH XUẤT ĐẶC TRƯNG & KHÁM PHÁ DỮ LIỆU (EDA)

### 4.1. Khám phá tương quan khía cạnh (EDA)
Phân tích ma trận tương quan giữa đánh giá tổng thể (Overall Rating) và các khía cạnh thành phần (Aspect Scores) chỉ ra:
- **Quản lý (Management cares about me):** Tương quan cao nhất ($r = 0,7368$).
- **Lương thưởng & Phúc lợi (Salary & benefits):** Tương quan rất cao ($r = 0,7343$).
- **Văn phòng & Cơ sở vật chất (Office & working space):** Tương quan thấp hơn ($r = 0,5423$).
> **Kết luận kiến trúc:** Các điểm thành phần là tín hiệu chẩn đoán giá trị nhưng **không có sẵn** khi người dùng nhập câu review tự do trong thực tế. Do đó, mô hình triển khai của nhóm tuân thủ nghiêm ngặt nguyên tắc **Text-only Input** (chỉ dự đoán từ văn bản).

### 4.2. Cấu hình trích xuất đặc trưng TF-IDF
Thực nghiệm so sánh trên 5-Fold Cross Validation khẳng định lợi ích của việc bổ sung Bigram:
- **Unigram thuần túy $(1, 1)$:** CV Macro F1 đạt $55,69\%$.
- **Unigram + Bigram $(1, 2)$:** CV Macro F1 đạt $57,22\%$ (tăng $+1,53\%$). Bigram nắm bắt chính xác các cụm phủ định như `"không_lương"`, `"thiếu_minh_bạch"`.
- **Cấu hình chốt:**
  - `ngram_range = (1, 2)`
  - `max_features = 5.000` (giữ ma trận thưa gọn gàng, tránh bùng nổ chiều)
  - `sublinear_tf = True` (dùng hàm logarit $1 + \log(\text{tf})$ làm mượt tần suất từ)
  - `min_df = 2` (loại bỏ các từ chỉ xuất hiện đúng 1 lần trong tập dữ liệu)

---

## 5. THIẾT KẾ MÔ HÌNH & CHỐNG RÒ RỈ DỮ LIỆU VỚI SMOTE

### 5.1. Năm thuật toán Machine Learning được đánh giá
Nhóm triển khai và so sánh công bằng 5 thuật toán trên cùng một contract dữ liệu TF-IDF:
1. **Multinomial Naive Bayes (MNB):** Baseline nhanh, giả định độc lập có điều kiện.
2. **Logistic Regression (LR):** Mô hình tuyến tính tối ưu hàm Log-loss với điều chuẩn L2 ($C=1.0$).
3. **Linear Support Vector Machine (Linear SVM):** Tìm siêu phẳng phân tách với lề cực đại.
4. **Random Forest Classifier (RF):** Tập hợp nhiều cây quyết định ngẫu nhiên (Ensemble Bagging).
5. **Stacking Ensemble Classifier (STK):** Kết hợp dự đoán của các mô hình cơ sở thông qua Meta-classifier.

### 5.2. Kỹ thuật SMOTE đặt trong Pipeline từng Fold (Leak-free Cross Validation)
- Để bù đắp số lượng mẫu ít ỏi của lớp Negative, nhóm áp dụng **SMOTE (Synthetic Minority Over-sampling Technique)**.
- **Nguyên tắc chống rò rỉ (Strict Data Isolation):**
  - SMOTE **chỉ được bọc bên trong Pipeline** huấn luyện của từng Fold (4/5 dữ liệu Train).
  - Validation Fold (1/5 còn lại) **hoàn toàn giữ nguyên bản**, không chứa bất kỳ mẫu tổng hợp nào.
  - Điều này đảm bảo điểm số Cross Validation phản ánh độ khái quát thực tế, không bị lạc quan giả tạo.

---

## 6. KẾT QUẢ THỰC NGHIỆM & ĐÁNH GIÁ MÔ HÌNH (RETRAINED V2)

### 6.1. Bảng xếp hạng mô hình trên Development Set (5-Fold Stratified CV ban đầu)
| Hạng | Mô hình | CV Macro F1 Mean | CV Macro F1 Std | Nhận xét |
| :---: | :--- | :---: | :---: | :--- |
| **1** | **Logistic Regression (SMOTE)** | **0,5727** | **$\pm 0,011$** | **Cao nhất, xuất trực tiếp xác suất xác thực** |
| 2 | Linear SVM (SMOTE) | 0,5724 | $\pm 0,010$ | Kém LR chỉ 0,0003, không hỗ trợ xác suất tự nhiên |
| 3 | Random Forest (Balanced) | 0,5421 | $\pm 0,009$ | Cây quyết định bị loãng trên không gian 5.000 chiều |
| 4 | Multinomial Naive Bayes | 0,5380 | $\pm 0,008$ | Nhạy cảm với giả định độc lập của từ ghép |
| 5 | Stacking Classifier | 0,5695 | $\pm 0,012$ | Không vượt trội do tương quan lỗi giữa các mô hình |

### 6.2. Kết quả Retrained v2 sau khi sửa lỗi tiền xử lý
Sau khi phát hiện và sửa lỗi từ dừng, nhóm giữ nguyên tắc **không dùng Final Test để tinh chỉnh mô hình**. Thay vào đó, nhóm chạy lại 5-Fold CV trên Development Set (fit TF-IDF độc lập trong từng fold):
- **CV Macro F1 trước khi sửa:** `0,5708`
- **CV Macro F1 sau khi sửa (Retrained v2):** `0,5815` (tăng $+0,0107$).

### 6.3. Đánh giá đối chiếu trên Final Test cũ (1.683 mẫu khóa)
| Chỉ số đánh giá | Bản gốc (Baseline) | Bản sửa (Retrained v2) | Thay đổi |
| :--- | :---: | :---: | :---: |
| **Accuracy** | 73,74% | **74,33%** | $+0,59\%$ |
| **Macro F1** | 0,5714 | **0,5764** | $+0,0050$ |
| **Weighted F1** | 0,7482 | **0,7540** | $+0,0058$ |
| **Tổng số mẫu dự đoán sai** | 442 / 1.683 | **432 / 1.683** | **Giảm 10 mẫu lỗi** |
| **Negative F1** | 0,3800 | **0,3910** | $+0,0110$ |
| **Negative Recall** | 44,74% | **45,61%** | $+0,87\%$ |

### 6.4. Ma trận nhầm lẫn (Confusion Matrix) chi tiết trên Final Test v2
| Nhãn thực tế \ Nhãn dự đoán | Negative (Đoán) | Neutral (Đoán) | Positive (Đoán) | Tổng thực tế |
| :--- | :---: | :---: | :---: | :---: |
| **Negative (Thực)** | **52** | 41 | 21 | 114 |
| **Neutral (Thực)** | 56 | **168** | 104 | 328 |
| **Positive (Thực)** | 44 | 166 | **1.031** | 1.241 |
| **Tổng dự đoán** | 152 | 375 | 1.156 | **1.683** |

*Hiệu năng chi tiết theo từng lớp:*
- **Positive:** Precision = $89,19\%$, Recall = $83,08\%$, **F1 = 0,8602** (Nhận diện rất tốt).
- **Neutral:** Precision = $44,80\%$, Recall = $51,22\%$, **F1 = 0,4780** (Mức độ trung bình).
- **Negative:** Precision = $34,21\%$, Recall = $45,61\%$, **F1 = 0,3910** (Khó nhận diện nhất do tập mẫu nhỏ).

---

## 7. PHÂN TÍCH LỖI CHUYÊN SÂU (ERROR ANALYSIS)

Nhóm phân tích 432 mẫu dự đoán sai trên tập Final Test và trích xuất 15 ca lỗi điển hình đại diện cho 3 nguyên nhân cốt lõi:

1. **Review nhiều vế, cảm xúc lẫn lộn (Mixed Sentiment — 7 / 15 ca):**
   - *Đặc điểm:* Người viết khen môi trường làm việc, đồng nghiệp thân thiện nhưng chê lương thấp, OT nhiều.
   - *Hạn chế mô hình:* Do mô hình chỉ xuất ra một nhãn duy nhất (Single-label), việc nén hai thái cực trái ngược vào một nhãn khiến xác suất bị phân mảnh giữa Positive và Negative.
2. **Cấu trúc phủ định phức tạp & Câu dài (Complex Negation — 6 / 15 ca):**
   - *Đặc điểm:* Sử dụng phủ định kép, mỉa mai (sarcasm) hoặc cụm từ cách nhau xa (ví dụ: *"không hẳn là tệ nhưng lương không cao"*).
   - *Hạn chế mô hình:* TF-IDF dạng Bag-of-Words bỏ qua thứ tự câu ngữ cảnh dài (Long-range dependency), dẫn đến việc mô hình chỉ cộng gộp trọng số từ khóa riêng lẻ.
3. **Nhãn yếu chưa chuẩn xác (Weak Label Noise — 2 / 15 ca):**
   - *Đặc điểm:* Người dùng chấm 1-2 sao (nhãn Negative) nhưng chỉ viết nhận xét trung tính hoặc lời cảm ơn; hoặc chấm 4 sao nhưng nội dung phàn nàn nặng nề.

---

## 8. INSIGHT VĂN HÓA & CẢM XÚC DOANH NGHIỆP IT

Từ 8.414 review, nhóm thực hiện phân tích chẩn đoán cho 5 doanh nghiệp công nghệ có số lượng đánh giá lớn nhất:

| Doanh nghiệp | Cỡ mẫu Review | Tỷ lệ Tích cực | Khía cạnh bị phàn nàn nhiều nhất | Chủ đề khen ngợi chính |
| :--- | :---: | :---: | :--- | :--- |
| **KMS Technology** | 251 | **Cao nhất** | Salary & benefits | Văn hóa doanh nghiệp, đồng nghiệp hòa đồng |
| **VNG Corporation** | 259 | **Cao** | Management cares about me | Phúc lợi, văn phòng hiện đại, sản phẩm lớn |
| **NashTech** | 358 | Trung bình khá | Salary & benefits | Quy trình chuyên nghiệp, cơ hội học hỏi |
| **FPT Software** | 2.014 | Trung bình | Salary & benefits | Quy mô lớn, môi trường tốt cho Fresher |
| **Bosch GST** | 412 | **Thấp nhất (15% Negative)** | Salary & benefits | Dự án toàn cầu, an toàn, ổn định |

> **Ghi chú quan trọng:** Đây là phân tích mô tả dữ liệu lịch sử trên nền tảng ITviec nhằm hỗ trợ phòng Nhân sự (HR) thấu hiểu tiếng nói nhân viên, **không mang ý nghĩa xếp hạng chất lượng uy tín** giữa các doanh nghiệp.

---

## 9. ỨNG DỤNG WEB DEMO TƯƠNG TÁC (STREAMLIT)

Hệ thống được đóng gói thành Web App Streamlit với **5 phân hệ chuyên biệt**, tích hợp theme Dark-tech đồng bộ với slide thuyết trình:

1. **Phân hệ Tổng quan (Overview):** Hiển thị KPI toàn diện của bộ dữ liệu (8.414 mẫu), phân bố số sao rating, và sơ đồ luồng kiến trúc Pipeline End-to-End.
2. **Phân hệ Insight Doanh nghiệp (Company Insights):** Khám phá cảm xúc theo từng công ty, trực quan hóa WordCloud các chủ đề tích cực và tiêu cực.
3. **Phân hệ Benchmark (Model Evaluation):** Trực quan hóa bảng so sánh hiệu năng 5 mô hình trên 5-Fold CV và kết quả đối chiếu Final Test Retrained v2.
4. **Phân hệ Phân tích lỗi (Error Analysis):** Trình diễn Confusion Matrix tương tác và bộ 15 ca lỗi minh họa với đầy đủ lý do phân loại.
5. **Phân hệ Dự đoán thời gian thực (Real-time Prediction):**
   - Cho phép người dùng nhập câu review tự do hoặc chọn 3 mẫu thử có sẵn (*"Khen rõ ràng"*, *"Chê rõ ràng"*, *"Nhiều vế phức tạp"*).
   - Hiển thị nhãn dự đoán cùng biểu đồ xác suất 3 lớp (Positive, Neutral, Negative).
   - Hiển thị văn bản sau tiền xử lý và bảng các token TF-IDF có trọng số cao nhất.

---

## 10. BÀI HỌC KINH NGHIỆM, HẠN CHẾ & HƯỚNG PHÁT TRIỂN

### 10.1. Bài học kinh nghiệm quý giá
- **Thước đo phù hợp với bài toán:** Không bao giờ dùng Accuracy làm kim chỉ nam duy nhất khi dữ liệu lệch; Macro F1 là cứu cánh phản ánh sự công bằng giữa các lớp.
- **Kỷ luật dữ liệu chống rò rỉ:** Khóa tập Final Test chỉ dùng đúng một lần; mọi kỹ thuật tiền xử lý và resample (SMOTE) phải nằm trọn vẹn trong Pipeline cross-validation.
- **Sức mạnh của mô hình đơn giản:** Trên không gian đặc trưng thưa (Sparse TF-IDF 5.000 chiều), mô hình tuyến tính Logistic Regression mang lại hiệu năng cao hơn, ổn định hơn và ít tốn tài nguyên hơn Random Forest hay Stacking Ensemble.

### 10.2. Hạn chế hiện tại
- Nhãn yếu suy từ rating vẫn tồn tại nhiễu ngữ nghĩa.
- Hiệu năng lớp Negative ($F_1 = 0,3910$) và Neutral ($F_1 = 0,4780$) còn hạn chế do thiếu mẫu và ranh giới sắc thái mờ nhạt.
- TF-IDF chưa xử lý trọn vẹn các câu dài có ngữ cảnh đối lập đa khía cạnh.

### 10.3. Hướng phát triển tương lai
1. **Gán nhãn chuẩn hóa (Gold Standard Sub-dataset):** Xây dựng tập dữ liệu nhỏ 500 – 1.000 review được chuyên gia gán nhãn thủ công để đo lường chính xác mức độ nhiễu của nhãn yếu.
2. **Phân tích cảm xúc theo khía cạnh (Aspect-Based Sentiment Analysis - ABSA):** Tách riêng dự đoán cảm xúc cho từng chiều: Lương thưởng, Đồng nghiệp, Quản lý, Chế độ OT.
3. **Mô hình ngôn ngữ tiền huấn luyện (Pretrained Transformers):** Thử nghiệm fine-tune các mô hình tiếng Việt tiên tiến như **PhoBERT** hoặc **ViSoBERT** khi có hạ tầng GPU.

---

## 11. HƯỚNG DẪN TÁI LẬP KẾT QUẢ (REPRODUCIBILITY GUIDE)

Tất cả các thành phần trong dự án đều được tự động hóa bằng script và kiểm thử toàn diện:

### 1. Cài đặt môi trường
```powershell
pip install -r requirements.txt
```

### 2. Chạy toàn bộ Unit Tests (37 tests)
```powershell
pytest -v -o pythonpath=.
```

### 3. Huấn luyện lại và đánh giá mô hình Retrained v2
```powershell
python scripts/retrain_sentiment_after_preprocessing.py
python scripts/evaluate_retrained_v2.py
```

### 4. Khởi chạy ứng dụng Web Demo (Streamlit)
```powershell
streamlit run app.py
```

### 5. Sinh bản Slide thuyết trình 15 trang (PowerPoint)
```powershell
python scripts/build_presentation_slides.py
```

---

*Bản tóm tắt được hoàn thiện vào ngày 23/09/2026 bởi Nhóm thực hiện Đồ án Máy học UIT.*
