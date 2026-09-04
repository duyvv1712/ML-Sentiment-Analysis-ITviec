# Phân tích khám phá dữ liệu và trích xuất đặc trưng cho bài toán phân loại cảm xúc đánh giá ITviec

## 2. Mô tả bộ dữ liệu

### 2.1. Quy mô và cấu trúc

Bộ dữ liệu sau tiền xử lý gồm **8.417 đánh giá** và **23 trường**, thu thập từ nền tảng tuyển dụng ITviec. Các trường được chia thành năm nhóm:

| Nhóm trường | Các trường thành phần |
|---|---|
| Định danh và doanh nghiệp | `id`, `Company Name`, `Cmt_day` |
| Nội dung đánh giá | `Title`, `What I liked`, `Suggestions for improvement` |
| Điểm đánh giá | `Rating` (thang 1–5) và năm điểm khía cạnh |
| Văn bản đã chuẩn hóa | `raw_review_text`, `clean_basic_text`, `clean_advance_text` |
| Đặc trưng từ điển và nhãn | `pos_w`, `neg_w`, `pos_e`, `neg_e`, `total_we`, `sentiment_ratio`, `sentiment` |

Năm điểm khía cạnh gồm lương và phúc lợi, đào tạo, quản lý, văn hóa doanh nghiệp và văn phòng làm việc, tất cả đều theo thang thứ bậc 1–5.

### 2.2. Chất lượng dữ liệu

Giá trị khuyết thiếu chỉ xuất hiện ở hai trường nội dung với tỷ lệ không đáng kể, gồm `What I liked` (1 dòng, 0,01%) và `Suggestions for improvement` (5 dòng, 0,06%). Các trường đã chuẩn hóa và trường nhãn không có giá trị khuyết thiếu, đồng thời không tồn tại dòng trùng lặp hoàn toàn.

Ở mức văn bản đã chuẩn hóa, sáu dòng thuộc ba nhóm có `clean_advance_text` giống hệt nhau. Trong đó một nhóm mang nhãn khác nhau dù nội dung đồng nhất, phản ánh mâu thuẫn nội tại của quy tắc gán nhãn. Nhóm mâu thuẫn được loại bỏ toàn bộ nhằm tránh đưa tín hiệu nhiễu vào quá trình học, các nhóm trùng cùng nhãn chỉ giữ lại một đại diện để tránh rò rỉ mẫu giữa tập huấn luyện và tập kiểm thử. Sau bước này, dữ liệu dùng cho mô hình hóa còn **8.413 dòng**.

### 2.3. Bản chất của nhãn

Nhãn `sentiment` nhận ba giá trị `Positive`, `Neutral` và `Negative`, được suy ra theo quy tắc ánh xạ từ `Rating`. Đây là **nhãn yếu** (weak label) sinh tự động, không phải nhãn vàng do người đọc nội dung và gán thủ công. Đặc điểm này chi phối toàn bộ cách diễn giải kết quả ở các phần sau, vì mọi trường có quan hệ hàm số với `Rating` đều tiềm ẩn nguy cơ rò rỉ nhãn.

## 3.1. Phân tích khám phá dữ liệu

### 3.1.1. Phân bố điểm đánh giá và nhãn cảm xúc

| Rating | Số đánh giá |
|---:|---:|
| 1 | 124 |
| 2 | 446 |
| 3 | 1.639 |
| 4 | 2.698 |
| 5 | 3.510 |

![Phân bố điểm đánh giá](figures/eda_rating_distribution.png)

| Nhãn | Số đánh giá | Tỷ lệ |
|---|---:|---:|
| Positive | 6.208 | 73,76% |
| Neutral | 1.639 | 19,47% |
| Negative | 570 | 6,77% |

![Phân bố nhãn cảm xúc](figures/eda_sentiment_counts.png)

Lớp Positive chiếm gần ba phần tư dữ liệu và lớn gấp khoảng 10,9 lần lớp Negative. Mức mất cân bằng này khiến Accuracy trở thành thang đo gây hiểu nhầm, vì một bộ phân loại luôn dự đoán Positive đã đạt xấp xỉ 73,8% Accuracy mà không mang giá trị sử dụng. Các thực nghiệm vì vậy lấy **Macro F1** làm thang đo chính và theo dõi thêm Recall của lớp Negative.

### 3.1.2. Độ dài nội dung đánh giá

| Trường | Trung bình | Trung vị | Phân vị 95 | Phân vị 99 | Lớn nhất |
|---|---:|---:|---:|---:|---:|
| What I liked | 50,29 | 36 | 131 | 237,36 | 1.400 |
| Suggestions for improvement | 29,97 | 20 | 78 | 184 | 876 |

![Phân bố độ dài nội dung](figures/eda_text_length_distribution.png)

Phân bố độ dài lệch phải rõ rệt ở cả hai trường, với phần lớn đánh giá ngắn và một số ít ngoại lệ rất dài. Đặc điểm này ủng hộ việc dùng TF-IDF thay cho tần suất tuyệt đối, vì trọng số TF-IDF chuẩn hóa theo độ dài văn bản nên hạn chế được ảnh hưởng của các đánh giá dài bất thường.

### 3.1.3. Quan hệ giữa điểm khía cạnh và cảm xúc tổng thể

Hệ số tương quan Spearman được sử dụng do các điểm đánh giá thuộc thang thứ bậc.

| Khía cạnh | Tương quan Spearman với `Rating` |
|---|---:|
| Management cares about me | 0,7368 |
| Salary & benefits | 0,7343 |
| Culture & fun | 0,6566 |
| Training & learning | 0,6398 |
| Office & workspace | 0,5423 |

![Tương quan giữa các điểm đánh giá](figures/eda_aspect_correlation.png)

Điểm trung bình của mọi khía cạnh đều giảm đơn điệu theo thứ tự Positive, Neutral, Negative, với chênh lệch lớn nhất ở quản lý, lương và phúc lợi, văn hóa doanh nghiệp. Mặc dù mang tính dự báo cao, `Rating` tổng thể không được đưa vào ma trận đặc trưng, bởi nhãn `sentiment` là hàm xác định của chính trường này nên việc sử dụng nó sẽ tạo rò rỉ nhãn hoàn toàn.

![Điểm khía cạnh theo cảm xúc](figures/eda_aspect_by_sentiment.png)

### 3.1.4. Phân bố theo doanh nghiệp và thời gian

Dữ liệu bao phủ **180 doanh nghiệp** trong khoảng thời gian từ tháng 07/2016 đến tháng 05/2025. Phân bố theo doanh nghiệp mất cân đối đáng kể, trong đó FPT Software chiếm 2.014 đánh giá tương đương 23,93% toàn bộ dữ liệu, còn 110 trên 180 doanh nghiệp có dưới 20 đánh giá. Hệ quả phương pháp luận là mọi kết luận ở cấp doanh nghiệp phải kèm theo cỡ mẫu và chỉ nên đưa ra khi đạt ngưỡng tối thiểu.

![Phân bố đánh giá theo doanh nghiệp và thời gian](figures/eda_company_time_distribution.png)

### 3.1.5. Chẩn đoán chất lượng nhãn yếu và độ phủ từ điển

| Chỉ số chẩn đoán | Giá trị |
|---|---|
| Tỷ lệ đánh giá có ít nhất một từ khớp từ điển cảm xúc | 12,23% |
| Số dòng có đặc trưng biểu tượng cảm xúc khác 0 | 0 trên 8.417 |
| Dòng thuộc nhóm văn bản trùng lặp | 6 |
| Nhóm văn bản trùng nhưng khác nhãn | 1 |

Đối chiếu nhãn yếu với trường `Recommend?` cho thấy các trường hợp bất đồng đáng kể, gồm 41 đánh giá mang nhãn Negative nhưng vẫn khuyến nghị công ty, 411 đánh giá Neutral và 87 đánh giá Positive lại không khuyến nghị.

![Chẩn đoán nhãn yếu và từ điển](figures/eda_label_quality_diagnostics.png)

Các dấu hiệu trên không đủ để kết luận nhãn yếu sai, nhưng đủ để bác bỏ giả định xem `Rating` là chân lý tuyệt đối. Trần hiệu năng của mọi mô hình học từ tập nhãn này vì vậy bị giới hạn bởi chính độ nhiễu của quy tắc gán nhãn.

## 3.2. Trích xuất đặc trưng

### 3.2.1. Biểu diễn TF-IDF

Văn bản được biểu diễn bằng TF-IDF trên trường `clean_advance_text` với tham số `max_features=5000`, `ngram_range=(1, 2)`, `min_df=2` và `sublinear_tf=True`. Trọng số TF-IDF của một từ trong một văn bản được tính theo công thức:

```
tfidf(t, d) = (1 + log tf(t, d)) × log((1 + N) / (1 + df(t))) + 1
```

trong đó `tf(t, d)` là tần suất xuất hiện, `df(t)` là số văn bản chứa từ và `N` là tổng số văn bản. Thành phần `sublinear_tf` giảm ảnh hưởng của việc lặp từ nhiều lần trong cùng một đánh giá, còn `min_df=2` loại bỏ các từ chỉ xuất hiện một lần vốn phần lớn là lỗi chính tả.

### 3.2.2. Thiết kế thực nghiệm và phân chia dữ liệu

Dữ liệu được chia phân tầng theo nhãn thành tập phát triển 80% và tập kiểm thử cuối 20% với `random_state=2026`. Tập kiểm thử cuối được khóa và không tham gia vào bất kỳ bước lựa chọn đặc trưng hay đo hiệu năng nào trong phần này, nhằm bảo toàn tính độc lập cho khâu đánh giá cuối cùng.

Mọi so sánh đặc trưng đều thực hiện bằng 5-fold Stratified Cross-Validation trên tập phát triển. Bộ vector hóa và bộ chuẩn hóa được huấn luyện lại bên trong từng fold, tránh rò rỉ thông tin thống kê từ phần dữ liệu kiểm định sang phần huấn luyện.

### 3.2.3. Lựa chọn cấu hình n-gram

Mô hình khảo sát là Logistic Regression với `class_weight='balanced'`.

| Cấu hình | Macro F1 trung bình | Độ lệch chuẩn |
|---|---:|---:|
| Unigram `(1, 1)` | 0,5385 | 0,0117 |
| Unigram và bigram `(1, 2)` | **0,5597** | 0,0141 |

![So sánh cấu hình n-gram](figures/eda_tfidf_ngram_comparison.png)

Cấu hình unigram kết hợp bigram cao hơn 0,0212 Macro F1. Khoảng dao động giữa các fold của hai cấu hình gần như tách rời nhau, do đó chênh lệch này được xem là ổn định chứ không phải dao động ngẫu nhiên. Cấu hình `(1, 2)` được chọn cho toàn bộ thực nghiệm tiếp theo, phù hợp với đặc thù tiếng Việt nơi nhiều cụm mang cảm xúc chỉ bộc lộ ở mức hai từ, chẳng hạn cấu trúc phủ định.

### 3.2.4. Nghiên cứu loại bỏ đặc trưng

| Nhóm đặc trưng | Macro F1 CV | Độ lệch chuẩn | Vai trò |
|---|---:|---:|---|
| Chỉ điểm khía cạnh | 0,7388 | 0,0069 | Cận trên dạng bảng, mang tính chẩn đoán |
| Kết hợp toàn bộ | 0,7370 | 0,0122 | Chẩn đoán |
| **Chỉ văn bản** | **0,5597** | 0,0141 | **Cấu hình chính** |
| Văn bản và từ điển | 0,5550 | 0,0151 | Loại bỏ sau khảo sát |

![Nghiên cứu loại bỏ đặc trưng](figures/eda_feature_ablation_cv.png)

Nhóm chỉ dùng điểm khía cạnh đạt Macro F1 cao hơn nhóm chỉ dùng văn bản khoảng 0,18, khoảng cách vượt xa độ dao động giữa các fold. Kết quả này không chứng tỏ mô hình hiểu ngôn ngữ tốt hơn, mà phản ánh việc các điểm khía cạnh là đường tắt thống kê rất mạnh dẫn tới nhãn yếu sinh từ `Rating`. Ngoài ra, bài toán đặt ra yêu cầu dự đoán cảm xúc từ văn bản tự do, trong đó năm điểm khía cạnh không tồn tại tại thời điểm suy luận. Cấu hình chính vì vậy chỉ gồm đặc trưng TF-IDF của văn bản, còn hai nhóm có chứa điểm khía cạnh được giữ lại như thí nghiệm chẩn đoán về bản chất nhãn.

Nhóm kết hợp văn bản và từ điển không cải thiện so với nhóm chỉ dùng văn bản. Khoảng dao động giữa các fold của hai nhóm chồng lấn gần như hoàn toàn, nên chênh lệch 0,0047 không có ý nghĩa phân biệt.

### 3.2.5. Hạn chế của nhóm đặc trưng từ điển

Kết quả ở mục 3.2.4 cần được diễn giải cùng với hai hạn chế của bước sinh đặc trưng từ điển, cả hai đều đã được kiểm chứng trực tiếp trên dữ liệu.

**Thứ nhất, cơ chế so khớp không tương thích với cấu trúc từ điển.** Hàm sinh đặc trưng so khớp ở mức từ đơn sau khi tách văn bản theo khoảng trắng, trong khi 135 trên 148 mục của từ điển tích cực (91%) và 130 trên 148 mục của từ điển tiêu cực (88%) là cụm nhiều từ, chẳng hạn "bảo hiểm tốt" hay "không có thưởng". Các mục này không bao giờ khớp được. Hệ quả là độ phủ chỉ đạt 12,23%, trung bình 0,16 lượt khớp trên mỗi đánh giá. Khi thay bằng cơ chế so khớp cụm ưu tiên độ dài giảm dần, độ phủ trên cùng bộ dữ liệu đạt **85,43%** với trung bình 2,37 lượt khớp, và tương quan giữa `sentiment_ratio` với `Rating` tăng từ 0,2021 lên 0,2956.

**Thứ hai, nhóm đặc trưng biểu tượng cảm xúc không mang thông tin.** Hai trường `pos_e` và `neg_e` bằng 0 trên toàn bộ dữ liệu, do bước chuẩn hóa văn bản đã thay biểu tượng cảm xúc bằng từ ngữ tương ứng trước khi bước đếm được thực hiện. Ngoài ra, chỉ 20 trên 8.417 đánh giá (0,24%) chứa biểu tượng cảm xúc thuộc hai tập từ điển, nên trần đóng góp của nhóm đặc trưng này là không đáng kể ngay cả khi cơ chế đếm hoạt động đúng.

Thực nghiệm kiểm chứng với bộ đặc trưng từ điển đã hiệu chỉnh theo cơ chế so khớp cụm cho Macro F1 đạt 0,5583 với độ lệch chuẩn 0,0131, vẫn không vượt cấu hình chỉ dùng văn bản. Như vậy, kết luận loại bỏ nhóm đặc trưng từ điển được rút ra trên bộ đặc trưng hoạt động đúng, chứ không phải hệ quả của lỗi cài đặt.

### 3.2.6. Xử lý mất cân bằng lớp

Hai chiến lược được khảo sát cho khâu mô hình hóa:

1. Sử dụng tham số `class_weight='balanced'`, gán trọng số tỷ lệ nghịch với tần suất lớp trong hàm mất mát.
2. Sinh mẫu tổng hợp bằng SMOTE, áp dụng bên trong từng fold huấn luyện. Trên tập phát triển, SMOTE đưa cả ba lớp về 4.964 mẫu, tổng cộng 14.892 mẫu. Tập kiểm thử cuối không được tái lấy mẫu trong bất kỳ trường hợp nào.

Do TF-IDF tạo không gian đặc trưng thưa và nhiều chiều, các vector tổng hợp do SMOTE sinh ra không tương ứng với văn bản có thật, làm giảm khả năng diễn giải. Việc lựa chọn giữa hai chiến lược cần dựa trên Macro F1, Recall của lớp Negative và ma trận nhầm lẫn, không dựa trên Accuracy.

### 3.2.7. Ma trận đặc trưng kết quả

| Thành phần | Kích thước | Phân bố nhãn |
|---|---|---|
| Tập phát triển | 6.730 × 5.000 | 4.964 Positive, 1.310 Neutral, 456 Negative |
| Tập kiểm thử cuối | 1.683 × 5.000 | Khóa, chỉ kiểm tra tính hợp lệ kỹ thuật |

Tỷ lệ ba lớp sau khi chia phân tầng được giữ đúng như phân bố gốc ở mục 3.1.1, xác nhận phép chia không làm lệch phân bố nhãn.

## Phụ lục: Điều kiện tái lập thực nghiệm

Toàn bộ số liệu trong tài liệu này được sinh lại từ đầu bằng môi trường khóa phiên bản theo `requirements.lock`, gồm Python 3.11.15, NumPy 2.4.6, pandas 3.0.5, scikit-learn 1.9.0, SciPy 1.17.1 và joblib 1.5.3.

| Thành phần | Đường dẫn |
|---|---|
| Quy trình thực nghiệm | `notebooks/01_data_exploration_eda.ipynb` |
| Mô-đun trích xuất đặc trưng | `src/features.py` |
| Bộ vector hóa TF-IDF | `models/text_tfidf_vectorizer.joblib` |
| Bộ trích xuất đã huấn luyện | `models/text_feature_extractor.joblib` |
| Ma trận đặc trưng và nhãn | `models/train_test_features.joblib` |
| Siêu dữ liệu và tổng kiểm tra | `models/artifact_manifest.json` |
| Biểu đồ độ phân giải 300 dpi | `reports/figures/` |
| Mẫu kiểm định nhãn thủ công | `data/annotation/` |

Tệp `artifact_manifest.json` lưu tổng kiểm tra SHA-256 của dữ liệu nguồn và của từng tệp kết quả, cùng phiên bản thư viện tại thời điểm sinh, cho phép xác minh rằng các ma trận đặc trưng tương ứng đúng với phiên bản dữ liệu được mô tả trong tài liệu này.
