# BỘ CÂU HỎI PHẢN BIỆN & HƯỚNG DẪN TRẢ LỜI BẢO VỆ ĐỒ ÁN MÁY HỌC

**ĐỀ TÀI:** PHÂN TÍCH CẢM XÚC ĐÁNH GIÁ CÔNG TY ITVIEC (SENTIMENT ANALYSIS ON ITVIEC REVIEWS)  
**Môn học:** Máy học (Machine Learning) — UIT  
**Giảng viên hướng dẫn:** Thầy Cáp Phạm Đình Thăng  
**Nhóm sinh viên thực hiện:** Hoàng Hôn · Vũ Văn Duy · Nguyễn Duy Khang · Phan Thành Trung  

---

## MỤC LỤC BỘ CÂU HỎI

- [NHÓM 1: BỘ DỮ LIỆU, NHÃN & MẤT CÂN BẰNG LỚP](#nhóm-1-bộ-dữ-liệu-nhãn--mất-cân-bằng-lớp)
  - [Câu 1: Tại sao không đưa các điểm thành phần (Lương, Quản lý...) vào mô hình?](#câu-1-tại-sao-không-đưa-các-điểm-thành-phần-lương-quản-lý-vào-mô-hình)
  - [Câu 2: Bản chất nhãn suy từ số sao (Weak Label) và nhược điểm?](#câu-2-bản-chất-nhãn-suy-từ-số-sao-weak-label-và-nhược-điểm)
  - [Câu 3: Tại sao dùng SMOTE thay vì Random Undersampling?](#câu-3-tại-sao-dùng-smote-thay-vì-random-undersampling)
  - [Câu 4: Bố trí SMOTE ở đâu trong quy trình để chống rò rỉ dữ liệu (Data Leakage)?](#câu-4-bố-trí-smote-ở-đâu-trong-quy-trình-để-chống-rò-rỉ-dữ-liệu-data-leakage)
- [NHÓM 2: TIỀN XỬ LÝ, TRÍCH XUẤT ĐẶC TRƯNG & TỪ ĐIỂN LEXICON](#nhóm-2-tiền-xử-lý-trích-xuất-đặc-trưng--từ-điển-lexicon)
  - [Câu 5: Tại sao nhóm không thêm các đặc trưng Lexicon vào mô hình triển khai chính?](#câu-5-tại-sao-nhóm-không-thêm-các-đặc-trưng-lexicon-vào-mô-hình-triển-khai-chính)
  - [Câu 6: Tại sao bản Retrained v2 lại phải sửa quy tắc lọc từ dừng (Stopwords)?](#câu-6-tại-sao-bản-retrained-v2-lại-phải-sửa-quy-tắc-lọc-từ-dừng-stopwords)
  - [Câu 7: Tại sao dùng N-gram (1, 2) và cấu hình sublinear_tf = True trong TF-IDF?](#câu-7-tại-sao-dùng-n-gram-1-2-và-cấu-hình-sublinear_tf--true-trong-tf-idf)
  - [Câu 8: Tại sao nhóm dùng TF-IDF mà không dùng Pretrained Transformers (PhoBERT/ViSoBERT)?](#câu-8-tại-sao-nhóm-dùng-tf-idf-mà-không-dùng-pretrained-transformers-phobertvisobert)
- [NHÓM 3: MÔ HÌNH HỌC MÁY & CHIẾN LƯỢC HUẤN LUYỆN](#nhóm-3-mô-hình-học-máy--chiến-lược-huấn-luyện)
  - [Câu 9: Tại sao Logistic Regression lại thắng Random Forest và Stacking trên dữ liệu này?](#câu-9-tại-sao-logistic-regression-lại-thắng-random-forest-và-stacking-trên-dữ-liệu-này)
  - [Câu 10: Top 2 chênh nhau chỉ 0,0003, tại sao nhóm chọn Logistic Regression thay vì SVM?](#câu-10-top-2-chênh-nhau-chỉ-00003-tại-sao-nhóm-chọn-logistic-regression-thay-vì-svm)
  - [Câu 11: Nguyên lý của Stacking Ensemble trong bài này và vì sao không hiệu quả?](#câu-11-nguyên-lý-của-stacking-ensemble-trong-bài-này-và-vì-sao-không-hiệu-quả)
- [NHÓM 4: THỰC NGHIỆM, CHỈ SỐ ĐÁNH GIÁ & PHÂN TÍCH LỖI](#nhóm-4-thực-nghiệm-chỉ-số-đánh-giá--phân-tích-lỗi)
  - [Câu 12: Tại sao Accuracy đạt 74,33% nhưng vẫn cần Macro F1? Bẫy Accuracy là gì?](#câu-12-tại-sao-accuracy-đạt-7433-nhưng-vẫn-cần-macro-f1-bẫy-accuracy-là-gì)
  - [Câu 13: Quy trình đánh giá có vi phạm nguyên tắc Blind Test khi chạy lại trên Final Test cũ?](#câu-13-quy-trình-đánh-giá-có-vi-phạm-nguyên-tắc-blind-test-khi-chạy-lại-trên-final-test-cũ)
  - [Câu 14: Khoảng cách giữa CV Macro F1 (0,5815) và Test (0,5764) nói lên điều gì?](#câu-14-khoảng-cách-giữa-cv-macro-f1-05815-và-test-05764-nói-lên-điều-gì)
  - [Câu 15: Phân tích 15 ca lỗi minh họa: Mô hình hay sai nhất ở dạng câu nào?](#câu-15-phân-tích-15-ca-lỗi-minh-họa-mô-hình-hay-sai-nhất-ở-dạng-câu-nào)
- [NHÓM 5: TÍNH DIỄN GIẢI, TRIỂN KHAI & HƯỚNG PHÁT TRIỂN](#nhóm-5-tính-diễn-giải-triển-khai--hướng-phát-triển)
  - [Câu 16: Trọng số TF-IDF của từ cao có đồng nghĩa từ đó quyết định nhãn cảm xúc không?](#câu-16-trọng-số-tf-idf-của-từ-cao-có-đồng-nghĩa-từ-đó-quyết-định-nhãn-cảm-xúc-không)
  - [Câu 17: Phân tích cảm xúc theo doanh nghiệp có ý nghĩa gì và có phải bảng xếp hạng uy tín không?](#câu-17-phân-tích-cảm-xúc-theo-doanh-nghiệp-có-ý-nghĩa-gì-và-có-phải-bảng-xếp-hạng-uy-tín-không)
  - [Câu 18: Để nâng cao hiệu năng đồ án trong thực tế, nhóm đề xuất giải pháp kỹ thuật nào?](#câu-18-để-nâng-cao-hiệu-năng-đồ-án-trong-thực-tế-nhóm-đề-xuất-giải-pháp-kỹ-thuật-nào)

---

## NHÓM 1: BỘ DỮ LIỆU, NHÃN & MẤT CÂN BẰNG LỚP

### Câu 1: Tại sao không đưa các điểm thành phần (Lương, Quản lý...) vào mô hình?
- **Ý đồ của Thầy:** Kiểm tra xem sinh viên có phân biệt giữa bài toán suy luận thực tế (Deployment contract) và việc chạy theo điểm số ảo.
- **Câu trả lời chuẩn:**
  > *"Dạ thưa Thầy, trong bước EDA nhóm nhận thấy điểm Quản lý ($r=0,7368$) và Lương thưởng ($r=0,7343$) tương quan rất cao với rating tổng thể. Tuy nhiên, nhóm **quyết định chỉ dùng văn bản (Text-only)** vì:  
  > 1. Khi triển khai thực tế trên Web hoặc hệ thống tiếp nhận feedback tự do, người dùng chỉ nhập một đoạn văn bản review mà không bắt buộc phải chấm từng điểm khía cạnh.  
  > 2. Nếu đưa các cột điểm này vào, mô hình sẽ bị phụ thuộc hoàn toàn vào các con số định lượng (đường tắt thống kê) và bỏ qua việc học các tín hiệu ngữ nghĩa từ văn bản.  
  > Vì vậy, pipeline của nhóm tuân thủ nghiêm ngặt nguyên tắc **Text-only Input** để mô hình có tính ứng dụng thực tế cao nhất."*

---

### Câu 2: Bản chất nhãn suy từ số sao (Weak Label) và nhược điểm?
- **Ý đồ của Thầy:** Kiểm tra hiểu biết về **Weak Supervision (Giám sát yếu)** và chất lượng dữ liệu.
- **Câu trả lời chuẩn:**
  > *"Dạ thưa Thầy, nhóm quy ước: 1–2★ là Negative, 3★ là Neutral, và 4–5★ là Positive.  
  > Nhược điểm lớn nhất là đây là **nhãn yếu (Weak Label / Silver Label)** chứ không phải nhãn do chuyên gia ngôn ngữ đọc từng câu để gán:
  > - Nhiều review chấm 4★ nhưng nội dung lại phàn nàn gay gắt về chế độ OT.
  > - Nhiều review chấm 1★ nhưng chỉ ghi vài chữ cộc lốc hoặc cảm ơn xã giao.  
  > Sự không nhất quán giữa 'số sao đánh giá' và 'sắc thái chữ viết' tạo ra **nhãn nhiễu (label noise)**, đặt ra trần hiệu năng giới hạn cho mọi mô hình học máy trên bộ dữ liệu này."*

---

### Câu 3: Tại sao dùng SMOTE thay vì Random Undersampling?
- **Ý đồ của Thầy:** Kiểm tra kỹ thuật xử lý Imbalanced Data.
- **Câu trả lời chuẩn:**
  > *"Dạ, nếu dùng Random Undersampling (cắt giảm lớp Positive cho bằng lớp Negative), chúng em sẽ phải vứt bỏ hơn 70% dữ liệu quý giá, làm mất đi tính đa dạng của từ vựng và giảm nghiêm trọng độ phủ từ.  
  > Vì vậy, nhóm chọn **SMOTE (Synthetic Minority Over-sampling Technique)** kết hợp trong tập huấn luyện để sinh thêm các điểm dữ liệu nội suy nhân tạo cho lớp thiểu số (Negative và Neutral). Đồng thời, nhóm so sánh trực tiếp với phương án `class_weight='balanced'` trên cả 5 mô hình."*

---

### Câu 4: Bố trí SMOTE ở đâu trong quy trình để chống rò rỉ dữ liệu (Data Leakage)?
- **Ý đồ của Thầy:** Bắt lỗi rò rỉ dữ liệu — lỗi kinh điển nhất của sinh viên học máy.
- **Câu trả lời chuẩn:**
  > *"Dạ thưa Thầy, nhóm **tuyệt đối không áp dụng SMOTE trên toàn bộ tập dữ liệu trước khi chia fold**.  
  > Nhóm bọc SMOTE bên trong `imblearn.pipeline.Pipeline`:
  > - Trong mỗi vòng 5-Fold Cross Validation, SMOTE **chỉ được fit và resample trên 4/5 dữ liệu Train Fold**.
  > - **1/5 Validation Fold hoàn toàn giữ nguyên bản** (dữ liệu thật, không chứa bất kỳ mẫu nhân tạo nào).  
  > Nhờ vậy, điểm đánh giá CV phản ánh đúng khả năng tổng quát hóa trên dữ liệu thực tế và không bị lạc quan giả tạo."*

---

## NHÓM 2: TIỀN XỬ LÝ, TRÍCH XUẤT ĐẶC TRƯNG & TỪ ĐIỂN LEXICON

### Câu 5: Tại sao nhóm không thêm các đặc trưng Lexicon vào mô hình triển khai chính?
- **Ý đồ của Thầy:** Kiểm tra xem sinh viên có làm **Ablation Study (Nghiên cứu loại bỏ đặc trưng)** và có hiểu nguyên lý **Occam's Razor** không.
- **Câu trả lời chuẩn:**
  > *"Dạ thưa Thầy, nhóm **đã triển khai hàm trích xuất 5 đặc trưng từ điển** (`pos_w`, `neg_w`, `pos_e`, `neg_e`, `sentiment_ratio`) và tiến hành Ablation Study trên 5-Fold CV:  
  > - **Text-only TF-IDF:** Macro F1 = `0,5722` (Std: $\pm 0,0131$).  
  > - **Text + Lexicon (Hybrid):** Macro F1 = `0,5819` (Std: $\pm 0,0166$).  
  > 
  > Mặc dù điểm số tăng nhẹ $+0,0097$ (chưa tới 1%), nhưng **khoảng dao động giữa các fold chồng lấn gần như hoàn toàn**, nghĩa là mức cải thiện này **chưa có ý nghĩa thống kê** (*not statistically significant*).  
  > Ngoài ra:  
  > 1. Bản thân ma trận TF-IDF Unigram + Bigram $5.000$ chiều đã tự học rất tốt trọng số của các từ ngữ cảm xúc, việc đếm từ điển gây ra **dư thừa thông tin**.  
  > 2. Đưa Lexicon vào sẽ làm tăng gấp đôi độ phức tạp khi triển khai (*phải mang theo từ điển ngoài, thuật toán quét tham lam, MinMaxScaler, hstack ma trận*).  
  > Theo nguyên lý Occam's Razor, nhóm giữ **pipeline Text-only** để mô hình tinh gọn, ổn định và tối ưu tốc độ suy luận."*

---

### Câu 6: Tại sao bản Retrained v2 lại phải sửa quy tắc lọc từ dừng (Stopwords)?
- **Ý đồ của Thầy:** Đánh giá tính trung thực và tư duy phản biện qua quá trình cải tiến mô hình.
- **Câu trả lời chuẩn:**
  > *"Dạ thưa Thầy, ở phiên bản đầu tiên, nhóm dùng bộ từ dừng tiếng Việt mặc định có sẵn. Khi test thực tế các câu ngắn mang tính phàn nàn, nhóm phát hiện mô hình đoán sai thành Positive vì bộ stopwords đã **xóa nhầm toàn bộ các từ phủ định** (`không`, `chưa`, `chẳng`) và **từ chỉ mức độ/tiêu cực** (`thấp`, `thiếu`, `kém`, `tệ`).  
  > Ở bản Retrained v2, nhóm đã tinh chỉnh danh sách từ dừng: giữ lại tuyệt đối các từ phủ định và sắc thái, đồng thời bổ sung từ điển teencode IT (`ko` $\to$ `không`, `dc` $\to$ `được`). Kết quả là CV Macro F1 tăng từ `0,5708` lên `0,5815` và số ca đoán sai trên Final Test giảm 10 mẫu (từ 442 xuống 432)."*

---

### Câu 7: Tại sao dùng N-gram (1, 2) và cấu hình sublinear_tf = True trong TF-IDF?
- **Ý đồ của Thầy:** Kiểm tra kiến thức toán học cơ bản của TF-IDF.
- **Câu trả lời chuẩn:**
  > *"Dạ thưa Thầy:
  > 1. **N-gram (1, 2):** Kết hợp cả từ đơn (Unigram) và từ ghép 2 tiếng (Bigram). Bigram giúp bắt được các cụm từ đảo ngược sắc thái như `không lương`, `thiếu minh_bạch`, `lương thấp` mà Unigram đứng riêng lẻ không thể hiện được. Thực nghiệm cho thấy Bigram giúp tăng Macro F1 thêm $+1,53\%$.
  > 2. **`sublinear_tf = True`:** Thay vì dùng tần suất từ đếm thô $\text{tf}$, công thức chuyển thành $1 + \log(\text{tf})$ khi $\text{tf} > 0$. Điều này giúp giảm độ chi phối quá mức của những từ lặp lại nhiều lần trong một bài review dài (ví dụ từ 'công ty' xuất hiện 20 lần không có nghĩa là nó quan trọng gấp 20 lần)."*

---

### Câu 8: Tại sao nhóm dùng TF-IDF mà không dùng Pretrained Transformers (PhoBERT/ViSoBERT)?
- **Ý đồ của Thầy:** Đánh giá sự hiểu biết về phạm vi môn học và sự đánh đổi kỹ thuật (Engineering Trade-offs).
- **Câu trả lời chuẩn:**
  > *"Dạ thưa Thầy:
  > 1. Đây là đồ án **Môn học Máy học truyền thống**, nhóm ưu tiên làm chủ và hiểu sâu các kỹ thuật nền tảng: tiền xử lý tiếng Việt chuyên sâu, biểu diễn vector thưa TF-IDF, và các thuật toán học máy cổ điển có cơ sở toán học rõ ràng.
  > 2. **Tính hiệu quả và độ trễ (Latency):** TF-IDF kết hợp Logistic Regression chạy suy luận cực kỳ nhanh (dưới 10ms/câu), tốn rất ít RAM và có thể triển khai mượt mà trên môi trường CPU của Web App Streamlit mà không cần GPU đắt đỏ.
  > 3. Nhóm đã đề xuất việc fine-tune PhoBERT/ViSoBERT trong phần Hướng phát triển tương lai khi có tài nguyên phần cứng lớn hơn."*

---

## NHÓM 3: MÔ HÌNH HỌC MÁY & CHIẾN LƯỢC HUẤN LUYỆN

### Câu 9: Tại sao Logistic Regression lại thắng Random Forest và Stacking trên dữ liệu này?
- **Ý đồ của Thầy:** Kiểm tra bản chất hình học của dữ liệu và nguyên lý thuật toán.
- **Câu trả lời chuẩn:**
  > *"Dạ thưa Thầy:
  > - Ma trận đặc trưng TF-IDF có số chiều lớn ($5.000$ đặc trưng) và **rất thưa (Sparse Matrix)**.
  > - Các mô hình tuyến tính như **Logistic Regression** và **Linear SVM** rất mạnh trên không gian thưa cao chiều vì chúng tìm một siêu phẳng phân tách tuyến tính tối ưu.
  > - Ngược lại, **Random Forest** phân tách không gian bằng các trục trực giao (orthogonal splits), trên không gian $5.000$ chiều với nhiều giá trị 0, các cây quyết định rất dễ bị loãng đặc trưng, tốn chi phí tính toán nhưng hiệu quả phân tách không cao.
  > - **Stacking** không cải thiện được nhiều do các mô hình cơ sở (base models) có tương quan lỗi khá cao trên cùng một tập đặc trưng TF-IDF."*

---

### Câu 10: Top 2 chênh nhau chỉ 0,0003, tại sao nhóm chọn Logistic Regression thay vì SVM?
- **Ý đồ của Thầy:** Kiểm tra tiêu chí kỹ thuật khi ra quyết định kỹ thuật và xem sinh viên có phân biệt được cơ chế phân loại hình học (SVM) và phân loại xác suất (Logistic Regression) hay không.
- **Kịch bản trả lời tự nhiên, dễ nhớ (Dành cho Sinh viên Năm 1):**
  > *"Dạ thưa Thầy/Cô, trong quá trình thử nghiệm, nhóm thấy **SVM chạy cũng rất tốt**, điểm số của SVM và Logistic Regression là **ngang ngửa nhau** (chênh lệch chỉ $0,0003$, về mặt thống kê coi như tương đương).  
  > 
  > Tuy nhiên, nhóm quyết định chọn **Logistic Regression** vì một lý do kỹ thuật rất thực tế:  
  > 1. **Logistic Regression có sẵn xác suất phần trăm:** Mô hình này tính toán trực tiếp ra được xác suất (ví dụ câu này $90\%$ Tích cực, $10\%$ Tiêu cực) nhờ hàm Softmax, giúp nhóm vẽ thanh độ tin cậy trực quan lên Web Demo.  
  > 2. **Trong khi đó, SVM không tự tính xác suất:** Bản chất của SVM là chỉ phân chia ranh giới dứt khoát (Hard Classifier). Nếu muốn ép SVM tính ra phần trăm, nhóm phải cài thêm các thuật toán phụ (Platt Scaling), làm mô hình chạy nặng và chậm hơn.  
  > 
  > Theo nguyên lý Dao cạo Ockham, khi hai mô hình cho kết quả tốt ngang nhau, nhóm ưu tiên chọn **Logistic Regression** vì nó **nhẹ hơn, đơn giản hơn và tối ưu nhất để đưa vào ứng dụng thực tế** ạ!"*
- **Phiên bản trả lời chuyên sâu (Dành cho phản biện chi tiết về toán/mã nguồn):**
  > *"Dạ thưa Thầy:  
  > - **Về hàm mất mát:** Linear SVM tối ưu Hinge Loss nhằm tìm siêu phẳng cực đại hóa Margin, đầu ra là khoảng cách đại số (`decision_function`) chứ không mang bản chất xác suất. Muốn có xác suất cho Web, scikit-learn phải bọc qua `CalibratedClassifierCV` (chạy thêm một vòng Cross-Validation nội bộ để khớp hàm Sigmoid), làm tăng thời gian huấn luyện lên gấp 3–5 lần.  
  > - **Về Log-Loss:** Logistic Regression trực tiếp mô hình hóa phân phối xác suất hậu nghiệm $P(Y|X)$, do đó xác suất đầu ra là tự nhiên, tin cậy và không tốn thêm chi phí hiệu chuẩn."*

---

### Câu 11: Nguyên lý của Stacking Ensemble trong bài này và vì sao không hiệu quả?
- **Ý đồ của Thầy:** Kiểm tra kiến thức về phương pháp Ensemble Learning.
- **Câu trả lời chuẩn:**
  > *"Dạ thưa Thầy, mô hình Stacking của nhóm dùng 4 Base Learners (Naive Bayes, Logistic Regression, Linear SVM, Random Forest) và một Meta-Learner (Logistic Regression) để học cách kết hợp dự đoán của các mô hình cơ sở.  
  > Stacking chỉ thực sự vượt trội khi các mô hình cơ sở có **tính đa dạng cao (diversity) và mắc lỗi ở những vùng không gian dữ liệu khác nhau**. Ở đây, do cùng học trên một ma trận TF-IDF thưa, các mô hình có xu hướng đồng thuận ở các ca dễ và cùng sai ở các ca phức tạp (tương quan lỗi cao), dẫn đến việc Meta-Learner không tìm được sự kết hợp tối ưu hơn mô hình đơn lẻ tốt nhất."*

---

## NHÓM 4: THỰC NGHIỆM, CHỈ SỐ ĐÁNH GIÁ & PHÂN TÍCH LỖI

### Câu 12: Tại sao Accuracy đạt 74,33% nhưng vẫn cần Macro F1? Bẫy Accuracy là gì?
- **Ý đồ của Thầy:** Kiểm tra việc sinh viên có thực sự hiểu bản chất của thang đo hiệu năng hay không.
- **Câu trả lời chuẩn:**
  > *"Dạ thưa Thầy, đây chính là **Bẫy Accuracy (Accuracy Paradox)** do mất cân bằng lớp gây ra:
  > - Lớp Positive chiếm tới $73,8\%$ tập dữ liệu. Nếu mô hình đoán 'bừa' mọi câu đều là Positive thì Accuracy mặc nhiên đã đạt gần $74\%$, nhưng mô hình sẽ hoàn toàn mù tịt trước lớp Negative và Neutral.
  > - Vì vậy nhóm bắt buộc phải nhìn vào **Macro F1** (trung bình cộng F1 của cả 3 lớp không trọng số).
  > - Khi phân tích sâu vào từng lớp: Positive F1 đạt $0,86$, nhưng **Negative F1 chỉ đạt $0,3910$** và **Neutral F1 đạt $0,4780$**. Con số Accuracy $74,33\%$ đã che giấu sự yếu kém của mô hình trên lớp thiểu số, và Macro F1 $0,5764$ mới là con số phản ánh đúng năng lực thực tế của hệ thống."*

---

### Câu 13: Quy trình đánh giá có vi phạm nguyên tắc Blind Test khi chạy lại trên Final Test cũ?
- **Ý đồ của Thầy:** Thử thách tính liêm chính học thuật (Scientific Integrity).
- **Câu trả lời chuẩn:**
  > *"Dạ thưa Thầy, nhóm **hoàn toàn nhận thức rõ nguyên tắc Blind Test**:
  > 1. Để quyết định chọn bản sửa tiền xử lý, nhóm **hoàn toàn dựa trên kết quả 5-Fold Cross Validation trên tập Development** (điểm tăng từ $0,5708 \to 0,5815$), tuyệt đối không nhìn vào tập Test để tinh chỉnh.
  > 2. Sau khi đã chốt bản sửa bằng CV, nhóm mới đem mô hình đã fit chạy trên tập Final Test 1.683 mẫu cũ **với mục đích đối chiếu minh bạch sự thay đổi trước và sau khi sửa lỗi** (giảm 10 mẫu lỗi từ 442 xuống 432).
  > 3. Nhóm không gọi tập Test này là tập kiểm thử độc lập mới, và đã ghi chú rõ ràng điều này trong slide 9, slide 13 và báo cáo kỹ thuật để đảm bảo tính khách quan học thuật."*

---

### Câu 14: Khoảng cách giữa CV Macro F1 (0,5815) và Test (0,5764) nói lên điều gì?
- **Ý đồ của Thầy:** Kiểm tra mức độ Overfitting.
- **Câu trả lời chuẩn:**
  > *"Dạ thưa Thầy, độ lệch giữa CV và Final Test chỉ là $-0,0052$ (chưa tới $0,6\%$).  
  > Điều này chứng minh **mô hình không hề bị Overfitting**, quy trình chia tập Stratified và bọc SMOTE trong từng fold của nhóm đã kiểm soát rò rỉ dữ liệu rất tốt, giúp hiệu năng trên tập dữ liệu chưa từng thấy (unseen test data) bám rất sát với kỳ vọng trong lúc huấn luyện."*

---

### Câu 15: Phân tích 15 ca lỗi minh họa: Mô hình hay sai nhất ở dạng câu nào?
- **Ý đồ của Thầy:** Kiểm tra xem sinh viên có thực sự đọc dữ liệu và hiểu Error Analysis không.
- **Câu trả lời chuẩn:**
  > *"Dạ thưa Thầy, nhóm phân tích và thấy 3 dạng sai phổ biến nhất:  
  > 1. **Dạng 1: Review nhiều vế (Chiếm 7/15 ca) — Nguyên nhân lớn nhất:** Câu vừa khen vừa chê (*'Môi trường tốt, đồng nghiệp thân thiện nhưng lương thấp, sếp bảo thủ'*). Vì đây là mô hình phân loại đơn nhãn (Single-label), việc nén hai thái cực cảm xúc vào 1 nhãn duy nhất khiến mô hình bị giằng co xác suất và dự đoán lệch sang Neutral hoặc Positive.  
  > 2. **Dạng 2: Cấu trúc phủ định phức tạp hoặc câu quá dài (6/15 ca):** Các câu dùng từ ngữ mỉa mai, phủ định kép (*'không phải là không tốt'*). TF-IDF là dạng Bag-of-Words nên không ghi nhớ được khoảng cách ngữ cảnh xa giữa các mệnh đề.  
  > 3. **Dạng 3: Nhãn yếu bị nhiễu theo số sao (2/15 ca):** Người viết đánh giá 1 sao nhưng câu chữ lại nhẹ nhàng, trung tính."*

---

## NHÓM 5: TÍNH DIỄN GIẢI, TRIỂN KHAI & HƯỚNG PHÁT TRIỂN

### Câu 16: Trọng số TF-IDF của từ cao có đồng nghĩa từ đó quyết định nhãn cảm xúc không?
- **Ý đồ của Thầy:** Kiểm tra hiểu biết giữa **Đặc trưng (Feature Representation)** và **Trọng số mô hình (Model Weights / Coefficients)**.
- **Câu trả lời chuẩn:**
  > *"Dạ thưa Thầy, đây là một điểm nhóm đã chủ động làm rõ trong Slide 12:  
  > **Trọng số TF-IDF chỉ phản ánh mức độ nổi bật (salience/importance) của từ đó trong văn bản**, chứ **không phản ánh chiều tác động (direction of impact)** lên nhãn cảm xúc.  
  > Để biết một từ đẩy câu về hướng Negative hay Positive, ta phải kết hợp trọng số TF-IDF đó với **hệ số hồi quy $\beta$ (coefficients) tương ứng trong ma trận trọng số của Logistic Regression**. Nếu $\beta > 0$ với lớp Negative thì nó mới thực sự kéo xác suất Negative tăng lên."*

---

### Câu 17: Phân tích cảm xúc theo doanh nghiệp có ý nghĩa gì và có phải bảng xếp hạng uy tín không?
- **Ý đồ của Thầy:** Kiểm tra đạo đức dữ liệu và tính khách quan khoa học.
- **Câu trả lời chuẩn:**
  > *"Dạ thưa Thầy, nhóm đã nhấn mạnh trong Slide 11:  
  > Đây là **phân tích mô tả (descriptive analytics)** trên dữ liệu lịch sử của 5 công ty có số lượng review lớn nhất nhằm hỗ trợ phòng Nhân sự (HR) thấu hiểu góc nhìn người lao động (*ví dụ: Salary & benefits là điểm bị phàn nàn nhiều nhất ở 4/5 công ty*).  
  > Nhóm **tuyệt đối không xem đây là bảng xếp hạng chất lượng doanh nghiệp**, bởi vì cỡ mẫu review giữa các công ty rất chênh lệch (FPT có hơn 2.000 review, trong khi KMS hay VNG chỉ khoảng 250 review)."*

---

### Câu 18: Để nâng cao hiệu năng đồ án trong thực tế, nhóm đề xuất giải pháp kỹ thuật nào?
- **Ý đồ của Thầy:** Đánh giá tầm nhìn và khả năng định hướng nghiên cứu tiếp theo.
- **Câu trả lời chuẩn:**
  > *"Dạ thưa Thầy, nhóm đề xuất 3 hướng đột phá:
  > 1. **Chuyển từ Phân loại tổng thể sang ABSA (Aspect-Based Sentiment Analysis):** Tách một review thành các khía cạnh độc lập: Lương, Quản lý, Môi trường, OT để giải quyết triệt để vấn đề 'review vừa khen vừa chê'.
  > 2. **Xây dựng tập chuẩn Gold Standard (500–1.000 mẫu):** Cho chuyên gia gán nhãn tay độc lập để đo lường và lọc sạch nhiễu của nhãn yếu theo sao.
  > 3. **Thử nghiệm Pretrained Language Models tiếng Việt (PhoBERT / ViSoBERT):** Tận dụng kiến trúc Self-Attention để bắt trọn ngữ cảnh câu dài và quan hệ ngữ pháp phức tạp."*

---

## NGUYÊN TẮC VÀNG KHI TRẢ LỜI PHẢN BIỆN TRƯỚC HỘI ĐỒNG

1. **Thái độ khoa học & Điềm tĩnh:** Khi Thầy chỉ ra điểm yếu (ví dụ: lớp Negative F1 thấp, nhãn theo sao bị nhiễu), hãy thẳng thắn thừa nhận đó là hạn chế cố hữu của bài toán và trình bày cách nhóm đã nhận diện, phân tích lỗi chứ không vòng vo chối bỏ.
2. **Nói có sách, mách có số liệu:** Sử dụng chính xác các con số trong đồ án:  
   - CV Macro F1: `0,5815`
   - Final Test Macro F1: `0,5764` (độ lệch $-0,0052$)
   - Độ chính xác: `74,33%`
   - Số mẫu lỗi: `432 / 1.683` (giảm 10 mẫu lỗi so với bản cũ)
   - Chênh lệch top 2 CV: `0,0003` giữa LR và SVM.
3. **Mẫu câu mở đầu chuẩn tác phong UIT:**  
   *"Dạ, em cảm ơn câu hỏi rất hay của Thầy. Về vấn đề này, nhóm xin phép được giải thích theo hai khía cạnh..."*
