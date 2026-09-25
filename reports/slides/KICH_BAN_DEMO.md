# Kịch bản demo web ML

## 0:00–0:30 — Chào và giới thiệu luồng

> Em xin chào thầy và các bạn... em tên Phạm Thành Trung, đại diện nhóm trình bày phần demo.

## 0:30–1:15 — Insight doanh nghiệp

> Đầu tiên... em mở trang Insight doanh nghiệp. Trang này cho thấy tỷ lệ review Tích cực, Trung tính, Tiêu cực của từng công ty... cùng điểm đánh giá về lương, quản lý và văn hóa.
>
> Phía dưới là bản đồ từ khóa, cho biết những từ nào xuất hiện nhiều trong review. Em có thể chuyển giữa nhóm Tích cực và Tiêu cực để xem từng nhóm.
>
> Từ càng lớn thì xuất hiện càng nhiều. Muốn hiểu rõ ý kiến của người viết, người dùng cần xem nội dung review.

## 1:15–2:45 — Mô hình được chọn và lỗi còn gặp

> Tiếp theo... em chuyển sang mục Mô hình & đánh giá để xem nhóm đã chọn model như thế nào.
>
> Ban đầu, nhóm so sánh năm mô hình; Logistic Regression nhỉnh hơn Linear SVM một chút... nên được chọn.
>
> Sau khi sửa tiền xử lý... nhóm huấn luyện lại mô hình này. Bản app đang dùng đạt CV Macro F1 khoảng 0,582. Biểu đồ phía trên vẫn là kết quả so sánh ban đầu.

---

> Tiếp theo là kết quả trên 1.683 review kiểm tra... model đạt Macro F1 khoảng 0,576.
>
> Nhìn vào hàng Tiêu cực: trong 114 review Tiêu cực... model nhận đúng 52 và bỏ sót 62 review. Vậy nên kết quả chung chưa nói hết khả năng nhận diện lời chê.
>
> Đây là kết quả kiểm tra lại trên cùng tập Final Test trước đó.

**Thao tác:** Cuộn tiếp đến **Khám phá 15 lỗi minh họa**. Lọc **Nhãn thật: Negative**, rồi chọn mẫu `#2592 · FPT Software`.

> Em mở một review model dự đoán sai để xem cụ thể hơn. Review này được xếp vào nhóm Tiêu cực theo số sao... nhưng model lại đoán Trung tính.
>
> Đọc nội dung thì thấy người viết khen môi trường, đồng nghiệp... đồng thời chê lương thấp và hay phải OT. Cả lời khen lẫn lời chê trong cùng một review có thể khiến model nhầm.
>
> Dạng lỗi hiển thị trên app chỉ là gợi ý tự động... nhóm chưa kiểm tra thủ công nguyên nhân của mẫu này.

## 2:45–6:00 — Dự đoán review mới

> Bây giờ... em thử phân tích một review mới.
>
> Người dùng có thể nhập câu tùy ý... hoặc bấm một tình huống có sẵn.
>
> App sẽ hiển thị nhãn dự đoán... và xác suất của cả ba lớp.

### Mẫu 1 — Tích cực

> Đầu tiên... em thử một lời khen về môi trường làm việc, đồng nghiệp và cơ hội học hỏi.
>
> Model dự đoán Tích cực... khoảng 66,7%. Ba thanh bên dưới cho thấy model nghiêng về từng nhóm cảm xúc với review này và đó không phải độ chính xác chung của model.

---
> Bên trái là câu gốc... và câu sau khi chuẩn hóa, tách từ.
>
> Chẳng hạn... “môi trường” được nối thành `môi_trường`.
>
> Văn bản này được đưa vào bộ TF-IDF **đã học từ dữ liệu huấn luyện**...
>
> app không học lại từ câu vừa nhập.

---

> Bên phải là những từ hoặc cụm từ nổi bật trong vector TF-IDF.
>
> Trọng số cao cho biết chúng nổi bật ở đầu vào...
>
> chứ không có nghĩa riêng một từ quyết định nhãn.
>
> Vector đó được đưa vào Logistic Regression... để tạo ba xác suất mình vừa xem.

### Mẫu 2 — Trung tính

> Tiếp theo... em thử một review nhẹ hơn: công việc ổn, quy trình bình thường, chưa có gì quá nổi bật.
>
> Vì câu này không khen hay chê rõ rệt... model dự đoán Trung tính, khoảng 93,1%.

### Mẫu 3 — Tiêu cực

> Bây giờ em thử một lời chê rõ hơn: lương thấp, quản lý thiếu minh bạch và thường xuyên OT không lương.
>
> Model dự đoán Tiêu cực... khoảng 99,0%. Ở câu này, những từ như “thấp”, “thiếu” và “không” rất quan trọng để hiểu đúng lời phàn nàn, nên nhóm giữ chúng ở bước tiền xử lý.

### Mẫu 4 — Nhiều vế

> Cuối cùng... em thử một review vừa khen môi trường tốt, vừa chê lương thấp và quản lý chưa quan tâm.
>
> Model dự đoán Trung tính... khoảng 58,7%, nhưng Tiêu cực cũng khoảng 30%. Vì câu có cả khen lẫn chê... người dùng nên đọc nội dung cùng ba thanh xác suất để hiểu kết quả, thay vì chỉ nhìn nhãn Trung tính.

## 6:00–6:30 — Kết thúc

> Vừa rồi... em đã demo cách xem cảm xúc theo doanh nghiệp, kiểm tra những lỗi model còn gặp và phân tích review mới.
> Phần demo của em đến đây là hết. Em cảm ơn thầy và các bạn đã theo dõi.
