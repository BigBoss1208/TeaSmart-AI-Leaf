# Dữ liệu và quy trình giảm rò rỉ

Bản xuất *My First Project v1*, ngày 30/05/2026, gồm 5.275 JPEG RGB, Auto-Orient, Stretch 224×224, không augmentation khi xuất. Train/Valid/Test ban đầu lần lượt là 3.689/1.055/531. Nhãn được kế thừa từ dataset. Chưa có kiểm định độc lập toàn bộ nhãn bởi chuyên gia bảo vệ thực vật. README của bản xuất và trang tác giả ghi CC BY 4.0. Các bảng quyết định của TeaSmart mô tả những thay đổi về chọn mẫu; ảnh nguồn do người dùng nhận từ nhà cung cấp. Tên lớp và tên ảnh gợi liên hệ với teaLeafBD, nhưng thiếu chuỗi xuất xứ cho từng ảnh. [Dataset và tác giả](https://universe.roboflow.com/rohithdraj810-gmail-com/my-first-project-yklen) · [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

1. Kiểm tra giải mã ảnh, SHA-256 cho byte tệp và pixel RGB. Hồ sơ ghi 0 ảnh lỗi và 0 cặp trùng chính xác.
2. Tạo ảnh RGB 64×64 bằng Lanczos. So A với tám hướng B (bốn phép xoay và lật ngang). Khoảng cách Hamming trên pHash DCT 32×32 và dHash ngang 9×8. Vào vòng so chi tiết khi pHash ≤ 6 hoặc đồng thời pHash ≤ 10 và dHash ≤ 8.
3. So SSIM và MAE (thang 0–255). SSIM dùng cửa sổ Gaussian 7×7, σ=1,5, bỏ ba pixel biên. Chọn hướng có SSIM lớn nhất, lấy MAE cùng hướng. Nhóm điểm cao có SSIM ≥ 0,95 và MAE ≤ 10. Vòng bổ sung lấy SSIM ≥ 0,90 và MAE ≤ 15 sau khi bỏ nhóm điểm cao.
4. Rà soát trực quan 15 cặp liên tập điểm cao, 225 cặp liên tập bổ sung và 18 cặp nội tập. Dùng đường viền, gân và vị trí tổn thương để phân loại bằng chứng. Các cặp có quan hệ được gom theo thành phần liên thông và quyết định đã duyệt. Có 50 nhóm G và 9 nhóm Q. Sau lọc còn 49 mã G có đại diện.
5. Giữ ưu tiên Test > Valid > Train; chọn một đại diện cho nhóm lặp đã xác nhận ở Valid/Test. Loại 34 ảnh gần trùng liên tập, 2 ảnh lặp nội bộ và cách ly 11 ảnh Train. Danh sách còn 5.228 ảnh.
6. Đối chiếu bản sao theo manifest, giải mã lại, kiểm SHA-256 tệp và RGB, kiểm nhóm giữa các tập và đối chiếu tính nguyên vẹn của nguồn. Sau lọc còn 165 cặp tương đồng mức thấp chưa đủ bằng chứng, trong đó 46 cặp chạm Test. Thiếu định danh lá, cây, vườn và đợt chụp để xác minh độc lập hoàn toàn.

[Bảng cặp](../data_records/2B/cap-anh.csv) · [Nhóm](../data_records/2B/nhom-anh.csv) · [Quyết định từng ảnh](../data_records/2B/de-xuat-theo-anh.csv) · [Xuất xứ mã](provenance.md).
