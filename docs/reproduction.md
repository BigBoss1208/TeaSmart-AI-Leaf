# Tái lập trên Windows

Để xem lại các kết quả đã lưu, mở thư mục `results/`. Chỉ các bước dưới đây mới cần ảnh, GPU hoặc bộ checkpoint. Đặt thư mục repo cùng cấp với `Dataset/`. Dùng Python 3.12.10, NVIDIA RTX 2050 (4 GB), driver hỗ trợ wheel CUDA 12.6. Môi trường huấn luyện gốc dùng torch 2.12.1+cu126, torchvision 0.27.1+cu126, Pillow 13.0.0 và filelock 4.2.4. Tách môi trường này khỏi FastAPI/PhoBERT.

```powershell
py -3.12 -m venv .venv
$py = (Resolve-Path .venv/Scripts/python.exe).Path
& $py -m pip install -r requirements.txt
& $py -m pip check
& $py -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Để đọc các đoạn rà soát ảnh hoặc vẽ biểu đồ, cài thêm NumPy, OpenCV và Matplotlib trong môi trường phù hợp. Những thư viện này được dùng bởi script phụ. Phiên bản lịch sử chính xác của môi trường sàng lọc chưa xác minh.

## Chuẩn bị tập sạch

Tải đúng bản xuất Roboflow v1 theo giấy phép, giải nén nguyên cây vào `../Dataset/tea-leaf-roboflow-v1/My First Project.v1i.folder`, gồm train, valid, test và hai README gốc. Để dùng bản sạch đã bàn giao, đặt nguyên cây cùng manifest ở `../Dataset/tea-leaf-clean-v1`.

Để tạo bản sạch từ bảng quyết định công khai, bắt đầu ở thư mục repo trong PowerShell và dùng **đường dẫn đích chưa tồn tại**. Script 17453 chỉ kiểm tra đầu vào; 17480 ghi bản sao sau khi fingerprint khớp. Fingerprint gồm thông tin hệ thống tệp nên phải lấy từ chính bản nguồn vừa giải nén. Bản báo cáo công khai 2B đã lược đường dẫn nội bộ; hash của riêng báo cáo này khác hồ sơ lưu trữ. Ba CSV quyết định giữ nguyên byte. Các lệnh dưới đây là hướng dẫn cho người tái lập.

```powershell
$parent = (Resolve-Path ../Dataset).Path
$env:TEASMART_2C_SOURCE = (Resolve-Path '../Dataset/tea-leaf-roboflow-v1/My First Project.v1i.folder').Path
$env:TEASMART_2C_DEST = Join-Path $parent 'tea-leaf-clean-v1'
$env:TEASMART_2C_AUDIT = (Resolve-Path data_records/2B).Path
$env:TEASMART_2C_MODE = 'verify'
$proof = & $py scripts/dataset_cleaning/17453-doi-chieu-truoc-sao-chep-2C.py
if ($LASTEXITCODE -ne 0) { throw 'Kiểm tra dữ liệu nguồn thất bại' }
$line = $proof | Where-Object { $_ -like 'SOURCE_OK *' }
$env:TEASMART_2C_EXPECTED_FP = ($line.Substring(10) | ConvertFrom-Json).fingerprint
$env:TEASMART_2C_MODE = 'copy'
& $py scripts/dataset_cleaning/17480-sao-chep-va-xac-minh-2C.py
if ($LASTEXITCODE -ne 0) { throw 'Sao chép chưa hoàn tất' }
$env:TEASMART_DATASET_PATH = $env:TEASMART_2C_DEST
& $py scripts/dataset_cleaning/17512-kiem-tra-ban-sao-2C.py
```

Để khảo sát từ đầu thay cho dùng ba CSV đã duyệt, 17303 nhận TEASMART_DATASET_PATH trỏ tới bản xuất gốc và TEASMART_LEAKAGE_OUTPUT trỏ tới một thư mục mới nằm ngoài dataset. 17386 đọc các cặp và contact sheet từ cùng thư mục, dùng quyết định thị giác đã lưu để gom nhóm. Hồ sơ xuất bản giữ kết quả của lần rà soát gốc.

## Huấn luyện

```powershell
& $py train.py --check
& $py train.py --run-name experiment-01 --epochs 30
& $py plot_training.py outputs/experiment-01
```

Lần đầu lấy trọng số ImageNet từ Torchvision. Để tiếp tục một lần bị ngắt, dùng `train.py --run-name experiment-01 --epochs 30 --resume`. Mỗi lần chạy mới cần tên khác. Batch 8, num_workers 0, AMP và seed 42 nằm trong config.py. Seed giúp kiểm soát ngẫu nhiên; độ tái lập số học còn phụ thuộc GPU, driver và phiên bản thư viện.

## Checkpoint lưu trữ và Test

Nhận ba tệp gốc theo [manifest](../MODEL_ARTIFACTS.json) và đặt vào `outputs/mobilenetv2-30epochs-20261119-215540/`. So checksum riêng từng tệp bằng `Get-FileHash -Algorithm SHA256`. `configs/experiment.json` là bản tham khảo đã bỏ đường dẫn máy, có checksum khác config.json gốc. Có thể dùng lệnh sau với GPU để kiểm tra checkpoint bằng tensor tổng hợp.

```powershell
& $py evaluate.py verify
```

Chế độ `evaluate.py test` chỉ dành cho **một lần đánh giá độc lập có chủ đích** trên đúng bộ Test và checkpoint lưu trữ. Evaluator cố định epoch, SHA-256, mapping và kích thước từng lớp, từ chối ghi đè thư mục đánh giá. Bản công khai chỉ điều chỉnh phép so tên thư mục dataset trong metadata để chuyển máy. Checkpoint mới từ một lần huấn luyện khác sẽ có SHA khác và cần kế hoạch đánh giá riêng. Đợt xuất bản không chạy lại huấn luyện hoặc Test.

Để dùng website, bàn giao ba tệp model cho Backend rồi đặt `LEAF_MODEL_DIR`. FastAPI dùng RGB, resize 224×224 và chuẩn hóa ImageNet như lúc đánh giá. [Hướng dẫn chạy FastAPI của TeaSmart](https://github.com/BigBoss1208/TeaSmart-Be/tree/main/ai-service/leaf_app).
