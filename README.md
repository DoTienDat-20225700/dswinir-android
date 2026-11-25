# DSwinIR Android Lab

Đã có app Android 10+, ba model TFLite FP32 đã kiểm tra đầu ra trên Android 10 và 15 ảnh test có nguồn rõ ràng. App hỗ trợ ghép tile 256×256, benchmark CPU/GPU với 5 warm-up + 50 lần đo, YOLO detection và instance segmentation độc lập.

**Chưa hoàn thành thí nghiệm mAP đủ năm tác vụ:** checkpoint tìm được là Rain100L **đơn tác vụ**, bộ ảnh chưa có nhãn bbox/mask, và chưa đo benchmark 5+50 trên điện thoại có GPU. Không coi kết quả kiểm tra số học là kết quả chất lượng của bài báo.

## Nội dung repository public

Repository chứa mã nguồn Android, công cụ chuyển đổi, các mô hình giả nhỏ dùng cho kiểm thử thành phần và báo cáo kiểm chứng. **Không chứa trọng số đã huấn luyện, model TFLite thật, ảnh test gốc, APK hoặc bộ máy ảo**. Các đường dẫn tới những tệp này bên dưới là đường dẫn đầu ra trên máy sau khi tải/chuyển đổi, không phải tệp đã đưa lên GitHub.

- Build app: cài Java 17, Android SDK 37; đặt `ANDROID_HOME` trỏ tới SDK rồi mở thư mục `android/` bằng Android Studio hoặc dùng lệnh bên dưới.
- Tải checkpoint Rain100L: cài `gdown` và PyTorch rồi chạy `sh download_weights.sh`. Chuyển đổi theo mục **Chuyển đổi và kiểm chứng** hoặc notebook Colab.
- Hai model nhận diện được tạo bằng `convert/prepare_yolo.py` và `convert/convert_yolo.py`.
- Tải ảnh thử và tạo bộ kiểm tra theo mục **Bộ thử 15 ảnh**.
- App có thể build khi chưa có model. Để khôi phục/nhận diện, cài các tệp model ở mục **Thiết lập** trong app.

Nguồn mã và mô hình tham khảo: [THIRD_PARTY.md](THIRD_PARTY.md).

## Chạy ngay trong Antigravity

Mở workspace này, nhấn **Cmd+Shift+B** để chạy task **Android: Build và chạy DSwinIR**. Task build mã nguồn hiện tại, mở cửa sổ emulator Android10 rồi cài/mở app. Không cần mở Android Studio.

Hoặc chạy trong terminal tích hợp:

```sh
sh tools/run_android.sh
```

Muốn mở nhanh APK đã build: `sh tools/run_android.sh --apk`. Có thể chọn task này qua Command Palette → Tasks: Run Task → Android: Mở APK có sẵn. Emulator hiển thị ở cửa sổ riêng; Antigravity dùng để sửa code, build và điều khiển chạy. Task mặc định không xóa dữ liệu của máy ảo hoặc đổi model người dùng đã nhập.

## Mở và chạy

Mở `android/` bằng Android Studio, dùng Java 17 và SDK 37, Gradle Sync rồi Run. `minSdk=29` (Android 10). Có thể build bằng:

```sh
cd android
./gradlew :app:assembleDebug :app:testDebugUnitTest :app:lintDebug
```

APK: [app-debug.apk](android/app/build/outputs/apk/debug/app-debug.apk).

| File để nhập vào app | Tác dụng |
|---|---|
| [dswinir_256_fp32.tflite](convert/prepared256/dswinir_256_fp32.tflite) | Khôi phục RGB, FP32, 256×256; checkpoint Rain100L single-task |
| [yolo11n_fp32.tflite](convert/yolo/detection/yolo11n_fp32.tflite) | COCO80 object detection, input 640×640 |
| [yolo11n-seg_fp32.tflite](convert/yolo/segmentation/yolo11n-seg_fp32.tflite) | COCO80 instance segmentation, input 640×640 |
| [testset-15-unlabelled.zip](deliverables/testset-15-unlabelled.zip) | Ba ảnh test mỗi suy thoái, chưa có nhãn object |

Chép các file sang điện thoại rồi nhập bằng các nút tương ứng. Chọn ảnh → Khôi phục CPU/GPU → xem ảnh gốc và ảnh khôi phục. Hai nút YOLO chạy detection và segmentation riêng. Với ZIP chưa có nhãn, dùng **Khôi phục tập ảnh ZIP**; app chặn mAP khi `annotation_status=pending`, không biến ảnh chưa gán nhãn thành ảnh negative.

Chọn suy thoái chỉ ghi metadata, không đổi trọng số. Dùng Rain100L single-task trên noise/haze/blur/low-light là thử ngoài phân phối, **không tái lập checkpoint five-task**.

Benchmark đo tile đầu 256×256, warm-up đúng 5 lần, đo 50 lần và hiện trung bình trong app. JSON xuất đủ samples, thiết bị, hash model và log. Chỉ xác nhận full/partial GPU khi có bằng chứng runtime; không xuất hiện thì N/A, thiếu log thì UNKNOWN. Số luồng CPU là `max(1, min(4, số lõi khả dụng - 1))` (máy ảo 2 lõi dùng 1 luồng); DSwinIR tắt XNNPACK vì đã quan sát OOM do hàng trăm partition trên emulator 2 GB; YOLO dùng cấu hình tối ưu riêng. Thời gian native parity trong log chỉ là từng lần invoke, **không phải** benchmark 5+50.

## Bộ thử 15 ảnh

| Suy thoái | Nguồn và lựa chọn |
|---|---|
| Noise | CBSD68 từ FFDNet của Kai Zhang; ba ảnh đầu theo tên; Gaussian σ=25, seed 42–44, round/clip PNG uint8 cho mobile |
| Rain | Rain100L test ZIP được repo DSwinIR liên kết; rain-001/002/003 và ảnh sạch tương ứng |
| Haze | RESIDE SOTS outdoor từ trang tác giả; ba scene đầu; JPEG nguồn giải mã rồi lưu PNG lossless |
| Blur | GoPro test từ archive do tác giả liên kết; một frame từ mỗi sequence đầu trong ba sequence khác nhau |
| Low-light | Ba ảnh trong `data/eval/low` của RetinexNet tác giả; repo subset không có ảnh sạch kèm theo |

Danh sách, nguồn từng file, SHA-256, seed và phạm vi thử nằm trong [manifest](data/pilot/manifest.json). Có 12 cặp ảnh sạch kèm theo; ảnh sạch không phải nhãn object. Quy ước PNG noise là pilot cho mobile, chưa khẳng định trùng protocol float của bài báo.

Tải lại và đóng gói:

```sh
PYTHONPATH=.export-deps python3 tools/download_pilot.py
python3 tools/pack_testset.py data/pilot/manifest.json --out deliverables/testset-15-unlabelled.zip --require-five-tasks --allow-unlabelled
```

Để đo mAP, gán nhãn đúng COCO class 0..79 theo [schema](docs/testset-format.json), bỏ trạng thái pending sau khi kiểm chứng. Không dùng dự đoán YOLO làm ground truth. Protocol AP101, IoU .50:.05:.95 được mô tả trong [trạng thái triển khai](docs/implementation-status.md); chưa hỗ trợ đầy đủ crowd/ignore/area của COCOeval.

## Chuyển đổi và kiểm chứng

```sh
PYTHONPATH=.export-deps python3 convert/prepare_tf_weights.py --checkpoint weights/rain100l_official.pth --out convert/prepared256 --size 256
venv/bin/python convert/direct_tf.py --prepared convert/prepared256
```

Đường mới PyTorch → TensorFlow → TFLite builtins, không Flex, không quantization. Attention chia nhóm nhỏ, giữ rank ≤5 cho phần slice và tránh bias bị constant-fold thành bảng không gian lớn. Chỉ ghi model sau khi hai fixture đều đạt max absolute error ≤1e-3. [Notebook Colab](convert/DSwinIR_Colab.ipynb) dùng source bundle từ `python3 tools/package_conversion.py`; notebook chưa được chạy trên tài khoản Google của người dùng.

YOLO dùng Ultralytics 8.3.221, output raw normalized xywh, `nms=False`. `convert/prepare_yolo.py` tạo ONNX và reference; `convert/convert_yolo.py` chuyển TFLite, chuẩn hóa layout output cho app và chặn file nếu parity không qua. Source và môi trường chính xác nằm trong báo cáo chuyển đổi.

Các lỗi LayerNorm/offset/biên NATTEN của port cũ đã được sửa. Không dùng fixture/log cũ trong `convert/out/` làm bằng chứng chất lượng. Vẫn cần đối chiếu với CUDA NATTEN gốc và cấu hình huấn luyện tác giả trước khi khẳng định tái lập bài báo.

## Bằng chứng

- [Android 10: 5 test qua](docs/verification/android10-instrumentation.txt), gồm bốn kiểm tra thành phần và một test chạy sáu fixture của ba model thật.
- [Sai số model trên Android 10](docs/verification/android10-real-model-parity.json).
- [Ảnh khôi phục thật](docs/verification/android10-rain100l-restoration.png), [detection](docs/verification/android10-rain100l-detection.png), [segmentation](docs/verification/android10-rain100l-segmentation.png) qua giao diện app; một ảnh Rain100L, chưa phải kết quả15 ảnh/mAP.
- [Hash và kích thước ba model](docs/verification/model-artifacts.json).
- [Build Android](docs/verification/android-build.txt), 7 Java unit tests qua, lint 0 errors; [3 Python tests qua](docs/verification/python-tests.txt).
- Ba báo cáo conversion trong `docs/verification/` giữ sai số, hash checkpoint và phạm vi chưa đánh giá mAP.

Các kiểm tra này không chứng minh mAP tăng hoặc hiệu năng điện thoại. Chi tiết còn thiếu, nguồn giấy tờ và protocol tại [implementation-status.md](docs/implementation-status.md).

Ảnh đơn không cần ZIP: sau khi chọn ảnh, dùng **3. Khôi phục ảnh này bằng CPU**. App hiện model/ảnh đã nạp và tiến độ từng tile ở đầu màn hình. Nút YOLO chỉ bật sau khi có ảnh khôi phục; nút mAP chỉ bật khi ZIP có nhãn đúng loại và model tương ứng đã nạp. Kiểm tra nhánh ảnh đơn và nhãn bbox/mask đã qua6 tests trong `docs/verification/android10-ui-readiness-tests.txt`.

Kiểm tra lại ngày 25/11/2026 trên máy ảo Android 10: ảnh Rain100L 481×321 đã chạy đủ 4 tile, giữ nguyên kích thước và bật nút YOLO sau khi xong. Thời gian toàn bộ thao tác CPU lần này là 307,3 giây; đây là một lần khôi phục ảnh, chưa phải phép đo warm-up 5 / chạy 50. Xem [log tiến độ](docs/verification/android10-ui-smoke-current.txt) và [trạng thái hoàn tất](docs/verification/android10-ui-restored.png).

Đã chạy tiếp detection và segmentation độc lập trên ảnh trước/sau trong bản giao diện đã sửa: [log YOLO](docs/verification/android10-ui-yolo-current.txt), [ảnh khôi phục](docs/verification/android10-ui-restored-result.png), [trạng thái detection](docs/verification/android10-ui-detection-status.png), [segmentation](docs/verification/android10-ui-segmentation-result.png). Các nút mAP vẫn bị khóa khi chưa có nhãn chuẩn.

Nếu máy ảo giữ khung hình cũ dù thao tác vẫn chạy, nhấn nút Power của cửa sổ emulator để tắt rồi bật màn hình. Đã dùng cách này để làm mới hiển thị khi kiểm tra segmentation; không cần chạy lại khôi phục.

Launcher đã thêm `-feature -Vulkan` cho máy ảo của dự án theo [tùy chọn của Android Emulator](https://developer.android.com/studio/run/emulator-troubleshooting). Sau khi khởi động lại, đã nạp lại ảnh và xác nhận khung hình thay đổi khi cuộn; bằng chứng tại [kiểm tra hiển thị](docs/verification/android10-emulator-display-check.json). Đây là cấu hình hiển thị máy ảo, không xác nhận khả năng chạy LiteRT trên GPU điện thoại.


Giao diện DSwinIR Studio được tổ chức thành ba mục: **Khôi phục** (chọn ảnh, chạy CPU/GPU, so sánh trước/sau, xuất kết quả), **Đánh giá** (YOLO, dataset/mAP và benchmark), **Thiết lập** (nhập ba model). Giao diện dùng nền sáng, thẻ bo góc, tiêu đề xanh đậm và nút xanh ngọc; trạng thái xử lý có thanh tiến độ khi đang chạy. Sau khi xem YOLO, app chuyển về mục Khôi phục để hiển thị ảnh trước/sau.

Ảnh giao diện: [Khôi phục](docs/verification/ui-studio-restore.png), [Đánh giá](docs/verification/ui-studio-evaluate.png), [Thiết lập](docs/verification/ui-studio-models.png). Bản giao diện này đã build, qua unit tests và lint 0 errors; 6/6 kiểm thử Android 10 qua trong `docs/verification/android-ui-redesign-tests.txt`. Không chạy lại benchmark hoặc model parity vì phần tính toán không đổi.


Câu chữ giao diện đã được đổi sang tiếng Việt phổ thông: chọn ảnh, khôi phục ảnh, khoanh vật thể và tô vùng vật thể. Năm loại ảnh được ghi bằng tiếng Việt; mã tác vụ trong báo cáo vẫn giữ nguyên. Phần đánh giá giải thích đáp án đúng và hiển thị điểm trước/sau thay cho JSON thô. CPU/GPU được giải thích là bộ xử lý chính/bộ xử lý đồ họa; số liệu chi tiết vẫn nằm trong báo cáo xuất ra. Bản sửa đã build, chạy unit tests và lint (log `docs/verification/android-wording-build.txt`).


Sửa luồng xem kết quả: nút “Bắt đầu khôi phục” nằm cố định ở cuối màn hình Khôi phục; lúc chạy có bộ đếm thời gian chờ và giữ màn hình sáng. Khi xong, app cuộn đến ảnh sau khôi phục. Ảnh đầu vào, đầu ra và báo cáo của phiên gần nhất được lưu trong bộ nhớ riêng để mở lại sau khi khởi động app. Lần chạy người dùng báo không thấy kết quả thực tế đã hoàn tất: CPU, 481×321, 926586,494 ms (một lần khôi phục toàn ảnh, không phải benchmark 5+50). Kết quả gốc đã xuất ra `docs/verification/user-restoration-result.zip`.


Tác vụ xử lý và phép đo dài có nút **Dừng xử lý**. Phép đo 5 lần chạy thử + 50 lần đo có hộp giải thích trước khi bắt đầu và báo số lần đang chạy. Khi dừng, dữ liệu đo chưa đủ không được công bố thành kết quả; ảnh/báo cáo đã lưu được giữ nguyên. LiteRT bật cơ chế hủy inference; giao diện không đóng interpreter khi nó đang chạy. Số luồng CPU hiện là `max(1, min(4, số lõi khả dụng - 1))` cho cả DSwinIR và YOLO; máy ảo 2 lõi dùng 1 luồng. Báo cáo benchmark ghi số luồng thực tế, không so trực tiếp với các lần đo cũ dùng 4 luồng.
