# DSwinIR trên Android — trạng thái và protocol

Kiểm chứng model: 23/11/2026; cập nhật luồng ứng dụng và tài liệu: 25/11/2026. Đây là mã triển khai và công cụ kiểm chứng; chưa phải báo cáo thí nghiệm hoàn tất.

## Xác định đúng bài và checkpoint

PDF người dùng cung cấp, trang 19: [242] là **Content-aware transformer for all-in-one image restoration**, arXiv:2504.04869. Bản arXiv mới đổi tiêu đề thành **DSwinIR: Rethinking Window-based Attention for Image Restoration**. Trang 12, bảng III của survey định nghĩa năm tác vụ:

| Tác vụ | Bộ test | Ghi chú |
|---|---|---|
| Denoising | BSD68 | Gaussian noise σ=25 cho bảng five-task; lưu seed và quy ước noise |
| Deraining | Rain100L | Dùng split test, không lấy ảnh train |
| Dehazing | RESIDE/SOTS | Ghi rõ indoor/outdoor và split theo phiên bản bài |
| Motion deblurring | GoPro | Cặp blur/sharp cùng frame trong test |
| Low-light enhancement | LOL | Xác định đúng LOLv1 và cặp low/normal test |

Snow, raindrop và outdoor rain là benchmark weather khác. Không thay snow vào nhóm năm tác vụ trên.

Nguồn: [repo tác giả](https://github.com/Aitical/DSwinIR), [bài DSwinIR](https://arxiv.org/html/2504.04869v3).

README chính thức có bảng **Results** chứa ZIP ảnh và mục **Pretrained/Model** cho Rain100L single-task. Sáu file cũ có đuôi `.pth` thực tế là ZIP ảnh. Chi tiết tại `weight-inventory.json`. Giữ nguyên các file cũ để không làm mất dữ liệu của người dùng.

Đã tải checkpoint từ [Model Rain100L chính thức](https://drive.google.com/file/d/1yQcnuRUepg9NW5DVh3MmBFlA-H1r7y7E/view): `weights/rain100l_official.pth`, SHA-256 `02ab7327c7f6f11d0db0f5c657c31d59e1f6f4e6791bb77c6d52fbc214666645`. Trùng byte với file cũ `w_rain100l_result.pth`. Không suy luận tác vụ từ tên file cũ.

**Chưa tìm thấy link checkpoint five-task được xác định rõ trong README. Người dùng xác nhận không có link bổ sung.** Chạy Rain100L single-task trên blur/low-light chỉ là kiểm tra ngoài phân phối, không tái lập kết quả five-task. Không tự huấn luyện lại hay thay mô hình khác để lấp chỗ thiếu.

## Các lỗi đã sửa trong bản port cũ

- `LayerNorm2d`: chuẩn hóa chiều channel tại từng pixel, khớp [NAFNet LayerNorm2d](https://github.com/megvii-research/NAFNet/blob/main/basicsr/models/archs/arch_util.py).
- Offset: đổi `B,2,H,W` sang `B,H,W,2` bằng permute; reshape không tương đương.
- Neighborhood attention: dịch cửa sổ vào trong biên, giữ residue của dilation, thay cho zero-padding. Công thức theo [NATTEN v0.17.1](https://github.com/SHI-Labs/NATTEN/blob/v0.17.1/csrc/include/natten/cuda/naive/natten_commons.cuh).
- PSNR cũ so sai ảnh GT và sai layout; nay yêu cầu input/GT khác thư mục, HWC và range [0,1].

Strict state-dict loading chỉ chứng minh khớp tên/shape. Phải kiểm thêm parity với NATTEN gốc trên CUDA và PSNR/SSIM trên cặp test thật trước khi dùng trong báo cáo học thuật. Giữ `global_residual=True` như mặc định mã gốc; flag này không thể suy ra từ state-dict, cần xác nhận cấu hình huấn luyện của checkpoint.

## Android

Mở thư mục `android/` bằng Android Studio, Gradle Sync, chọn thiết bị API >=29. Java 17, Gradle wrapper 9.5.0, compile/target API37, minSdk29. LiteRT Interpreter 1.4.2 (TensorFlow Lite kế thừa, vẫn đọc `.tflite`), GPU cùng phiên bản, [hỗ trợ Android API21 trở lên](https://developers.google.com/edge/litert/android). Không cần tài khoản/server để suy luận.

Nhập file TFLite bằng Storage Access Framework, chọn ảnh, chọn suy thoái để ghi metadata. Lựa chọn suy thoái không thay đổi trọng số. DSwinIR nhận FP32 RGB [0,1], batch1, [1,256,256,3] hoặc [1,3,256,256]. Ảnh chia lưới không overlap, cạnh dư replicate-pad rồi crop, ghép đúng kích thước ban đầu; cách này có thể tạo seam và khác suy luận toàn ảnh. Không resize ảnh restoration về 256. YOLO có bước letterbox riêng.

CPU/GPU delegate được tạo, gọi và hủy trên cùng worker thread theo [hướng dẫn Google](https://developers.google.com/edge/litert/android/gpu). GPU FP32 không cho phép precision loss. GPU có thể không hỗ trợ một số op và chạy xen CPU; [tài liệu GPU](https://developers.google.com/edge/litert/performance/gpu) nêu rõ successful initialization không đảm bảo toàn bộ graph trên GPU.

Benchmark: tile đầu tiên của ảnh, encode sẵn; warm-up đúng 5 lần; đo 50 lần `Interpreter.run` đồng bộ, gồm output copy, trung bình và toàn bộ 50 samples. Init, encode, ghép ảnh không nằm trong số ms/tile. Thời gian restore cả ảnh gồm khởi tạo được báo riêng. CPU dùng `max(1,min(4,availableProcessors-1))` luồng (máy ảo2 lõi dùng1); DSwinIR tắt XNNPACK. Graph đã tạo 353 partitions với XNNPACK và làm Android10 emulator2GB kill tiến trình ở khoảng1.35GB PSS; tắt XNNPACK giúp dùng chung arena và đã qua kiểm chứng model thật. GPU cũng tắt XNNPACK. YOLO vẫn dùng XNNPACK riêng. Nếu GPU attempt lỗi, nút restore GPU tự fallback CPU và ghi lỗi; benchmark CPU riêng vẫn có số đo để đối chiếu.

Mức phân bổ lấy từ log runtime **của tiến trình app, trong khoảng benchmark đó**. `FULL_GPU`, `PARTIAL_GPU`, `CPU_ONLY` chỉ hiện khi log cung cấp bằng chứng phù hợp; nếu log không có thông tin sẽ hiện `UNKNOWN_GPU_COVERAGE`. Không coi “khởi tạo GPU thành công” là “full GPU”. Có đọc số toán tử từ FlatBuffer một subgraph để đối chiếu log `Replacing N node(s)`; model nhiều subgraph bị từ chối. Chưa có native execution-plan inspector; runtime không phát count log vẫn cần benchmark binary/profiling thêm. Không thể ép mọi điện thoại xuất hiện đủ ba tình huống với cùng một graph; trường hợp không xảy ra phải báo N/A, không bịa thời gian.

Xuất ZIP chứa JSON và ảnh PNG phục hồi kích thước gốc. Đây là bản lab; chưa tối ưu peak RAM cho ảnh lớn/đánh giá segmentation nhiều mẫu, có thể dừng tác vụ; chưa tiếp tục được phép đo dang dở sau khi tiến trình bị Android hủy.

## YOLO và mAP

Hai file riêng: YOLO11n detection và YOLO11n-seg segmentation, COCO80, FP32, batch1, input NHWC vuông, output raw `nms=False`. Decoder Android nhắm phiên bản Ultralytics **8.3.221**, [nguồn output TFLite normalized xywh](https://github.com/ultralytics/ultralytics/blob/v8.3.221/ultralytics/nn/modules/head.py). Không đưa một model detection vào nút segmentation.

Trong môi trường export riêng/Colab:

```python
# pip install ultralytics==8.3.221
from ultralytics import YOLO
YOLO('yolo11n.pt').export(format='tflite', imgsz=640, batch=1, half=False, int8=False, nms=False)
YOLO('yolo11n-seg.pt').export(format='tflite', imgsz=640, batch=1, half=False, int8=False, nms=False)
```

App kiểm tra shape input/output; detection output [1,84,N], segmentation detection [1,116,N] + prototypes [1,H,W,32]. Đã xuất hai YOLO TFLite thật và đối chiếu tensor output với Ultralytics8.3.221 trên desktop và Android10; decoder bbox/mask qua fixture có kiểm tra letterbox. Chưa có nhãn object test để đánh giá mAP thật.

Không dùng YOLO(clean) làm nhãn chuẩn. Các bộ restoration thường cung cấp clean image, không đồng nghĩa có object bbox hoặc instance masks. Cần nhãn người gán/nhãn test chuẩn, class mapping COCO0..79 đúng với YOLO. Có thể gán nhãn lên ảnh clean tương ứng nếu hình học khớp với ảnh degraded. Nếu cảnh có vật thể ngoài COCO, phải có YOLO đúng bộ class và sửa decoder/class mapping; không gán class tùy ý.

`testset-format.json` chỉ minh họa schema, không phải mẫu test thật. Tạo manifest với các ảnh test thật, `source_dataset`, `source_split`, `id`, `degradation`, `input`, `annotations`. `bbox` là xyxy pixel ảnh gốc; `polygon` là instance polygon xy pixel cho segmentation. Ảnh không vật thể hợp lệ có `annotations: []`; không bỏ ảnh negative khỏi tập đánh giá. Polygon hiện chỉ hỗ trợ một đường biên, không holes/multipolygon/crowd/ignore.

```sh
python3 tools/pack_testset.py dataset/manifest.json --out testset.zip --require-five-tasks
```

App restore CPU mỗi ảnh, chạy YOLO trước và sau độc lập theo nút detection/segmentation. mAP50 và mAP50:95 theo 101 recall points, confidence .001, class-aware NMS .7, tối đa300 detections/ảnh, trung bình trên class có GT. Mask IoU dùng raster instance mask, không bbox IoU. Báo delta âm nếu kết quả xấu hơn; không có giả định restoration chắc chắn tăng mAP. Đây là protocol subset được mô tả rõ, **không phải toàn bộ COCOeval** (thiếu crowd/ignore/area/maxDets slicing). Cần dùng cùng tập ảnh, nhãn và tham số cho before/after. Bộ vài ảnh chỉ là pilot, không đủ để khẳng định hiệu quả tổng quát.

Demo synthetic-rain/count-confidence cũ đã được lưu ngoài Git trong `.local-archive/`; không dùng demo này thay cho công cụ mAP của đề tài.

## Điều kiện còn thiếu để hoàn thành đề tài

1. Checkpoint five-task cùng config/source provenance.
2. Đã chuẩn bị15 input test (3/tác vụ) và12 cặp clean; còn thiếu clean LOL từ bản repo subset và xác nhận protocol denoising float đúng bản bài cần tái lập.
3. Nhãn bbox/mask chuẩn tương ứng với class YOLO.
4. Kiểm chứng PyTorch-port ↔ NATTEN gốc và chất lượng PSNR/SSIM.
5. Đã chuyển TFLite256 và hai YOLO; hai fixture mỗi model đều qua trên desktop và Android10. Báo cáo trong verification/.
6. Chạy thật trên Android10+ và điện thoại có GPU, lưu benchmark/profiling và mAP. Emulator không đại diện hiệu năng điện thoại.

## Bằng chứng kiểm tra hiện có

- Build APK + Java unit tests: PASS, 7 tests; Android lint 0 errors (còn warning về phiên bản thư viện, icon và text resources).
- PyTorch: 3 tests toán học PASS.
- Android10/API29 emulator: 5 instrumentation tests PASS. Bốn test thành phần kiểm tra identity NHWC/NCHW ảnh357×259, RGB/kích thước/50 samples, input256, hai YOLO decoder và chặn ảnh chưa gán nhãn. Một test khác chạy sáu fixture của ba model thật, đối chiếu tensor với PyTorch. Các fixture số học không đo chất lượng hay mAP; timing single invoke không phải benchmark5+50.
- Log kiểm tra lưu trong `verification/`. Kiểm thử Android17 trước đó không hoàn tất do emulator không ổn định và bị dừng; kết quả API29 mới là lần chạy hoàn chỉnh.

## Artifact và dữ liệu đã tạo

DSwinIR FP32 có input/output NHWC[1,256,256,3],221593464 bytes; max abs so với PyTorch trên Android10 là2.98e-6. Hai YOLO output detection[1,84,8400] và segmentation[1,116,8400]+[1,160,160,32], đều FP32 và qua ngưỡng1e-3. Model thật được kiểm tra CPU; chưa có số đo phân bổ/thời gian GPU trên điện thoại.

`data/pilot/manifest.json` lưu15 ảnh thử chưa gán nhãn: CBSD68 màu sigma25 PNG(seed42-44), Rain100L test, SOTS outdoor test, GoPro test, LOL tác giả eval subset. App hỗ trợ ZIP chưa gán nhãn để khôi phục, nhưng chặn mAP khi pending. `annotations: []` chỉ dành cho ảnh đã kiểm chứng không có object; không dùng cho ảnh chưa gán nhãn. `tools/download_pilot.py` không tải train RainTrainL; GoPro/SOTS chỉ đọc member được chọn bằng HTTP Range.

## Kiểm chứng giao diện và khả năng dừng

Ảnh Rain100L 481×321 đã được khôi phục trong app bằng 4 tile và giữ kích thước gốc. Lần đầu ghi 143829,737 ms; lần kiểm tra sau ghi 307,3 giây; một lần khác ghi 926586,494 ms. Đây là các lần khôi phục toàn ảnh trên máy ảo với cấu hình tại thời điểm chạy, không phải trung bình 5+50 và không đại diện hiệu năng điện thoại. Các nguồn số liệu nằm trong [danh mục kiểm chứng](verification/README.md).

Detection và segmentation được chạy độc lập trên cùng cặp ảnh trước/sau. Box, confidence và mask trong ảnh minh họa không phải mAP. PSNR RGB của một mẫu Rain100L là25,97 →36,74 dB; chỉ là sanity của một mẫu, không phải trung bình bài báo.

Giao diện hiện dùng câu chữ tiếng Việt, nút bắt đầu cố định, thời gian chờ, tự cuộn đến ảnh sau và lưu phiên gần nhất. Khi app chạy phép đo dài, đã quan sát CPU176%, khoảng1GB RAM trên máy ảo2 lõi; Android không ghi nhận ANR. Bản sửa có nút Dừng xử lý, bật LiteRT cancellable inference, hiển thị từng lượt và nhắc trước phép đo dài.

Kiểm thử trên DSwinIR thật đã hủy native CPU warm-up đầu tiên, bật lại các nút và giữ nguyên byte của ảnh/báo cáo đã lưu. Đây không phải benchmark5+50 hoàn chỉnh. Số luồng CPU hiện được ghi trong báo cáo; không so trực tiếp với các lần kiểm tra lịch sử dùng4 luồng. Tác vụ đánh giá mới vô hiệu hóa báo cáo cũ trước khi thay các tệp đầu ra để tránh xuất nhầm báo cáo với ảnh xử lý dở.

Ảnh/log giao diện qua các lần chỉnh sửa và script chuyển đổi thử nghiệm đã được chuyển vào `.local-archive/` trên máy phát triển, không nằm trong repository public. Chỉ giữ các bằng chứng chính theo [verification/README.md](verification/README.md).
