# Bằng chứng kiểm chứng được giữ lại

## Model và chuyển đổi

| Tệp | Nội dung |
|---|---|
| [dswinir-conversion.json](dswinir-conversion.json) | Sai số và phạm vi chuyển đổi DSwinIR |
| [yolo-detection-conversion.json](yolo-detection-conversion.json), [yolo-segmentation-conversion.json](yolo-segmentation-conversion.json) | Hai model YOLO riêng |
| [model-artifacts.json](model-artifacts.json) | Hash và kích thước model |
| [android10-real-model-parity.json](android10-real-model-parity.json) | Output model thật trên Android10 so với PyTorch |
| [android10-instrumentation.txt](android10-instrumentation.txt), [python-tests.txt](python-tests.txt) | Kiểm thử model/thành phần và toán học |
| [android10-xnnpack-oom.txt](android10-xnnpack-oom.txt) | Lỗi bộ nhớ đã quan sát, cơ sở để tắt XNNPACK cho DSwinIR |

Các kiểm tra số học không chứng minh chất lượng phục hồi hay mAP tăng. Model DSwinIR hiện có chỉ dành cho Rain100L.

## Ảnh thử và chất lượng minh họa

- Cặp [ảnh ban đầu](rain100l-example/degraded.png) / [ảnh khôi phục](rain100l-example/restored.png), [báo cáo](rain100l-example/report.json) và [PSNR sanity](rain100l-example/pixel-quality-sanity.json).
- [Detection](android10-rain100l-detection.png) và [segmentation](android10-rain100l-segmentation.png) chạy riêng trên cặp ảnh đó.
- [Lần chạy đầu](android10-rain100l-ui.json): 143829,737 ms; [log lần kiểm tra tiếp](android10-ui-smoke-current.txt): 307,3 giây; [báo cáo lần chạy sau](last-restoration.json): 926586,494 ms.
- [Log hai thao tác YOLO](android10-ui-yolo-current.txt).

Đây là các lần chạy riêng trên máy ảo. Không dùng chúng làm trung bình benchmark5+50 hoặc mAP; lựa chọn suy thoái trong app chỉ là metadata. Bộ dữ liệu gốc không được đóng gói trong Git.

## Giao diện và dừng tác vụ

- [Ảnh kết quả hiển thị trong app](android-visible-result.png).
- [Kiểm tra nút thao tác](android-visible-action-tests.txt), [kiểm tra thành phần trước đó](android10-ui-readiness-tests.txt).
- [Build, unit tests và lint bản hỗ trợ dừng](android-cancellation-build.txt).
- [Đang chạy](android-cancel-running.png) → [đã dừng](android-cancel-stopped.png), [log kiểm thử](android-cancel-smoke.txt) và [kết quả kiểm tra](android-cancel-result.json).

Kiểm thử hủy dùng model DSwinIR thật trong native CPU inference và kiểm tra byte của ảnh/báo cáo cũ được giữ nguyên. Nó không chạy đủ phép đo5+50. Các ảnh/log giao diện trùng và script thử nghiệm cũ được giữ ở `.local-archive/` trên máy phát triển, ngoài Git.
