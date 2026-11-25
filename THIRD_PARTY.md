# Nguồn tham khảo và thành phần bên thứ ba

- **DSwinIR**: [mã nguồn tác giả](https://github.com/Aitical/DSwinIR), [bài báo](https://arxiv.org/abs/2504.04869). `model.py` là bản chuyển từ kiến trúc MSDeformableNAIR và deformable attention; `natten_pure.py` thay phép tính NATTEN bằng PyTorch. Đây không phải repository chính thức của tác giả.
- **BasicSR / NAFNet**: cách chuẩn hóa LayerNorm2d theo [BasicSR](https://github.com/XPixelGroup/BasicSR) và [NAFNet](https://github.com/megvii-research/NAFNet).
- **NATTEN**: ngữ nghĩa cửa sổ attention tham khảo [NATTEN](https://github.com/SHI-Labs/NATTEN), phiên bản 0.17.1. Chưa xác nhận parity với CUDA gốc.
- **Ultralytics YOLO11**: công cụ chuyển đổi sử dụng [Ultralytics](https://github.com/ultralytics/ultralytics), phiên bản 8.3.221. Trọng số YOLO không được đưa vào repository này.
- **LiteRT**: ứng dụng dùng các thư viện [Google AI Edge LiteRT](https://github.com/google-ai-edge/LiteRT) 1.4.2 qua Gradle.
- Nguồn các bộ ảnh thử và quy trình lựa chọn được mô tả trong README và `tools/download_pilot.py`; ảnh nguồn không được đóng gói trong Git. Các ảnh trong tài liệu là minh họa kiểm chứng trên tập thử đã nêu nguồn.

Giấy phép và điều kiện sử dụng của từng thành phần, mã nguồn và trọng số bên thứ ba do các dự án nguồn quy định. Repository này không cấp một giấy phép thay thế cho các thành phần đó và không tuyên bố quyền sở hữu đối với mã nguồn hay dữ liệu của tác giả gốc.
