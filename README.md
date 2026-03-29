# AI-Challenge – Hệ thống Video Retrieval theo Keyframe (Đang hoàn thiện)

> **Trạng thái dự án:** 🚧 **Work in Progress**  
> Dự án hiện đang trong quá trình hoàn thiện kiến trúc, API và trải nghiệm người dùng.

---

## 1) Giới thiệu

Đây là dự án xây dựng hệ thống **tìm kiếm khung hình (keyframe) từ video** bằng AI, hỗ trợ:

- Tìm kiếm bằng **text query**
- Tìm kiếm bằng **image query**
- Pipeline xử lý video bất đồng bộ theo từng **session người dùng**
- Lưu trữ embedding và chỉ mục truy hồi theo session riêng
- Giao diện web để upload video, theo dõi tiến trình, và xem kết quả retrieval

Mục tiêu chính là hướng tới một hệ thống retrieval linh hoạt theo kiểu SaaS/internal tool, không phụ thuộc hoàn toàn vào dataset cố định.

---

## 2) Kiến trúc tổng quan

Luồng chính của hệ thống:

1. Người dùng upload video
2. Backend tạo `session_id` và thư mục làm việc riêng
3. Worker nền chạy pipeline:
   - Trích xuất keyframe
   - Sinh embedding
   - Build database/index truy hồi
4. Frontend polling trạng thái xử lý
5. Khi hoàn tất, người dùng thực hiện search theo session tương ứng

### Thành phần chính

- **Web API (Flask):** `AIC2025/sever.py`
- **Retrieval Core:** `AIC2025/eva02_retrieval_trake.py`
- **Background Pipeline:** `AIC2025/workers/video_pipeline.py`
- **Session Storage Utility:** `AIC2025/utils/storage.py`
- **Embedding Utility:** `AIC2025/utils/embedding_utils.py`
- **Frontend:** `AIC2025/templates/index.html`

---

## 3) Cấu trúc thư mục chính

```text
AI-Challenge/
├── AIC2025/
│   ├── sever.py
│   ├── cut_keyframe.py
│   ├── eva02_retrieval_trake.py
│   ├── templates/
│   │   └── index.html
│   ├── workers/
│   │   └── video_pipeline.py
│   ├── utils/
│   │   ├── model_loader.py
│   │   ├── embedding_utils.py
│   │   └── storage.py
│   └── userdata/
│       └── <session_id>/
│           ├── videos/
│           ├── keyframes/
│           ├── map_keyframes/
│           ├── embeddings/
│           └── db/
├── GoogleVideo/
│   └── (nhánh thử nghiệm BoW/SIFT)
├── requirements.txt
└── README.md
```

---

## 4) Công nghệ sử dụng

- **Python 3.10+**
- **Flask + Flask-CORS**
- **PyTorch / TorchVision**
- **OpenCLIP (EVA02)**
- **FAISS**
- **OpenCV, NumPy, Pandas, Pillow**
- **Transformers (một số module rerank/caption)**

---

## 5) Cài đặt nhanh

## 5.1 Tạo môi trường

```bash
python -m venv venv
```

Windows:
```bash
venv\Scripts\activate
```

Linux/Mac:
```bash
source venv/bin/activate
```

## 5.2 Cài dependencies

```bash
pip install -r requirements.txt
```

---

## 6) Chạy ứng dụng

Từ thư mục gốc dự án:

```bash
cd AIC2025
python sever.py
```

Mặc định server chạy tại:

- `http://0.0.0.0:5000`

Bạn có thể override bằng biến môi trường:

- `HOST`
- `PORT`

---

## 7) API chính (hiện tại)

- `POST /api/upload_video` – Upload video và khởi tạo xử lý nền
- `GET /api/status/<session_id>` – Lấy trạng thái pipeline
- `POST /api/search_text` – Search theo text
- `POST /api/upload_image` – Upload ảnh và search ảnh tương tự
- `POST /api/search_text_with_image` – Text + ảnh để rerank
- `GET /api/stored_queries` – Danh sách query đã lưu
- `GET /api/reload_query/<idx>` – Nạp lại kết quả query theo index
- `GET /api/get_frames/<video_name>` – Lấy toàn bộ frame của video
- `GET /api/get_frames_range/<video_name>/<frame_id>/<range_val>` – Lấy dải frame quanh mốc
- `GET /api/stats` – Thống kê dữ liệu retrieval
- `GET /image?path=...` – Serve ảnh keyframe (đã có kiểm soát đường dẫn)

> Lưu ý: API có thể thay đổi trong quá trình chuẩn hóa backend.

---

## 8) Trạng thái hiện tại & lưu ý phát triển

Dự án đang trong giai đoạn hoàn thiện nên có thể tồn tại:

- Một số module/script mang tính thử nghiệm hoặc legacy
- Một số endpoint/logic còn đang được tinh chỉnh
- Cấu hình deploy chưa đồng nhất tuyệt đối giữa các file
- Cần thêm test tự động và chuẩn hóa cấu trúc production

Khuyến nghị khi sử dụng:

- Dùng cho mục đích dev/internal trước
- Kiểm tra kỹ đường dẫn dữ liệu và model trước khi chạy pipeline dài
- Theo dõi log khi upload video lớn hoặc chạy đa session

---

## 9) Roadmap 

- [ ] Ổn định retrieval core và thuật toán temporal retrieval
- [ ] Chuẩn hóa tên file/module (server, retrieval, ...)
- [ ] Hợp nhất các script trùng chức năng
- [ ] Bổ sung unit/integration tests cho API + pipeline
- [ ] Tối ưu tài nguyên cho môi trường nhiều người dùng đồng thời
- [ ] Hoàn thiện tài liệu deploy (Docker/Heroku/local)

---

## 10) Ghi chú

- Dự án có nhiều nhánh thực nghiệm (retrieval/caption/object detection/OCR,....).  


---

## 11) Liên hệ / đóng góp


