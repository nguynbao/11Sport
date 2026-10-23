# 11Sport Dubber — Hệ thống Lồng tiếng & Phụ đề Video bằng AI

Ứng dụng web tự động nhận diện giọng nói từ video gốc, dịch thuật theo ngữ cảnh sang **Tiếng Việt (`VI`)** và **Tiếng Anh (`EN`)**, cho phép duyệt/chỉnh sửa câu thoại, lồng tiếng thuyết minh khớp khẩu hình **98–100%** và xuất file video `MP4` kèm phụ đề `SRT`.

---

## Tính năng nổi bật

- **Nhận diện giọng nói chính xác đến từng từ (`word-level timestamps`)**: Sử dụng `faster-whisper` (`medium` local miễn phí) hoặc OpenAI Whisper.
- **Dịch thuật AI hiểu ngữ cảnh**: Tự động dùng mô hình ngôn ngữ lớn 20B (`GPT-OSS 20B` miễn phí + Google Translate dự phòng) kèm bảng thuật ngữ (`Glossary`) riêng của kênh.
- **3 lựa chọn giọng đọc kèm nút Nghe thử (`EN` / `VI`) trực tiếp trên giao diện**:
  1. **Adam (Dominant, Firm) — ElevenLabs**: Giọng trầm, uy lực, dứt khoát chuẩn TikTok (`pNInz6obpgDQGcFmaJgB`, model `eleven_turbo_v2_5` hỗ trợ đầy đủ Tiếng Việt & Tiếng Anh). Khi chọn giọng này, giao diện sẽ hiển thị ô nhập **ElevenLabs API Key** trực tiếp trên web.
  2. **Nữ — Miễn phí**: Giọng nữ tự nhiên, rõ chữ, không cần API Key.
  3. **Nam — Miễn phí**: Giọng nam ấm, thân thiện, không cần API Key.
- **Khớp thời lượng thuyết minh 98–100% (`Exact-Sync`)**: Tự động căn chỉnh điểm đầu – điểm cuối của từng câu thoại theo thời gian thực người nói bằng thuật toán 2 tầng (`TTS Rate` + `FFmpeg atempo`) và tự động làm nhỏ tiếng gốc (`audio ducking`).

---

## 1. Yêu cầu cài đặt (Dành cho người mới)

Trước khi bắt đầu, máy tính của bạn cần cài sẵn 3 công cụ cơ bản sau:

1. **Node.js (từ bản 18 trở lên)**: Kiểm tra bằng lệnh `node -v`.
   - Cài trên macOS: `brew install node` (hoặc tải từ [nodejs.org](https://nodejs.org/)).
2. **uv (Trình quản lý Python siêu tốc)**: Kiểm tra bằng lệnh `uv --version`.
   - Cài đặt nhanh trên macOS / Linux:
     ```bash
     curl -LsSf https://astral.sh/uv/install.sh | sh
     ```
3. **FFmpeg & FFprobe (có hỗ trợ `libass` để gắn phụ đề cứng)**: Kiểm tra bằng lệnh `ffmpeg -version`.
   - Cài trên macOS: `brew install ffmpeg`

---

## 2. Cài đặt dự án (Chỉ cần chạy 1 lần đầu tiên)

Mở Terminal tại thư mục gốc của dự án (`11Sport`) và chạy lệnh:

```bash
make setup
```

> **Lệnh `make setup` sẽ tự động làm gì?**
> 1. Tạo môi trường Python ảo `backend/.venv` và cài toàn bộ thư viện backend qua `uv sync`.
> 2. Tạo file cấu hình `backend/.env` từ `backend/.env.example` (nếu chưa có).
> 3. Cài đặt toàn bộ thư viện giao diện tại `frontend/node_modules` qua `npm install`.

*(Nếu không dùng `make`, bạn có thể chạy thủ công)*:
```bash
cd backend && uv sync && cp -n .env.example .env && cd ..
cd frontend && npm install && cd ..
```

---

## 3. Cách khởi chạy ứng dụng (Mở 3 cửa sổ Terminal)

Để hệ thống hoạt động đầy đủ, bạn mở **3 tab Terminal** tại thư mục gốc `11Sport` và chạy lần lượt:

### Terminal 1 — Chạy Backend API (Cổng `8000`)
```bash
make api
```
*(Hoặc chạy thủ công: `cd backend && uv run uvicorn app.main:app --reload --port 8000`)*

### Terminal 2 — Chạy Worker xử lý video nền
```bash
make worker
```
*(Hoặc chạy thủ công: `cd backend && uv run python -m app.worker`)*
> **Lưu ý:** Lần đầu tiên xử lý video ở chế độ miễn phí, Worker sẽ tự động tải model `faster-whisper` (`medium`) về máy (chỉ tải 1 lần duy nhất).

### Terminal 3 — Chạy Giao diện Web (Cổng `5173`)
```bash
make web
```
*(Hoặc chạy thủ công: `cd frontend && npm run dev`)*

Sau khi chạy xong cả 3 terminal, mở trình duyệt truy cập:
👉 **http://localhost:5173** (Tài liệu API Swagger: http://localhost:8000/docs)

---

## 4. Hướng dẫn sử dụng nhanh cho người mới

1. Truy cập **http://localhost:5173** → Bấm **"Tạo video mới"** (hoặc vào thẳng `/new`).
2. **Mục 1 (Video nguồn)**: Kéo thả file video (`.mp4`, `.mov`, `.mkv`, `.webm`, `.avi`, tối đa 500MB / 10 phút).
3. **Mục 2 (Ngôn ngữ đích)**: Chọn **Tiếng Anh (`EN`)**, **Tiếng Việt (`VI`)**, hoặc chọn cả hai (hệ thống sẽ xuất ra 2 video riêng mà chỉ cần nhận diện giọng nói 1 lần).
4. **Mục 3 (Giọng đọc)**:
   - Bấm nút **Nghe thử EN** / **Nghe thử VI** trên từng thẻ để nghe trước chất giọng.
   - Nếu chọn **Adam (Dominant, Firm)**: Ô nhập **ElevenLabs API Key** sẽ hiện ra bên dưới. Dán mã API Key (`sk_...`) của bạn vào và bấm **"Lưu & Kiểm tra Key"** (chỉ cần lưu 1 lần, hệ thống sẽ nhớ cho các lần sau).
   - Nếu chọn **Nữ** hoặc **Nam**: Dùng hoàn toàn miễn phí, không cần nhập Key.
5. **Mục 4, 5, 6**: Chọn kiểu phụ đề (**Gắn cứng**, **Phụ đề rời**, hoặc **Không phụ đề**), quy trình (**Duyệt trước khi xuất** hoặc **Tự động hoàn toàn**) và âm lượng tiếng gốc.
6. Bấm **"Bắt đầu xử lý"**:
   - Nếu chọn **Duyệt trước khi xuất**: Khi trạng thái chuyển sang **Chờ duyệt (`REVIEW`)**, bạn có thể bấm vào từng câu để nghe đoạn video đó, sửa lại câu dịch nếu muốn, rồi bấm **"Duyệt & Xuất video"**.
   - Khi hoàn tất (`DONE`), bạn có thể xem trực tiếp trên web và tải về file **MP4** hoặc **SRT**.

---

## 5. Cấu hình nâng cao trong `backend/.env`

Mặc định hệ thống chạy ngay ở chế độ `PROVIDER_MODE=free` mà không cần chỉnh sửa gì thêm. Nếu muốn tùy biến, bạn có thể sửa file `backend/.env`:

| Biến môi trường | Mặc định | Ý nghĩa |
| :--- | :--- | :--- |
| `PROVIDER_MODE` | `free` | `free` = Whisper local + LLM 20B miễn phí + Edge/Google TTS; `real` = OpenAI + Azure; `fake` = giả lập chạy test |
| `WHISPER_MODEL` | `medium` | Model nhận diện giọng nói ở chế độ `free` (`tiny`, `base`, `small`, `medium`, `large-v3`) |
| `ELEVENLABS_API_KEY` | *(trống)* | API Key ElevenLabs cho giọng **Adam** (có thể nhập trực tiếp trên giao diện web) |
| `ELEVENLABS_MODEL` | `eleven_turbo_v2_5` | Model ElevenLabs hỗ trợ chuẩn cả Tiếng Việt (`vi`) và Tiếng Anh (`en`) |
| `DEFAULT_ORIG_VOLUME` | `0.05` | Âm lượng tiếng gốc khi có giọng thuyết minh đè lên (`0.05` = 5%) |
| `MAX_UPLOAD_MB` | `500` | Dung lượng file video tải lên tối đa (MB) |
| `MAX_DURATION_SEC` | `600` | Thời lượng video tối đa (giây, `600` = 10 phút) |
| `RETENTION_DAYS` | `14` | Tự động dọn dẹp các job cũ sau `N` ngày |

---

## 6. Các lệnh kiểm thử & bảo trì

```bash
make test     # Chạy bộ 47 unit & E2E tests (pytest)
make lint     # Kiểm tra chuẩn code Python (ruff)
make build    # Kiểm tra TypeScript & đóng gói giao diện production (Vite)
make sample   # Tạo video mẫu tiếng Anh (storage/sample.mp4) để test nhanh
```

---

## 7. Cấu trúc thư mục

```text
11Sport/
├── Makefile                  # Các lệnh tắt khởi chạy & kiểm thử
├── storage/                  # Thư mục lưu trữ video gốc, cache ASR, file TTS và video xuất ra
├── backend/
│   ├── .env.example          # File mẫu cấu hình môi trường
│   ├── assets/
│   │   ├── fonts/            # Font chữ gắn phụ đề cứng (Be Vietnam Pro, Noto Sans...)
│   │   └── voices/           # Các file âm thanh mẫu cho tính năng "Nghe thử"
│   ├── app/
│   │   ├── api/              # REST API (jobs.py, glossary.py, meta.py)
│   │   ├── pipeline/         # Các bước xử lý (stages.py, fit.py khớp thời lượng 98-100%, runner.py)
│   │   ├── providers/        # Tích hợp AI (free.py, elevenlabs_tts.py, openai_*, azure_tts.py)
│   │   ├── media/            # Xử lý FFmpeg, trộn âm thanh (ducking) & tạo phụ đề ASS/SRT
│   │   ├── text/             # Chia câu thông minh theo nhịp nghỉ (segmentation.py) & tính ký tự
│   │   └── worker.py         # Tiến trình xử lý hàng đợi video chạy nền
│   └── tests/                # Bộ kiểm thử tự động (47 tests)
└── frontend/
    └── src/
        ├── api/              # HTTP client & kiểu dữ liệu TypeScript
        ├── pages/            # Các trang: Danh sách video, Tạo video mới, Chi tiết & Duyệt, Từ điển
        └── components/       # Bộ sửa câu thoại (SegmentEditor), thanh trạng thái, bố cục
```
