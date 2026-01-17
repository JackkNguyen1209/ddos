# DDoS Detection ML

Ứng dụng phát hiện tấn công DDoS sử dụng Machine Learning với giao diện tiếng Việt.

## Tính năng

- **Upload đa định dạng**: Hỗ trợ CSV và Excel (.xlsx, .xls) lên đến 50MB
- **6 thuật toán ML**: Decision Tree, Random Forest, KNN, Naive Bayes, Logistic Regression, LUCID Neural Network
- **Phân loại tấn công**: Phát hiện 17+ loại tấn công thuộc 7 nhóm (reconnaissance, bruteforce, remote_access, volumetric, amplification, application_layer, protocol_exploit)
- **Chế độ kép**:
  - Supervised Mode: Dataset có nhãn - hiển thị Accuracy/Precision/Recall/F1
  - Unlabeled Inference Mode: Dataset không nhãn - sử dụng IsolationForest, LOF
- **Hệ thống tự học**: Ghi nhớ dữ liệu qua các phiên, liên tục cải thiện độ chính xác

## Cài đặt với Docker (Khuyến nghị)

### Yêu cầu
- Docker và Docker Compose

### Các bước

1. Clone repository:
```bash
git clone <your-repo-url>
cd ddos-detection-ml
```

2. Tạo file `.env` từ template:
```bash
cp .env.example .env
```

3. Chạy ứng dụng:
```bash
docker compose up -d
```

4. Truy cập ứng dụng tại: http://localhost:8000

### Dừng ứng dụng
```bash
docker compose down
```

### Xóa toàn bộ dữ liệu (bao gồm database)
```bash
docker compose down -v
```

## Cài đặt thủ công (Local)

### Yêu cầu
- Node.js 20+
- PostgreSQL 15+

### Các bước

1. Cài đặt dependencies:
```bash
npm install
```

2. Tạo file `.env`:
```bash
cp .env.example .env
```

3. Cập nhật `DATABASE_URL` trong `.env` với thông tin PostgreSQL của bạn

4. Đẩy schema vào database:
```bash
npm run db:push
```

5. Chạy ứng dụng:
```bash
npm run dev
```

6. Truy cập: http://localhost:5000

## Biến môi trường

| Biến | Mô tả | Giá trị mặc định |
|------|-------|------------------|
| DATABASE_URL | Connection string PostgreSQL | postgresql://postgres:postgres@db:5432/ddos_detection |
| SESSION_SECRET | Khóa bí mật cho session | (tự sinh) |
| NODE_ENV | Môi trường chạy | development |
| PORT | Cổng server | 5000 |

## Cấu trúc dự án

```
├── client/src/          # React frontend
│   ├── components/      # UI components
│   ├── pages/           # Trang route
│   └── lib/             # Utilities
├── server/              # Express backend
│   ├── routes.ts        # API endpoints
│   ├── ml-algorithms.ts # Thuật toán ML
│   ├── learning-service.ts # Hệ thống tự học
│   └── storage.ts       # Lưu trữ dữ liệu
├── shared/              # Types dùng chung
│   └── schema.ts        # Database schema
├── docker-compose.yml   # Docker config
└── Dockerfile           # Build image
```

## API Endpoints

- `POST /api/dataset` - Upload dataset
- `GET /api/dataset` - Lấy thông tin dataset
- `POST /api/analyze` - Phân tích với ML
- `GET /api/results/:id` - Lấy kết quả phân tích
- `GET /api/learning/stats` - Thống kê hệ thống tự học
- `GET /api/learning/patterns` - Các pattern đã học
- `POST /api/learning/learn` - Học từ dữ liệu mới

## License

MIT
