# LAB 1 – Training & Regularization for Steel Surface Defect Classification

## 1. Mục tiêu
Lab 1 tập trung vào **huấn luyện và tối ưu mạng nơ-ron thường (MLP)** cho bài toán phân loại lỗi bề mặt thép. Trọng tâm là hiểu đúng pipeline train / validation / test, Cross-Entropy, optimizer, regularization, overfitting và cách ghi lại thí nghiệm bằng W&B.

## 2. Vì sao phải dùng W&B?
Lab yêu cầu so sánh **ít nhất 3 cấu hình**, có learning curves, bảng tổng hợp configs → metrics và chọn best config. W&B giúp lưu lại:
- train loss, val loss
- train accuracy, val accuracy
- learning rate
- cấu hình từng run
- ảnh confusion matrix và learning curves

W&B không làm mô hình mạnh hơn, nhưng làm thí nghiệm **rõ ràng, có bằng chứng và dễ so sánh hơn**.


## 3. Chuẩn bị dữ liệu
Repo hỗ trợ:
1. Thư mục lớp riêng `Crazing/`, `Inclusion/`, ...
2. File ZIP hoặc thư mục phẳng có tên ảnh như `crazing_10.jpg`, `rolled-in_scale_8.jpg`, ...

Với bộ dữ liệu do giảng viên cung cấp, sinh viên dùng trực tiếp file ZIP:
```bash
python -m src.train --data_dir /duong_dan/NEU-CLS.zip --run_name quick_test
```

## 5. Chạy baseline
```bash
python -m src.train   --data_dir /duong_dan/NEU-CLS.zip   --project csc4005-lab1-neu-mlp   --run_name baseline_adamw   --optimizer adamw   --lr 0.001   --weight_decay 0.0001   --dropout 0.3   --epochs 20   --batch_size 32   --img_size 64   --patience 5   --augment   --use_wandb
```

## 6. Ba run nên thử
### Run A – Baseline AdamW
Dùng đúng lệnh baseline bên trên.

### Run B – SGD
```bash
python -m src.train   --data_dir /duong_dan/NEU-CLS.zip   --project csc4005-lab1-neu-mlp   --run_name run_b_sgd   --optimizer sgd   --lr 0.01   --weight_decay 0.0   --dropout 0.3   --epochs 20   --batch_size 32   --img_size 64   --patience 5   --augment   --use_wandb
```

### Run C – AdamW với regularization mạnh hơn
```bash
python -m src.train   --data_dir /duong_dan/NEU-CLS.zip   --project csc4005-lab1-neu-mlp   --run_name run_c_strong_reg   --optimizer adamw   --lr 0.0005   --weight_decay 0.001   --dropout 0.5   --epochs 20   --batch_size 32   --img_size 64   --patience 5   --augment   --use_wandb
```

## 7. Dấu hiệu cần quan sát
- `train_loss` giảm, `val_loss` cũng giảm: tốt.
- `train_acc` rất cao nhưng `val_acc` tăng ít: có thể overfit.
- cả train và val đều thấp: có thể underfit.

## 8. Điều phải nộp
- code chạy được
- ít nhất 3 cấu hình
- learning curves
- bảng configs → metrics
- link hoặc ảnh W&B dashboard
- kết luận chọn best config theo validation
