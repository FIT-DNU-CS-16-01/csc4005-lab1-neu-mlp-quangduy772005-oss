# Hướng dẫn dùng Weights & Biases (W&B)

## Vì sao lab này dùng W&B?
Lab 1 yêu cầu so sánh ít nhất 3 cấu hình và kết luận mô hình tốt nhất dựa trên learning curves và metrics. W&B giúp lưu lại toàn bộ cấu hình, loss, accuracy và ảnh đầu ra của từng run để so sánh rõ ràng hơn.

## 1. Tạo tài khoản
Đăng ký tài khoản tại trang W&B bằng email cá nhân hoặc email trường.

## 2. Lấy API key
Sau khi đăng nhập, vào phần profile/settings để lấy API key.

## 3. Đăng nhập trong terminal
```bash
wandb login
```
Dán API key khi được hỏi.

## 4. Chạy một run có bật W&B
```bash
python -m src.train   --data_dir /duong_dan/NEU-CLS.zip   --project csc4005-lab1-neu-mlp   --run_name baseline_adamw   --optimizer adamw   --lr 0.001   --weight_decay 0.0001   --dropout 0.3   --epochs 20   --batch_size 32   --img_size 64   --patience 5   --augment   --use_wandb
```

## 5. Những gì được log
- project: `csc4005-lab1-neu-mlp`
- run name
- train loss, val loss
- train accuracy, val accuracy
- learning rate
- best validation accuracy
- test accuracy
- confusion matrix và learning curves

## 6. Chạy offline khi cần
```bash
wandb offline
```
Khi có mạng lại, có thể sync sau theo hướng dẫn của W&B.
