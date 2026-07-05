# Barn-Inspection-Assistant

基于 YOLO11 的谷仓动物智能检测系统，支持目标检测模型的训练、评估和推理。

## 功能

- 目标检测模型微调训练
- 模型性能评估（mAP、Precision、Recall）
- 图片/视频/摄像头/RTSP 流实时推理
- 自动检测训练设备（CUDA / Apple MPS / CPU）

## 安装

```bash
pip install -r requirements.txt
```

## 项目结构

```
Barn-Inspection-Assistant/
├── scripts/
│   ├── train.py          # 模型训练脚本
│   ├── evaluate.py       # 模型评估脚本
│   └── predict.py        # 推理预测脚本
├── config/               # 数据集配置文件（不提交）
├── data/                 # 数据集（不提交）
├── runs/                 # 训练输出 / 模型权重（不提交）
├── requirements.txt
└── README.md
```

## 用法

### 1. 准备数据集

按 YOLO 格式组织数据，创建 `data/xxx/xxx.yaml` 配置文件：

```yaml
path: data/your_dataset
train: train/images
val: val/images
names:
  0: class_name
```

### 2. 训练

```bash
# 基础训练
python scripts/train.py --data config/dataset.yaml

# 自定义参数
python scripts/train.py \
    --data config/dataset.yaml \
    --model yolo11s.pt \
    --epochs 150 \
    --batch 16 \
    --device mps

# 断点续训
python scripts/train.py --data config/dataset.yaml --resume

# 冻结 backbone（数据极少时）
python scripts/train.py --data config/dataset.yaml --freeze 10
```

| 参数        | 默认值                | 说明                       |
| ----------- | --------------------- | -------------------------- |
| `--data`    | `config/dataset.yaml` | 数据集配置路径             |
| `--model`   | `yolo11n.pt`          | 预训练模型（n/s/m/l/x）    |
| `--epochs`  | `100`                 | 训练轮数                   |
| `--batch`   | `16`                  | 批大小                     |
| `--imgsz`   | `640`                 | 输入图片尺寸               |
| `--device`  | 自动检测              | 训练设备（cuda:0/mps/cpu） |
| `--lr0`     | `0.001`               | 初始学习率                 |
| `--freeze`  | 无                    | 冻结 backbone 前 N 层      |
| `--resume`  | 否                    | 断点续训                   |
| `--workers` | `8`                   | 数据加载线程数             |

### 3. 评估

```bash
python scripts/evaluate.py --model runs/detect/train/weights/best.pt
```

### 4. 推理

```bash
# 图片
python scripts/predict.py --model runs/detect/train/weights/best.pt --source image.jpg

# 视频
python scripts/predict.py --model runs/detect/train/weights/best.pt --source video.mp4

# 摄像头
python scripts/predict.py --model runs/detect/train/weights/best.pt --source 0

# RTSP 流
python scripts/predict.py --model runs/detect/train/weights/best.pt --source rtsp://...
```

| 参数       | 默认值 | 说明                              |
| ---------- | ------ | --------------------------------- |
| `--model`  | 必填   | 模型权重路径                      |
| `--source` | 必填   | 输入源（图片/视频/摄像头ID/RTSP） |
| `--conf`   | `0.25` | 置信度阈值                        |
| `--iou`    | `0.5`  | NMS IOU 阈值                      |
| `--save`   | 是     | 保存结果图片                      |
| `--show`   | 否     | 实时显示（视频/摄像头）           |

## 依赖

- Python >= 3.10
- ultralytics >= 8.3.0
- torch >= 2.0.0
- opencv-python >= 4.8.0

## 模型

训练输出保存至 `runs/detect/train/`：

- `weights/best.pt` — 最佳模型（推理用这个）
- `weights/last.pt` — 最后模型（断点续训用）
- `results.png` — 训练曲线
- `confusion_matrix.png` — 混淆矩阵
