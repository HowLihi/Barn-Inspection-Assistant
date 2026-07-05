#!/usr/bin/env python3
"""
YOLO 微调训练脚本

用法:
    python scripts/train.py --data config/dataset.yaml
    python scripts/train.py --data config/dataset.yaml --model yolo11s.pt --epochs 150 --device mps
    python scripts/train.py --data config/dataset.yaml --resume

必备前置条件:
    1. 数据集已按 YOLO 格式准备好 (images/train, labels/train, images/val, labels/val)
    2. 已创建 dataset.yaml 配置文件
    3. 已安装依赖: pip install ultralytics
"""
from __future__ import annotations

import argparse
import ssl
import sys
from pathlib import Path

import torch
from ultralytics import YOLO

# 修复 macOS SSL 证书问题
ssl._create_default_https_context = ssl._create_unverified_context


def detect_device() -> str:
    """自动检测最佳可用设备"""
    if torch.cuda.is_available():
        return "cuda:0"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="YOLO 微调训练脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础训练
  python scripts/train.py --data config/dataset.yaml

  # 选择更大模型 + 更多轮数
  python scripts/train.py --data config/dataset.yaml --model yolo11m.pt --epochs 200

  # 从检查点恢复训练
  python scripts/train.py --data config/dataset.yaml --resume

  # 冻结 backbone 前 10 层（数据极少时使用）
  python scripts/train.py --data config/dataset.yaml --freeze 10
        """,
    )

    parser.add_argument(
        "--data",
        type=str,
        default="config/dataset.yaml",
        help="数据集配置文件路径",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="预训练模型 (yolo11n.pt / yolo11s.pt / yolo11m.pt / yolo11l.pt / yolo11x.pt)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="训练轮数 (微调建议 50-150)",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="输入图片尺寸 (640 / 320 / 1280)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=16,
        help="Batch Size (显存不足时减小: 8 / 4 / 2)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="训练设备 (cuda:0 / mps / cpu, 默认自动检测)",
    )
    parser.add_argument(
        "--lr0",
        type=float,
        default=0.001,
        help="初始学习率 (微调建议 0.0005-0.002)",
    )
    parser.add_argument(
        "--freeze",
        type=int,
        default=None,
        help="冻结 backbone 前 N 层 (数据极少时使用, 如 --freeze 10)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="从最近一次训练断点续训",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="数据加载线程数 (CPU 核心多可加大)",
    )
    parser.add_argument(
        "--cache",
        action="store_true",
        default=True,
        help="缓存图片到内存，加速训练",
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="不缓存图片（显存紧张时使用）",
    )
    parser.add_argument(
        "--cos_lr",
        action="store_true",
        default=True,
        help="使用余弦退火学习率",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="早停耐心值 (连续 N 轮不提升则停止)",
    )

    return parser.parse_args()


def validate_environment(args: argparse.Namespace) -> None:
    """验证环境和配置"""
    # 检查数据集配置
    data_path = Path(args.data)
    if not data_path.exists():
        print(f"\n❌ 数据集配置文件不存在: {args.data}")
        print("   请先创建数据集配置文件，可参考 config/dataset_template.yaml")
        sys.exit(1)

    # 检查设备
    device = args.device or detect_device()
    print(f"\n🖥️  检测到设备: {device}")

    if device == "cpu":
        print("   ⚠️  未检测到 GPU，将使用 CPU 训练（速度较慢）")
        print("   建议: 使用 Google Colab 免费 GPU 或 AutoDL 租用 GPU")
        if args.batch > 8:
            print(f"   ⚠️  Batch Size 自动从 {args.batch} 调整为 8")
            args.batch = 8

    elif device == "mps":
        print("   ✅ 使用 Apple MPS 加速训练")
        if args.batch > 16:
            args.batch = 16

    elif device.startswith("cuda"):
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
        print(f"   ✅ GPU: {gpu_name} ({gpu_mem:.1f} GB)")

        if gpu_mem < 6 and args.batch > 8:
            print(f"   ⚠️  显存不足 {gpu_mem:.1f}GB，Batch Size 自动调整为 8")
            args.batch = 8
        elif gpu_mem < 4 and args.batch > 4:
            print(f"   ⚠️  显存不足 {gpu_mem:.1f}GB，Batch Size 自动调整为 4")
            args.batch = 4

    # 检查模型文件
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"\n📥 模型 {args.model} 不存在，将自动从 Ultralytics 下载...")

    print(f"\n📋 训练配置:")
    print(f"   数据集:     {args.data}")
    print(f"   预训练模型:  {args.model}")
    print(f"   训练轮数:    {args.epochs}")
    print(f"   图片尺寸:    {args.imgsz}")
    print(f"   Batch Size: {args.batch}")
    print(f"   学习率:      {args.lr0}")
    print(f"   设备:        {device}")
    if args.freeze:
        print(f"   冻结层数:    {args.freeze}")
    print()


def main():
    args = build_args()

    # 处理冲突参数
    if args.no_cache:
        args.cache = False

    validate_environment(args)

    device = args.device or detect_device()

    # 加载预训练模型
    model = YOLO(args.model)

    # 开始训练
    print("🚀 开始训练...\n")
    print("=" * 60)

    results = model.train(
        # === 数据集 ===
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=device,
        workers=args.workers,
        cache=args.cache,

        # === 微调关键参数 ===
        lr0=args.lr0,
        lrf=0.01,
        warmup_epochs=3 if args.epochs >= 30 else 1,
        cos_lr=args.cos_lr,
        optimizer="auto",
        weight_decay=0.0005,
        momentum=0.937,

        # === 冻结层 ===
        freeze=args.freeze,

        # === 数据增强 ===
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,

        # === 训练控制 ===
        pretrained=True,
        patience=args.patience,
        amp=True,
        save=True,
        save_period=10,
        val=True,
        plots=True,
        resume=args.resume,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("✅ 训练完成!")
    print(f"   最佳模型: {results.save_dir}/weights/best.pt")
    print(f"   最后模型: {results.save_dir}/weights/last.pt")
    print(f"   训练图表: {results.save_dir}/results.png")
    print(f"   混淆矩阵: {results.save_dir}/confusion_matrix.png")
    print()

    # 输出关键指标
    print("📊 训练结果摘要:")
    if hasattr(results, "results_dict"):
        rd = results.results_dict
        print(f"   mAP@0.5:      {rd.get('metrics/mAP50(B)', 'N/A')}")
        print(f"   mAP@0.5:0.95: {rd.get('metrics/mAP50-95(B)', 'N/A')}")
        print(f"   Precision:    {rd.get('metrics/precision(B)', 'N/A')}")
        print(f"   Recall:       {rd.get('metrics/recall(B)', 'N/A')}")

    print(f"\n💡 下一步:")
    print(f"   python scripts/evaluate.py --model {results.save_dir}/weights/best.pt")
    print(f"   python scripts/predict.py --model {results.save_dir}/weights/best.pt --source your_image.jpg")


if __name__ == "__main__":
    main()