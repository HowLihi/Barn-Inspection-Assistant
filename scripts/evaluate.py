#!/usr/bin/env python3
"""
模型评估脚本

用法:
    python scripts/evaluate.py --model runs/detect/train/weights/best.pt
    python scripts/evaluate.py --model best.pt --data config/dataset.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO 模型评估")
    parser.add_argument("--model", type=str, required=True, help="模型权重路径")
    parser.add_argument("--data", type=str, default=None, help="数据集配置 (默认从模型自动读取)")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.001, help="验证置信度阈值")
    parser.add_argument("--iou", type=float, default=0.6, help="验证 IOU 阈值")
    parser.add_argument("--save_json", action="store_true", help="保存评估结果为 JSON")
    return parser.parse_args()


def main():
    args = build_args()

    model = YOLO(args.model)

    print(f"\n📊 评估模型: {args.model}")
    print(f"   模型任务: {model.task}")
    print(f"   类别数量: {len(model.names)}")
    print(f"   类别列表: {list(model.names.values())}")
    print()

    metrics = model.val(
        data=args.data,
        imgsz=args.imgsz,
        batch=args.batch,
        conf=args.conf,
        iou=args.iou,
        plots=True,
    )

    print("\n" + "=" * 60)
    print("📊 评估结果")
    print("=" * 60)
    print(f"   mAP@0.5:      {metrics.box.map50:.4f}")
    print(f"   mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"   Precision:    {metrics.box.mp:.4f}")
    print(f"   Recall:       {metrics.box.mr:.4f}")
    print()

    if hasattr(metrics.box, "ap50") and metrics.box.ap50 is not None:
        print("   各类别 mAP@0.5:")
        for i, ap in enumerate(metrics.box.ap50):
            name = model.names.get(i, f"class_{i}")
            print(f"      {name:20s} {ap:.4f}")

    if args.save_json:
        result = {
            "model": args.model,
            "mAP50": metrics.box.map50,
            "mAP50_95": metrics.box.map,
            "precision": metrics.box.mp,
            "recall": metrics.box.mr,
            "per_class": {
                model.names.get(i, f"class_{i}"): float(ap)
                for i, ap in enumerate(metrics.box.ap50)
            } if hasattr(metrics.box, "ap50") and metrics.box.ap50 is not None else {},
        }
        json_path = Path(args.model).with_suffix(".eval.json")
        json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        print(f"\n   📄 评估结果已保存: {json_path}")


if __name__ == "__main__":
    main()