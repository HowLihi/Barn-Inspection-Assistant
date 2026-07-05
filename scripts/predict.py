#!/usr/bin/env python3
"""
模型推理测试脚本

用法:
    python scripts/predict.py --model best.pt --source test.jpg
    python scripts/predict.py --model best.pt --source video.mp4
    python scripts/predict.py --model best.pt --source 0           # 摄像头
    python scripts/predict.py --model best.pt --source rtsp://...  # RTSP 流
"""
from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLO 模型推理")
    parser.add_argument("--model", type=str, required=True, help="模型权重路径")
    parser.add_argument("--source", type=str, required=True, help="输入源 (图片/视频/摄像头ID/RTSP)")
    parser.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IOU 阈值")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--save", action="store_true", default=True, help="保存结果")
    parser.add_argument("--show", action="store_true", help="实时显示 (视频/摄像头)")
    parser.add_argument("--hide_labels", action="store_true", help="不显示标签")
    parser.add_argument("--hide_conf", action="store_true", help="不显示置信度")
    return parser.parse_args()


def main():
    args = build_args()

    model = YOLO(args.model)
    print(f"\n🔍 模型: {args.model}")
    print(f"   类别: {list(model.names.values())}")
    print(f"   输入: {args.source}")
    print()

    results = model.predict(
        source=args.source,
        conf=args.conf,
        iou=args.iou,
        imgsz=args.imgsz,
        save=args.save,
        show=args.show,
        stream=True,
        hide_labels=args.hide_labels,
        hide_conf=args.hide_conf,
    )

    total_detections = 0
    for i, result in enumerate(results):
        boxes = result.boxes
        if boxes is not None and len(boxes) > 0:
            total_detections += len(boxes)
            print(f"帧 {i}: 检测到 {len(boxes)} 个目标")
            for cls_id, conf in zip(boxes.cls, boxes.conf):
                name = model.names[int(cls_id)]
                print(f"   {name} 置信度 {conf:.2f}")

    if total_detections > 0:
        print(f"\n📊 总计检测到 {total_detections} 个目标")
    else:
        print("\n📊 未检测到目标 (可尝试降低 --conf 阈值)")

    print(f"\n📁 结果保存路径: {Path('runs/detect/predict').absolute()}")


if __name__ == "__main__":
    main()