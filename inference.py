# inference.py
# Webcam/video helmet compliance inference: head+helmet detection + no-helmet logic
# Features:
# - mode: stream (webcam) or video
# - per-class thresholds (head_conf, helmet_conf)
# - match rule: helmet center inside head OR IoU >= iou_thr
# - skip N frames for YOLO inference, with non-blinking boxes via simple temporal hold (TTL)
# - optional save annotated video

import argparse
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2
from ultralytics import YOLO


def xyxy_iou(a: List[float], b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return float(inter / (area_a + area_b - inter + 1e-9))


def center_inside(box_small: List[float], box_big: List[float]) -> bool:
    x1, y1, x2, y2 = box_small
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    X1, Y1, X2, Y2 = box_big
    return (X1 <= cx <= X2) and (Y1 <= cy <= Y2)


def clamp_xyxy(xyxy: List[float], w: int, h: int) -> List[float]:
    x1, y1, x2, y2 = xyxy
    x1 = max(0.0, min(float(x1), w - 2.0))
    y1 = max(0.0, min(float(y1), h - 2.0))
    x2 = max(x1 + 2.0, min(float(x2), w - 1.0))
    y2 = max(y1 + 2.0, min(float(y2), h - 1.0))
    return [x1, y1, x2, y2]


def draw_box(img, xyxy: List[float], text: str, color: Tuple[int, int, int], thickness: int = 2):
    x1, y1, x2, y2 = map(int, map(round, xyxy))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    if text:
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        y = max(0, y1 - th - baseline - 6)
        cv2.rectangle(img, (x1, y), (x1 + tw + 8, y + th + baseline + 8), color, -1)
        cv2.putText(img, text, (x1 + 4, y + th + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)


@dataclass
class Det:
    xyxy: List[float]
    cls: int
    conf: float
    name: str


@dataclass
class FrameState:
    heads: List[Tuple[List[float], float]]          # (xyxy, conf)
    helmets: List[Tuple[List[float], float]]        # (xyxy, conf)
    no_helmet_heads: List[Tuple[List[float], float]]


def run_yolo(
    model: YOLO,
    frame,
    imgsz: int,
    device: str,
    global_conf: float,
    nms_iou: float,
    max_det: int,
    half: bool,
) -> List[Det]:
    res = model.predict(
        source=frame,
        imgsz=imgsz,
        device=device,
        conf=global_conf,
        iou=nms_iou,
        max_det=max_det,
        half=half,
        verbose=False,
    )[0]

    out: List[Det] = []
    if res.boxes is None or len(res.boxes) == 0:
        return out

    names = model.names
    for b in res.boxes:
        cls = int(b.cls[0].item())
        conf = float(b.conf[0].item())
        xyxy = b.xyxy[0].cpu().numpy().tolist()
        out.append(Det(xyxy=xyxy, cls=cls, conf=conf, name=names.get(cls, str(cls))))
    return out


def postprocess(
    dets: List[Det],
    head_cls: int,
    helmet_cls: int,
    head_conf: float,
    helmet_conf: float,
    iou_thr: float,
    frame_w: int,
    frame_h: int,
) -> FrameState:
    heads: List[Tuple[List[float], float]] = []
    helmets: List[Tuple[List[float], float]] = []

    for d in dets:
        xyxy = clamp_xyxy(d.xyxy, frame_w, frame_h)
        if d.cls == head_cls and d.conf >= head_conf:
            heads.append((xyxy, d.conf))
        elif d.cls == helmet_cls and d.conf >= helmet_conf:
            helmets.append((xyxy, d.conf))

    no_helmet_heads: List[Tuple[List[float], float]] = []
    for hxyxy, hconf in heads:
        matched = False
        for kxyxy, _ in helmets:
            if center_inside(kxyxy, hxyxy) or (xyxy_iou(hxyxy, kxyxy) >= iou_thr):
                matched = True
                break
        if not matched:
            no_helmet_heads.append((hxyxy, hconf))

    return FrameState(heads=heads, helmets=helmets, no_helmet_heads=no_helmet_heads)


def overlay(frame, state: FrameState):
    vis = frame

    # helmets = blue
    for xyxy, conf in state.helmets:
        draw_box(vis, xyxy, f"helmet {conf:.2f}", (255, 0, 0), 2)

    # heads: red if NO_HELMET else green
    no_set = [nh[0] for nh in state.no_helmet_heads]
    for xyxy, conf in state.heads:
        flagged = any(xyxy_iou(xyxy, nhxyxy) > 0.90 for nhxyxy in no_set)
        color = (0, 0, 255) if flagged else (0, 255, 0)
        label = "NO_HELMET" if flagged else "head"
        draw_box(vis, xyxy, f"{label} {conf:.2f}", color, 2)

    cv2.putText(
        vis,
        f"heads={len(state.heads)} helmets={len(state.helmets)} no_helmet={len(state.no_helmet_heads)}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )   
    cv2.putText(
        vis,
        f"ESC to quit",
        (470, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 0),
        2,
    )
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["video", "stream"], required=True)

    ap.add_argument("--input", default=None, help="Input video path (mode=video)")
    ap.add_argument("--output", default=None, help="Output video path (mode=video). If omitted: <input>_annotated.mp4")

    ap.add_argument("--webcam", type=int, default=0, help="Webcam id (mode=stream)")
    ap.add_argument("--model", default=r"model\\weights\\best.pt")

    ap.add_argument("--imgsz", type=int, default=960, help="YOLO inference resolution (imgsz)")
    ap.add_argument("--device", default="0", help="'0' for GPU0 or 'cpu'")
    ap.add_argument("--half", action="store_true", help="Use FP16 (GPU only)")
    ap.add_argument("--max_det", type=int, default=100)

    # YOLO inference knobs
    ap.add_argument("--nms_iou", type=float, default=0.45, help="NMS IoU used inside model.predict()")
    ap.add_argument("--skip", type=int, default=0, help="Run YOLO every (skip+1) frames. 0 = every frame")

    # class ids + per-class thresholds
    ap.add_argument("--head_cls", type=int, default=0)
    ap.add_argument("--helmet_cls", type=int, default=1)
    ap.add_argument("--head_conf", type=float, default=0.40)
    ap.add_argument("--helmet_conf", type=float, default=0.20)

    # head↔helmet matching threshold (post-process)
    ap.add_argument("--iou_thr", type=float, default=0.10)

    # temporal hold (prevents blinking on skipped frames)
    ap.add_argument("--hold", type=int, default=8, help="How many frames to keep last detections when skipping")

    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()

    model = YOLO(args.model)
    print("[info] model loaded:", args.model)
    print("[info] model.names:", model.names)

    if args.mode == "video":
        if not args.input:
            raise SystemExit("ERROR: --input is required for mode=video")
        cap = cv2.VideoCapture(args.input)
        if not cap.isOpened():
            raise SystemExit(f"ERROR: cannot open input video: {args.input}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_path = args.output
        if not out_path:
            in_p = Path(args.input)
            out_path = str(in_p.with_name(in_p.stem + "_annotated.mp4"))

        os.makedirs(str(Path(out_path).parent), exist_ok=True)
        writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), float(fps), (w, h))
        if not writer.isOpened():
            raise SystemExit(f"ERROR: cannot open output writer: {out_path}")

        print(f"[info] writing -> {out_path}")
    else:
        cap = cv2.VideoCapture(args.webcam)
        if not cap.isOpened():
            raise SystemExit(f"ERROR: cannot open webcam id={args.webcam}")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        writer = None
        out_path = None

    frame_idx = 0

    # For "no blinking": keep last computed state for a few frames
    last_state: Optional[FrameState] = None
    hold_left = 0

    # IMPORTANT: global_conf must be <= both per-class conf thresholds
    global_conf = min(args.head_conf, args.helmet_conf)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame_idx += 1
        fh, fw = frame.shape[:2]

        do_detect = (frame_idx == 1) or (args.skip <= 0) or (frame_idx % (args.skip + 1) == 0)

        if do_detect:
            dets = run_yolo(
                model=model,
                frame=frame,
                imgsz=args.imgsz,
                device=args.device,
                global_conf=global_conf,
                nms_iou=args.nms_iou,
                max_det=args.max_det,
                half=args.half,
            )
            state = postprocess(
                dets=dets,
                head_cls=args.head_cls,
                helmet_cls=args.helmet_cls,
                head_conf=args.head_conf,
                helmet_conf=args.helmet_conf,
                iou_thr=args.iou_thr,
                frame_w=fw,
                frame_h=fh,
            )
            last_state = state
            hold_left = args.hold
        else:
            # Skip inference: reuse last state for a few frames
            if last_state is not None and hold_left > 0:
                state = last_state
                hold_left -= 1
            else:
                state = FrameState(heads=[], helmets=[], no_helmet_heads=[])

        vis = overlay(frame, state)

        if writer is not None:
            writer.write(vis)

        if args.show:
            cv2.imshow("NoHelmet inference", vis)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break

    cap.release()
    if writer is not None:
        writer.release()
        print(f"[done] saved: {out_path}")
    if args.show:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()