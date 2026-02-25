import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO


def xyxy_iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


def center_inside(box_small, box_big):
    x1, y1, x2, y2 = box_small
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    X1, Y1, X2, Y2 = box_big
    return (X1 <= cx <= X2) and (Y1 <= cy <= Y2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=str, required=True, help="Path to trained weights .pt")
    ap.add_argument("--glob", type=str, default=r".\test_video\*.mp4", help="Video glob")
    ap.add_argument("--out", type=str, default=r".\runs\no_helmet_video", help="Output folder")
    ap.add_argument("--device", type=str, default="0")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--half", action="store_true", help="Use FP16 (GPU only)")

    # class ids in YOUR model
    ap.add_argument("--head_cls", type=int, default=2)
    ap.add_argument("--helmet_cls", type=int, default=0)

    # thresholds
    ap.add_argument("--head_conf", type=float, default=0.4)
    ap.add_argument("--helmet_conf", type=float, default=0.25)
    ap.add_argument("--iou_thr", type=float, default=0.10)

    # speed knobs
    ap.add_argument("--skip", type=int, default=0, help="Process every (skip+1) frames. 0=all frames")
    ap.add_argument("--max_det", type=int, default=100)

    args = ap.parse_args()

    model = YOLO(args.model)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted(Path(".").glob(args.glob))
    if not videos:
        raise FileNotFoundError(f"No videos matched: {args.glob}")

    for vp in videos:
        cap = cv2.VideoCapture(str(vp))
        if not cap.isOpened():
            print(f"[warn] can't open {vp}")
            continue

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out_path = out_dir / f"{vp.stem}_annotated.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))

        frame_i = 0
        processed = 0

        print(f"[info] processing {vp.name} -> {out_path.name} ({w}x{h}@{fps:.1f})")

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if args.skip > 0 and (frame_i % (args.skip + 1) != 0):
                writer.write(frame)
                frame_i += 1
                continue

            # Inference
            res = model.predict(
                source=frame,
                imgsz=args.imgsz,
                device=args.device,
                conf=min(args.head_conf, args.helmet_conf),
                max_det=args.max_det,
                half=args.half,
                verbose=False
            )[0]

            heads = []
            helmets = []

            if res.boxes is not None and len(res.boxes) > 0:
                for b in res.boxes:
                    cls = int(b.cls.item())
                    conf = float(b.conf.item())
                    xyxy = b.xyxy[0].cpu().numpy().tolist()
                    if cls == args.head_cls and conf >= args.head_conf:
                        heads.append((xyxy, conf))
                    elif cls == args.helmet_cls and conf >= args.helmet_conf:
                        helmets.append((xyxy, conf))

            # Match helmets to heads
            no_helmet = []
            for hxyxy, hconf in heads:
                matched = False
                for kxyxy, kconf in helmets:
                    if center_inside(kxyxy, hxyxy) or (xyxy_iou(hxyxy, kxyxy) >= args.iou_thr):
                        matched = True
                        break
                if not matched:
                    no_helmet.append((hxyxy, hconf))

            # Draw
            vis = frame

            # helmets = blue
            for (x1, y1, x2, y2), conf in [(tuple(map(int, b[0])), b[1]) for b in helmets]:
                cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 0, 0), 2)
                cv2.putText(vis, f"helmet {conf:.2f}", (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            # heads: red if NO_HELMET else green
            for hxyxy, hconf in heads:
                x1, y1, x2, y2 = map(int, hxyxy)
                flagged = False
                for nhxyxy, _ in no_helmet:
                    if xyxy_iou(hxyxy, nhxyxy) > 0.9:
                        flagged = True
                        break
                color = (0, 0, 255) if flagged else (0, 255, 0)
                label = "NO_HELMET" if flagged else "head"
                cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
                cv2.putText(vis, f"{label} {hconf:.2f}", (x1, max(0, y1 - 6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

            # summary overlay
            cv2.putText(vis, f"heads={len(heads)} helmets={len(helmets)} no_helmet={len(no_helmet)}",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

            writer.write(vis)

            frame_i += 1
            processed += 1
            if processed % 100 == 0:
                print(f"[info] {vp.name}: processed {processed} frames (frame {frame_i})")

        cap.release()
        writer.release()
        print(f"[done] wrote {out_path}")

    print("[all done]")


if __name__ == "__main__":
    main()