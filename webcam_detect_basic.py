import argparse
import cv2
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=r"DS\runs\train\helmet_yolo26s_exp3_1280\weights\best.pt")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--iou", type=float, default=0.6)
    ap.add_argument("--device", default="0")  # "0" or "cpu"
    ap.add_argument("--max_det", type=int, default=50)
    args = ap.parse_args()

    print("[info] loading model:", args.model)
    model = YOLO(args.model)
    print("[info] model names:", model.names)

    cap = cv2.VideoCapture(args.cam)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam id={args.cam}")

    # small latency improvements for some webcams
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Inference
        res = model.predict(
            source=frame,
            imgsz=args.imgsz,
            head_conf = 0.40,
            helmet_conf = 0.20,
            iou=0.1,
            device=args.device,
            max_det=args.max_det,
            verbose=False,
        )[0]

        vis = frame.copy()

        # Draw boxes
        if res.boxes is not None and len(res.boxes) > 0:
            for b in res.boxes:
                x1, y1, x2, y2 = b.xyxy[0].cpu().numpy().tolist()
                cls = int(b.cls[0].item())
                score = float(b.conf[0].item())
                name = model.names.get(cls, str(cls))

                x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
                cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
                cv2.putText(
                    vis,
                    f"{name} {score:.2f}",
                    (x1, max(0, y1 - 7)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

        cv2.putText(
            vis,
            "ESC to quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

        cv2.imshow("YOLO webcam basic", vis)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()