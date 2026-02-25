# 🦺 NoHelmet -- Real-Time Helmet Detection with YOLO

Real-time helmet detection system based on YOLO (Ultralytics), trained
for construction-site safety monitoring.

Supports: - Webcam inference - Video file inference - GPU acceleration -
Separate confidence thresholds per class - Optimized for close, medium,
and distant shots

------------------------------------------------------------------------

# Project Structure

NOHELMET/ │ ├── inference.py \# Main inference script ├── train.ps1 \#
Training script ├── pps.yaml \# Dataset config ├── requirements.txt ├──
DS/ \# Dataset (excluded from git) ├── runs/ \# Training outputs ├──
model/ \# Training result images └── weights/ \# Trained weights

------------------------------------------------------------------------

# Installation

pip install -r requirements.txt

Make sure CUDA is installed if using GPU.

------------------------------------------------------------------------

# Training

\$MODEL = "yolo11n.pt" \$DATA = "DS/pps.yaml" \$NAME = "helmet_exp1"

yolo detect train `model=$MODEL`
data=$DATA `  imgsz=1280 `  epochs=30 `  device=0 `  batch=96 `  name=$NAME
`project="DS/runs/train"` exist_ok=True

------------------------------------------------------------------------

# Classes

0 → head\
1 → helmet

------------------------------------------------------------------------

# Model Performance

-   mAP@0.5: \~0.975\
-   Precision: \~0.94--0.95\
-   Recall: \~0.93--0.94

Helmet class slightly outperforms head class.

------------------------------------------------------------------------

# Inference

Webcam:

python inference.py --mode stream --device 0

Video:

python inference.py \^ --mode video \^ --input
.`\test`{=tex}\_video`\video`{=tex}.mp4 \^ --output output.mp4 \^
--model
DS`\runs`{=tex}`\train`{=tex}`\helmet`{=tex}\_yolo26s_exp3_1280`\weights`{=tex}`\best`{=tex}.pt
\^ --imgsz 1280 \^ --head_conf 0.40 \^ --helmet_conf 0.20 \^ --iou_thr
0.10

------------------------------------------------------------------------

# Advanced Parameters

--head_conf Confidence threshold for head\
--helmet_conf Confidence threshold for helmet\
--iou_thr IoU threshold\
--imgsz Inference resolution\
--skip_frames Skip N frames (tracking prevents blinking)

------------------------------------------------------------------------

# Use Case

-   Construction site monitoring\
-   PPE compliance detection\
-   Industrial safety automation\
-   Smart surveillance systems

------------------------------------------------------------------------

# Future Improvements

-   Add explicit "no_helmet" class\
-   Temporal smoothing with tracker\
-   Multi-scale training\
-   RTSP deployment
