# $MODEL = "yolo11n.pt"
# $DATA = "DS/pps.yaml"
# $NAME = "helmet_exp1"

yolo detect train `
  model="yolo26s.pt" `
  data="DS/pps.yaml" `
  imgsz=1024 `
  epochs=30 `
  batch=20 `
  device=0 `
  close_mosaic=10 `
  project="DS/runs/train" `
  name="helmet_yolo26s_exp3_1280" `
  exist_ok=True