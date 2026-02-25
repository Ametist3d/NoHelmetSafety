from ultralytics import YOLO

# Load a pretrained YOLO model (adjust model type as needed)
model = YOLO("yolo26n.pt")  # n, s, m, l, x versions available

# Perform object detection on an image
results = model.predict(source="DS\\datasets\\vodan37\\helm\\helm\\images\\test\\helm_000014.jpg")  # Can also use video, directory, URL, etc.

# Display the results
results[0].show()  # Show the first image results