import torch
import cv2
import os
from collections import Counter

# Example class names for punches
punch_classes = ["jab", "cross", "hook", "uppercut"]

# Initialize statistics
punch_counts = Counter()

# Function to load YOLOv7 model using torch.hub
def load_model(model_path=None):
    if model_path is None:
        # Load pre-trained YOLOv7 from the repository
        model = torch.hub.load('ultralytics/yolov7', 'yolov7', pretrained=True)
    else:
        # Load custom trained model
        model = torch.load(model_path, weights_only=True)
    model.eval()  # Set the model to evaluation mode
    return model

# Function to process model outputs and update statistics
def process_detections(detections, image):
    for detection in detections:
        class_id, confidence, x1, y1, x2, y2 = detection
        class_name = punch_classes[int(class_id)]
        punch_counts[class_name] += 1

        # Draw bounding box and class name on the image
        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(image, f"{class_name} ({confidence:.2f})", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Function to process images and generate statistics
def generate_punch_statistics_for_images(image_folder, model, output_folder=None):
    if output_folder and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for image_file in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_file)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Skipping {image_file}: Unable to read the file.")
            continue

        # Replace this with your actual model inference
        detections = [
            [0, 0.95, 50, 60, 200, 180],
            [2, 0.90, 300, 100, 400, 250],
        ]

        process_detections(detections, image)

        # Save the annotated image
        if output_folder:
            output_path = os.path.join(output_folder, image_file)
            cv2.imwrite(output_path, image)

    # Print final statistics
    print("\nPunch Statistics:")
    total_punches = sum(punch_counts.values())
    for punch, count in punch_counts.items():
        print(f"{punch.capitalize()}: {count} ({(count / total_punches) * 100:.2f}%)")
    print(f"Total Punches: {total_punches}")

model_path = "best.pt"  # Your custom model path
model = torch.load(model_path)
generate_punch_statistics_for_images("images", model=model, output_folder="output_images")
