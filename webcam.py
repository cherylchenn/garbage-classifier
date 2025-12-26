import cv2
import torch
import numpy as np
from torchvision import models, transforms

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "model.pth"
LABELS_PATH = "labels.txt"

BIN_MAP = {
    "cardboard": "Black",
    "paper": "Black",
    "metal": "Blue",
    "glass": "Blue",
    "plastic": "Blue",
    "trash": "Garbage"
}

def load_classes():
    with open(LABELS_PATH) as f:
        classes = [line.strip() for line in f.readlines()]
    return classes

def build_model(num_classes):
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def preprocess_frame(frame, transform):
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = transform(image)
    image = image.unsqueeze(0)
    return image.to(DEVICE)

def predict(model, tensor):
    with torch.no_grad():
        outputs = model(tensor)
        _, pred = torch.max(outputs, 1)
    return pred.item()

def draw_prediction(frame, label):
    bin_text = BIN_MAP[label]

    cv2.putText(frame, f"Item: {label}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, 
                (0, 255, 0), 2, cv2.LINE_AA)
    
    cv2.putText(frame, f"Bin: {bin_text}", (20, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, 
                (0, 200, 255), 2, cv2.LINE_AA)
    
    return frame

def run_webcam(model, classes, transform):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam...")
        return
    print("Webcam opened!")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame...")
            break

        tensor = preprocess_frame(frame, transform)
        pred_index = predict(model, tensor)
        label = classes[pred_index]

        frame = draw_prediction(frame, label)
        cv2.imshow("Garbage Classifier", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed!")

if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    classes = load_classes()
    print("Classes:", classes)
    model = build_model(len(classes))
    print("Model built! Running webcam...")
    run_webcam(model, classes, transform)