import cv2
import torch
import torch.nn.functional as F
from torchvision import models, transforms
import mediapipe as mp

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

def load_model(num_classes):
    model = models.resnet18(weights="IMAGENET1K_V1")
    model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def get_hand_roi(frame): # maybe detect item itself instead of hand
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if not result.multi_hand_landmarks:
        return None, None

    hand = result.multi_hand_landmarks[0]
    xs = [lm.x for lm in hand.landmark]
    ys = [lm.y for lm in hand.landmark]
    
    xmin = int(min(xs) * w)
    xmax = int(max(xs) * w)
    ymin = int(min(ys) * h)
    ymax = int(max(ys) * h)

    pad = 100 # expand box to include item in hand
    xmin = max(0, xmin - pad)
    ymin = max(0, ymin - pad)
    xmax = min(w, xmax + pad)
    ymax = min(h, ymax + pad)

    roi = frame[ymin:ymax, xmin:xmax]
    return roi, (xmin, ymin, xmax, ymax)

def predict(model, roi, transform):
    img = transform(roi).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(img)
        probs = F.softmax(outputs, dim=1)
        conf, pred = torch.max(probs, dim=1)
    
    return conf.item(), pred.item()

def draw_info(frame, box, label, conf):
    (xmin, ymin, xmax, ymax) = box
    bin_text = BIN_MAP[label]

    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0,255,0), 2)

    text = f"{label} ({conf*100:.1f}%) - {bin_text}"
    cv2.putText(frame, text,
                (xmin, ymin - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0,255,0), 2)
    
    return frame

def run_webcam(model, classes, transform):
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Could not open webcam...")
        return
    print("Webcam opened! Press q to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame...")
            break

        roi, box = get_hand_roi(frame)
        if roi is not None and roi.size > 0:
            conf, pred_index = predict(model, roi, transform)
            label = classes[pred_index]
            frame = draw_info(frame, box, label, conf)
        
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
    model = load_model(len(classes))
    print("Model built! Running webcam...")
    run_webcam(model, classes, transform)