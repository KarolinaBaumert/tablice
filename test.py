from ultralytics import YOLO

# Załaduj wytrenowany model
model = YOLO("runs/detect/train2/weights/best.pt")
model.to('cpu')  # Force the model to use the CPU
# Wykonaj predykcję na danych walidacyjnych
results = model.predict(source="datasets/processed_data/images/val", save=True, conf=0.017, save_txt=True)

detected = 0
undetected = 0

for r in results:
    if len(r.boxes) > 0:
        detected += 1
    else:
        undetected += 1

print(f"📸 Wykryto obiekty na {detected} obrazach.")
print(f"🚫 Brak wykryć na {undetected} obrazach.")