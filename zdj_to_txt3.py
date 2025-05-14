import os
import random
import time  # Dodano do pomiaru czasu
import numpy as np
from torchaudio.functional import contrast
from torchvision.transforms.v2.functional import adjust_contrast
from ultralytics import YOLO
import cv2
import easyocr  # Dodano do odczytu tekstu
import re  # Dodano do przetwarzania tekstu
import xml.etree.ElementTree as ET  # Dodano do obsługi XML

# Folder, w którym znajdują się zdjęcia
photos_path = "poland-vehicle-license-plate-dataset/photos"

# Pobierz listę wszystkich plików w folderze
all_images = [f for f in os.listdir(photos_path) if f.endswith('.jpg')]

# Wybierz 100 losowych zdjęć
random_images = random.sample(all_images, 100)

# Ścieżki do tych zdjęć
random_image_paths = [os.path.join(photos_path, img) for img in random_images]

# Ścieżka do pliku XML
annotations_path = "poland-vehicle-license-plate-dataset/annotations.xml"

# Funkcja do wczytania danych z XML
def load_annotations(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    annotations = {}
    for image in root.findall('image'):
        image_name = image.get('name')
        for box in image.findall('box'):
            plate_number = box.find("attribute[@name='plate number']").text
            annotations[image_name] = plate_number
    return annotations

# Wczytaj dane z XML
annotations = load_annotations(annotations_path)

# Załaduj model YOLO
model = YOLO("runs/detect/train2/weights/best.pt")
model.to('cpu')  # Force the model to use the CPU

# Inicjalizacja EasyOCR
reader = easyocr.Reader(['en'], gpu=True)  # Inicjalizacja EasyOCR z użyciem GPU

def detect_license_plates(image_path):
    image = cv2.imread(image_path)

    # Wczytaj obraz
    results = model.predict(image, conf=0.02)

    # Wynik detekcji, z którego pobieramy liczby wykrytych tablic
    detections = results[0].boxes  # Wyniki detekcji dla pierwszego obrazu

    cropped_plate_paths = []  # Lista ścieżek do wyciętych tablic

    # Wycięcie i zapisanie wykrytych obszarów
    for i, box in enumerate(detections.xyxy):  # Pobierz współrzędne wykrytych obszarów
        x1, y1, x2, y2 = map(int, box[:4])  # Współrzędne prostokąta

        # Przytnij 12% szerokości obrazu z lewej strony
        width = x2 - x1
        x1 += int(0.08 * width)
        height = y2 - y1
        x2 -= int(0.025 * width)  # Odetnij 5% szerokości z prawej strony
        y2 -= int(0.01 * height)

        cropped_plate = image[y1:y2, x1:x2]  # Wycięcie obszaru
        cropped_plate1 = cv2.cvtColor(cropped_plate, cv2.COLOR_BGR2RGB)

        # Konwersja na skalę szarości
        cropped_plate_gray = cv2.cvtColor(cropped_plate1, cv2.COLOR_BGR2GRAY)
        # Wygładzanie medianowe (usuwanie szumów)
        #cropped_plate_gray = cv2.medianBlur(cropped_plate_gray, 3)
        cropped_plate_gray = cv2.GaussianBlur(cropped_plate_gray, (3, 3), 0)
        alpha = 1.2 # Współczynnik kontrastu
        beta = 0.1     # Wartość jasności
        cropped_plate_gray = cv2.convertScaleAbs(cropped_plate_gray, alpha=alpha, beta=beta)
        # Wyostrzanie
        #kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        #cropped_plate_gray = cv2.filter2D(cropped_plate_gray, -1, kernel)
        # Binaryzacja obrazu
        _, cropped_plate_binary = cv2.threshold(cropped_plate_gray, 100, 255, cv2.THRESH_BINARY_INV)
        output_path = f"detected_plates/plate_{os.path.basename(image_path).split('.')[0]}_{i}.jpg"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)  # Utwórz folder, jeśli nie istnieje
        cv2.imwrite(output_path, cropped_plate_binary)  # Zapisz wycięty obszar w postaci binarnej
        cropped_plate_paths.append(output_path)

    # Zwróć liczbę wykrytych tablic i ścieżki do wyciętych obrazów
    return len(detections), cropped_plate_paths

total_ocr_time = 0  # Zmienna do śledzenia łącznego czasu OCR

def perform_ocr_on_cropped(cropped_plate_paths):
    global total_ocr_time  # Użycie zmiennej globalnej
    detected_texts = []  # Lista na odczytane teksty

    for cropped_path in cropped_plate_paths:
        cropped_plate = cv2.imread(cropped_path)

        # Odczyt tekstu z wyciętego obrazu
        start_time = time.time()  # Rozpocznij pomiar czasu
        result = reader.readtext(cropped_plate, contrast_ths=0.9, adjust_contrast=0.6)
        elapsed_time = time.time() - start_time  # Zakończ pomiar czasu
        total_ocr_time += elapsed_time  # Dodaj czas OCR do łącznego czasu

        # Pobierz odczytany tekst (jeśli istnieje)
        if result:
            raw_text = result[0][-2]
            detected_text = raw_text.upper().replace('|', 'I')
            detected_text = re.sub(r'[^A-Z0-9]', '', detected_text)
            if len(detected_text) <= 2 and len(result) > 1:
                second_raw_text = result[1][-2]
                second_detected_text = second_raw_text.upper().replace('|', 'I')
                second_detected_text = re.sub(r'[^A-Z0-9]', '', second_detected_text)
                if 3 <= len(second_detected_text) <= 5:
                    detected_text = detected_text + second_detected_text
            # Usuń początkowe "I", jeśli występuje
            if detected_text.startswith("I"):
                detected_text = detected_text[1:]
            # Zamień '0' na 'O' na 1. lub 2. pozycji
            if len(detected_text) > 0 and detected_text[0] == '0':
                detected_text = 'O' + detected_text[1:]
            if len(detected_text) > 1 and detected_text[1] == '0':
                detected_text = detected_text[0] + 'O' + detected_text[2:]
            # Zamień '5' na 'S' na 1. lub 2. pozycji
            if len(detected_text) > 0 and detected_text[0] == '5':
                detected_text = 'S' + detected_text[1:]
            if len(detected_text) > 1 and detected_text[1] == '5':
                detected_text = detected_text[0] + 'S' + detected_text[2:]
            # Zamień '6' na 'G' na 1. lub 2. pozycji
            if len(detected_text) > 0 and detected_text[0] == '6':
                detected_text = 'G' + detected_text[1:]
            if len(detected_text) > 1 and detected_text[1] == '6':
                detected_text = detected_text[0] + 'G' + detected_text[2:]
            # Zamień '2' na 'Z' na 1. lub 2. pozycji
            if len(detected_text) > 0 and detected_text[0] == '2':
                detected_text = 'Z' + detected_text[1:]
            if len(detected_text) > 1 and detected_text[1] == '2':
                detected_text = detected_text[0] + 'Z' + detected_text[2:]
            # Zamień znaki po trzeciej literze
            if len(detected_text) > 3:
                prefix = detected_text[:3]
                suffix = detected_text[3:]
                suffix = suffix.replace('B', '8') \
                               .replace('I', '1') \
                               .replace('O', '0') \
                               .replace('Z', '2')
                detected_text = prefix + suffix
            detected_texts.append(detected_text)
            print(f"Odczytano tekst: {detected_text} (czas: {elapsed_time:.2f}s)")
        else:
            print(f"Nie udało się odczytać tekstu (czas: {elapsed_time:.2f}s)")

    return detected_texts

# Funkcja do zapisu wyników OCR do pliku
def save_ocr_results(results, output_file="ocr_results.txt"):
    with open(output_file, "w") as f:
        for image_name, detected_text in results.items():
            f.write(f"{image_name}: {detected_text}\n")

# Funkcja do porównania wyników OCR z XML
def compare_results(ocr_results, annotations):
    correct = 0
    total = len(ocr_results)
    for image_name, detected_text in ocr_results.items():
        if image_name in annotations and detected_text == annotations[image_name]:
            correct += 1
    accuracy = (correct / total) * 100 if total > 0 else 0
    return accuracy

# Funkcja do wyświetlenia źle wykrytych tekstów
def display_incorrect_results(ocr_results, annotations):
    print("\nŹle wykryte teksty:")
    for image_name, detected_text in ocr_results.items():
        if image_name in annotations and detected_text != annotations[image_name]:
            print(f"OCR: {detected_text} | XML: {annotations[image_name]}")

# ...existing code...

def calculate_final_grade(accuracy_percent: float, processing_time_sec: float) -> float:
    """
    Oblicza ocenę końcową na podstawie dokładności OCR i czasu przetwarzania.
    Parametry:
    - accuracy_percent: Dokładność OCR w procentach (0–100)
    - processing_time_sec: Całkowity czas przetwarzania 100 obrazów w sekundach
    Zwraca:
    - Ocena w skali od 2.0 do 5.0 (zaokrąglona do najbliższego 0.5)
    """
    # Sprawdź minimalne wymagania
    if accuracy_percent < 60 or processing_time_sec > 60:
        return 2.0
    # Normalizacja dokładności: 60% → 0.0, 100% → 1.0
    accuracy_norm = (accuracy_percent - 60) / 40
    # Normalizacja czasu: 60s → 0.0, 10s → 1.0
    time_norm = (60 - processing_time_sec) / 50
    # Oblicz wynik ważony
    score = 0.7 * accuracy_norm + 0.3 * time_norm

    grade = 2.0 + 3.0 * score
    # Zaokrąglij do najbliższego 0.5
    return round(grade * 2) / 2

# ...existing code...

# Liczba wykrytych tablic w 100 zdjęciach
total_detected = 0

ocr_results = {}  # Słownik do przechowywania wyników OCR

for image_path in random_image_paths:
    num_detected, cropped_paths = detect_license_plates(image_path)
    total_detected += num_detected
    texts = perform_ocr_on_cropped(cropped_paths)
    image_name = os.path.basename(image_path)
    ocr_results[image_name] = texts[0] if texts else ""  # Zapisz pierwszy odczytany tekst lub pusty ciąg
    print(f"Odczytane teksty dla {image_name}: {texts}")

# Zapisz wyniki OCR do pliku
save_ocr_results(ocr_results)


# Wyświetl źle wykryte teksty
display_incorrect_results(ocr_results, annotations)

# Porównaj wyniki OCR z danymi z XML
accuracy = compare_results(ocr_results, annotations)
print(f"Dokładność OCR: {accuracy:.2f}%")

# Wyświetl całkowitą liczbę wykrytych tablic
print(f"Liczba wykrytych tablic: {total_detected}")
print(f"Łączny czas OCR: {total_ocr_time:.2f} sekund")  # Wyświetl łączny czas OCR


# Oblicz ocenę końcową
final_grade = calculate_final_grade(accuracy, total_ocr_time)
print(f"Ostateczna ocena: {final_grade}")


