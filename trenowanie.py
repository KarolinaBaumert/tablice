import xml.etree.ElementTree as ET
import os
import random
import shutil

# Ścieżki do danych
xml_path = "poland-vehicle-license-plate-dataset/annotations.xml"
images_path = "poland-vehicle-license-plate-dataset/photos"
base_path = "datasets/processed_data"

# Ścieżki do folderów YOLO
images_train = os.path.join(base_path, "images/train")
images_val = os.path.join(base_path, "images/val")
labels_train = os.path.join(base_path, "labels/train")
labels_val = os.path.join(base_path, "labels/val")

# Tworzenie folderów
os.makedirs(images_train, exist_ok=True)
os.makedirs(images_val, exist_ok=True)
os.makedirs(labels_train, exist_ok=True)
os.makedirs(labels_val, exist_ok=True)

# Funkcja do konwersji adnotacji do formatu YOLO
def convert_to_yolo_format(annotation, image_width, image_height):
    x_min = int(float(annotation['xtl']))
    y_min = int(float(annotation['ytl']))
    x_max = int(float(annotation['xbr']))
    y_max = int(float(annotation['ybr']))
    x_center = (x_min + x_max) / 2 / image_width
    y_center = (y_min + y_max) / 2 / image_height
    width = (x_max - x_min) / image_width
    height = (y_max - y_min) / image_height
    return x_center, y_center, width, height

# Załaduj XML
tree = ET.parse(xml_path)
root = tree.getroot()

# Tymczasowy folder na pliki .txt
tmp_labels_path = os.path.join(base_path, "tmp_labels")
os.makedirs(tmp_labels_path, exist_ok=True)

# Tworzenie plików .txt z adnotacjami
all_txt_files = []
for image in root.findall('image'):
    image_name = image.get('name')
    width = int(image.get('width'))
    height = int(image.get('height'))
    txt_filename = os.path.splitext(image_name)[0] + ".txt"
    txt_filepath = os.path.join(tmp_labels_path, txt_filename)
    with open(txt_filepath, 'w') as f:
        for box in image.findall('box'):
            if box.get('label') == 'plate':
                x_center, y_center, w, h = convert_to_yolo_format(box.attrib, width, height)
                f.write(f"0 {x_center} {y_center} {w} {h}\n")
    all_txt_files.append(txt_filename)

# Losowy podział na trening i walidację
random.shuffle(all_txt_files)
split_index = int(0.7 * len(all_txt_files))
train_files = all_txt_files[:split_index]
val_files = all_txt_files[split_index:]

# Przenoszenie danych do struktur YOLO
for file in train_files:
    base_name = os.path.splitext(file)[0]
    shutil.copy(os.path.join(tmp_labels_path, file), os.path.join(labels_train, file))
    shutil.copy(os.path.join(images_path, base_name + ".jpg"), os.path.join(images_train, base_name + ".jpg"))

for file in val_files:
    base_name = os.path.splitext(file)[0]
    shutil.copy(os.path.join(tmp_labels_path, file), os.path.join(labels_val, file))
    shutil.copy(os.path.join(images_path, base_name + ".jpg"), os.path.join(images_val, base_name + ".jpg"))

# Usunięcie tymczasowego folderu
shutil.rmtree(tmp_labels_path)

# Tworzenie pliku data.yaml
with open(os.path.join(base_path, "data.yaml"), "w") as f:
    f.write(
        f"path: {base_path}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: plate\n"
    )

print("Gotowe")
