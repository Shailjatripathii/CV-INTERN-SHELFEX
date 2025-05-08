
# Custom Object Detection with YOLOv8

This project trains a custom YOLOv8 model for object detection and counts the detected objects from test images. The final counts are saved into a JSON file.

---

## Project Structure

```
├── data.yaml                         # Dataset configuration
├── train/                             # Training images
│   ├── images/
│   ├── labels/
├── test/                              # Test images
│   ├── images/
│   ├── labels/
├── final_counts.json                  # Output: final object counts
├── your_script.py                     # Main training and inference script
└── README.md                          # This file
```

---
## Tools and Technologies

- **Programming Language**: Python 3.x
- **Deep Learning Framework**: YOLOv8 (Ultralytics)
- **Notebook Environment**: Google Colab / Jupyter Notebook / Local IDEs (VS Code, PyCharm)
- **Libraries Used**:
  - `ultralytics` for YOLOv8
  - `json` for saving output in JSON format
  - `collections.Counter` for counting occurrences
  - `os` for handling file paths
- **Output Format**: JSON file containing class counts
- **Storage**: Google Drive (if using Colab)

---

##  How to Run

### 1. Install Required Libraries

```bash
pip install ultralytics
```

### 2. Prepare Your Dataset

- Dataset folder should be organized like:
  ```
  train/
    images/
    labels/
  test/
    images/
    labels/
  ```
- Your `data.yaml` should specify paths to training and validation datasets:

Example `data.yaml`:
```yaml
path: /content/drive/MyDrive/computer vision_yolov8
train: train/images
val: test/images

names:
  0: Product-Name
  1: Empty-Space
  2: Other-Product
```

### 3. Train the Model

```python
from ultralytics import YOLO

model = YOLO('yolov8s.pt')  # or yolov8n.pt, yolov8m.pt, etc.
model.train(
    data='/path/to/data.yaml',
    epochs=30,
    imgsz=640,
    batch=16,
    name='custom_object_detection_model'
)
```

### 4. Inference on Test Images

```python
model = YOLO('runs/detect/custom_object_detection_model/weights/best.pt')
results = model.predict('/path/to/test/images', save=False)
```

### 5. Count Detected Objects

```python
from collections import Counter
import json

counts = Counter({"Product-Name": 0, "Empty-Space": 0, "Other-Product": 0})

for result in results:
    boxes = result.boxes
    for box in boxes:
        cls_id = int(box.cls[0].item())
        cls_name = ["Product-Name", "Empty-Space", "Other-Product"][cls_id]
        counts[cls_name] += 1

with open('final_counts.json', 'w') as f:
    json.dump(counts, f, indent=4)
```

---

##  Customization

- **Change Model Size:** Swap `'yolov8s.pt'` with `'yolov8n.pt'`, `'yolov8m.pt'`, `'yolov8l.pt'`, or `'yolov8x.pt'` based on your accuracy/speed needs.
- **Adjust Hyperparameters:** Modify `epochs`, `imgsz`, `batch`, etc. in the `.train()` method for better results.
- **Add More Classes:** Update `names:` section in `data.yaml` if you have additional object types.

---

##  Example Output

**final_counts.json**
```json
{
    "Product-Name": 5,
    "Empty-Space": 1,
    "Other-Product": 0
}
```
Saving and Downloading the Output
##If you're using Google Colab:
**To download the output JSON file directly in Colab**, use:
`````
python
Copy
Edit
from google.colab import files
files.download('final_counts.json')
This will trigger a browser download of the file.
```````
## If you're using Jupyter Notebook, VS Code, or running the script locally:
The output JSON file will be saved in your current working directory. You can access and download it manually from your file explorer.
`````
python
Copy
Edit
with open('final_counts.json', 'w') as f:
    json.dump(counts, f, indent=4)

print("JSON file 'final_counts.json' saved successfully.")
The file will be available in your project directory.
````

---

## Notes

- Make sure your **GPU** is available for faster training.
- Always check that your `.yaml` paths are **correct and accessible**.
- Larger YOLO models (`m`, `l`, `x`) need more memory — use smaller (`n`, `s`) models if running into memory issues.

---

##Author
**Shailja Tripathi**

