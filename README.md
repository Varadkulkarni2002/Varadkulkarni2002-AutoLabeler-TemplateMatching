AutoLabeler – Template Matching Annotation Tool

AutoLabeler is a desktop GUI application designed to accelerate dataset annotation for Computer Vision projects using a classical Template Matching approach.

Instead of manually drawing bounding boxes for every image, AutoLabeler allows you to provide one reference image + its annotation, and it automatically propagates that label across a folder of similar images.

It supports exporting annotations in YOLO, Pascal VOC (XML), and JSON (LabelMe-style) formats, making it compatible with most modern training pipelines.

🚀 Why AutoLabeler?

Manual annotation is:

Time-consuming

Error-prone

Overkill for controlled environments

AutoLabeler is built for real-world industrial and academic datasets where:

Camera angle is fixed

Object appearance is consistent

Deep learning is unnecessary or too heavy

Think assembly lines, inspection setups, screenshots, standardized image capture.

✨ Features
🔹 Automated Label Propagation

Uses OpenCV Template Matching to detect objects based on a reference crop

One labeled image → many labeled images

🔹 Multi-Format Support

Input formats

YOLO (.txt)

Pascal VOC (.xml)

JSON (LabelMe style)

Output formats

YOLO

Pascal VOC (XML)

JSON

🔹 GUI-Based (No CLI Pain)

Built with Tkinter

No command-line arguments

Beginner-friendly and demo-ready

🔹 Real-Time Preview

Preview Sample option to validate detection quality

Tune confidence threshold before batch processing

🔹 Non-Destructive Workflow

Original images are never modified

Images + labels are copied to a new output directory

🔹 Responsive UI

Uses threading

UI remains active during long batch operations

🧠 System Architecture

The tool follows a classical Computer Vision pipeline:

Ingestion

Reads the reference label file (YOLO / XML / JSON)

Extracts bounding box coordinates

Template Extraction

Crops the object region from the reference image

Template Matching

Applies cv2.matchTemplate

Uses Normalized Correlation Coefficient

Filtering

Applies user-defined confidence threshold (0.0 – 1.0)

Serialization

Converts detected bounding boxes into the selected output format

Handles YOLO normalization automatically

🛠️ Installation
Prerequisites

Python 3.7+

pip

Setup

Clone the repository:

git clone https://github.com/Varadkulkarni2002/Varadkulkarni2002-AutoLabeler-TemplateMatching.git
cd Varadkulkarni2002-AutoLabeler-TemplateMatching


Install dependencies:

pip install opencv-python numpy Pillow


Note:
tkinter is usually included with Python.
On Linux, if missing:

sudo apt install python3-tk


Run the application:

python autolabeller.py

📖 Usage Guide
1️⃣ Reference Section

Reference Image
Select an image containing the object of interest

Reference Label
Select its annotation file (.xml, .json, or .txt)

Classes File (YOLO only)
Required to map class names to IDs

2️⃣ Target Section

Select the folder containing raw, unlabeled images

3️⃣ Output Section

Choose output directory

Select annotation format:

YOLO

Pascal VOC

JSON

4️⃣ Processing

Adjust Score Threshold (0.8 recommended start)

Click Preview Sample to validate detection

Click Start Auto Labeling to process the full dataset

⚠️ Limitations (Read This Honestly)

This tool uses Template Matching, not Deep Learning.

Known Constraints

Scale Sensitivity
Object size should be similar to the reference

Rotation Sensitivity
Significant rotation reduces accuracy

Lighting Variations
Extreme lighting changes may affect matching

Ideal Use Cases

Fixed camera setups

Assembly lines

Industrial inspection

Screenshots / UI datasets

Standardized data collection environments

If your dataset has heavy variation → YOLO / Detectron is a better fit.
This tool is about speed and practicality, not hype.

🔮 Future Scope

Planned enhancements:

Multi-scale template matching

Multiple reference templates per class

Manual review & correction tab

Heatmap-based match visualization

Hybrid CV + lightweight ML approach

📜 License

This project is licensed under the MIT License.
You are free to use, modify, and distribute it with attribution.

👨‍💻 Author

Varad Kulkarni
Applied AI / Computer Vision Developer

If this tool helped you reduce annotation time, ⭐ the repo — it genuinely helps.
